"""World model, motion encoder, shared decoder, and critic.

Ablations, off by default: world-model normalization (none, layer, rms),
residual blocks, and a state-independent Gaussian actor.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from supermimic.math_utils import quat_apply, quat_mul, quat_normalize, rotvec_to_quat, yaw_quat
from supermimic.mdp import (
    BODY_POS_DIM,
    GOAL_DIM,
    INV_DIM,
    N_CONTACT,
    NUM_J,
    PELVIS_FEAT_DIM,
    PRIV_DIM,
    WM_FRAMES,
    XI_DIM,
    WorldModelState,
    heading_body_pos,
    invariant_features,
    pelvis_body_pos,
    pelvis_features,
)

# Set by the trainer. None keeps fp32. bf16 covers the MLP GEMMs only; callers
# get fp32 back so quaternion integration stays in fp32.
AMP_DTYPE: torch.dtype | None = None


def _norm(kind: str, dim: int) -> nn.Module:
    if kind == "none":
        return nn.Identity()
    if kind == "layer":
        return nn.LayerNorm(dim)
    if kind == "rms":
        return nn.RMSNorm(dim)
    raise ValueError(f"Unknown normalization {kind!r}.")


class EnsembleLinear(nn.Module):
    """``ensemble`` independent linears in one einsum.

    Each weight is a separate ``(out, in)`` parameter, so Muon sees a matrix per
    member. Bias stays 1D and is left to AdamW. ``x`` is ``(ensemble, ..., in)``;
    the caller expands a shared input. The output is ``(ensemble, ..., out)``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        ensemble: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if ensemble < 1:
            raise ValueError(f"ensemble must be >= 1, got {ensemble}.")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.ensemble = int(ensemble)
        factory = {"device": device, "dtype": dtype}
        self.weight = nn.ParameterList(
            nn.Parameter(torch.empty(self.out_features, self.in_features, **factory))
            for _ in range(self.ensemble)
        )
        if bias:
            self.bias = nn.ParameterList(
                nn.Parameter(torch.empty(self.out_features, **factory)) for _ in range(self.ensemble)
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features) if self.in_features else 0.0
        for weight in self.weight:
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        if self.bias is not None:
            for bias in self.bias:
                nn.init.uniform_(bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.stack(tuple(self.weight), dim=0)
        y = torch.einsum("eoi,e...i->e...o", weight, x)
        if self.bias is not None:
            bias = torch.stack(tuple(self.bias), dim=0)
            y = y + bias.view(self.ensemble, *([1] * (y.dim() - 2)), self.out_features)
        return y

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"ensemble={self.ensemble}, bias={self.bias is not None}"
        )


class EnsembleLayerNorm(nn.Module):
    """``ensemble`` independent layer norms over ``normalized_shape``.

    Affine parameters are marked off Muon: a scale of shape ``(ensemble, *normalized_shape)``
    is not a hidden matrix. ``x`` is ``(ensemble, ..., *normalized_shape)``.
    """

    def __init__(
        self,
        normalized_shape: int | tuple[int, ...],
        ensemble: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if ensemble < 1:
            raise ValueError(f"ensemble must be >= 1, got {ensemble}.")
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(int(n) for n in normalized_shape)
        self.ensemble = int(ensemble)
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        factory = {"device": device, "dtype": dtype}
        if self.elementwise_affine:
            shape = (self.ensemble, *self.normalized_shape)
            self.weight = nn.Parameter(torch.empty(shape, **factory))
            self.weight._non_muon = True  # type: ignore[attr-defined]
            if bias:
                self.bias = nn.Parameter(torch.empty(shape, **factory))
                self.bias._non_muon = True  # type: ignore[attr-defined]
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(-len(self.normalized_shape), 0))
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, unbiased=False, keepdim=True)
        y = (x - mean) * torch.rsqrt(var + self.eps)
        if self.weight is not None:
            view = (self.ensemble, *([1] * (y.dim() - 1 - len(self.normalized_shape))), *self.normalized_shape)
            y = y * self.weight.view(view)
            if self.bias is not None:
                y = y + self.bias.view(view)
        return y


class EnsembleRMSNorm(nn.Module):
    """``ensemble`` independent RMS norms over ``normalized_shape``.

    ``eps=None`` uses the dtype epsilon, as ``nn.RMSNorm`` does. Affine parameters
    are marked off Muon. ``x`` is ``(ensemble, ..., *normalized_shape)``.
    """

    def __init__(
        self,
        normalized_shape: int | tuple[int, ...],
        ensemble: int,
        eps: float | None = None,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if ensemble < 1:
            raise ValueError(f"ensemble must be >= 1, got {ensemble}.")
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(int(n) for n in normalized_shape)
        self.ensemble = int(ensemble)
        self.eps = eps
        self.elementwise_affine = bool(elementwise_affine)
        if self.elementwise_affine:
            self.weight = nn.Parameter(
                torch.empty(self.ensemble, *self.normalized_shape, device=device, dtype=dtype)
            )
            self.weight._non_muon = True  # type: ignore[attr-defined]
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(-len(self.normalized_shape), 0))
        eps = torch.finfo(x.dtype).eps if self.eps is None else self.eps
        y = x * torch.rsqrt(x.pow(2).mean(dim=dims, keepdim=True) + eps)
        if self.weight is not None:
            view = (self.ensemble, *([1] * (y.dim() - 1 - len(self.normalized_shape))), *self.normalized_shape)
            y = y * self.weight.view(view)
        return y


class Block(nn.Module):
    def __init__(self, din: int, dout: int, norm: str, residual: bool | float):
        super().__init__()
        self.fc = nn.Linear(din, dout)
        self.norm = _norm(norm, dout)
        self.act = nn.SiLU()
        self.residual = float(residual) > 0.0
        if self.residual and din == dout:
            self.skip: nn.Module = nn.Identity()
            self.alpha = nn.Parameter(torch.tensor(float(residual)))
        elif self.residual:
            self.skip = nn.Linear(din, dout, bias=False)
            self.alpha = nn.Parameter(torch.tensor(float(residual)))
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.norm(self.fc(x)))
        if self.residual:
            y = y + self.alpha * self.skip(x)
        return y

class RunningNorm(nn.Module):
    """Running mean and variance. Updates do not receive gradients."""

    def __init__(self, dim: int, max_count: float = 2e8, eps: float = 1e-2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.max_count = max_count
        self.eps = eps

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.detach().reshape(-1, x.shape[-1]).float()
        if self.count >= self.max_count or x.numel() == 0:
            return
        n = x.shape[0]
        batch_mean = x.mean(0)
        batch_var = x.var(0, unbiased=False)
        total = self.count + n
        delta = batch_mean - self.mean
        self.mean += delta * n / total
        self.var = (self.var * self.count + batch_var * n + delta.square() * self.count * n / total) / total
        self.count = total

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / torch.sqrt(self.var + self.eps**2)


class InputNorms(nn.Module):
    """One running normalizer per shared input, used by every module that reads it.

    State features are shared by the world model and the critic. The goal is shared
    by the policy encoder and the critic. The observation is the policy's, the action
    is the world model's, and ξ is shared by the world model and the critic.
    """

    def __init__(self, obs_dim: int):
        super().__init__()
        self.state = RunningNorm(INV_DIM)
        self.dyn = RunningNorm(PELVIS_FEAT_DIM)
        self.obs = RunningNorm(obs_dim)
        self.goal = RunningNorm(GOAL_DIM)
        self.action = RunningNorm(NUM_J)
        self.xi = RunningNorm(XI_DIM)
        self.body = RunningNorm(BODY_POS_DIM)
        self.priv = RunningNorm(PRIV_DIM)

    def wm(
        self,
        inv: torch.Tensor,
        action: torch.Tensor,
        xi_n: torch.Tensor,
        body: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat((self.state(inv), self.action(action), self.xi(xi_n), self.body(body)), dim=-1)

    def critic(
        self,
        inv: torch.Tensor,
        goal: torch.Tensor,
        xi_n: torch.Tensor,
        priv: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            (self.state(inv), self.goal(goal), self.xi(xi_n), self.priv(priv)),
            dim=-1,
        )

    def update(self, inv: torch.Tensor, obs: torch.Tensor, goal: torch.Tensor, action: torch.Tensor, xi_n: torch.Tensor) -> None:
        self.state.update(inv)
        self.obs.update(obs)
        self.goal.update(goal)
        self.action.update(action)
        self.xi.update(xi_n)


def _mark_non_muon(linear: nn.Linear) -> None:
    linear.weight._non_muon = True  # type: ignore[attr-defined]


class MLP(nn.Module):
    def __init__(self, inp: int, hidden: list[int], out: int, norm: str = "none", residual: bool = False):
        super().__init__()
        blocks = []
        din = inp
        for width in hidden:
            blocks.append(Block(din, width, norm, residual))
            din = width
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(din if hidden else inp, out)
        # Muon is for hidden matrices. The input projection and the output head stay Adam.
        if blocks:
            _mark_non_muon(blocks[0].fc)
            if isinstance(blocks[0].skip, nn.Linear):
                _mark_non_muon(blocks[0].skip)
        _mark_non_muon(self.head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if AMP_DTYPE is None or x.device.type == "cpu":
            return self.head(self.blocks(x))
        with torch.autocast(x.device.type, dtype=AMP_DTYPE):
            return self.head(self.blocks(x)).float()


class WorldModel(nn.Module):
    """Deterministic dynamics. The network predicts a frame-local increment."""

    out_dim = 3 + 3 + 3 + 3 + NUM_J + NUM_J + N_CONTACT

    def __init__(self, norm: str = "none", residual: bool = False, frame: str = "heading"):
        super().__init__()
        if frame not in WM_FRAMES:
            raise ValueError(f"wm frame must be one of {WM_FRAMES}, got {frame!r}.")
        self.frame = frame
        from supermimic.mdp import XI_DIM

        feat_dim = PELVIS_FEAT_DIM if frame == "pelvis" else INV_DIM
        self.net = MLP(
            feat_dim + NUM_J + XI_DIM + BODY_POS_DIM + N_CONTACT,
            [2048, 2048, 2048],
            self.out_dim,
            norm,
            residual,
        )
        nn.init.zeros_(self.net.head.weight)
        nn.init.zeros_(self.net.head.bias)

    def step(
        self,
        state: WorldModelState,
        action: torch.Tensor,
        xi_n: torch.Tensor,
        norms: InputNorms,
        contact: torch.Tensor,
        mdp,
    ) -> tuple[WorldModelState, torch.Tensor]:
        """Advance one step. ``contact`` is a per-body probability in ``[0, 1]``.

        The network reads body positions of the current state in ``self.frame``
        (heading or the full pelvis) and predicts the root-and-joint increment
        plus the next contact. The returned state carries FK of that next pose.
        The caller passes ``sigmoid(logits)`` into the following step.
        """
        p, quat, v, w, q, qd = state.root()
        body_pos = state.body_pos
        if body_pos is None:
            raise ValueError("state.body_pos is required for WorldModel.step.")
        if self.frame == "pelvis":
            feat_in = pelvis_features(p, quat, v, w, q, qd)
            body = pelvis_body_pos(p, quat, body_pos)
            state_n = norms.dyn(feat_in)
            rotate = quat
        else:
            feat_in = invariant_features(p, quat, v, w, q, qd)
            body = heading_body_pos(p, quat, body_pos)
            state_n = norms.state(feat_in)
            rotate = yaw_quat(quat)
        feat = torch.cat((state_n, norms.action(action), norms.xi(xi_n), norms.body(body), contact), dim=-1)
        dp, drot, dv, dw, dq, dqd, contact_logit = self.net(feat).split(
            (3, 3, 3, 3, NUM_J, NUM_J, N_CONTACT), dim=-1
        )
        nxt_p = p + quat_apply(rotate, dp)
        nxt_quat = quat_normalize(quat_mul(quat, rotvec_to_quat(drot)))
        nxt_v = v + quat_apply(quat, dv)
        nxt_w = w + quat_apply(quat, dw)
        nxt_q = q + dq
        nxt_qd = qd + dqd
        nxt_body_pos, nxt_body_quat, _, _ = mdp.bodies(nxt_p, nxt_quat, nxt_v, nxt_w, nxt_q, nxt_qd)
        nxt = WorldModelState(
            nxt_p, nxt_quat, nxt_v, nxt_w, nxt_q, nxt_qd, nxt_body_pos, nxt_body_quat
        )
        return nxt, contact_logit


# Table A.4 encoder and decoder widths. Depth is the length of the list.
PAPER_POLICY_HIDDEN = (2048, 1024, 512, 256)


# Single trunk: goal and proprioceptive history concatenated. No tanh code.
TRUNK_POLICY_HIDDEN = (512, 512, 512)

# Goal embedding for per-layer scale and shift. The proprioceptive trunk stays at TRUNK_POLICY_HIDDEN.
FILM_GOAL_HIDDEN = 256


def policy_widths(saved: dict, key: str) -> list[int]:
    """Read encoder or decoder widths from a checkpoint config, with the paper default."""
    return [int(width) for width in saved.get(key, PAPER_POLICY_HIDDEN)]


def build_policy(obs_dim: int, goal_dim: int, hist: int, saved: dict) -> Policy:
    """Rebuild a policy from a checkpoint config. Missing ``policy_kind`` is the split actor."""
    kind = str(saved.get("policy_kind", "split"))
    norm = str(saved.get("policy_norm", "none"))
    stochastic = bool(saved.get("stochastic_actor", False))
    if kind in ("trunk", "film"):
        hidden = [int(width) for width in saved.get("policy_trunk", TRUNK_POLICY_HIDDEN)]
        return Policy(
            obs_dim, goal_dim, hist, stochastic=stochastic, norm=norm, kind=kind, trunk_hidden=hidden
        )
    return Policy(
        obs_dim,
        goal_dim,
        hist,
        z_dim=int(saved.get("policy_z_dim", 32)),
        stochastic=stochastic,
        encoder_hidden=policy_widths(saved, "policy_encoder"),
        decoder_hidden=policy_widths(saved, "policy_decoder"),
        norm=norm,
        kind="split",
    )


class FiLMTrunk(nn.Module):
    """Proprioceptive trunk. Each hidden layer is scaled and shifted by the goal.

    The film weights start at zero, so a fresh network is an ordinary MLP on the
    observation history. ``h ← (1 + γ(goal)) ⊙ h + β(goal)`` after the layer norm.
    """

    def __init__(self, obs_in: int, goal_dim: int, hidden: list[int], norm: str):
        super().__init__()
        self.goal_fc = nn.Linear(goal_dim, FILM_GOAL_HIDDEN)
        self.goal_act = nn.SiLU()
        _mark_non_muon(self.goal_fc)
        self.fcs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.films = nn.ModuleList()
        din = obs_in
        for i, width in enumerate(hidden):
            fc = nn.Linear(din, width)
            if i == 0:
                _mark_non_muon(fc)
            self.fcs.append(fc)
            self.norms.append(_norm(norm, width))
            film = nn.Linear(FILM_GOAL_HIDDEN, 2 * width)
            nn.init.zeros_(film.weight)
            nn.init.zeros_(film.bias)
            _mark_non_muon(film)
            self.films.append(film)
            din = width
        self.act = nn.SiLU()
        self.head = nn.Linear(din, NUM_J)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.weight.mul_(0.01)
        _mark_non_muon(self.head)

    def forward(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        def run() -> torch.Tensor:
            g = self.goal_act(self.goal_fc(goal))
            x = obs
            for fc, norm, film in zip(self.fcs, self.norms, self.films):
                y = norm(fc(x))
                scale, shift = film(g).chunk(2, dim=-1)
                x = self.act(y * (1.0 + scale) + shift)
            return self.head(x)

        if AMP_DTYPE is None or obs.device.type == "cpu":
            return run()
        with torch.autocast(obs.device.type, dtype=AMP_DTYPE):
            return run().float()


class Policy(nn.Module):
    """Actor. ``split`` is the tanh encoder plus decoder. ``trunk`` concatenates the goal. ``film`` modulates each layer with it."""

    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        hist: int = 10,
        z_dim: int = 32,
        stochastic: bool = False,
        encoder_hidden: list[int] | tuple[int, ...] = PAPER_POLICY_HIDDEN,
        decoder_hidden: list[int] | tuple[int, ...] = PAPER_POLICY_HIDDEN,
        trunk_hidden: list[int] | tuple[int, ...] = TRUNK_POLICY_HIDDEN,
        norm: str = "none",
        kind: str = "split",
    ):
        super().__init__()
        if kind not in ("split", "trunk", "film"):
            raise ValueError(f"Policy kind must be 'split', 'trunk', or 'film', got {kind!r}.")
        self.hist = hist
        self.stochastic = stochastic
        self.kind = kind
        self.encoder = None
        self.decoder = None
        self.trunk = None
        self.film = None
        if kind == "trunk":
            self.trunk = MLP(goal_dim + obs_dim * hist, list(trunk_hidden), NUM_J, norm)
            nn.init.zeros_(self.trunk.head.bias)
            with torch.no_grad():
                self.trunk.head.weight.mul_(0.01)
        elif kind == "film":
            self.film = FiLMTrunk(obs_dim * hist, goal_dim, list(trunk_hidden), norm)
        else:
            self.encoder = MLP(goal_dim, list(encoder_hidden), z_dim, norm)
            self.decoder = MLP(z_dim + obs_dim * hist, list(decoder_hidden), NUM_J, norm)
            nn.init.zeros_(self.decoder.head.bias)
            with torch.no_grad():
                self.decoder.head.weight.mul_(0.01)
        self.log_std = nn.Parameter(torch.full((NUM_J,), -1.0)) if stochastic else None

    def forward(self, obs_hist: torch.Tensor, goal: torch.Tensor, norms: InputNorms, deterministic: bool = True) -> torch.Tensor:
        obs = norms.obs(obs_hist).flatten(-2)
        goal_n = norms.goal(goal)
        if self.film is not None:
            mean = self.film(obs, goal_n)
        elif self.trunk is not None:
            mean = self.trunk(torch.cat((goal_n, obs), dim=-1))
        else:
            z = torch.tanh(self.encoder(goal_n))
            mean = self.decoder(torch.cat((z, obs), dim=-1))
        if self.log_std is not None and not deterministic:
            return mean + self.log_std.exp() * torch.randn_like(mean)
        return mean

    def entropy(self) -> torch.Tensor:
        """Entropy of the state-independent diagonal Gaussian (sum over action dims)."""
        if self.log_std is None:
            raise RuntimeError("Entropy is only defined for the stochastic actor.")
        return (self.log_std + 0.5 * (1.0 + math.log(2.0 * math.pi))).sum()


class ScalarCritic(nn.Module):
    """Independent value heads. Callers average them only for the policy bootstrap."""

    def __init__(
        self,
        in_dim: int,
        norm: str = "layer",
        ensemble: int | None = None,
    ):
        super().__init__()
        n = 2 if ensemble is None else int(ensemble)
        if n < 1:
            raise ValueError(f"critic ensemble must be >= 1, got {n}")
        self.ensemble = n
        self.heads = nn.ModuleList(MLP(in_dim, [512, 256, 256], 1, norm=norm) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Values with the ensemble on the last axis, shape ``(*, ensemble)``."""
        return critic_forward(self, x)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        # Checkpoints from the single critic stored one MLP under ``net``.
        if any(key.startswith("net.") for key in state_dict) and not any(
            key.startswith("heads.") for key in state_dict
        ):
            state_dict = {
                ("heads.0." + key.removeprefix("net.")): value for key, value in state_dict.items()
            }
        return super().load_state_dict(state_dict, strict=strict, assign=assign)
    
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (pred - target).square().mean()

    def expected_values(self, pred: torch.Tensor) -> torch.Tensor:
        return pred


def hl_gauss(values: torch.Tensor, support: torch.Tensor, sigma_ratio: float = 0.75) -> torch.Tensor:
    """Soft histogram of a scalar onto bin centers. ``values`` is ``(...,)``, result ``(..., atoms)``."""
    z = support.to(device=values.device, dtype=values.dtype)
    delta = (z[-1] - z[0]) / (z.numel() - 1)
    sigma = (sigma_ratio * delta).clamp_min(torch.finfo(values.dtype).tiny * 256)
    center = values.unsqueeze(-1).clamp(z[0], z[-1])
    scale = sigma * math.sqrt(2.0)
    half = 0.5 * delta
    cdf_hi = 0.5 * (1.0 + torch.erf((z + half - center) / scale))
    cdf_lo = 0.5 * (1.0 + torch.erf((z - half - center) / scale))
    probs = (cdf_hi - cdf_lo).clamp_min(0.0)
    return probs / probs.sum(-1, keepdim=True).clamp_min(1e-8)


class ClassificationCritic(nn.Module):
    """HL-Gauss value classifier (arXiv 2403.03950).

    Each head outputs atom logits. The loss is cross-entropy against the
    histogram of the scalar target. GAE and the policy bootstrap read
    ``expected_values``.
    """

    def __init__(
        self,
        in_dim: int,
        norm: str = "layer",
        ensemble: int = 2,
        num_atoms: int = 51,
        v_min: float = -2.0,
        v_max: float = 15.0,
        sigma_ratio: float = 0.75,
    ):
        super().__init__()
        n = int(ensemble)
        atoms = int(num_atoms)
        if n < 1:
            raise ValueError(f"critic ensemble must be >= 1, got {n}")
        if atoms < 2:
            raise ValueError(f"num_atoms must be >= 2, got {atoms}")
        if not v_min < v_max:
            raise ValueError(f"v_min must be below v_max, got {v_min}, {v_max}")
        self.ensemble = n
        self.num_atoms = atoms
        self.sigma_ratio = float(sigma_ratio)
        self.heads = nn.ModuleList(MLP(in_dim, [512, 256, 256], atoms, norm=norm) for _ in range(n))
        self.register_buffer("support", torch.linspace(v_min, v_max, atoms))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expected values, shape ``(*, ensemble)``. The loss path uses the logits."""
        return self.expected_values(critic_forward(self, x))

    def expected_values(self, logits: torch.Tensor) -> torch.Tensor:
        """``logits`` is ``(..., ensemble, atoms)``. Returns ``(..., ensemble)``."""
        probs = logits.softmax(dim=-1)
        z = self.support.to(device=logits.device, dtype=logits.dtype)
        return (probs * z).sum(dim=-1)

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Cross-entropy. ``pred`` is logits ``(N, ensemble, atoms)``, ``target`` is ``(N, ensemble)``."""
        dist = hl_gauss(target, self.support, self.sigma_ratio)
        log_probs = pred.log_softmax(dim=-1)
        return -(dist * log_probs).sum(dim=-1).mean()


def critic_forward(critic: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Both heads under one autocast, so a compiled call sees every GEMM.

    Each ``MLP.forward`` opens its own autocast, which splits the graph. This
    runs the blocks directly and casts back to fp32 once.
    """

    def values(x: torch.Tensor) -> torch.Tensor:
        outs = [head.head(head.blocks(x)) for head in critic.heads]
        if getattr(critic, "num_atoms", None):
            return torch.stack(outs, dim=-2)
        return torch.stack([out.squeeze(-1) for out in outs], dim=-1)

    if AMP_DTYPE is None or x.device.type == "cpu":
        return values(x)
    with torch.autocast(x.device.type, dtype=AMP_DTYPE):
        return values(x).float()
