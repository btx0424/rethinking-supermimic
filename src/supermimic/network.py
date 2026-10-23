"""World model, motion encoder, shared decoder, and critic.

Ablations, off by default: world-model normalization (none, layer, rms),
residual blocks, and a state-independent Gaussian actor.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .math_utils import quat_apply, quat_mul, quat_normalize, rotvec_to_quat, yaw_quat
from .mdp import (
    BODY_POS_DIM,
    GOAL_DIM,
    INV_DIM,
    N_CONTACT,
    NUM_J,
    PRIV_DIM,
    XI_DIM,
    WorldModelState,
    heading_body_pos,
    invariant_features,
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

    def __init__(self, norm: str = "none", residual: bool = False):
        super().__init__()
        from .mdp import INV_DIM, XI_DIM

        self.net = MLP(
            INV_DIM + NUM_J + XI_DIM + BODY_POS_DIM + N_CONTACT,
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

        The network reads heading-frame body positions of the current state (from
        ``state.body_pos``, or from FK when that is missing) and predicts the
        root-and-joint increment plus the next contact. The returned state carries
        FK of that next pose. The caller passes ``sigmoid(logits)`` into the
        following step.
        """
        p, quat, v, w, q, qd = state.root()
        body_pos = state.body_pos
        if body_pos is None:
            # body_pos, _, _, _ = mdp.bodies(p, quat, v, w, q, qd)
            raise ValueError("state.body_pos is required for WorldModel.step.")
        body = heading_body_pos(p, quat, body_pos)
        feat = torch.cat(
            (norms.wm(invariant_features(p, quat, v, w, q, qd), action, xi_n, body), contact),
            dim=-1,
        )
        dp, drot, dv, dw, dq, dqd, contact_logit = self.net(feat).split(
            (3, 3, 3, 3, NUM_J, NUM_J, N_CONTACT), dim=-1
        )
        yaw = yaw_quat(quat)
        nxt_p = p + quat_apply(yaw, dp)
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


class Critic(nn.Module):
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
        return torch.stack([head(x).squeeze(-1) for head in self.heads], dim=-1)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        # Checkpoints from the single critic stored one MLP under ``net``.
        if any(key.startswith("net.") for key in state_dict) and not any(
            key.startswith("heads.") for key in state_dict
        ):
            state_dict = {
                ("heads.0." + key.removeprefix("net.")): value for key, value in state_dict.items()
            }
        return super().load_state_dict(state_dict, strict=strict, assign=assign)
