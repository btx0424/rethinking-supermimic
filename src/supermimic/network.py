"""World model, motion encoder, shared decoder, and critic.

Ablations, off by default: world-model normalization (none, layer, rms),
residual blocks, and a state-independent Gaussian actor.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from typing import NamedTuple

from .math_utils import quat_apply, quat_mul, quat_normalize, rotvec_to_quat, yaw_quat
from .mdp import GOAL_DIM, INV_DIM, NUM_J, PRIV_DIM, XI_DIM, invariant_features

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
        self.priv = RunningNorm(PRIV_DIM)

    def wm(self, inv: torch.Tensor, action: torch.Tensor, xi_n: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.state(inv), self.action(action), self.xi(xi_n)), dim=-1)

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


class WorldModelState(NamedTuple):
    """Root pose and twist in the world frame, plus the 29 hinge joints."""

    p: torch.Tensor
    """World position, (..., 3)."""
    quat: torch.Tensor
    """World orientation as wxyz, (..., 4)."""
    v: torch.Tensor
    """World linear velocity, (..., 3)."""
    w: torch.Tensor
    """World angular velocity, (..., 3)."""
    q: torch.Tensor
    """Joint positions, (..., 29)."""
    qd: torch.Tensor
    """Joint velocities, (..., 29)."""


class WorldModel(nn.Module):
    """Deterministic dynamics. The network predicts a frame-local increment."""

    out_dim = 3 + 3 + 3 + 3 + NUM_J + NUM_J

    def __init__(self, norm: str = "none", residual: bool = False):
        super().__init__()
        from .mdp import INV_DIM, XI_DIM

        self.net = MLP(INV_DIM + NUM_J + XI_DIM, [2048, 2048, 2048], self.out_dim, norm, residual)
        nn.init.zeros_(self.net.head.weight)
        nn.init.zeros_(self.net.head.bias)

    def step(self, state: WorldModelState, action: torch.Tensor, xi_n: torch.Tensor, norms: InputNorms) -> WorldModelState:
        p, quat, v, w, q, qd = state
        feat = norms.wm(invariant_features(p, quat, v, w, q, qd), action, xi_n)
        dp, drot, dv, dw, dq, dqd = self.net(feat).split((3, 3, 3, 3, NUM_J, NUM_J), dim=-1)
        yaw = yaw_quat(quat)
        return WorldModelState(
            p + quat_apply(yaw, dp),
            quat_normalize(quat_mul(quat, rotvec_to_quat(drot))),
            v + quat_apply(quat, dv),
            w + quat_apply(quat, dw),
            q + dq,
            qd + dqd,
        )


# Table A.4 encoder and decoder widths. Depth is the length of the list.
PAPER_POLICY_HIDDEN = (2048, 1024, 512, 256)


# Single trunk: goal and proprioceptive history concatenated. No tanh code.
TRUNK_POLICY_HIDDEN = (512, 512, 512)


def policy_widths(saved: dict, key: str) -> list[int]:
    """Read encoder or decoder widths from a checkpoint config, with the paper default."""
    return [int(width) for width in saved.get(key, PAPER_POLICY_HIDDEN)]


def build_policy(obs_dim: int, goal_dim: int, hist: int, saved: dict) -> Policy:
    """Rebuild a policy from a checkpoint config. Missing ``policy_kind`` is the split actor."""
    kind = str(saved.get("policy_kind", "split"))
    norm = str(saved.get("policy_norm", "none"))
    stochastic = bool(saved.get("stochastic_actor", False))
    if kind == "trunk":
        hidden = [int(width) for width in saved.get("policy_trunk", TRUNK_POLICY_HIDDEN)]
        return Policy(
            obs_dim, goal_dim, hist, stochastic=stochastic, norm=norm, kind="trunk", trunk_hidden=hidden
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


class Policy(nn.Module):
    """Actor. ``split`` is the tanh encoder plus decoder. ``trunk`` is one MLP on goal and history."""

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
        if kind not in ("split", "trunk"):
            raise ValueError(f"Policy kind must be 'split' or 'trunk', got {kind!r}.")
        self.hist = hist
        self.stochastic = stochastic
        self.kind = kind
        if kind == "trunk":
            self.trunk = MLP(goal_dim + obs_dim * hist, list(trunk_hidden), NUM_J, norm)
            nn.init.zeros_(self.trunk.head.bias)
            with torch.no_grad():
                self.trunk.head.weight.mul_(0.01)
            self.encoder = None
            self.decoder = None
        else:
            self.encoder = MLP(goal_dim, list(encoder_hidden), z_dim, norm)
            self.decoder = MLP(z_dim + obs_dim * hist, list(decoder_hidden), NUM_J, norm)
            nn.init.zeros_(self.decoder.head.bias)
            with torch.no_grad():
                self.decoder.head.weight.mul_(0.01)
            self.trunk = None
        self.log_std = nn.Parameter(torch.full((NUM_J,), -1.0)) if stochastic else None

    def forward(self, obs_hist: torch.Tensor, goal: torch.Tensor, norms: InputNorms, deterministic: bool = True) -> torch.Tensor:
        obs = norms.obs(obs_hist).flatten(-2)
        if self.trunk is not None:
            mean = self.trunk(torch.cat((norms.goal(goal), obs), dim=-1))
        else:
            z = torch.tanh(self.encoder(norms.goal(goal)))
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
    def __init__(self, in_dim: int, norm: str = "layer"):
        super().__init__()
        self.net = MLP(in_dim, [512, 256, 256], 1, norm=norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
