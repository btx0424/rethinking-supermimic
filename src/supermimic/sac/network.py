from __future__ import annotations

import math

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from supermimic.sac.util import IndependentNormal


class MultiStepReturn(nn.Module):
    """Compute n-step return and fixed-horizon action stacks.

    Always consumes ``actions`` with shape ``[T, N, act_dim]`` and returns
    stacked actions with shape ``[N, T, act_dim]`` where steps beyond episode
    end are filled by repeating the terminal action.
    """

    def __init__(self, gamma: float, n_steps: int):
        super().__init__()
        self.n_steps = n_steps
        self.register_buffer("gamma", torch.tensor(gamma))
        self.gamma: torch.Tensor

    def forward(
        self,
        actions: torch.Tensor,
        next_observations: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        done: torch.Tensor,
        env_discount: torch.Tensor | None,
    ):
        T, N = next_observations.shape[:2]
        assert T == self.n_steps

        device = rewards.device
        dtype = rewards.dtype
        gamma = self.gamma.to(device=device, dtype=dtype)

        done_bool = done.bool()
        cum_not_done = (~done_bool).cumprod(dim=0)

        step_discount = torch.full((T, N), gamma, device=device, dtype=dtype)
        if env_discount is not None:
            step_discount = step_discount * env_discount.squeeze(-1).to(device=device, dtype=dtype)

        gammas_cumprod = step_discount.cumprod(dim=0)
        gammas_pow = torch.cat(
            [torch.ones(1, N, device=device, dtype=dtype), gammas_cumprod[:-1]],
            dim=0,
        ).unsqueeze(-1)

        cum_reward = (rewards * gammas_pow).cumsum(dim=0)
        alive_steps = cum_not_done.sum(dim=0)

        last_indices = alive_steps.clamp_max(self.n_steps - 1).reshape(N)
        batch_indices = torch.arange(N, device=device)

        next_observations = next_observations[last_indices, batch_indices]
        rewards_out = cum_reward[last_indices, batch_indices]
        terminated = terminated[last_indices, batch_indices]

        discount = (
            gammas_cumprod[last_indices, batch_indices]
            .reshape_as(terminated.float())
            * (1.0 - terminated.float())
        )

        assert actions.shape[0] == T and actions.shape[1] == N
        actions_out = einops.rearrange(actions, "t n d -> n t d").contiguous()
        last_actions = actions[last_indices, batch_indices]
        t_idx = torch.arange(T, device=device).unsqueeze(0)
        repeat_mask = t_idx > last_indices.unsqueeze(1)
        actions_out = torch.where(
            repeat_mask.unsqueeze(-1), last_actions.unsqueeze(1), actions_out
        )
        return actions_out, next_observations, rewards_out, discount, terminated


def _init_sac_linear(m: nn.Module, gain: float = 1.0) -> None:
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)
        nn.init.zeros_(m.bias)


class ConditionalBlock(nn.Module):
    """Residual block with optional RMS / layer norm (active-adaptation ``sac`` / sac_rlpd)."""

    def __init__(
        self,
        hidden_dim: int,
        expansion: int = 1,
        condition_dim: int = 0,
        norm: str | None = None,
        activation: type[nn.Module] = nn.SiLU,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        if norm is None:
            self.norm = nn.Identity()
        elif norm.lower() == "rms":
            self.norm = nn.RMSNorm(hidden_dim)
        elif norm.lower() == "layer":
            self.norm = nn.LayerNorm(hidden_dim)
        else:
            raise ValueError(f"Invalid norm: {norm}")

        act = activation
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion),
            act(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim * expansion, hidden_dim),
            act(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )
        self.cond_proj = nn.Linear(condition_dim, 2 * hidden_dim) if condition_dim > 0 else None

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        if self.cond_proj is not None:
            assert cond is not None
            scale, shift = self.cond_proj(cond).chunk(2, dim=-1)
            x = x * (1.0 + scale) + shift
        return self.layers(x) + residual


class CriticTrunk(nn.Module):
    """Twin-Q trunk (``active_adaptation.learning.offpolicy.sac.CriticTrunk``)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        output_dim: int = 1,
        activation: type[nn.Module] = nn.SiLU,
        norm: str | None = "rms",
        condition_dim: int = 0,
    ):
        super().__init__()
        self.in_layer = nn.Linear(input_dim, hidden_dim)
        self.in_layer.weight._non_muon = True
        self.out_layer = nn.Linear(hidden_dim, output_dim)
        self.out_layer.weight._non_muon = True
        self.block1 = ConditionalBlock(
            hidden_dim=hidden_dim, activation=activation, norm=norm, condition_dim=condition_dim
        )
        self.block2 = ConditionalBlock(
            hidden_dim=hidden_dim, activation=activation, norm=norm, condition_dim=condition_dim
        )
        self.norm = nn.RMSNorm(hidden_dim)
        self.apply(_init_sac_linear)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        x = self.in_layer(x)
        x = self.block1(x, cond)
        x = self.block2(x, cond)
        x = self.norm(x)
        return self.out_layer(x)


class NormalActor(nn.Module):
    """Gaussian policy trunk matching sac_rlpd / AA ``NormalActor``."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        std_max: float = 1.0,
        std_min: float = 0.001,
        action_init: str = "zeros",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.in_layer = nn.Linear(obs_dim, 384)
        self.in_layer.weight._non_muon = True
        self.trunk = nn.Sequential(
            ConditionalBlock(hidden_dim=384, condition_dim=0, norm="rms"),
            ConditionalBlock(hidden_dim=384, condition_dim=0, norm="rms"),
            nn.RMSNorm(384),
        )
        self.action = nn.Linear(384, act_dim * 2)
        self.action.weight._non_muon = True
        self.trunk.apply(_init_sac_linear)
        if action_init == "orthogonal":
            self.action.apply(lambda m: _init_sac_linear(m, gain=0.01))
        elif action_init == "zeros":
            nn.init.zeros_(self.action.weight)
            nn.init.zeros_(self.action.bias)
        else:
            raise ValueError(f"Unknown action_init {action_init!r}")
        if not std_max > 0.0:
            raise ValueError("std_max must be positive")
        self.log_std_max = math.log(std_max)
        self.log_std_min = math.log(std_min)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.trunk(self.in_layer(obs))
        mean, raw = self.action(feat).chunk(2, dim=-1)
        log_std = self.log_std_min + (self.log_std_max - self.log_std_min) * 0.5 * (1 + torch.tanh(raw))
        return mean, torch.exp(log_std)


class AlphaModule(nn.Module):
    """Learnable SAC temperature α = exp(log_α)."""

    def __init__(self, init_value: float = 1.0):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.tensor(math.log(init_value)))

    def forward(self) -> torch.Tensor:
        return self.log_alpha.exp()


def hl_gauss(values: torch.Tensor, support: torch.Tensor, sigma_ratio: float = 0.75) -> torch.Tensor:
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


class _TwinQTrunk(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.q1 = CriticTrunk(in_dim, hidden_dim=hidden_dim, output_dim=out_dim)
        self.q2 = CriticTrunk(in_dim, hidden_dim=hidden_dim, output_dim=out_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(x), self.q2(x)


class ScalarCritic(nn.Module):
    """Clipped double-Q regression."""

    def __init__(self, obs_dim: int, act_dim: int, ensemble: int = 2, huber_delta: float = 1.0):
        super().__init__()
        if ensemble != 2:
            raise ValueError("ScalarCritic currently supports ensemble=2 (twin Q).")
        self.ensemble = ensemble
        self.huber_delta = float(huber_delta)
        self.trunk = _TwinQTrunk(obs_dim + act_dim, 1)

    def _concat(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        if act.dim() == 2:
            return torch.cat([obs, act], dim=-1)
        if act.dim() == 3:
            b, k, _ = act.shape
            obs_flat = einops.repeat(obs, "b d -> (b k) d", k=k)
            act_flat = einops.rearrange(act, "b k d -> (b k) d")
            return torch.cat([obs_flat, act_flat], dim=-1)
        raise ValueError(f"act must be rank 2 or 3, got {tuple(act.shape)}")

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = self._concat(obs, act)
        q1, q2 = self.trunk(x)
        if act.dim() == 3:
            b, k, _ = act.shape
            q1 = einops.rearrange(q1.squeeze(-1), "(b k) -> b k", b=b, k=k)
            q2 = einops.rearrange(q2.squeeze(-1), "(b k) -> b k", b=b, k=k)
            return torch.stack([q1, q2], dim=-1)
        return torch.cat([q1, q2], dim=-1)

    def get_values(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.forward(obs, act)

    def compute_target(
        self,
        next_obs: torch.Tensor,
        next_act: torch.Tensor,
        reward: torch.Tensor,
        discount: torch.Tensor,
    ) -> torch.Tensor:
        qs = self(next_obs, next_act)
        q = qs.min(dim=-1, keepdim=True).values
        return reward + discount * q

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.float()
        target = target.float().expand_as(pred)
        err = pred - target
        if self.huber_delta > 0:
            return F.huber_loss(pred, target, reduction="none", delta=self.huber_delta).sum(dim=-1)
        return err.square().sum(dim=-1)


class ClassificationCritic(nn.Module):
    """HL-Gauss distributional critic with twin heads."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        ensemble: int = 2,
        num_atoms: int = 51,
        v_min: float = -2.0,
        v_max: float = 15.0,
        sigma_ratio: float = 0.75,
    ):
        super().__init__()
        if ensemble != 2:
            raise ValueError("ClassificationCritic currently supports ensemble=2.")
        self.ensemble = ensemble
        self.num_atoms = int(num_atoms)
        self.sigma_ratio = float(sigma_ratio)
        self.trunk = _TwinQTrunk(obs_dim + act_dim, num_atoms)
        self.register_buffer("support", torch.linspace(v_min, v_max, num_atoms))

    def _concat(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        if act.dim() == 2:
            return torch.cat([obs, act], dim=-1)
        if act.dim() == 3:
            b, k, _ = act.shape
            obs_flat = einops.repeat(obs, "b d -> (b k) d", k=k)
            act_flat = einops.rearrange(act, "b k d -> (b k) d")
            return torch.cat([obs_flat, act_flat], dim=-1)
        raise ValueError(f"act must be rank 2 or 3, got {tuple(act.shape)}")

    def _logits(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = self._concat(obs, act)
        l1, l2 = self.trunk(x)
        if act.dim() == 3:
            b, k, _ = act.shape
            l1 = einops.rearrange(l1, "(b k) a -> b k a", b=b, k=k)
            l2 = einops.rearrange(l2, "(b k) a -> b k a", b=b, k=k)
            return torch.stack([l1, l2], dim=-2)
        return torch.stack([l1, l2], dim=-2)

    def expected_values(self, logits: torch.Tensor) -> torch.Tensor:
        probs = logits.softmax(dim=-1)
        z = self.support.to(device=logits.device, dtype=logits.dtype)
        return (probs * z).sum(dim=-1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self._logits(obs, act)

    def get_values(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.expected_values(self._logits(obs, act))

    def compute_target(
        self,
        next_obs: torch.Tensor,
        next_act: torch.Tensor,
        reward: torch.Tensor,
        discount: torch.Tensor,
    ) -> torch.Tensor:
        values = self.get_values(next_obs, next_act)
        q = values.min(dim=-1, keepdim=True).values
        y = (reward + discount * q).reshape(reward.shape[0], -1).squeeze(-1)
        return hl_gauss(y, self.support, self.sigma_ratio)

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch = pred.shape[0]
        target = target.float()
        if target.shape[-1] != self.num_atoms:
            y = target.reshape(batch, -1).squeeze(-1)
            target = hl_gauss(y, self.support, self.sigma_ratio)
        target = target.unsqueeze(-2)
        log_p = pred.float().log_softmax(dim=-1).clamp(min=-30.0)
        return -(target * log_p).sum(dim=(-1, -2))


class C51Critic(ClassificationCritic):
    """Alias: categorical backup uses the HL-Gauss head in this port."""


def make_critic(
    kind: str,
    state_dim: int,
    act_dim: int,
    *,
    ensemble: int = 2,
    num_atoms: int = 51,
    v_min: float = -2.0,
    v_max: float = 15.0,
    sigma_ratio: float = 0.75,
) -> nn.Module:
    if kind == "scalar":
        return ScalarCritic(state_dim, act_dim, ensemble=ensemble)
    if kind in ("classification", "c51"):
        return ClassificationCritic(
            state_dim,
            act_dim,
            ensemble=ensemble,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
            sigma_ratio=sigma_ratio,
        )
    raise ValueError(f"Unknown critic kind {kind!r}")


def rollout_action(actor: NormalActor, obs: torch.Tensor, dist_cls=IndependentNormal) -> torch.Tensor:
    loc, scale = actor(obs)
    return dist_cls(loc, scale).sample()
