"""Running return-based reward normalization (FlashSAC / PopArt-style scale).

Tracks a discounted return per env, keeps a running variance of those returns,
and divides rewards by ``max(√Var(G), G_r_max / G_max)`` so policy gradients
are roughly invariant to a global reward scale.
"""

from __future__ import annotations

import torch


class RunningMeanStd:
    """Welford / parallel mean-variance tracker."""

    def __init__(self, device: torch.device | str, epsilon: float = 1e-4, shape: tuple[int, ...] = ()):
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = torch.tensor(epsilon, dtype=torch.float32, device=device)
        self.epsilon = epsilon

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.detach().reshape(-1, *self.mean.shape).float()
        if x.shape[0] == 0:
            return
        batch_mean = x.mean(0)
        batch_var = x.var(0, unbiased=False)
        batch_count = float(x.shape[0])
        delta = batch_mean - self.mean
        total = self.count + batch_count
        ratio = batch_count / total
        self.mean = self.mean + delta * ratio
        m_a = self.var * (self.count + self.epsilon)
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.square() * self.count * ratio
        self.var = m2 / total
        self.count = total

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.mean = state["mean"]
        self.var = state["var"]
        self.count = state["count"]


class RewardNormalizer:
    """Normalize rewards by the running std of discounted returns."""

    def __init__(
        self,
        num_envs: int,
        gamma: float,
        g_max: float,
        device: torch.device | str = "cpu",
        eps: float = 1e-8,
    ):
        """``g_max`` is the MDP bound on |discounted return|, e.g. ``r_max / (1-γ)``."""
        self.gamma = gamma
        self.g_max = g_max
        self.eps = eps
        self.device = torch.device(device)
        self.g_r = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self.g_r_max = torch.zeros((), dtype=torch.float32, device=self.device)
        self.g_rms = RunningMeanStd(device=self.device, shape=())

    @torch.no_grad()
    def update(self, reward: torch.Tensor, done: torch.Tensor) -> None:
        """Advance per-env discounted returns from a real simulator step."""
        reward = reward.detach().float().reshape(-1)
        done = done.detach().float().reshape(-1)
        if self.g_r.shape[0] != reward.shape[0]:
            self.g_r = torch.zeros(reward.shape[0], dtype=torch.float32, device=self.device)
        self.g_r = self.gamma * (1.0 - done) * self.g_r + reward
        self.g_r_max = torch.maximum(self.g_r_max, self.g_r.abs().max())
        self.g_rms.update(self.g_r)

    def scale(self) -> torch.Tensor:
        """Current reward divisor (detached scalar)."""
        var_den = torch.sqrt(self.g_rms.var + self.eps)
        floor = self.g_r_max / self.g_max
        return torch.maximum(var_den, floor).clamp_min(self.eps)

    def normalize(self, rewards: torch.Tensor) -> torch.Tensor:
        """Divide rewards by the running return scale. Gradients flow through ``rewards``."""
        return rewards / self.scale()

    def state_dict(self) -> dict:
        return {
            "g_r": self.g_r,
            "g_r_max": self.g_r_max,
            "g_rms": self.g_rms.state_dict(),
            "gamma": self.gamma,
            "g_max": self.g_max,
        }

    def load_state_dict(self, state: dict) -> None:
        self.g_r = state["g_r"]
        self.g_r_max = state["g_r_max"]
        self.g_rms.load_state_dict(state["g_rms"])
        self.gamma = float(state.get("gamma", self.gamma))
        self.g_max = float(state.get("g_max", self.g_max))
