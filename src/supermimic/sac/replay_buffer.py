"""Ring replay storage using plain dicts of tensors (no tensordict)."""

from __future__ import annotations

import torch


class ReplayBuffer:
    """Storage shape ``[max_size, num_envs, …]`` for each field.

    Only ``obs`` and ``critic`` at the transition start are stored. With
    ``sample(..., next_obs=True)``, ``next_obs`` / ``next_critic`` are rebuilt from
    the following ring row, with **fake bootstrap**: on ``done``, the successor is
    the same row's current features (not the post-reset state).
    """

    def __init__(
        self,
        max_size: int,
        num_envs: int,
        obs_dim: int,
        critic_dim: int,
        act_dim: int,
        device: torch.device | str,
    ):
        self.max_size = int(max_size)
        self.num_envs = int(num_envs)
        self.obs_dim = int(obs_dim)
        self.critic_dim = int(critic_dim)
        self.act_dim = int(act_dim)
        self.device = torch.device(device)
        shape = (self.max_size, self.num_envs)
        dev = self.device
        self.obs = torch.zeros(*shape, obs_dim, device=dev)
        self.critic = torch.zeros(*shape, critic_dim, device=dev)
        self.action = torch.zeros(*shape, act_dim, device=dev)
        self.reward = torch.zeros(*shape, 1, device=dev)
        self.terminated = torch.zeros(*shape, 1, device=dev)
        self.done = torch.zeros(*shape, 1, device=dev)
        self.is_init = torch.zeros(*shape, 1, device=dev)
        self._ptr = 0
        self._size = 0

    @staticmethod
    def _bootstrap(current: torch.Tensor, successor: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        mask = done.bool()
        if mask.dim() == current.dim() - 1:
            mask = mask.unsqueeze(-1)
        return torch.where(mask, current, successor)

    def __len__(self) -> int:
        return self._size

    @property
    def num_samples(self) -> int:
        return self._size * self.num_envs

    def push(self, batch: dict[str, torch.Tensor]) -> None:
        """``batch`` tensors are ``[num_envs, …]``."""
        i = self._ptr
        self.obs[i] = batch["obs"]
        self.critic[i] = batch["critic"]
        self.action[i] = batch["action"]
        self.reward[i] = batch["reward"].reshape(self.num_envs, 1)
        self.terminated[i] = batch["terminated"].reshape(self.num_envs, 1).float()
        self.done[i] = batch["done"].reshape(self.num_envs, 1).float()
        self.is_init[i] = batch["is_init"].reshape(self.num_envs, 1).float()
        self._ptr = (self._ptr + 1) % self.max_size
        self._size = min(self._size + 1, self.max_size)

    def _next_features(
        self, t: torch.Tensor, e: torch.Tensor, steps: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ts = (t.unsqueeze(0) + torch.arange(1, steps + 1, device=self.device).unsqueeze(1)) % self._size
        ts_cur = (t.unsqueeze(0) + torch.arange(0, steps, device=self.device).unsqueeze(1)) % self._size
        succ_obs = self.obs[ts, e]
        succ_crit = self.critic[ts, e]
        cur_obs = self.obs[ts_cur, e]
        cur_crit = self.critic[ts_cur, e]
        boot = self.done[ts_cur, e].bool().squeeze(-1)
        next_obs = self._bootstrap(cur_obs, succ_obs, boot)
        next_critic = self._bootstrap(cur_crit, succ_crit, boot)
        return next_obs, next_critic

    def sample(self, batch_size: int, steps: int = 1, *, next_obs: bool = False) -> dict[str, torch.Tensor]:
        if self._size == 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}.")
        batch_size = int(batch_size)
        idx = torch.randint(0, self.num_samples, (batch_size,), device=self.device)
        t, e = torch.unravel_index(idx, (self._size, self.num_envs))

        if steps == 1:
            out = {
                "obs": self.obs[t, e],
                "critic": self.critic[t, e],
                "action": self.action[t, e],
                "reward": self.reward[t, e],
                "terminated": self.terminated[t, e],
                "done": self.done[t, e],
                "is_init": self.is_init[t, e],
            }
            if next_obs:
                t1 = (t + 1) % self._size
                boot = self.done[t, e].bool().squeeze(-1)
                out["next_obs"] = self._bootstrap(self.obs[t, e], self.obs[t1, e], boot)
                out["next_critic"] = self._bootstrap(self.critic[t, e], self.critic[t1, e], boot)
            return out

        ts = (t.unsqueeze(0) + torch.arange(steps, device=self.device).unsqueeze(1)) % self._size
        out = {
            "obs": self.obs[ts, e],
            "critic": self.critic[ts, e],
            "action": self.action[ts, e],
            "reward": self.reward[ts, e],
            "terminated": self.terminated[ts, e],
            "done": self.done[ts, e],
            "is_init": self.is_init[ts, e],
        }
        if next_obs:
            next_obs_t, next_crit = self._next_features(t, e, steps)
            out["next_obs"] = next_obs_t
            out["next_critic"] = next_crit
        return out
