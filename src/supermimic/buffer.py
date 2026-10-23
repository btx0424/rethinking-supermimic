"""Fixed-size replay buffer. New transitions replace the oldest."""

from __future__ import annotations

import torch

from .mdp import NUM_J, STATE_DIM, XI_DIM


class ReplayBuffer:
    def __init__(self, capacity: int, num_envs: int, device: str):
        self.capacity = capacity
        self.num_envs = num_envs
        self.device = device
        shape = (capacity, num_envs)
        self.s = torch.zeros(*shape, STATE_DIM, device=device)
        self.s1 = torch.zeros(*shape, STATE_DIM, device=device)
        self.a = torch.zeros(*shape, NUM_J, device=device)
        self.a_prev = torch.zeros(*shape, NUM_J, device=device)
        self.xi = torch.zeros(*shape, XI_DIM, device=device)
        self.f = torch.zeros(*shape, dtype=torch.long, device=device)
        self.ep = torch.full(shape, -1, dtype=torch.long, device=device)
        self.step = torch.zeros(*shape, dtype=torch.long, device=device)
        self.ptr = 0
        self.filled = 0

    def add(self, batch: dict[str, torch.Tensor]) -> None:
        i = self.ptr
        self.s[i] = batch["s"]
        self.s1[i] = batch["s1"]
        self.a[i] = batch["a"]
        self.a_prev[i] = batch["a_prev"]
        self.xi[i] = batch["xi"]
        self.f[i] = batch["f"]
        self.ep[i] = batch["ep"]
        self.step[i] = batch["step"]
        self.ptr = (self.ptr + 1) % self.capacity
        self.filled = min(self.filled + 1, self.capacity)

    def _time_env(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.filled < self.capacity:
            time = torch.randint(0, self.filled, (n,), device=self.device)
        else:
            time = torch.randint(0, self.capacity, (n,), device=self.device)
        env = torch.randint(0, self.num_envs, (n,), device=self.device)
        return time, env

    def sample_windows(self, batch: int, k: int) -> dict[str, torch.Tensor]:
        if self.filled < k:
            raise RuntimeError(f"Need at least {k} buffered steps, have {self.filled}.")
        got_t, got_e = [], []
        guard = 0
        while sum(t.shape[0] for t in got_t) < batch and guard < 40:
            guard += 1
            need = batch - sum(t.shape[0] for t in got_t)
            # Draw only starts that leave k in-range steps. Gathering first and
            # rejecting afterwards indexes past the buffer while it is still filling.
            if self.filled < self.capacity:
                time = torch.randint(0, self.filled - k + 1, (need * 4,), device=self.device)
                idx = time[:, None] + torch.arange(k, device=self.device)
            else:
                time = torch.randint(0, self.capacity, (need * 4,), device=self.device)
                idx = (time[:, None] + torch.arange(k, device=self.device)) % self.capacity
            env = torch.randint(0, self.num_envs, (time.shape[0],), device=self.device)
            steps = self.step[idx, env[:, None]]
            eps = self.ep[idx, env[:, None]]
            continuous = (steps[:, 1:] == steps[:, :1] + torch.arange(1, k, device=self.device)).all(-1)
            same = (eps == eps[:, :1]).all(-1)
            take = (continuous & same).nonzero(as_tuple=False).flatten()[:need]
            got_t.append(time[take])
            got_e.append(env[take])
        time = torch.cat(got_t)
        env = torch.cat(got_e)
        if time.shape[0] < batch:
            raise RuntimeError("Could not sample enough contiguous world-model windows.")
        if self.filled < self.capacity:
            idx = time[:, None] + torch.arange(k, device=self.device)
        else:
            idx = (time[:, None] + torch.arange(k, device=self.device)) % self.capacity
        return {
            "s": self.s[idx, env[:, None]],
            "s1": self.s1[idx, env[:, None]],
            "a": self.a[idx, env[:, None]],
            "a_prev": self.a_prev[idx, env[:, None]],
            "xi": self.xi[idx, env[:, None]],
            "f": self.f[idx, env[:, None]],
        }

    def sample_starts(self, batch: int, hist: int) -> dict[str, torch.Tensor]:
        """Current transitions plus an observation history, oldest frame first."""
        if self.filled < 1:
            raise RuntimeError("The replay buffer is empty.")
        time, env = self._time_env(batch)
        offsets = torch.arange(hist - 1, -1, -1, device=self.device)
        if self.filled < self.capacity:
            raw = time[:, None] - offsets
            valid = raw >= 0
            raw = raw.clamp(min=0)
        else:
            raw = (time[:, None] - offsets) % self.capacity
            valid = torch.ones(raw.shape, dtype=torch.bool, device=self.device)
        steps = self.step[raw, env[:, None]]
        expected = self.step[time, env][:, None] - offsets
        eps = self.ep[raw, env[:, None]]
        valid = valid & (steps == expected) & (eps == self.ep[time, env][:, None])
        for col in range(hist - 2, -1, -1):
            bad = ~valid[:, col]
            raw[bad, col] = raw[bad, col + 1]
        return {
            "s": self.s[time, env],
            "a_prev": self.a_prev[time, env],
            "xi": self.xi[time, env],
            "f": self.f[time, env],
            "hist_s": self.s[raw, env[:, None]],
            "hist_a_prev": self.a_prev[raw, env[:, None]],
            "hist_xi": self.xi[raw, env[:, None]],
            "hist_f": self.f[raw, env[:, None]],
        }
