"""Raw world-body transitions. Windows never cross a reset or a ring discontinuity."""

import torch


class ReplayBuffer:
    def __init__(self, capacity, num_envs, state_dim, action_dim, device):
        self.capacity = capacity
        self.num_envs = num_envs
        self.device = device
        shape = (capacity, num_envs)
        self.s = torch.zeros(*shape, state_dim, device=device)
        self.s1 = torch.zeros_like(self.s)
        self.a = torch.zeros(*shape, action_dim, device=device)
        self.a_prev = torch.zeros_like(self.a)
        self.xi = torch.zeros(*shape, action_dim + 4, device=device)
        self.motion_t = torch.zeros(*shape, dtype=torch.long, device=device)
        self.motion_id = torch.zeros_like(self.motion_t)
        self.episode_step = torch.zeros_like(self.motion_t)
        self.ep = torch.full_like(self.motion_t, -1)
        self.step = torch.zeros_like(self.motion_t)
        self.ptr = 0
        self.filled = 0
        self.windows = None
        self.starts = None

    def add(self, record):
        state = record.state
        result = record.transition
        i = self.ptr
        if state.full is None or result.state.full is None or state.xi is None:
            raise ValueError("Replay accepts raw real transitions only")
        self.s[i] = state.full
        self.s1[i] = result.state.full
        self.a[i] = record.action
        self.a_prev[i] = state.previous_action
        self.xi[i] = state.xi
        self.motion_t[i] = state.command.motion_t
        self.motion_id[i] = state.command.motion_id
        self.episode_step[i] = state.episode_step
        self.ep[i] = result.info["episode"]
        self.step[i] = result.info["step"]
        self.ptr = (i + 1) % self.capacity
        self.filled = min(self.filled + 1, self.capacity)

    def prepare(self, window, horizon, motion):
        if self.filled < 1:
            raise ValueError("Empty replay")
        live = (torch.arange(self.capacity, device=self.device)[:, None] < self.filled).expand(
            -1, self.num_envs
        )
        valid = live.clone()
        for j in range(1, window):
            valid &= (
                torch.roll(live, -j, 0)
                & (torch.roll(self.ep, -j, 0) == self.ep)
                & (torch.roll(self.step, -j, 0) == self.step + j)
            )
        self.windows = valid.flatten().nonzero().flatten()
        self.starts = (
            (
                live
                & (
                    self.motion_t + horizon
                    < motion.clip_end[self.motion_id] - motion.clip_start[self.motion_id]
                )
            )
            .flatten()
            .nonzero()
            .flatten()
        )

    def _sample(self, candidates, n):
        if candidates is None or candidates.numel() == 0:
            raise RuntimeError("No valid replay samples; collect more transitions")
        ids = candidates[torch.randint(candidates.numel(), (n,), device=self.device)]
        return ids // self.num_envs, ids % self.num_envs

    def sample_windows(self, n, window):
        t, e = self._sample(self.windows, n)
        rows = (t[:, None] + torch.arange(window, device=self.device)) % self.capacity
        return {k: getattr(self, k)[rows, e[:, None]] for k in ("s", "s1", "a", "xi")}

    def sample_starts(self, n):
        t, e = self._sample(self.starts, n)
        return {
            k: getattr(self, k)[t, e]
            for k in ("s", "motion_id", "motion_t", "episode_step", "a_prev", "xi")
        }
