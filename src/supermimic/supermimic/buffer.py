"""Fixed-size replay buffer. New transitions replace the oldest.

Sampling never synchronises with the host: contiguous windows and valid rollout
starts are indexed once per update by :meth:`ReplayBuffer.prepare`, and the draws
themselves are plain gathers from those precomputed candidate lists.
"""

from __future__ import annotations

import torch

from supermimic.tasks import TaskSpec


class ReplayBuffer:
    def __init__(self, capacity: int, num_envs: int, device: str, task: TaskSpec):
        self.capacity = capacity
        self.num_envs = num_envs
        self.device = device
        self.has_object = task.has_object
        shape = (capacity, num_envs)
        self.s = torch.zeros(*shape, task.state_dim, device=device)
        self.s1 = torch.zeros(*shape, task.state_dim, device=device)
        self.body_pos = torch.zeros(*shape, task.n_body, 3, device=device)
        self.body_quat = torch.zeros(*shape, task.n_body, 4, device=device)
        self.a = torch.zeros(*shape, task.num_joints, device=device)
        self.a_prev = torch.zeros(*shape, task.num_joints, device=device)
        self.xi = torch.zeros(*shape, task.xi_dim, device=device)
        self.c = torch.zeros(*shape, task.n_contact, device=device)
        self.c1 = torch.zeros(*shape, task.n_contact, device=device)
        self.f = torch.zeros(*shape, dtype=torch.long, device=device)
        self.ep = torch.full(shape, -1, dtype=torch.long, device=device)
        self.step = torch.zeros(*shape, dtype=torch.long, device=device)
        if task.has_object:
            self.obj = torch.zeros(*shape, task.object_state_dim, device=device)
            self.obj1 = torch.zeros(*shape, task.object_state_dim, device=device)
        else:
            self.obj = None
            self.obj1 = None
        self.ptr = 0
        self.filled = 0
        self._window_candidates: torch.Tensor | None = None
        self._start_candidates: torch.Tensor | None = None

    def add(self, batch: dict[str, torch.Tensor]) -> None:
        i = self.ptr
        self.s[i] = batch["s"]
        self.s1[i] = batch["s1"]
        self.body_pos[i] = batch["body_pos"]
        self.body_quat[i] = batch["body_quat"]
        self.a[i] = batch["a"]
        self.a_prev[i] = batch["a_prev"]
        self.xi[i] = batch["xi"]
        self.c[i] = batch["c"]
        self.c1[i] = batch["c1"]
        self.f[i] = batch["f"]
        self.ep[i] = batch["ep"]
        self.step[i] = batch["step"]
        if self.obj is not None:
            self.obj[i] = batch["obj"]
            self.obj1[i] = batch["obj1"]
        self.ptr = (self.ptr + 1) % self.capacity
        self.filled = min(self.filled + 1, self.capacity)

    # ------------------------------------------------------------------ candidates

    def _live(self) -> torch.Tensor:
        """``(capacity, num_envs)`` mask of rows written so far."""
        if self.filled >= self.capacity:
            return torch.ones((self.capacity, self.num_envs), dtype=torch.bool, device=self.device)
        rows = torch.arange(self.capacity, device=self.device)[:, None]
        return (rows < self.filled).expand(self.capacity, self.num_envs)

    @torch.no_grad()
    def prepare(
        self,
        window: int,
        start_horizon: int = 0,
        clip_last: torch.Tensor | None = None,
    ) -> None:
        """Index the rows that can start a ``window``-step transition window or an imagined rollout.

        Called once per training iteration; both candidate lists are gathered from
        afterwards without touching the host. ``clip_last`` (``(capacity, num_envs)``)
        is the last usable reference frame of the row's clip; starts must leave
        ``start_horizon`` frames of reference beyond them.
        """
        if self.filled < 1:
            raise RuntimeError("The replay buffer is empty.")
        live = self._live()
        step, ep = self.step, self.ep

        # Each row already stores its successor in s1/c1/obj1. Only the K
        # transition rows must share an episode; requiring row K as well would
        # exclude the final (possibly terminal) transition of every episode.
        ok = live.clone()
        for j in range(1, window):
            same = (
                torch.roll(live, -j, 0)
                & (torch.roll(ep, -j, 0) == ep)
                & (torch.roll(step, -j, 0) == step + j)
            )
            ok &= same
        self._window_candidates = ok.reshape(-1).nonzero().squeeze(-1)

        starts = live
        if start_horizon > 0 and clip_last is not None:
            starts = starts & (self.f + start_horizon <= clip_last)
        self._start_candidates = starts.reshape(-1).nonzero().squeeze(-1)

    def _pick(self, candidates: torch.Tensor | None, n: int, what: str) -> tuple[torch.Tensor, torch.Tensor]:
        """``(time, env)`` of ``n`` random entries of a precomputed candidate list.

        Draws with replacement, so the list only has to be non-empty (uniform
        sampling with replacement, as before). An empty list means the buffer is too
        short for the requested window or horizon.
        """
        if candidates is None:
            raise RuntimeError("Call ReplayBuffer.prepare before sampling.")
        if candidates.numel() == 0:
            raise RuntimeError(
                f"No valid {what} candidates. The buffer is too short for this "
                "window/horizon (or no row is a legal start); collect more steps or "
                "lower wm_window/horizon."
            )
        idx = candidates[torch.randint(0, candidates.numel(), (n,), device=self.device)]
        return idx // self.num_envs, idx % self.num_envs

    # ------------------------------------------------------------------ sampling

    def sample_windows(self, batch: int, k: int) -> dict[str, torch.Tensor]:
        if self.filled < k:
            raise RuntimeError(f"Need at least {k} buffered steps, have {self.filled}.")
        time, env = self._pick(self._window_candidates, batch, "window")
        if self.filled < self.capacity:
            idx = time[:, None] + torch.arange(k, device=self.device)
        else:
            idx = (time[:, None] + torch.arange(k, device=self.device)) % self.capacity
        env = env[:, None]
        out = {
            "s": self.s[idx, env],
            "s1": self.s1[idx, env],
            "body_pos": self.body_pos[idx, env],
            "body_quat": self.body_quat[idx, env],
            "a": self.a[idx, env],
            "a_prev": self.a_prev[idx, env],
            "xi": self.xi[idx, env],
            "c": self.c[idx, env],
            "c1": self.c1[idx, env],
            "f": self.f[idx, env],
        }
        if self.obj is not None:
            out["obj"] = self.obj[idx, env]
            out["obj1"] = self.obj1[idx, env]
        return out

    def sample_starts(self, batch: int, hist: int) -> dict[str, torch.Tensor]:
        """Current transitions plus an observation history, oldest frame first."""
        if self.filled < 1:
            raise RuntimeError("The replay buffer is empty.")
        time, env = self._pick(self._start_candidates, batch, "rollout start")
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
        # Rows before the episode start repeat the episode's first row (mjlab fills
        # the history buffer with the first observation after a reset).
        for col in range(hist - 2, -1, -1):
            bad = ~valid[:, col]
            raw[bad, col] = raw[bad, col + 1]
        out = {
            "s": self.s[time, env],
            "body_pos": self.body_pos[time, env],
            "body_quat": self.body_quat[time, env],
            "a_prev": self.a_prev[time, env],
            "xi": self.xi[time, env],
            "c": self.c[time, env],
            "f": self.f[time, env],
            "hist_s": self.s[raw, env[:, None]],
            "hist_body_pos": self.body_pos[raw, env[:, None]],
            "hist_body_quat": self.body_quat[raw, env[:, None]],
            "hist_a_prev": self.a_prev[raw, env[:, None]],
            "hist_xi": self.xi[raw, env[:, None]],
            "hist_f": self.f[raw, env[:, None]],
        }
        if self.obj is not None:
            out["obj"] = self.obj[time, env]
            out["hist_obj"] = self.obj[raw, env[:, None]]
        return out
