"""Explicit reference progress shared by simulator and imagined rollouts."""

from typing import NamedTuple
import torch
from . import BIN_FRAMES


class CommandState(NamedTuple):
    motion_id: torch.Tensor
    motion_t: torch.Tensor  # integer frame offset inside the clip


class MotionCommand:
    def __init__(self, motion):
        self.motion = motion

    def from_frame(self, frame):
        motion_id = self.motion.clip_id[frame]
        return CommandState(motion_id, frame - self.motion.clip_start[motion_id])

    def frame(self, state):
        return self.motion.clip_start[state.motion_id] + state.motion_t

    def advance(self, state):
        last = self.motion.clip_end[state.motion_id] - self.motion.clip_start[state.motion_id] - 1
        return CommandState(state.motion_id, torch.minimum(state.motion_t + 1, last))

    def reference(self, state, steps):
        return self.motion.get_slice(state.motion_id, state.motion_t, steps)

    def sample(self, count, probabilities, generator=None):
        """Sample command state; caller owns adaptive sampling statistics and RNG."""
        bins = torch.multinomial(probabilities, count, replacement=True, generator=generator)
        start = self.motion.bin_start[bins]
        clip = self.motion.bin_clip[bins]
        hi = torch.maximum(
            torch.minimum(start + BIN_FRAMES, self.motion.clip_end[clip] - 1), start + 1
        )
        offset = (
            torch.rand(count, device=start.device, generator=generator) * (hi - start).float()
        ).long()
        return self.from_frame(start + offset)
