"""Full prepared motion dataset; independent of tracking-body or task choices."""

from pathlib import Path
import numpy as np
import torch
from tensordict import TensorClass


class MotionData(TensorClass):
    root_p: torch.Tensor
    root_quat: torch.Tensor
    root_v: torch.Tensor
    root_w: torch.Tensor
    q: torch.Tensor
    qd: torch.Tensor
    body_p: torch.Tensor
    body_quat: torch.Tensor
    body_v: torch.Tensor
    body_w: torch.Tensor


MOTION_FIELDS = tuple(MotionData.__annotations__)


class MotionDataset:
    def __init__(self, path, device="cpu"):
        with np.load(Path(path), allow_pickle=False) as data:
            names = ("joint_names", "body_names", "clip_names")
            fields = MOTION_FIELDS
            tensors = {}
            expected = set(names + fields + ("fps", "starts", "ends"))
            if set(data.files) != expected:
                raise ValueError(f"Expected a prepared motion cache with fields {sorted(expected)}")
            self.joint_names = tuple(data["joint_names"].tolist())
            self.clip_names = tuple(data["clip_names"].tolist())
            cached_bodies = tuple(data["body_names"].tolist())
            self.body_names = cached_bodies
            self.fps = float(data["fps"])
            if self.fps <= 0:
                raise ValueError("Motion fps must be positive")
            starts = np.asarray(data["starts"], dtype=np.int64)
            ends = np.asarray(data["ends"], dtype=np.int64)
            self.num_frames = len(data["root_p"])
            self.num_clips = len(self.clip_names)
            if (
                not len(starts)
                or len(starts) != self.num_clips
                or starts[0] != 0
                or ends[-1] != self.num_frames
                or np.any(ends - starts < 2)
                or not np.array_equal(starts[1:], ends[:-1])
            ):
                raise ValueError("Invalid contiguous motion clip ranges")
            for key in fields:
                values = np.asarray(data[key])
                if key.startswith("body_"):
                    if values.shape[1] != len(cached_bodies):
                        raise ValueError(f"{key} does not match cache body names")
                if len(values) != self.num_frames or not np.isfinite(values).all():
                    raise ValueError(f"Invalid {key}")
                tensors[key] = torch.as_tensor(values.copy(), dtype=torch.float32, device=device)
        self.data = MotionData(**tensors, batch_size=[self.num_frames], device=device)
        if self.q.shape[1] != len(self.joint_names) or self.body_p.shape[1] != len(self.body_names):
            raise ValueError("Motion dimensions disagree with names")
        self.clip_start = torch.tensor(starts, device=device)
        self.clip_end = torch.tensor(ends, device=device)
        clip_id = np.empty(self.num_frames, dtype=np.int64)
        for c, (start, end) in enumerate(zip(starts, ends, strict=True)):
            clip_id[start:end] = c
        self.clip_id = torch.tensor(clip_id, device=device)

    def last_frame(self, frame):
        return self.clip_end[self.clip_id[frame]] - 1

    def future_frames(self, frame, horizon):
        offsets = torch.arange(1, horizon + 1, device=frame.device)
        return torch.minimum(frame[..., None] + offsets, self.last_frame(frame)[..., None])

    def __getattr__(self, name):
        if name in MOTION_FIELDS:
            return getattr(self.data, name)
        raise AttributeError(name)

    def get_slice(self, motion_id, motion_t, steps):
        """Return a typed [batch..., time] window, clamped within each clip."""
        start = self.clip_start[motion_id]
        end = self.clip_end[motion_id] - 1
        indices = start[..., None] + motion_t[..., None] + steps
        indices = torch.maximum(start[..., None], torch.minimum(indices, end[..., None]))
        return self.data[indices]
