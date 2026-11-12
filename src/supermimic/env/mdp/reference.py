"""Task-selected reference view and reset sampling bins over a full dataset."""

import torch
from supermimic.dataset.motion import MotionData, MOTION_FIELDS
from . import CONTROL_DT, BIN_FRAMES


class TrackingReference:
    def __init__(self, dataset, tracking_bodies):
        self.dataset = dataset
        self.body_names = tuple(tracking_bodies)
        if len(self.body_names) < 2 or len(set(self.body_names)) != len(self.body_names):
            raise ValueError("Invalid tracking body selection")
        missing = set(self.body_names) - set(dataset.body_names)
        if missing:
            raise ValueError(f"Tracking bodies absent from dataset: {sorted(missing)}")
        if len(dataset.joint_names) != 29 or dataset.fps != round(1 / CONTROL_DT):
            raise ValueError("G1 tracking requires 29 joints at 50 Hz")
        ids = [dataset.body_names.index(n) for n in self.body_names]
        self.data = MotionData(
            **{
                key: getattr(dataset.data, key)[:, ids]
                if key.startswith("body_")
                else getattr(dataset.data, key)
                for key in MOTION_FIELDS
            },
            batch_size=dataset.data.batch_size,
            device=dataset.data.device,
        )
        starts = dataset.clip_start.tolist()
        ends = dataset.clip_end.tolist()
        bins, clips = [], []
        frame_bin = torch.empty_like(dataset.clip_id)
        for clip, (start, end) in enumerate(zip(starts, ends, strict=True)):
            for lo in range(start, end, BIN_FRAMES):
                frame_bin[lo : min(lo + BIN_FRAMES, end)] = len(bins)
                bins.append(lo)
                clips.append(clip)
        self.frame_bin = frame_bin
        self.bin_start = torch.tensor(bins, device=frame_bin.device)
        self.bin_clip = torch.tensor(clips, device=frame_bin.device)
        self.num_bins = len(bins)

    def __getattr__(self, name):
        if name in MOTION_FIELDS:
            return getattr(self.data, name)
        return getattr(self.dataset, name)

    def get_slice(self, motion_id, motion_t, steps):
        start = self.clip_start[motion_id]
        end = self.clip_end[motion_id] - 1
        indices = start[..., None] + motion_t[..., None] + steps
        indices = torch.maximum(start[..., None], torch.minimum(indices, end[..., None]))
        return self.data[indices]
