"""Resampled reference motion libraries shared across tracking tasks."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch

from .math_utils import quat_apply_inverse, quat_mul, quat_normalize, quat_to_rotvec, yaw_quat

CONTROL_DT = 0.02
SOURCE_FPS = 30
TARGET_FPS = 50
BIN_FRAMES = 50  # 1 s bins at the control rate.

GOAL_PER_FRAME = 3 + 6 + 3 + 3 + 29 + 29


def _slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    dot = (q0 * q1).sum(-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs().clamp(-1.0, 1.0)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)
    small = sin_omega.abs() < 1e-4
    denom = sin_omega.clamp(min=1e-4)
    a = torch.sin((1.0 - t) * omega) / denom
    b = torch.sin(t * omega) / denom
    blended = a * q0 + b * q1
    linear = (1.0 - t) * q0 + t * q1
    out = torch.where(small, linear, blended)
    return quat_normalize(out)


def _resample(values: np.ndarray, quat: bool = False) -> np.ndarray:
    length = values.shape[0]
    duration = (length - 1) / SOURCE_FPS
    src_t = np.linspace(0.0, duration, length)
    dst_t = np.arange(0.0, duration, 1.0 / TARGET_FPS)
    index = np.clip(np.searchsorted(src_t, dst_t, side="right") - 1, 0, length - 2)
    tau = (dst_t - src_t[index]) / (src_t[index + 1] - src_t[index])
    tau_t = torch.tensor(tau, dtype=torch.float32).unsqueeze(-1)
    if quat:
        out = _slerp(
            torch.tensor(values[index], dtype=torch.float32),
            torch.tensor(values[index + 1], dtype=torch.float32),
            tau_t,
        )
        return out.numpy()
    flat = values.reshape(length, -1)
    out = (1.0 - tau)[:, None] * flat[index] + tau[:, None] * flat[index + 1]
    return out.reshape(len(dst_t), *values.shape[1:]).astype(np.float32)


def _linear_velocity(position: np.ndarray, dt: float) -> np.ndarray:
    velocity = np.diff(position, axis=0) / dt
    return np.concatenate((velocity, velocity[-1:]), axis=0).astype(np.float32)


def _angular_velocity(quat: torch.Tensor, dt: float) -> torch.Tensor:
    """World-frame angular velocity from a wxyz quaternion sequence."""
    delta = quat_mul(quat[1:], quat_inv_batch(quat[:-1]))
    velocity = quat_to_rotvec(delta) / dt
    return torch.cat((velocity, velocity[-1:]), dim=0)


def quat_inv_batch(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[..., 1:] = -out[..., 1:]
    return out / q.square().sum(-1, keepdim=True).clamp(min=1e-8)


def write_npz_cache(
    cache_path: Path,
    *,
    joint_names: Sequence[str],
    clip_names: Sequence[str],
    body_names: Sequence[str],
    fps: int,
    starts: Sequence[int],
    ends: Sequence[int],
    root_p: np.ndarray,
    root_quat: np.ndarray,
    root_v: np.ndarray,
    root_w: np.ndarray,
    q: np.ndarray,
    qd: np.ndarray,
    body_p: np.ndarray,
    body_quat: np.ndarray,
    body_v: np.ndarray,
    body_w: np.ndarray,
    **extra: np.ndarray,
) -> Path:
    """Write the standard humanoid motion NPZ (optional extra arrays for object HOI)."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        joint_names=np.array(joint_names),
        clip_names=np.array(clip_names),
        body_names=np.array(body_names),
        fps=np.int64(fps),
        starts=np.asarray(starts, dtype=np.int64),
        ends=np.asarray(ends, dtype=np.int64),
        root_p=root_p,
        root_quat=root_quat,
        root_v=root_v,
        root_w=root_w,
        q=q,
        qd=qd,
        body_p=body_p,
        body_quat=body_quat,
        body_v=body_v,
        body_w=body_w,
        **extra,
    )
    np.savez(cache_path, **payload)
    return cache_path


class MotionLib:
    """Concatenated reference library. Frame indices point into this buffer."""

    def __init__(
        self,
        joint_names: list[str],
        clip_names: list[str],
        body_names: list[str],
        fps: int,
        starts: torch.Tensor,
        ends: torch.Tensor,
        root_p: torch.Tensor,
        root_quat: torch.Tensor,
        root_v: torch.Tensor,
        root_w: torch.Tensor,
        q: torch.Tensor,
        qd: torch.Tensor,
        body_p: torch.Tensor,
        body_quat: torch.Tensor,
        body_v: torch.Tensor,
        body_w: torch.Tensor,
    ):
        self.joint_names = tuple(joint_names)
        self.clip_names = tuple(clip_names)
        self.body_names = tuple(body_names)
        self.fps = int(fps)
        self.clip_start = starts.long()
        self.clip_end = ends.long()
        self.root_p = root_p
        self.root_quat = root_quat
        self.root_v = root_v
        self.root_w = root_w
        self.q = q
        self.qd = qd
        self.body_p = body_p
        self.body_quat = body_quat
        self.body_v = body_v
        self.body_w = body_w
        self.num_frames = int(root_p.shape[0])
        self.num_clips = len(clip_names)
        clip_id = torch.empty(self.num_frames, dtype=torch.long)
        for c in range(self.num_clips):
            clip_id[int(starts[c]) : int(ends[c])] = c
        self.clip_id = clip_id
        bin_clip, bin_start = [], []
        frame_bin = torch.zeros(self.num_frames, dtype=torch.long)
        for c in range(self.num_clips):
            length = int(ends[c] - starts[c])
            nb = int(np.ceil(length / BIN_FRAMES))
            for b in range(nb):
                bin_clip.append(c)
                bin_start.append(int(starts[c]) + b * BIN_FRAMES)
                lo = int(starts[c]) + b * BIN_FRAMES
                hi = min(lo + BIN_FRAMES, int(ends[c]))
                frame_bin[lo:hi] = len(bin_clip) - 1
        self.bin_clip = torch.tensor(bin_clip, dtype=torch.long)
        self.bin_start = torch.tensor(bin_start, dtype=torch.long)
        self.frame_bin = frame_bin
        self.num_bins = len(bin_clip)
        self._goal_ready = False

    @classmethod
    def from_npz(cls, cache_path: Path, device: str | torch.device = "cpu") -> MotionLib:
        data = np.load(cache_path, allow_pickle=False)
        lib = cls(
            joint_names=data["joint_names"].tolist(),
            clip_names=data["clip_names"].tolist(),
            body_names=data["body_names"].tolist(),
            fps=int(data["fps"]),
            starts=torch.tensor(data["starts"]),
            ends=torch.tensor(data["ends"]),
            root_p=torch.tensor(data["root_p"]),
            root_quat=torch.tensor(data["root_quat"]),
            root_v=torch.tensor(data["root_v"]),
            root_w=torch.tensor(data["root_w"]),
            q=torch.tensor(data["q"]),
            qd=torch.tensor(data["qd"]),
            body_p=torch.tensor(data["body_p"]),
            body_quat=torch.tensor(data["body_quat"]),
            body_v=torch.tensor(data["body_v"]),
            body_w=torch.tensor(data["body_w"]),
        )
        if "object_p" in data.files:
            lib.object_p = torch.tensor(data["object_p"])
            lib.object_quat = torch.tensor(data["object_quat"])
            lib.object_v = torch.tensor(data["object_v"])
            lib.object_w = torch.tensor(data["object_w"])
        if "hand_p" in data.files:
            lib.hand_p = torch.tensor(data["hand_p"])
            lib.hand_body_names = tuple(data["hand_body_names"].tolist())
        return lib.to(device)

    @classmethod
    def load(
        cls,
        cache_path: Path | None = None,
        device: str | torch.device = "cpu",
        *,
        build: Callable[[Path], Path] | None = None,
    ) -> MotionLib:
        """Load LAFAN1 G1 cache by default (see :mod:`supermimic.motions.lafan1`)."""
        from .motions.lafan1 import CACHE_PATH, build_cache

        path = CACHE_PATH if cache_path is None else cache_path
        if not path.is_file():
            builder = build_cache if build is None else build
            builder(path)
        return cls.from_npz(path, device)

    def to(self, device: str | torch.device) -> MotionLib:
        for name, value in list(self.__dict__.items()):
            if torch.is_tensor(value):
                setattr(self, name, value.to(device))
        self._prepare_goal()
        return self

    def _prepare_goal(self) -> None:
        root_v_local = quat_apply_inverse(self.root_quat, self.root_v)
        root_w_local = quat_apply_inverse(self.root_quat, self.root_w)
        from .math_utils import mat_to_6d, quat_to_mat

        self.root_rot6 = mat_to_6d(quat_to_mat(self.root_quat))
        self.root_v_local = root_v_local
        self.root_w_local = root_w_local
        if getattr(self, "object_p", None) is not None:
            self.object_v_local = quat_apply_inverse(self.object_quat, self.object_v)
            self.object_w_local = quat_apply_inverse(self.object_quat, self.object_w)
        self._goal_ready = True

    def future_frames(self, f: torch.Tensor, k: int) -> torch.Tensor:
        """Frames ``f+1`` through ``f+k``, clamped to each clip's last frame."""
        steps = torch.arange(1, k + 1, device=f.device)
        future = f[:, None] + steps
        last = self.clip_end[self.clip_id[f]] - 1
        return torch.minimum(future, last[:, None])

    def goal(self, f: torch.Tensor, k: int) -> torch.Tensor:
        """Frames f+1..f+k in the current reference yaw frame. Shape (B, K * width).

        Joint velocity is scaled by 0.1 so it does not swamp the root displacement.
        """
        if not self._goal_ready:
            self._prepare_goal()
        future = self.future_frames(f, k)
        yaw = yaw_quat(self.root_quat[f])
        dp = quat_apply_inverse(yaw.unsqueeze(1), self.root_p[future] - self.root_p[f].unsqueeze(1))
        from .math_utils import mat_to_6d, quat_inv, quat_to_mat

        relative = quat_mul(quat_inv(yaw).unsqueeze(1).expand_as(self.root_quat[future]), self.root_quat[future])
        rot6 = mat_to_6d(quat_to_mat(relative))
        feats = torch.cat(
            (
                dp,
                rot6,
                self.root_v_local[future],
                self.root_w_local[future],
                self.q[future],
                self.qd[future] * 0.1,
            ),
            dim=-1,
        )
        object_p = getattr(self, "object_p", None)
        if object_p is not None:
            from .math_utils import mat_to_6d, quat_inv, quat_to_mat

            obj_dp = quat_apply_inverse(yaw.unsqueeze(1), object_p[future] - self.root_p[f].unsqueeze(1))
            obj_rel = quat_mul(
                quat_inv(yaw).unsqueeze(1).expand_as(self.object_quat[future]),
                self.object_quat[future],
            )
            obj_rot6 = mat_to_6d(quat_to_mat(obj_rel))
            feats = torch.cat(
                (
                    feats,
                    obj_dp,
                    obj_rot6,
                    self.object_v_local[future],
                    self.object_w_local[future],
                ),
                dim=-1,
            )
        return feats.reshape(f.shape[0], -1)


def build_cache(cache_path: Path | None = None) -> Path:
    """Backward-compatible entry point; delegates to LAFAN1."""
    from .motions.lafan1 import CACHE_PATH, build_cache as _build

    return _build(cache_path or CACHE_PATH)


# Default cache path for scripts that imported it from here.
def __getattr__(name: str):
    if name == "CACHE_PATH":
        from .motions.lafan1 import CACHE_PATH

        return CACHE_PATH
    if name == "RAW_DIR":
        from .motions.lafan1 import RAW_DIR

        return RAW_DIR
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    build_cache()
