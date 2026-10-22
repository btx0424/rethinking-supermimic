"""LAFAN1 clips resampled to 50 Hz, with body poses from the mjlab G1."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .kinematics import TorchKinematics, compile_g1, verify_kinematics
from .math_utils import quat_apply_inverse, quat_mul, quat_normalize, quat_to_rotvec, yaw_quat

CONTROL_DT = 0.02
SOURCE_FPS = 30
TARGET_FPS = 50
BIN_FRAMES = 50  # 1 s bins at the control rate.

RAW_DIR = Path("/mnt/workspace/btx0424/aa-projects/aa-bfm/data/lafan1/raw/g1")
META_PATH = Path("/mnt/workspace/btx0424/aa-projects/aa-bfm/data/lafan1/meta.json")
CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "lafan1_g1.npz"

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
        # Yaw-frame root velocity used by the goal encoder. Built lazily per device.
        self._goal_ready = False

    @classmethod
    def load(cls, cache_path: Path = CACHE_PATH, device: str | torch.device = "cpu") -> "MotionLib":
        if not cache_path.is_file():
            build_cache(cache_path)
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
        return lib.to(device)

    def to(self, device: str | torch.device) -> "MotionLib":
        for name, value in list(self.__dict__.items()):
            if torch.is_tensor(value):
                setattr(self, name, value.to(device))
        self._prepare_goal()
        return self

    def _prepare_goal(self) -> None:
        root_v_local = quat_apply_inverse(self.root_quat, self.root_v)
        root_w_local = quat_apply_inverse(self.root_quat, self.root_w)
        # 6D root rotation in the world frame; the goal shifts it into the current yaw frame.
        from .math_utils import mat_to_6d, quat_to_mat

        self.root_rot6 = mat_to_6d(quat_to_mat(self.root_quat))
        self.root_v_local = root_v_local
        self.root_w_local = root_w_local
        self._goal_ready = True

    def future_frames(self, f: torch.Tensor, k: int) -> torch.Tensor:
        steps = torch.arange(k, device=f.device)
        future = f[:, None] + steps
        last = self.clip_end[self.clip_id[f]] - 1
        return torch.minimum(future, last[:, None])

    def goal(self, f: torch.Tensor, k: int) -> torch.Tensor:
        """K future reference frames in the current reference yaw frame. Shape (B, K * 73)."""
        if not self._goal_ready:
            self._prepare_goal()
        future = self.future_frames(f, k)
        yaw = yaw_quat(self.root_quat[f])
        dp = quat_apply_inverse(yaw.unsqueeze(1), self.root_p[future] - self.root_p[f].unsqueeze(1))
        # Reference orientation of each future frame, expressed in the current yaw frame.
        from .math_utils import mat_to_6d, quat_inv, quat_to_mat

        relative = quat_mul(quat_inv(yaw).unsqueeze(1).expand_as(self.root_quat[future]), self.root_quat[future])
        rot6 = mat_to_6d(quat_to_mat(relative))
        feats = torch.cat(
            (dp, rot6, self.root_v_local[future], self.root_w_local[future], self.q[future], self.qd[future]),
            dim=-1,
        )
        return feats.reshape(f.shape[0], -1)


def build_cache(cache_path: Path = CACHE_PATH, raw_dir: Path = RAW_DIR) -> Path:
    """Resample the 40 raw 30 Hz CSVs and store torch-FK body trajectories."""
    meta = json.loads(META_PATH.read_text())
    csv_joint_names = list(meta["joint_names"])
    clip_names = list(meta["clip_names"])
    model, joint_names = compile_g1()
    if list(joint_names) != csv_joint_names:
        # CSVs follow meta.json. Reorder columns onto the compiled model's hinge order.
        src = {name: i for i, name in enumerate(csv_joint_names)}
        missing = [name for name in joint_names if name not in src]
        if missing:
            raise KeyError(f"CSV joints do not cover the G1 model: {missing}")
        perm = [src[name] for name in joint_names]
    else:
        perm = list(range(len(joint_names)))
    kin = TorchKinematics(model, joint_names)
    verify_kinematics(kin, model)

    root_p, root_quat, root_v, root_w, joint_q, joint_qd = [], [], [], [], [], []
    starts, ends = [], []
    cursor = 0
    for name in clip_names:
        path = raw_dir / f"{name}.csv"
        rows = np.loadtxt(path, delimiter=",")
        if rows.ndim != 2 or rows.shape[1] != 7 + len(csv_joint_names):
            raise ValueError(f"{path} has shape {rows.shape}")
        pos = _resample(rows[:, 0:3].astype(np.float64))
        quat_xyzw = rows[:, 3:7].astype(np.float64)
        quat_wxyz = np.concatenate((quat_xyzw[:, 3:4], quat_xyzw[:, 0:3]), axis=-1)
        quat = _resample(quat_wxyz, quat=True)
        q = _resample(rows[:, 7:].astype(np.float64))[:, perm]
        pos = pos.copy()
        pos[:, 0:2] -= pos[0, 0:2]
        quat_t = torch.tensor(quat, dtype=torch.float32)
        lin = _linear_velocity(pos, CONTROL_DT)
        ang = _angular_velocity(quat_t, CONTROL_DT).numpy()
        qvel = _linear_velocity(q, CONTROL_DT)
        n = pos.shape[0]
        starts.append(cursor)
        cursor += n
        ends.append(cursor)
        root_p.append(pos.astype(np.float32))
        root_quat.append(quat.astype(np.float32))
        root_v.append(lin)
        root_w.append(ang.astype(np.float32))
        joint_q.append(q.astype(np.float32))
        joint_qd.append(qvel.astype(np.float32))
        print(f"resampled {name}: {n} frames", flush=True)

    root_p_t = torch.tensor(np.concatenate(root_p))
    root_quat_t = torch.tensor(np.concatenate(root_quat))
    root_v_t = torch.tensor(np.concatenate(root_v))
    root_w_t = torch.tensor(np.concatenate(root_w))
    q_t = torch.tensor(np.concatenate(joint_q))
    qd_t = torch.tensor(np.concatenate(joint_qd))
    body_p, body_quat, body_v, body_w = [], [], [], []
    chunk = 8192
    for start in range(0, root_p_t.shape[0], chunk):
        end = min(start + chunk, root_p_t.shape[0])
        bp, bq, bv, bw = kin.forward(
            root_p_t[start:end],
            root_quat_t[start:end],
            root_v_t[start:end],
            root_w_t[start:end],
            q_t[start:end],
            qd_t[start:end],
        )
        body_p.append(bp)
        body_quat.append(bq)
        body_v.append(bv)
        body_w.append(bw)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        joint_names=np.array(joint_names),
        clip_names=np.array(clip_names),
        body_names=np.array(kin.body_names),
        fps=np.int64(TARGET_FPS),
        starts=np.asarray(starts, dtype=np.int64),
        ends=np.asarray(ends, dtype=np.int64),
        root_p=root_p_t.numpy(),
        root_quat=root_quat_t.numpy(),
        root_v=root_v_t.numpy(),
        root_w=root_w_t.numpy(),
        q=q_t.numpy(),
        qd=qd_t.numpy(),
        body_p=torch.cat(body_p).numpy(),
        body_quat=torch.cat(body_quat).numpy(),
        body_v=torch.cat(body_v).numpy(),
        body_w=torch.cat(body_w).numpy(),
    )
    print(f"wrote {cache_path} ({cursor} frames)", flush=True)
    return cache_path


if __name__ == "__main__":
    build_cache()
