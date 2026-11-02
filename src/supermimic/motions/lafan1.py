"""
Loading, processing, caching, and visualization of LAFAN1 motions.

The robot is the G1 shipped with mjlab.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from supermimic.utils.kinematics import TorchKinematics, compile_g1, verify_kinematics
from supermimic.motion import (
    CONTROL_DT,
    TARGET_FPS,
    _angular_velocity,
    _linear_velocity,
    _resample,
    write_npz_cache,
)

from .base import Motion
from supermimic.utils.vis_types import VisClipInfo, VisFrame, VisSession

_REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_PATH = _REPO_ROOT / "data" / "lafan1_g1.npz"


def lafan1_root() -> Path:
    """LAFAN1 data root: ``$LAFAN1_ROOT`` or ``<repo>/data/lafan1``."""
    env = os.environ.get("LAFAN1_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return _REPO_ROOT / "data" / "lafan1"


def ensure_lafan1_raw(root: Path | None = None) -> tuple[Path, Path]:
    """Return ``(raw/g1, meta.json)``, downloading via aa-bfm when missing."""
    root = lafan1_root() if root is None else Path(root).expanduser().resolve()
    raw_dir = root / "raw" / "g1"
    meta_path = root / "meta.json"
    has_csv = raw_dir.is_dir() and any(raw_dir.glob("*.csv"))
    if has_csv and meta_path.is_file():
        return raw_dir, meta_path
    try:
        from aa_bfm.data import Dataloader
    except ImportError as exc:
        raise RuntimeError(
            f"LAFAN1 data not found under {root} (need raw/g1/*.csv and meta.json). "
            "Set LAFAN1_ROOT to that directory, or install aa-bfm to auto-download "
            f"into {lafan1_root()}."
        ) from exc
    print(f"LAFAN1 raw missing at {raw_dir}; downloading via aa_bfm into {root}…", flush=True)
    Dataloader(root=root).ensure()
    if not meta_path.is_file() or not any(raw_dir.glob("*.csv")):
        raise FileNotFoundError(
            f"aa_bfm download finished but {raw_dir} / {meta_path} are still incomplete."
        )
    return raw_dir, meta_path


def __getattr__(name: str):
    """Lazy ``RAW_DIR`` / ``META_PATH`` so callers see the resolved env root."""
    if name == "RAW_DIR":
        return lafan1_root() / "raw" / "g1"
    if name == "META_PATH":
        return lafan1_root() / "meta.json"
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class Lafan1Motion(Motion):
    name = "lafan1"
    _model_joint_names: tuple[str, ...] | None = None

    @classmethod
    def default_cache_path(cls) -> Path:
        return CACHE_PATH

    @classmethod
    def build_cache(cls, cache_path: Path | None = None, raw_dir: Path | None = None) -> Path:
        return build_cache(cache_path or CACHE_PATH, raw_dir=raw_dir)

    @classmethod
    def _hinge_joint_names(cls) -> tuple[str, ...]:
        if cls._model_joint_names is None:
            _, cls._model_joint_names = compile_g1()
        return cls._model_joint_names

    @classmethod
    def dataset_joint_names(cls) -> tuple[str, ...]:
        return cls._hinge_joint_names()

    @classmethod
    def model_joint_names(cls) -> tuple[str, ...]:
        return cls._hinge_joint_names()

    @classmethod
    def free_joint_name(cls) -> str:
        import mujoco

        model, _ = compile_g1()
        for jid in range(model.njnt):
            if int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
                return name or "root"
        raise RuntimeError("G1 vis model has no free joint.")

    @classmethod
    def compile_vis_model(cls, clip: int, session: VisSession) -> mujoco.MjModel:
        del clip
        model, _ = compile_g1()
        return model

    @classmethod
    def open_vis_session(cls, cache_path: Path | None = None, **_) -> VisSession:
        from supermimic.motion import MotionLib

        lib = cls.load(cache_path=cache_path, device="cpu")
        return _Lafan1VisSession(lib)


@dataclass
class _Lafan1VisSession:
    lib: object

    @property
    def fps(self) -> float:
        return float(self.lib.fps)

    @property
    def num_clips(self) -> int:
        return int(self.lib.num_clips)

    def clip_info(self, clip: int) -> VisClipInfo:
        start = int(self.lib.clip_start[clip])
        end = int(self.lib.clip_end[clip])
        return VisClipInfo(name=str(self.lib.clip_names[clip]), length=end - start)

    def vis_model_key(self, clip: int) -> str:
        del clip
        return "lafan1_g1"

    def frame(self, clip: int, local: int) -> VisFrame:
        index = int(self.lib.clip_start[clip]) + int(local)
        return VisFrame(
            self.lib.root_p[index].numpy().astype(np.float64),
            self.lib.root_quat[index].numpy().astype(np.float64),
            self.lib.q[index].numpy().astype(np.float64),
        )


def build_cache(cache_path: Path | None = None, raw_dir: Path | None = None) -> Path:
    """Resample the 40 raw 30 Hz CSVs and store torch-FK body trajectories."""
    cache_path = CACHE_PATH if cache_path is None else Path(cache_path)
    if raw_dir is None:
        raw_dir, meta_path = ensure_lafan1_raw()
    else:
        raw_dir = Path(raw_dir)
        meta_path = raw_dir.parent.parent / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(
                f"Expected meta.json at {meta_path} (parent of raw/g1). "
                "Pass LAFAN1_ROOT or place meta.json next to raw/."
            )
    meta = json.loads(meta_path.read_text())
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
    write_npz_cache(
        cache_path,
        joint_names=joint_names,
        clip_names=clip_names,
        body_names=kin.body_names,
        fps=TARGET_FPS,
        starts=starts,
        ends=ends,
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
