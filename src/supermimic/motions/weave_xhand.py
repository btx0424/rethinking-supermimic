"""
Loading, processing, caching, and visualization of the Weave G1-Xhand-OMOMO motions.

Newer Weave release: XHand hardware, retargeted to ``g1_29dof_rev_1_0_xhand``,
with primitive-fitted objects. Layout (see the dataset's ``README.txt``)::

    assets/g1_xhand/           robot URDF + meshes
    assets/objects/<obj>/      object meshes (<obj>.obj)
    assets/primitives/         object-local collision primitives (<obj>.json)
    motions/<obj>/<seq>.npz    one file per clip, 30 FPS

Every clip NPZ holds ``joint_positions (T, 53)``, ``joint_names (53,)``,
``root_poses (T, 7)`` and ``object_poses (T, 7)`` as ``[x, y, z, qw, qx, qy, qz]``
in a Z-up world, at 30 FPS. This module resamples to 50 Hz, computes velocities,
runs torch FK once for the tracked bodies, and writes the shared MotionLib cache.

Set ``WEAVE_XHAND_ROOT`` to the ``G1-Xhand-OMOMO`` directory.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch

from supermimic.utils.vis_types import VisClipInfo, VisFrame, VisSession

from .base import Motion

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATA_DIR = _REPO_ROOT / "data"

AVAILABLE_OBJECTS = (
    "clothesstand",
    "floorlamp",
    "largebox",
    "largetable",
    "smallbox",
    "smalltable",
    "tripod",
    "woodchair",
)

PELVIS_BODY = "pelvis"

# Train/test clip split, by position in each object's sorted listing. The release
# has no official split and no subject labels, so hold out every fifth clip:
# 20% of the 2,404 motions, spread across every object.
TEST_EVERY = 5


def dataset_root() -> Path:
    """``$WEAVE_XHAND_ROOT`` (required; the dataset is not auto-downloaded)."""
    env = os.environ.get("WEAVE_XHAND_ROOT")
    if not env or not env.strip():
        raise RuntimeError(
            "WEAVE_XHAND_ROOT is not set. Export it to the G1-Xhand-OMOMO directory "
            "(contains assets/ and motions/)."
        )
    path = Path(env).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"WEAVE_XHAND_ROOT={path} is not a directory.")
    return path


def object_mesh_path(object_name: str, root: Path | None = None) -> Path:
    """Triangle mesh for one object, in the object-local frame."""
    base = (dataset_root() if root is None else Path(root)) / "assets" / "objects"
    path = base / object_name / f"{object_name}.obj"
    if not path.is_file():
        raise FileNotFoundError(
            f"No mesh for object {object_name!r} at {path}. Set WEAVE_XHAND_ROOT."
        )
    return path


def load_primitives(object_name: str, root: Path | None = None) -> list[dict]:
    """Primitive-fit colliders for one object, in the object-local frame.

    Each entry is ``{"type": box|cylinder|capsule, "size": [...], "position":
    [3], "rotation": 3x3}`` with a primitive-to-object transform. Sizes follow the
    dataset card: box = full XYZ extents, cylinder = ``[radius, height]``,
    capsule = ``[radius, sphere-center distance]``.
    """
    base = (dataset_root() if root is None else Path(root)) / "assets" / "primitives"
    path = base / f"{object_name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"No primitives for object {object_name!r} at {path}.")
    parts = json.loads(path.read_text())
    if not isinstance(parts, list) or not parts:
        raise ValueError(f"{path} is not a non-empty list of primitives.")
    for part in parts:
        kind = part.get("type")
        if kind not in ("box", "cylinder", "capsule"):
            raise ValueError(f"{path}: unknown primitive type {kind!r}.")
    return parts


def cache_path_for(objects: tuple[str, ...], split: str = "train") -> Path:
    """One NPZ per (object set, split) so train and test never share a cache."""
    if not objects:
        raise ValueError("objects must be non-empty.")
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}.")
    tag = "_".join(sorted(objects))
    return _DATA_DIR / f"weave_xhand_{tag}_{split}.npz"


def clip_paths(root: Path, object_name: str, split: str) -> list[Path]:
    """Clip NPZ files for one object, sorted, split into train/test."""
    directory = root / "motions" / object_name
    if not directory.is_dir():
        raise FileNotFoundError(f"No motions for object {object_name!r} at {directory}.")
    paths = sorted(directory.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"{directory} has no .npz files.")
    if split == "test":
        return paths[TEST_EVERY - 1 :: TEST_EVERY]
    return [p for i, p in enumerate(paths) if (i + 1) % TEST_EVERY]


_MODEL_JOINT_NAMES: tuple[str, ...] | None = None


def model_joint_names() -> tuple[str, ...]:
    """Hinge names of the XHand model, in qpos order."""
    global _MODEL_JOINT_NAMES
    if _MODEL_JOINT_NAMES is None:
        from supermimic.robots.g1_xhand import compile_g1_xhand

        _MODEL_JOINT_NAMES = compile_g1_xhand()[1]
    return _MODEL_JOINT_NAMES


class WeaveXHandMotion(Motion):
    name = "weave_xhand"

    # Overridden by the task, since each object has its own cache file.
    default_objects: tuple[str, ...] = ("smalltable",)

    @classmethod
    def default_cache_path(cls) -> Path:
        return cache_path_for(cls.default_objects)

    @classmethod
    def build_cache(
        cls,
        cache_path: Path | None = None,
        root: Path | None = None,
        objects: tuple[str, ...] = default_objects,
        split: str = "train",
        max_clips: int | None = None,
        **_,
    ) -> Path:
        path = cache_path_for(tuple(objects), split) if cache_path is None else cache_path
        return build_cache(path, root=root, objects=tuple(objects), split=split, max_clips=max_clips)

    @classmethod
    def load(
        cls,
        cache_path: Path | None = None,
        device: str = "cpu",
        objects: tuple[str, ...] = default_objects,
        split: str = "train",
        **build_kwargs,
    ):
        from supermimic.motion import MotionLib

        path = cache_path_for(tuple(objects), split) if cache_path is None else cache_path
        if not path.is_file():
            cls.build_cache(path, objects=tuple(objects), split=split, **build_kwargs)
        lib = MotionLib.from_npz(path, device=device)
        if getattr(lib, "object_p", None) is None:
            raise RuntimeError(f"{path} has no object fields; delete it and rebuild.")
        if getattr(lib, "hand_p", None) is None:
            raise RuntimeError(f"{path} has no hand_p; delete it and rebuild.")
        return lib

    @classmethod
    def dataset_joint_names(cls) -> tuple[str, ...]:
        """Clips carry their own ``joint_names``; this is the model's order."""
        return model_joint_names()

    @classmethod
    def model_joint_names(cls) -> tuple[str, ...]:
        return model_joint_names()

    @classmethod
    def free_joint_name(cls) -> str:
        from supermimic.robots.g1_xhand import FREE_JOINT

        return FREE_JOINT

    @classmethod
    def compile_vis_model(cls, clip: int, session: VisSession) -> mujoco.MjModel:
        if not isinstance(session, _WeaveXHandVisSession):
            raise TypeError("Weave XHand vis session expected.")
        object_name = session.object_for_clip(clip)
        return mujoco.MjModel.from_xml_string(_scene_xml(object_name))

    @classmethod
    def open_vis_session(
        cls,
        root: Path | None = None,
        objects: Sequence[str] = ("smalltable",),
        split: str = "train",
        max_clips: int | None = 50,
        **_,
    ) -> VisSession:
        resolved = dataset_root() if root is None else Path(root)
        return _WeaveXHandVisSession.load(
            root=resolved, objects=tuple(objects), split=split, max_clips=max_clips
        )


def _scene_xml(object_name: str) -> str:
    """Robot MJCF text with a floor and a mocap copy of the object mesh.

    The robot XML already carries ``meshdir="meshes/"`` relative to itself, which
    ``MjModel.from_xml_string`` resolves from the current directory — wrong here —
    so the compiler line is made absolute. The object mesh is absolute either way.
    """
    from supermimic.robots.g1_xhand import MESH_DIR, XML_PATH

    robot_xml = XML_PATH.read_text().replace(
        'meshdir="meshes/"', f'meshdir="{MESH_DIR.resolve()}"', 1
    )
    object_mesh = object_mesh_path(object_name).resolve()
    asset = f'\n    <mesh name="{object_name}_vis" file="{object_mesh}"/>\n  '
    body = f"""
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.85 0.85 0.85 1"/>
    <body name="{object_name}" mocap="true" pos="0 0 0" quat="1 0 0 0">
      <geom type="mesh" mesh="{object_name}_vis" rgba="0.92 0.48 0.16 1" contype="0" conaffinity="0" group="1"/>
    </body>
  """
    return robot_xml.replace("</asset>", f"{asset}</asset>").replace("</worldbody>", f"{body}</worldbody>")


@dataclass
class _WeaveXHandVisSession:
    fps: float
    clip_names: list[str]
    object_names: list[str]
    joint_names: tuple[str, ...]
    root_p: np.ndarray
    root_quat: np.ndarray
    joint_q: np.ndarray
    object_p: np.ndarray
    object_quat: np.ndarray
    clip_starts: np.ndarray
    clip_lengths: np.ndarray

    @property
    def num_clips(self) -> int:
        return len(self.clip_names)

    @classmethod
    def load(
        cls,
        *,
        root: Path,
        objects: tuple[str, ...],
        split: str,
        max_clips: int | None,
    ) -> _WeaveXHandVisSession:
        from supermimic.motion import TARGET_FPS, _resample

        unknown = [name for name in objects if name not in AVAILABLE_OBJECTS]
        if unknown:
            raise ValueError(f"Unknown objects {unknown}. Available: {list(AVAILABLE_OBJECTS)}.")
        joint_names = model_joint_names()
        clip_names: list[str] = []
        object_names: list[str] = []
        lengths: list[int] = []
        roots_p, roots_q, joints, obj_p, obj_q = [], [], [], [], []
        fps = float(TARGET_FPS)
        for obj in objects:
            for path in clip_paths(root, obj, split):
                if max_clips is not None and len(lengths) >= max_clips:
                    break
                with np.load(path, allow_pickle=False) as data:
                    names = [str(n) for n in data["joint_names"]]
                    perm = [names.index(name) for name in joint_names]
                    poses = np.asarray(data["root_poses"], dtype=np.float64)
                    obj_poses = np.asarray(data["object_poses"], dtype=np.float64)
                    q = _resample(np.asarray(data["joint_positions"], dtype=np.float64)[:, perm])
                    rp = _resample(poses[:, :3])
                    rq = _resample(poses[:, 3:7], quat=True)
                    op = _resample(obj_poses[:, :3])
                    oq = _resample(obj_poses[:, 3:7], quat=True)
                roots_p.append(rp.astype(np.float32))
                roots_q.append(rq.astype(np.float32))
                joints.append(q.astype(np.float32))
                obj_p.append(op.astype(np.float32))
                obj_q.append(oq.astype(np.float32))
                lengths.append(int(q.shape[0]))
                clip_names.append(f"{obj}/{path.stem}")
                object_names.append(obj)
            if max_clips is not None and len(lengths) >= max_clips:
                break
        if not lengths:
            raise ValueError(f"No clips for objects={objects} split={split}.")

        clip_lengths = np.asarray(lengths, dtype=np.int64)
        clip_starts = np.zeros_like(clip_lengths)
        if len(clip_lengths) > 1:
            np.cumsum(clip_lengths[:-1], out=clip_starts[1:])
        print(
            f"weave_xhand vis: {len(clip_lengths)} clips, "
            f"{int(clip_lengths.sum())} frames @ {fps:g} Hz",
            flush=True,
        )
        return cls(
            fps=fps,
            clip_names=clip_names,
            object_names=object_names,
            joint_names=joint_names,
            root_p=np.concatenate(roots_p),
            root_quat=np.concatenate(roots_q),
            joint_q=np.concatenate(joints),
            object_p=np.concatenate(obj_p),
            object_quat=np.concatenate(obj_q),
            clip_starts=clip_starts,
            clip_lengths=clip_lengths,
        )

    def object_for_clip(self, clip: int) -> str:
        return self.object_names[clip]

    def clip_info(self, clip: int) -> VisClipInfo:
        return VisClipInfo(
            name=self.clip_names[clip],
            length=int(self.clip_lengths[clip]),
            tags={"object": self.object_for_clip(clip)},
        )

    def vis_model_key(self, clip: int) -> str:
        return self.object_for_clip(clip)

    def frame(self, clip: int, local: int) -> VisFrame:
        index = int(self.clip_starts[clip]) + int(local)
        return VisFrame(
            self.root_p[index].astype(np.float64),
            self.root_quat[index].astype(np.float64),
            self.joint_q[index].astype(np.float64),
            mocap_body=self.object_for_clip(clip),
            mocap_pos=self.object_p[index].astype(np.float64),
            mocap_quat=self.object_quat[index].astype(np.float64),
        )


def build_cache(
    cache_path: Path | None = None,
    root: Path | None = None,
    objects: tuple[str, ...] = ("smalltable",),
    split: str = "train",
    max_clips: int | None = None,
) -> Path:
    """Pack the XHand clips into a MotionLib cache with object pose and twist.

    Clips are 30 Hz and are resampled onto the 50 Hz control rate. Joint columns
    are reordered onto the XHand model's hinge order, the root is the pelvis, and
    the tracked / hand body trajectories come from torch FK on the compiled XHand
    model, because the release ships joint and root trajectories only.
    """
    from supermimic.motion import (
        TARGET_FPS,
        _angular_velocity,
        _linear_velocity,
        _resample,
        write_npz_cache,
    )
    from supermimic.robots.g1_xhand import compile_g1_xhand
    from supermimic.utils.kinematics import TRACKED_BODIES, TorchKinematics, XHAND_BODIES

    root = dataset_root() if root is None else Path(root)
    unknown = [name for name in objects if name not in AVAILABLE_OBJECTS]
    if unknown:
        raise ValueError(f"Unknown objects {unknown}. Available: {list(AVAILABLE_OBJECTS)}.")
    if not objects:
        raise ValueError("Select at least one object.")
    out_path = cache_path_for(tuple(objects), split) if cache_path is None else cache_path

    model, joint_names = compile_g1_xhand()
    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
    missing = [name for name in (*TRACKED_BODIES, *XHAND_BODIES) if name not in body_names]
    if missing:
        raise KeyError(f"XHand model is missing tracked/hand bodies: {missing}")
    kin = TorchKinematics(model, joint_names, hand_bodies=XHAND_BODIES)
    pelvis = TRACKED_BODIES.index(PELVIS_BODY)
    dt = 1.0 / TARGET_FPS

    joint_q: list[np.ndarray] = []
    joint_qd: list[np.ndarray] = []
    root_p: list[np.ndarray] = []
    root_quat: list[np.ndarray] = []
    root_v: list[np.ndarray] = []
    root_w: list[np.ndarray] = []
    body_p: list[np.ndarray] = []
    body_quat: list[np.ndarray] = []
    body_v: list[np.ndarray] = []
    body_w: list[np.ndarray] = []
    hand_p: list[np.ndarray] = []
    object_p: list[np.ndarray] = []
    object_quat: list[np.ndarray] = []
    object_v: list[np.ndarray] = []
    object_w: list[np.ndarray] = []
    clip_names: list[str] = []
    clip_lengths: list[int] = []

    for obj in objects:
        paths = clip_paths(root, obj, split)
        if max_clips is not None:
            paths = paths[: max(0, max_clips - len(clip_names))]
        before = len(clip_names)
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                names = [str(n) for n in data["joint_names"]]
                absent = [name for name in joint_names if name not in names]
                if absent:
                    raise KeyError(f"{path} is missing model joints: {absent}")
                perm = [names.index(name) for name in joint_names]
                poses = np.asarray(data["root_poses"], dtype=np.float64)
                obj_poses = np.asarray(data["object_poses"], dtype=np.float64)
                q = _resample(np.asarray(data["joint_positions"], dtype=np.float64)[:, perm])
                rp = _resample(poses[:, :3])
                rq = _resample(poses[:, 3:7], quat=True)
                op = _resample(obj_poses[:, :3])
                oq = _resample(obj_poses[:, 3:7], quat=True)
            if not (len(q) == len(rp) == len(op)):
                raise ValueError(f"{path}: resampled joint/root/object lengths disagree.")

            rp_t = torch.tensor(rp, dtype=torch.float32)
            rq_t = torch.tensor(rq, dtype=torch.float32)
            q_t = torch.tensor(q, dtype=torch.float32)
            oq_t = torch.tensor(oq, dtype=torch.float32)
            rv = _linear_velocity(rp, dt)
            rw = _angular_velocity(rq_t, dt).numpy()
            qd = _linear_velocity(q, dt)
            ov = _linear_velocity(op, dt)
            ow = _angular_velocity(oq_t, dt).numpy()

            # Torch FK for the tracked and hand body trajectories. The recorded
            # root velocity feeds the FK so body twists match the trajectory.
            qd_t = torch.tensor(qd, dtype=torch.float32)
            rv_t = torch.tensor(rv, dtype=torch.float32)
            rw_t = torch.tensor(rw, dtype=torch.float32)
            bp, bq, bv, bw, hp = [], [], [], [], []
            chunk = 4096
            for start in range(0, rp_t.shape[0], chunk):
                end = min(start + chunk, rp_t.shape[0])
                out = kin.forward_with_hands(
                    rp_t[start:end],
                    rq_t[start:end],
                    rv_t[start:end],
                    rw_t[start:end],
                    q_t[start:end],
                    qd_t[start:end],
                )
                bp.append(out[0])
                bq.append(out[1])
                bv.append(out[2])
                bw.append(out[3])
                hp.append(out[4])

            joint_q.append(q.astype(np.float32))
            joint_qd.append(qd.astype(np.float32))
            root_p.append(rp.astype(np.float32))
            root_quat.append(rq.astype(np.float32))
            root_v.append(rv.astype(np.float32))
            root_w.append(rw.astype(np.float32))
            body_p.append(torch.cat(bp).numpy().astype(np.float32))
            body_quat.append(torch.cat(bq).numpy().astype(np.float32))
            body_v.append(torch.cat(bv).numpy().astype(np.float32))
            body_w.append(torch.cat(bw).numpy().astype(np.float32))
            hand_p.append(torch.cat(hp).numpy().astype(np.float32))
            object_p.append(op.astype(np.float32))
            object_quat.append(oq.astype(np.float32))
            object_v.append(ov.astype(np.float32))
            object_w.append(ow.astype(np.float32))
            clip_names.append(f"{obj}/{path.stem}")
            clip_lengths.append(int(q.shape[0]))
        print(f"{obj}: {len(clip_names) - before} clips ({split})", flush=True)

    if not clip_lengths:
        raise ValueError(f"No clips loaded for objects={objects} split={split}.")

    def cat(parts: list[np.ndarray]) -> np.ndarray:
        return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)

    lengths = np.asarray(clip_lengths, dtype=np.int64)
    starts = np.zeros_like(lengths)
    if len(lengths) > 1:
        np.cumsum(lengths[:-1], out=starts[1:])
    ends = starts + lengths
    q_all = cat(joint_q)
    if int(lengths.sum()) != q_all.shape[0]:
        raise ValueError("clip lengths do not sum to the flat frame count.")
    # The cache root is the pelvis, which FK also reports as tracked body 0.
    if not np.allclose(cat(root_p), cat(body_p)[:, pelvis], atol=2e-4):
        raise RuntimeError("FK pelvis disagrees with the recorded root pose.")

    write_npz_cache(
        out_path,
        joint_names=joint_names,
        clip_names=clip_names,
        body_names=TRACKED_BODIES,
        fps=int(round(TARGET_FPS)),
        starts=starts,
        ends=ends,
        root_p=cat(root_p),
        root_quat=cat(root_quat),
        root_v=cat(root_v),
        root_w=cat(root_w),
        q=q_all,
        qd=cat(joint_qd),
        body_p=cat(body_p),
        body_quat=cat(body_quat),
        body_v=cat(body_v),
        body_w=cat(body_w),
        object_p=cat(object_p),
        object_quat=cat(object_quat),
        object_v=cat(object_v),
        object_w=cat(object_w),
        hand_p=cat(hand_p),
        hand_body_names=np.array(XHAND_BODIES),
    )
    print(f"wrote {out_path} ({q_all.shape[0]} frames, {len(clip_names)} clips)", flush=True)
    return out_path


if __name__ == "__main__":
    WeaveXHandMotion.viser_play(objects=("smalltable",), max_clips=5)
