"""
Loading, processing, caching, and visualization of Weave motions.

Target asset:
``aa-projects/object_hoi/src/assets/unitree_g1/g1_29dof_rev_1_0_with_inspire_hand_DFQ.xml``

Raw rollouts live under ``/mnt/workspace/btx0424/Weave`` (see ``genrl.data.weave``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .base import Motion
from .vis_types import VisClipInfo, VisFrame, VisSession

DEFAULT_ROOT = Path("/mnt/workspace/btx0424/Weave")
_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
# Robot-only cache kept for backward compatibility; object HOI uses a tagged path.
CACHE_PATH = _DATA_DIR / "weave_g1_hand.npz"


def cache_path_for(objects: tuple[str, ...]) -> Path:
    """One NPZ per object set so smalltable and smallbox do not share a cache."""
    if not objects:
        raise ValueError("objects must be non-empty.")
    return _DATA_DIR / f"weave_g1_hand_{'_'.join(objects)}.npz"

ROBOT_XML = Path(
    "/mnt/workspace/btx0424/aa-projects/object_hoi/src/assets/unitree_g1/"
    "g1_29dof_rev_1_0_with_inspire_hand_DFQ.xml"
)
OBJECT_MESH_ROOT = DEFAULT_ROOT / "objects"

# Weave rollout NPZ column order (dataset card).
DATASET_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
    "L_index_proximal_joint",
    "L_middle_proximal_joint",
    "L_pinky_proximal_joint",
    "L_ring_proximal_joint",
    "L_thumb_proximal_yaw_joint",
    "R_index_proximal_joint",
    "R_middle_proximal_joint",
    "R_pinky_proximal_joint",
    "R_ring_proximal_joint",
    "R_thumb_proximal_yaw_joint",
    "L_index_intermediate_joint",
    "L_middle_intermediate_joint",
    "L_pinky_intermediate_joint",
    "L_ring_intermediate_joint",
    "L_thumb_proximal_pitch_joint",
    "R_index_intermediate_joint",
    "R_middle_intermediate_joint",
    "R_pinky_intermediate_joint",
    "R_ring_intermediate_joint",
    "R_thumb_proximal_pitch_joint",
    "L_thumb_intermediate_joint",
    "R_thumb_intermediate_joint",
    "L_thumb_distal_joint",
    "R_thumb_distal_joint",
)

DATASET_BODY_NAMES: tuple[str, ...] = (
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "L_index_proximal",
    "L_middle_proximal",
    "L_pinky_proximal",
    "L_ring_proximal",
    "L_thumb_proximal_base",
    "R_index_proximal",
    "R_middle_proximal",
    "R_pinky_proximal",
    "R_ring_proximal",
    "R_thumb_proximal_base",
    "L_index_intermediate",
    "L_middle_intermediate",
    "L_pinky_intermediate",
    "L_ring_intermediate",
    "L_thumb_proximal",
    "R_index_intermediate",
    "R_middle_intermediate",
    "R_pinky_intermediate",
    "R_ring_intermediate",
    "R_thumb_proximal",
    "L_thumb_intermediate",
    "R_thumb_intermediate",
    "L_thumb_distal",
    "R_thumb_distal",
)

PELVIS_BODY = "pelvis"
FREE_JOINT = "floating_base_joint"

_MODEL_JOINT_NAMES: tuple[str, ...] | None = None


def _concat_arrays(chunks: list[np.ndarray]) -> np.ndarray:
    if len(chunks) == 1:
        return chunks[0]
    return np.concatenate(chunks, axis=0)


def _hinge_joint_names(model: mujoco.MjModel) -> tuple[str, ...]:
    names: list[str] = []
    for jid in range(model.njnt):
        if int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name:
            names.append(name)
    return tuple(names)


def _compile_robot_model() -> mujoco.MjModel:
    if not ROBOT_XML.is_file():
        raise FileNotFoundError(ROBOT_XML)
    return mujoco.MjModel.from_xml_path(str(ROBOT_XML))


def _object_mesh(object_name: str, mesh_root: Path = OBJECT_MESH_ROOT) -> Path:
    path = mesh_root / object_name / f"{object_name}.obj"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _scene_xml(robot_xml: Path, object_name: str, object_mesh: Path) -> str:
    xml = robot_xml.read_text()
    meshdir = robot_xml.parent.resolve()
    if 'meshdir="' not in xml:
        xml = xml.replace(
            '<compiler angle="radian"/>',
            f'<compiler angle="radian" meshdir="{meshdir}"/>',
        )
    asset = f'\n    <mesh name="{object_name}_vis" file="{object_mesh.resolve()}"/>\n  '
    body = f"""
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.85 0.85 0.85 1"/>
    <body name="{object_name}" mocap="true" pos="0 0 0" quat="1 0 0 0">
      <geom type="mesh" mesh="{object_name}_vis" rgba="0.92 0.48 0.16 1" contype="0" conaffinity="0" group="1"/>
    </body>
  """
    return xml.replace("</asset>", f"{asset}</asset>").replace("</worldbody>", f"{body}</worldbody>")


class WeaveMotion(Motion):
    name = "weave"

    @classmethod
    def default_cache_path(cls) -> Path:
        return cache_path_for(("smalltable",))

    @classmethod
    def build_cache(
        cls,
        cache_path: Path | None = None,
        root: Path = DEFAULT_ROOT,
        objects: tuple[str, ...] = ("smalltable",),
        split: str = "train",
        **_,
    ) -> Path:
        path = cache_path_for(objects) if cache_path is None else cache_path
        return build_cache(path, root=root, objects=objects, split=split)

    @classmethod
    def load(
        cls,
        cache_path: Path | None = None,
        device: str = "cpu",
        objects: tuple[str, ...] = ("smalltable",),
        **build_kwargs,
    ):
        from supermimic.motion import MotionLib

        path = cache_path_for(objects) if cache_path is None else cache_path
        if not path.is_file():
            cls.build_cache(path, objects=objects, **build_kwargs)
        lib = MotionLib.from_npz(path, device=device)
        if getattr(lib, "object_p", None) is None:
            raise RuntimeError(
                f"{path} has no object fields; delete it and rebuild with WeaveMotion.build_cache."
            )
        if getattr(lib, "hand_p", None) is None:
            raise RuntimeError(
                f"{path} has no hand_p; delete it and rebuild with WeaveMotion.build_cache."
            )
        return lib

    @classmethod
    def dataset_joint_names(cls) -> tuple[str, ...]:
        return DATASET_JOINT_NAMES

    @classmethod
    def model_joint_names(cls) -> tuple[str, ...]:
        global _MODEL_JOINT_NAMES
        if _MODEL_JOINT_NAMES is None:
            _MODEL_JOINT_NAMES = _hinge_joint_names(_compile_robot_model())
        missing = [name for name in _MODEL_JOINT_NAMES if name not in DATASET_JOINT_NAMES]
        if missing:
            raise KeyError(f"Weave dataset joints missing from the inspire-hand model: {missing}")
        return _MODEL_JOINT_NAMES

    @classmethod
    def free_joint_name(cls) -> str:
        return FREE_JOINT

    @classmethod
    def compile_vis_model(cls, clip: int, session: VisSession) -> mujoco.MjModel:
        if not isinstance(session, _WeaveVisSession):
            raise TypeError("Weave vis session expected.")
        object_name = session.object_for_clip(clip)
        xml = _scene_xml(ROBOT_XML, object_name, _object_mesh(object_name))
        return mujoco.MjModel.from_xml_string(xml)

    @classmethod
    def open_vis_session(
        cls,
        root: Path = DEFAULT_ROOT,
        objects: Sequence[str] = ("smalltable", "smallbox"),
        split: str = "train",
        max_clips: int | None = 200,
        **_,
    ) -> VisSession:
        return _WeaveVisSession.load(root=root, objects=tuple(objects), split=split, max_clips=max_clips)


@dataclass
class _WeaveVisSession:
    fps: float
    clip_starts: np.ndarray
    clip_lengths: np.ndarray
    clip_names: np.ndarray
    object_names: np.ndarray
    joint_pos: np.ndarray
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray
    object_pos_w: np.ndarray
    object_quat_w: np.ndarray
    joint_perm: np.ndarray
    pelvis_index: int

    @property
    def num_clips(self) -> int:
        return int(self.clip_lengths.shape[0])

    @classmethod
    def load(
        cls,
        *,
        root: Path,
        objects: tuple[str, ...],
        split: str,
        max_clips: int | None,
    ) -> _WeaveVisSession:
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}.")
        array_keys = (
            "joint_pos",
            "body_pos_w",
            "body_quat_w",
            "object_pos_w",
            "object_quat_w",
        )
        fps: float | None = None
        clip_names: list[str] = []
        object_names_out: list[str] = []
        clip_lengths_list: list[int] = []
        chunks: dict[str, list[np.ndarray]] = {key: [] for key in array_keys}
        clips_loaded = 0

        for obj in objects:
            if max_clips is not None and clips_loaded >= max_clips:
                break
            path = root / "rollout" / split / f"{obj}_rollout.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=True) as data:
                file_fps = float(np.asarray(data["fps"]).reshape(-1)[0])
                if fps is None:
                    fps = file_fps
                elif file_fps != fps:
                    raise ValueError(f"{path} is {file_fps} fps; expected {fps}.")
                lengths = np.asarray(data["motion_lengths"], dtype=np.int64)
                motion_names = np.asarray(data["motion_names"])
                object_names = np.asarray(data["object_names"])
                n_take = int(lengths.shape[0])
                if max_clips is not None:
                    n_take = min(n_take, max_clips - clips_loaded)
                if n_take <= 0:
                    break
                frame_end = int(lengths[:n_take].sum())
                print(
                    f"loading {path.name}: {n_take} clips, {frame_end} frames",
                    flush=True,
                )
                for key in array_keys:
                    chunks[key].append(np.asarray(data[key][:frame_end]))
                clip_names.extend(str(motion_names[i]) for i in range(n_take))
                object_names_out.extend(str(object_names[i]) for i in range(n_take))
                clip_lengths_list.extend(int(lengths[i]) for i in range(n_take))
                clips_loaded += n_take

        if fps is None or not clip_lengths_list:
            raise ValueError("No Weave clips loaded; check root/objects/split.")

        clip_lengths = np.asarray(clip_lengths_list, dtype=np.int64)
        clip_starts = np.empty(len(clip_lengths), dtype=np.int64)
        clip_starts[0] = 0
        if len(clip_lengths) > 1:
            np.cumsum(clip_lengths[:-1], out=clip_starts[1:])
        total_frames = int(clip_lengths.sum())

        perm = np.asarray(WeaveMotion.joint_perm_dataset_to_model(), dtype=np.int64)
        pelvis_index = DATASET_BODY_NAMES.index(PELVIS_BODY)
        print(
            f"weave vis: {len(clip_lengths)} clips, {total_frames} frames @ {fps:g} Hz",
            flush=True,
        )
        return cls(
            fps=fps,
            clip_starts=clip_starts,
            clip_lengths=clip_lengths,
            clip_names=np.asarray(clip_names),
            object_names=np.asarray(object_names_out),
            joint_pos=_concat_arrays(chunks["joint_pos"]),
            body_pos_w=_concat_arrays(chunks["body_pos_w"]),
            body_quat_w=_concat_arrays(chunks["body_quat_w"]),
            object_pos_w=_concat_arrays(chunks["object_pos_w"]),
            object_quat_w=_concat_arrays(chunks["object_quat_w"]),
            joint_perm=perm,
            pelvis_index=pelvis_index,
        )

    def object_for_clip(self, clip: int) -> str:
        return str(self.object_names[clip])

    def clip_info(self, clip: int) -> VisClipInfo:
        return VisClipInfo(
            name=str(self.clip_names[clip]),
            length=int(self.clip_lengths[clip]),
            tags={"object": self.object_for_clip(clip)},
        )

    def vis_model_key(self, clip: int) -> str:
        return self.object_for_clip(clip)

    def frame(self, clip: int, local: int) -> VisFrame:
        index = int(self.clip_starts[clip]) + int(local)
        joint_q = self.joint_pos[index, self.joint_perm].astype(np.float64)
        return VisFrame(
            self.body_pos_w[index, self.pelvis_index].astype(np.float64),
            self.body_quat_w[index, self.pelvis_index].astype(np.float64),
            joint_q,
            mocap_body=self.object_for_clip(clip),
            mocap_pos=self.object_pos_w[index].astype(np.float64),
            mocap_quat=self.object_quat_w[index].astype(np.float64),
        )


_AVAILABLE_OBJECTS = (
    "clothesstand",
    "floorlamp",
    "largebox",
    "largetable",
    "smallbox",
    "smalltable",
    "tripod",
    "whitechair",
    "woodchair",
)


def build_cache(
    cache_path: Path | None = None,
    root: Path = DEFAULT_ROOT,
    objects: tuple[str, ...] = ("smalltable",),
    split: str = "train",
) -> Path:
    """Pack Weave rollouts into a MotionLib cache with object pose and twist.

    Clips are already 50 Hz. Joint columns are reordered onto the inspire-hand
    hinge order. The root is the pelvis, and the 14 tracked bodies are sliced
    out of the recorded body trajectory.
    """
    from supermimic.kinematics import HAND_BODIES, TRACKED_BODIES
    from supermimic.motion import TARGET_FPS, write_npz_cache
    from supermimic.robots.g1_inspire import compile_g1_inspire

    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}.")
    unknown = [name for name in objects if name not in _AVAILABLE_OBJECTS]
    if unknown:
        raise ValueError(f"Unknown objects {unknown}. Available: {list(_AVAILABLE_OBJECTS)}.")
    if not objects:
        raise ValueError("Select at least one object.")
    out_path = cache_path_for(objects) if cache_path is None else cache_path

    _model, joint_names = compile_g1_inspire()
    if len(joint_names) != len(DATASET_JOINT_NAMES):
        raise RuntimeError(
            f"Inspire-hand model has {len(joint_names)} hinges; "
            f"the dataset lists {len(DATASET_JOINT_NAMES)}."
        )
    dataset_index = {name: i for i, name in enumerate(DATASET_JOINT_NAMES)}
    missing = [name for name in joint_names if name not in dataset_index]
    if missing:
        raise KeyError(f"Dataset is missing model joints: {missing}")
    joint_perm = np.asarray([dataset_index[name] for name in joint_names], dtype=np.int64)
    body_index = {name: i for i, name in enumerate(DATASET_BODY_NAMES)}
    missing_bodies = [name for name in TRACKED_BODIES if name not in body_index]
    if missing_bodies:
        raise KeyError(f"Dataset is missing tracked bodies: {missing_bodies}")
    missing_hands = [name for name in HAND_BODIES if name not in body_index]
    if missing_hands:
        raise KeyError(f"Dataset is missing hand bodies: {missing_hands}")
    body_perm = np.asarray([body_index[name] for name in TRACKED_BODIES], dtype=np.int64)
    hand_perm = np.asarray([body_index[name] for name in HAND_BODIES], dtype=np.int64)
    pelvis = body_index[PELVIS_BODY]

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
    fps: int | None = None

    for obj in objects:
        path = root / "rollout" / split / f"{obj}_rollout.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=True) as data:
            file_fps = int(np.asarray(data["fps"]).reshape(-1)[0])
            if fps is None:
                fps = file_fps
            elif file_fps != fps:
                raise ValueError(f"{path} is {file_fps} fps; expected {fps}.")
            if file_fps != TARGET_FPS:
                raise ValueError(f"{path} is {file_fps} fps; the cache expects {TARGET_FPS}.")
            lengths = np.asarray(data["motion_lengths"], dtype=np.int64)
            names = np.asarray(data["motion_names"])
            print(
                f"loading {path.name}: {len(lengths)} clips, {int(lengths.sum())} frames",
                flush=True,
            )
            q = np.asarray(data["joint_pos"])[:, joint_perm]
            qd = np.asarray(data["joint_vel"])[:, joint_perm]
            pos = np.asarray(data["body_pos_w"])
            quat = np.asarray(data["body_quat_w"])
            lin = np.asarray(data["body_lin_vel_w"])
            ang = np.asarray(data["body_ang_vel_w"])
            obj_p = np.asarray(data["object_pos_w"])
            obj_quat = np.asarray(data["object_quat_w"])
            obj_v = np.asarray(data["object_lin_vel_w"])
            obj_w = np.asarray(data["object_ang_vel_w"])
        joint_q.append(q.astype(np.float32, copy=False))
        joint_qd.append(qd.astype(np.float32, copy=False))
        root_p.append(pos[:, pelvis].astype(np.float32, copy=False))
        root_quat.append(quat[:, pelvis].astype(np.float32, copy=False))
        root_v.append(lin[:, pelvis].astype(np.float32, copy=False))
        root_w.append(ang[:, pelvis].astype(np.float32, copy=False))
        body_p.append(pos[:, body_perm].astype(np.float32, copy=False))
        body_quat.append(quat[:, body_perm].astype(np.float32, copy=False))
        body_v.append(lin[:, body_perm].astype(np.float32, copy=False))
        body_w.append(ang[:, body_perm].astype(np.float32, copy=False))
        hand_p.append(pos[:, hand_perm].astype(np.float32, copy=False))
        object_p.append(obj_p.astype(np.float32, copy=False))
        object_quat.append(obj_quat.astype(np.float32, copy=False))
        object_v.append(obj_v.astype(np.float32, copy=False))
        object_w.append(obj_w.astype(np.float32, copy=False))
        clip_names.extend(str(name) for name in names)
        clip_lengths.extend(int(length) for length in lengths)

    if fps is None or not clip_lengths:
        raise ValueError("No Weave clips loaded; check root/objects/split.")

    def cat(parts: list[np.ndarray]) -> np.ndarray:
        return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)

    lengths_arr = np.asarray(clip_lengths, dtype=np.int64)
    starts = np.empty(len(lengths_arr), dtype=np.int64)
    starts[0] = 0
    if len(lengths_arr) > 1:
        np.cumsum(lengths_arr[:-1], out=starts[1:])
    ends = starts + lengths_arr
    q_all = cat(joint_q)
    if int(lengths_arr.sum()) != q_all.shape[0]:
        raise ValueError("motion_lengths do not sum to the flat frame count.")
    write_npz_cache(
        out_path,
        joint_names=joint_names,
        clip_names=clip_names,
        body_names=TRACKED_BODIES,
        fps=fps,
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
        hand_body_names=np.array(HAND_BODIES),
    )
    print(f"wrote {out_path} ({q_all.shape[0]} frames, {len(clip_names)} clips)", flush=True)
    return out_path


if __name__ == "__main__":
    WeaveMotion.viser_play(objects=("smalltable",), max_clips=20)
