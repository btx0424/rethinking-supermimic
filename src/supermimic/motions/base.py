"""Dataset-specific motion loading builds a shared :class:`~supermimic.motion.MotionLib`."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import mujoco
    import torch

    from supermimic.motion import MotionLib

    from .vis_types import VisFrame, VisSession


class Motion(ABC):
    """Build or load a resampled reference library for one dataset and robot asset."""

    name: str

    @classmethod
    @abstractmethod
    def default_cache_path(cls) -> Path:
        """NPZ path written by :meth:`build_cache` and read by :meth:`load`."""

    @classmethod
    @abstractmethod
    def build_cache(cls, cache_path: Path | None = None, **kwargs) -> Path:
        """Resample raw clips, run FK if needed, and write ``cache_path``."""

    @classmethod
    def load(
        cls,
        cache_path: Path | None = None,
        device: str | torch.device = "cpu",
        **build_kwargs,
    ) -> MotionLib:
        from supermimic.motion import MotionLib

        path = cls.default_cache_path() if cache_path is None else cache_path
        if not path.is_file():
            cls.build_cache(path, **build_kwargs)
        return MotionLib.from_npz(path, device=device)

    # --- Visualization (Viser + mjviser) ---------------------------------

    @classmethod
    @abstractmethod
    def dataset_joint_names(cls) -> tuple[str, ...]:
        """Joint names in the order stored by this dataset."""

    @classmethod
    @abstractmethod
    def model_joint_names(cls) -> tuple[str, ...]:
        """Actuated hinge names in MuJoCo qpos write order."""

    @classmethod
    def joint_perm_dataset_to_model(cls) -> tuple[int, ...]:
        """Map model joint index → column in dataset ``joint_q`` / ``q`` arrays."""
        dataset_index = {name: i for i, name in enumerate(cls.dataset_joint_names())}
        return tuple(dataset_index[name] for name in cls.model_joint_names())

    @classmethod
    @abstractmethod
    def free_joint_name(cls) -> str:
        """MuJoCo free joint that carries the floating base."""

    @classmethod
    @abstractmethod
    def compile_vis_model(cls, clip: int, session: VisSession) -> mujoco.MjModel:
        """MuJoCo model for Viser. May depend on the clip (object mesh, etc.)."""

    @classmethod
    @abstractmethod
    def open_vis_session(cls, **kwargs: Any) -> VisSession:
        """In-memory clip list for the viewer (may bypass the training cache)."""

    @classmethod
    def apply_vis_frame(
        cls,
        data: mujoco.MjData,
        model: mujoco.MjModel,
        clip: int,
        frame: VisFrame,
    ) -> None:
        import mujoco as mj

        if frame.mocap_body is None:
            return
        body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, frame.mocap_body)
        if body_id < 0:
            raise KeyError(f"Mocap body {frame.mocap_body!r} missing from vis model.")
        mocap_id = int(model.body_mocapid[body_id])
        if mocap_id < 0:
            raise ValueError(f"Body {frame.mocap_body!r} is not a mocap body.")
        data.mocap_pos[mocap_id] = frame.mocap_pos
        data.mocap_quat[mocap_id] = frame.mocap_quat

    @classmethod
    def viser_play(cls, port: int = 8080, hz: float = 30.0, **session_kwargs: Any) -> None:
        from .vis import play_viser

        play_viser(cls, port=port, hz=hz, session_kwargs=session_kwargs)
