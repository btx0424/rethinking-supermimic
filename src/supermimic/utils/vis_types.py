"""Shared types for dataset motion visualization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class VisFrame:
    """Robot pose in the MuJoCo free-joint + hinge layout."""

    root_pos: Any  # (3,) float64
    root_quat: Any  # (4,) wxyz float64
    joint_q: Any  # (n_model_joints,) float64 in model joint order
    mocap_body: str | None = None
    mocap_pos: Any | None = None  # (3,)
    mocap_quat: Any | None = None  # (4,) wxyz
    # Optional hand–object contact bits, aligned with ``hand_contact_body_names``.
    hand_contact: Any | None = None  # (H,) {0,1}
    hand_contact_body_names: tuple[str, ...] | None = None


@dataclass(frozen=True)
class VisClipInfo:
    name: str
    length: int
    tags: dict[str, str] = field(default_factory=dict)


class VisSession(Protocol):
    fps: float

    @property
    def num_clips(self) -> int: ...

    def clip_info(self, clip: int) -> VisClipInfo: ...

    def frame(self, clip: int, local: int) -> VisFrame: ...

    def vis_model_key(self, clip: int) -> str:
        """Cache key for :meth:`Motion.compile_vis_model`."""
