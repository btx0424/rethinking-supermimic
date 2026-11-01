"""Motion datasets (LAFAN1 tracking, Weave object interaction, …)."""

from __future__ import annotations

from typing import Type

from .base import Motion
from .lafan1 import Lafan1Motion
from .weave import WeaveMotion

MOTION_REGISTRY: dict[str, Type[Motion]] = {
    Lafan1Motion.name: Lafan1Motion,
    WeaveMotion.name: WeaveMotion,
}


def get_motion(name: str) -> Type[Motion]:
    try:
        return MOTION_REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(MOTION_REGISTRY))
        raise KeyError(f"Unknown motion dataset {name!r}. Available: {known}.") from exc


__all__ = [
    "Motion",
    "Lafan1Motion",
    "WeaveMotion",
    "MOTION_REGISTRY",
    "get_motion",
]

# Re-export vis entry points on the dataset classes (`Motion.viser_play`).
