"""Inspire-hand G1 used for Weave tracking.

The recorded clips treat every finger hinge as an independent angle, so the
mimic equalities are removed and each hinge gets a position actuator. The 29
body joints reuse the mjlab G1 gains. The free-joint object lives in
``supermimic.robots.object`` and is added by ``TrackingSim`` when the task
sets ``has_object``.
"""

from __future__ import annotations

import re

import mujoco
import numpy as np
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
    G1_ACTUATOR_4010,
    G1_ACTUATOR_5020,
    G1_ACTUATOR_7520_14,
    G1_ACTUATOR_7520_22,
    G1_ACTUATOR_ANKLE,
    G1_ACTUATOR_WAIST,
)
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.string import resolve_expr

from supermimic.motions.weave import ROBOT_XML

_EQUALITY = re.compile(r"<equality>.*?</equality>", re.DOTALL)
# Visual shells with a token mass and no joint. The mjlab G1 keeps these as geoms
# on the parent. Head and hand bases stay bodies: they carry real inertia.
_DUMMY_BODIES = ("pelvis_contour_link", "logo_link")

_FINGER_DRIVER = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*_thumb_proximal_yaw_joint",
        ".*_thumb_proximal_pitch_joint",
        ".*_index_proximal_joint",
        ".*_middle_proximal_joint",
        ".*_ring_proximal_joint",
        ".*_pinky_proximal_joint",
    ),
    stiffness=5.0,
    damping=0.5,
    effort_limit=1.0,
    armature=0.001,
)
# Recorded clips do not obey the URDF mimic ratios, so these hinges are driven too.
_FINGER_COUPLED = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*_thumb_intermediate_joint",
        ".*_thumb_distal_joint",
        ".*_index_intermediate_joint",
        ".*_middle_intermediate_joint",
        ".*_ring_intermediate_joint",
        ".*_pinky_intermediate_joint",
    ),
    stiffness=5.0,
    damping=0.5,
    effort_limit=1.0,
    armature=0.001,
)

ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        G1_ACTUATOR_5020,
        G1_ACTUATOR_7520_14,
        G1_ACTUATOR_7520_22,
        G1_ACTUATOR_4010,
        G1_ACTUATOR_WAIST,
        G1_ACTUATOR_ANKLE,
        _FINGER_DRIVER,
        _FINGER_COUPLED,
    ),
    soft_joint_pos_limit_factor=0.9,
)

ACTION_SCALE: dict[str, float] = {}
for _actuator in ARTICULATION.actuators:
    assert isinstance(_actuator, BuiltinPositionActuatorCfg)
    assert _actuator.effort_limit is not None
    for _pattern in _actuator.target_names_expr:
        ACTION_SCALE[_pattern] = 0.25 * _actuator.effort_limit / _actuator.stiffness


def _quat_mul(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    w0, x0, y0, z0 = q
    w1, x1, y1, z1 = r
    return np.array(
        [
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ]
    )


def _quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    xyz = q[1:]
    t = 2.0 * np.cross(xyz, v)
    return v + q[0] * t + np.cross(xyz, t)


def _find_body(root: mujoco.MjsBody, name: str) -> mujoco.MjsBody | None:
    for child in root.bodies:
        if child.name == name:
            return child
        found = _find_body(child, name)
        if found is not None:
            return found
    return None


def _merge_dummy_bodies(spec: mujoco.MjSpec) -> None:
    """Move a jointless visual body onto its parent and delete the body."""
    for name in _DUMMY_BODIES:
        body = _find_body(spec.worldbody, name)
        if body is None:
            continue
        if len(list(body.joints)) or len(list(body.bodies)):
            raise RuntimeError(f"{name} is not a jointless leaf, so it cannot be merged.")
        parent = body.parent
        parent_quat = np.asarray(body.quat, dtype=np.float64)
        parent_pos = np.asarray(body.pos, dtype=np.float64)
        for geom in list(body.geoms):
            merged = parent.add_geom()
            merged.name = geom.name or name
            merged.type = geom.type
            merged.meshname = geom.meshname
            merged.material = geom.material
            merged.pos = parent_pos + _quat_apply(parent_quat, np.asarray(geom.pos, dtype=np.float64))
            merged.quat = _quat_mul(parent_quat, np.asarray(geom.quat, dtype=np.float64))
            merged.size = geom.size
            merged.rgba = geom.rgba
            merged.group = geom.group
            merged.contype = geom.contype
            merged.conaffinity = geom.conaffinity
            merged.density = 0.0
        spec.delete(body)


def inspire_spec() -> mujoco.MjSpec:
    """MJCF with mimic equalities removed so each recorded finger angle is free."""
    if not ROBOT_XML.is_file():
        raise FileNotFoundError(ROBOT_XML)
    text = _EQUALITY.sub("", ROBOT_XML.read_text())
    spec = mujoco.MjSpec.from_string(text)
    spec.compiler.meshdir = str(ROBOT_XML.parent.resolve())
    _merge_dummy_bodies(spec)
    return spec


def hinge_joint_names(model: mujoco.MjModel) -> tuple[str, ...]:
    names: list[str] = []
    for joint_id in range(model.njnt):
        if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name:
            names.append(name)
    return tuple(names)


def compile_g1_inspire() -> tuple[mujoco.MjModel, tuple[str, ...]]:
    model = inspire_spec().compile()
    return model, hinge_joint_names(model)


def robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.76), joint_pos={".*": 0.0}),
        spec_fn=inspire_spec,
        articulation=ARTICULATION,
    )


def action_scale(joint_names: tuple[str, ...]) -> list[float]:
    scale = list(resolve_expr(ACTION_SCALE, joint_names))
    missing = [name for name, value in zip(joint_names, scale) if value is None]
    if missing:
        raise KeyError(f"No action scale for joints {missing}")
    return [float(value) for value in scale]
