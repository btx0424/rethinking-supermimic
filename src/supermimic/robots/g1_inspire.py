"""Inspire-hand G1 used for Weave tracking.

The recorded clips treat every finger hinge as an independent angle, so the
mimic equalities are removed and each hinge gets a position actuator. The 29
body joints reuse the mjlab G1 gains. The object is not part of this asset.
"""

from __future__ import annotations

import re

import mujoco
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


def inspire_spec() -> mujoco.MjSpec:
    """MJCF with mimic equalities removed so each recorded finger angle is free."""
    if not ROBOT_XML.is_file():
        raise FileNotFoundError(ROBOT_XML)
    text = _EQUALITY.sub("", ROBOT_XML.read_text())
    spec = mujoco.MjSpec.from_string(text)
    spec.compiler.meshdir = str(ROBOT_XML.parent.resolve())
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
