"""Floating rigid-body object for Weave HOI scenes."""

from __future__ import annotations

from pathlib import Path

import mujoco
from mjlab.entity import EntityCfg
from mjlab.utils.spec_config import CollisionCfg

# Prefer the authored XML+STL under object_hoi; fall back to Weave meshes.
_OBJECT_HOI_ROOT = Path("/mnt/workspace/btx0424/aa-projects/object_hoi/data/objects")
_WEAVE_ROOT = Path("/mnt/workspace/btx0424/Weave/objects")


def _object_xml(object_name: str) -> Path | None:
    path = _OBJECT_HOI_ROOT / object_name / f"{object_name}.xml"
    return path if path.is_file() else None


def _object_mesh(object_name: str) -> Path:
    for root in (_OBJECT_HOI_ROOT, _WEAVE_ROOT):
        for suffix in (".stl", ".obj"):
            path = root / object_name / f"{object_name}{suffix}"
            if path.is_file():
                return path
    raise FileNotFoundError(f"No mesh for object {object_name!r} under {_OBJECT_HOI_ROOT} or {_WEAVE_ROOT}.")


def object_spec(object_name: str, *, disable_gravity: bool = False) -> mujoco.MjSpec:
    """Free-joint rigid body named ``object`` (entity key), mesh from ``object_name``."""
    xml_path = _object_xml(object_name)
    if xml_path is not None:
        spec = mujoco.MjSpec.from_file(str(xml_path))
        # Normalize the free body name so the scene entity is always ``object``.
        for body in spec.bodies:
            if body.name and body.name != "world":
                body.name = "object"
                break
        for joint in spec.joints:
            if joint.name:
                joint.name = "object_freejoint"
                break
        for geom in spec.geoms:
            if geom.name and geom.name.endswith("_collision"):
                geom.name = "object_collision"
    else:
        mesh = _object_mesh(object_name)
        gravcomp = "1" if disable_gravity else "0"
        xml = f"""
<mujoco model="{object_name}">
  <compiler meshdir="{mesh.parent.resolve()}" angle="radian"/>
  <asset>
    <mesh name="{object_name}" file="{mesh.name}"/>
  </asset>
  <worldbody>
    <body name="object" pos="0 0 0" gravcomp="{gravcomp}">
      <freejoint name="object_freejoint"/>
      <geom type="mesh" mesh="{object_name}" contype="0" conaffinity="0"
            group="1" density="0" rgba="0.92 0.48 0.16 1"/>
      <geom name="object_collision" type="mesh" mesh="{object_name}"
            contype="1" conaffinity="1" rgba="0.92 0.48 0.16 1"/>
    </body>
  </worldbody>
</mujoco>
"""
        spec = mujoco.MjSpec.from_string(xml)
    if disable_gravity:
        for body in spec.bodies:
            if body.name and body.name != "world":
                body.gravcomp = 1.0
                break
    return spec


def object_cfg(object_name: str, *, disable_gravity: bool = False) -> EntityCfg:
    """mjlab entity for the HOI object (free joint, no articulation)."""

    def spec_fn() -> mujoco.MjSpec:
        return object_spec(object_name, disable_gravity=disable_gravity)

    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spec_fn=spec_fn,
        articulation=None,
        collisions=(
            CollisionCfg(
                geom_names_expr=(".*_collision.*",),
                contype=1,
                conaffinity=1,
                condim=3,
                priority=0,
                solref=(0.02, 1),
                friction=(1.0, 5e-3, 5e-4),
            ),
        ),
    )
