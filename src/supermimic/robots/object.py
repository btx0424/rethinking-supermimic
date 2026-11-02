"""Floating rigid-body object for Weave HOI scenes.

Object poses in the Weave rollouts are authored against the meshes under
``Weave/objects/<name>/<name>.obj`` (same assets ``genrl.data.view`` loads).
Those quaternions are MuJoCo/Viser ``wxyz``. The object_hoi STL/XML set uses a
different local frame (table height along Y), so it must not be used here.

Default MuJoCo density makes ``smalltable`` ~22 kg; object_hoi trains with a
0.2 kg object (Isaac ``set_masses(0.2)``). We match that so hand contacts can
actually displace the free body.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
from mjlab.entity import EntityCfg
from mjlab.utils.spec_config import CollisionCfg

_WEAVE_ROOT = Path("/mnt/workspace/btx0424/Weave/objects")

# Same mass object_hoi applies at runtime for the HOI table.
_OBJECT_MASS = 0.2


def object_mesh_path(object_name: str) -> Path:
    """Weave object triangle mesh. Matches ``object_quat_w`` in the rollout NPZ."""
    path = _WEAVE_ROOT / object_name / f"{object_name}.obj"
    if path.is_file():
        return path
    raise FileNotFoundError(
        f"No Weave mesh for object {object_name!r} at {path}. "
        f"Expected the same OBJ genrl loads from {_WEAVE_ROOT}."
    )


# Backward-compatible private alias.
_object_mesh = object_mesh_path


def _apply_object_mass(spec: mujoco.MjSpec, mass: float = _OBJECT_MASS) -> None:
    """Replace density-derived inertia with a light HOI mass (scaled inertia)."""
    # Compile once to read the mesh-derived COM / inertia, then overwrite.
    probe = spec.compile()
    body_id = mujoco.mj_name2id(probe, mujoco.mjtObj.mjOBJ_BODY, "object")
    if body_id < 0:
        raise RuntimeError("Object spec is missing body 'object'.")
    dens_mass = float(probe.body_mass[body_id])
    if dens_mass <= 0.0:
        raise RuntimeError(f"Object density mass is non-positive ({dens_mass}).")
    scale = mass / dens_mass
    ipos = probe.body_ipos[body_id]
    inertia = probe.body_inertia[body_id] * scale
    for body in spec.bodies:
        if body.name != "object":
            continue
        body.mass = mass
        body.ipos = ipos
        body.inertia = inertia
        for geom in body.geoms:
            geom.density = 0.0
        return
    raise RuntimeError("Object spec is missing body 'object'.")


def object_spec(object_name: str, *, disable_gravity: bool = False) -> mujoco.MjSpec:
    """Free-joint rigid body named ``object`` (entity key), Weave OBJ mesh."""
    mesh = object_mesh_path(object_name)
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
    _apply_object_mass(spec)
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
