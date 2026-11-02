"""Floating rigid-body objects for the Weave G1-Xhand-OMOMO scenes.

Unlike the older ``Weave`` release, this one ships *primitive-fitted* colliders:
``assets/primitives/<object>.json`` lists boxes / cylinders / capsules in the
object-local frame (``assets/object_rotations.json`` maps the original frame to
the current one). Using them instead of the visual mesh keeps the contact
geometry convex and cheap, which is what the release's Isaac task does.

The visual mesh is still attached, with ``contype=0`` so it never collides.

Mass follows ``g1_hoi_learning``'s ``COMMON_MASS_PROPS``: density 200 kg/m³ on the
primitive volume, except ``largebox`` at 25 (≈1 kg in that task's note). That
yields 0.16–6.6 kg here — heavy enough to be moved by the fingers, unlike the
22 kg a mesh-density table would give.
"""

from __future__ import annotations

import json

import mujoco
import numpy as np
from mjlab.entity import EntityCfg
from mjlab.utils.spec_config import CollisionCfg

# Per-object density, matching g1_hoi_learning.objects.object_cfg.
OBJECT_DENSITY: dict[str, float] = {"largebox": 25.0}
DEFAULT_DENSITY = 200.0
OBJECT_NAME = "object"
FREE_JOINT = "object_freejoint"


def _primitive_xml(part: dict, index: int) -> str:
    """One primitive as a MuJoCo geom, placed by its primitive-to-object transform."""
    kind = part["type"]
    size = np.asarray(part["size"], dtype=np.float64)
    position = np.asarray(part["position"], dtype=np.float64)
    rotation = np.asarray(part["rotation"], dtype=np.float64)
    quat = _quat_from_mat(rotation)
    pos = " ".join(f"{v:.9g}" for v in position)
    quat_s = " ".join(f"{v:.9g}" for v in quat)
    if kind == "box":
        size_s = " ".join(f"{v / 2.0:.9g}" for v in size)  # MuJoCo box size = half extents
        geom_type = "box"
    elif kind == "cylinder":
        radius, height = size
        size_s = f"{radius:.9g} {height / 2.0:.9g}"  # MuJoCo cylinder: radius, half height
        geom_type = "cylinder"
    elif kind == "capsule":
        radius, span = size
        size_s = f"{radius:.9g} {span / 2.0:.9g}"  # MuJoCo capsule: radius, half cylinder
        geom_type = "capsule"
    else:
        raise ValueError(f"Unknown primitive type {kind!r}.")
    return (
        f'      <geom name="{OBJECT_NAME}_collision_{index}" type="{geom_type}" '
        f'size="{size_s}" pos="{pos}" quat="{quat_s}" density="0" group="0"/>\n'
    )


def _quat_from_mat(m: np.ndarray) -> np.ndarray:
    """``wxyz`` quaternion from a rotation matrix (Shepperd's method)."""
    m = np.asarray(m, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float64)
    return quat / np.linalg.norm(quat)


def object_spec(object_name: str, *, disable_gravity: bool = False) -> mujoco.MjSpec:
    """Free-joint body ``object``: visual mesh plus primitive colliders."""
    from supermimic.motions.weave_xhand import load_primitives, object_mesh_path

    parts = load_primitives(object_name)
    mesh = object_mesh_path(object_name).resolve()
    density = OBJECT_DENSITY.get(object_name, DEFAULT_DENSITY)
    gravcomp = 1 if disable_gravity else 0
    collisions = "".join(_primitive_xml(part, i) for i, part in enumerate(parts))
    # Primitive geoms carry the mass (``density``), the visual mesh none.
    xml = f"""
<mujoco model="{object_name}">
  <compiler meshdir="{mesh.parent}" angle="radian"/>
  <asset>
    <mesh name="{object_name}" file="{mesh.name}"/>
  </asset>
  <worldbody>
    <body name="{OBJECT_NAME}" pos="0 0 0" gravcomp="{gravcomp}">
      <freejoint name="{FREE_JOINT}"/>
      <geom name="{OBJECT_NAME}_visual" type="mesh" mesh="{object_name}"
            contype="0" conaffinity="0" group="2" density="0" rgba="0.92 0.48 0.16 1"/>
{collisions}    </body>
  </worldbody>
</mujoco>
"""
    spec = mujoco.MjSpec.from_string(xml)
    # ``density`` on the primitive geoms only; scale to the target density.
    for geom in spec.geoms:
        if geom.name and geom.name.startswith(f"{OBJECT_NAME}_collision"):
            geom.density = density
    return spec


def object_cfg(object_name: str, *, disable_gravity: bool = False) -> EntityCfg:
    """mjlab entity for the HOI object (free joint, primitive colliders)."""

    def spec_fn() -> mujoco.MjSpec:
        return object_spec(object_name, disable_gravity=disable_gravity)

    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5), rot=(1.0, 0.0, 0.0, 0.0)),
        spec_fn=spec_fn,
        articulation=None,
        collisions=(
            CollisionCfg(
                geom_names_expr=(f"{OBJECT_NAME}_collision_.*",),
                contype=1,
                conaffinity=1,
                condim=3,
                priority=0,
                solref=(0.02, 1),
                friction=(1.0, 5e-3, 5e-4),
            ),
        ),
    )


def object_mass(object_name: str) -> float:
    """Compiled mass of the object's primitives, for logging and sanity checks."""
    model = object_spec(object_name).compile()
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, OBJECT_NAME)
    if body_id < 0:
        raise RuntimeError(f"Object spec is missing body {OBJECT_NAME!r}.")
    return float(model.body_mass[body_id])


def primitive_summary(object_name: str) -> dict[str, int]:
    """Count of each primitive type, for the task docstring and logs."""
    from supermimic.motions.weave_xhand import load_primitives

    counts: dict[str, int] = {}
    for part in load_primitives(object_name):
        counts[part["type"]] = counts.get(part["type"], 0) + 1
    return counts


if __name__ == "__main__":
    import sys

    from supermimic.motions.weave_xhand import AVAILABLE_OBJECTS

    names = sys.argv[1:] or list(AVAILABLE_OBJECTS)
    for name in names:
        print(f"{name:12s} {object_mass(name):6.3f} kg  {json.dumps(primitive_summary(name))}")
