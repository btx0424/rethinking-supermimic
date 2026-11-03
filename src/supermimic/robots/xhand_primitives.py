"""Build a primitive-collided XHand MJCF from the current mesh-collided one.

Feet are already capsules; this replaces the *hand* mesh colliders with the
box/cylinder primitives from ``g1_hoi_learning``'s
``g1_29dof_rev_1_0_xhand_simplified.urdf``. Everything else -- links, joints,
inertials, visual meshes, the feet -- is copied unchanged, so the two XMLs differ
only in how the hands collide.

URDF fixed joints were merged into their parent bodies here, so a link's frame is
not always its body's frame. The link->body transform is read from the existing
mesh geom named ``<link>_collision_1``; links that are real bodies have identity.

Run:  PYTHONPATH=src python -m supermimic.robots.xhand_primitives
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[3]
SRC_XML = REPO / "robot_assets" / "unitree_g1" / "g1_29dof_rev_1_0_xhand.xml"
OUT_XML = REPO / "robot_assets" / "unitree_g1" / "g1_29dof_rev_1_0_xhand_simple.xml"
URDF = (
    REPO.parent
    / "g1_hoi_learning/source/g1_hoi_learning/g1_hoi_learning/assets/g1_xhand/"
    "g1_29dof_rev_1_0_xhand_simplified.urdf"
)
MESH_DIR = REPO / "robot_assets" / "unitree_g1" / "meshes"
# Hand links whose colliders the URDF fits primitives to.
HAND_KEYS = ("_hand", "_xhand_adapter")


# --------------------------------------------------------------------------- math

def quat_from_rpy(rpy: np.ndarray) -> np.ndarray:
    """URDF fixed-axis XYZ roll-pitch-yaw -> ``wxyz`` quaternion."""
    r, p, y = rpy
    cr, sr = np.cos(r / 2), np.sin(r / 2)
    cp, sp = np.cos(p / 2), np.sin(p / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    vec = q[1:]
    t = 2.0 * np.cross(vec, v)
    return v + q[0] * t + np.cross(vec, t)


def _vec(text: str | None, default=(0.0, 0.0, 0.0)) -> np.ndarray:
    if not text:
        return np.array(default, dtype=np.float64)
    return np.array([float(v) for v in text.split()], dtype=np.float64)


# --------------------------------------------------------------------------- URDF

def urdf_hand_colliders() -> dict[str, list[dict]]:
    """``link -> [geometry dicts]`` for hand links that the URDF gives colliders."""
    root = ET.parse(URDF).getroot()
    out: dict[str, list[dict]] = {}
    for link in root.findall("link"):
        name = link.get("name", "")
        if not any(k in name for k in HAND_KEYS):
            continue
        parts = []
        for col in link.findall("collision"):
            geom = col.find("geometry")
            if geom is None or len(geom) == 0:
                continue
            node = geom[0]
            origin = col.find("origin")
            xyz = _vec(origin.get("xyz") if origin is not None else None)
            rpy = _vec(origin.get("rpy") if origin is not None else None)
            part = {"xyz": xyz, "rpy": rpy}
            if node.tag == "box":
                part |= {"type": "box", "size": _vec(node.get("size"))[:3]}
            elif node.tag == "cylinder":
                part |= {
                    "type": "cylinder",
                    "radius": float(node.get("radius", "0")),
                    "length": float(node.get("length", "0")),
                }
            elif node.tag == "sphere":
                part |= {"type": "sphere", "radius": float(node.get("radius", "0"))}
            else:
                continue  # mesh: not wanted
            parts.append(part)
        if parts:
            out[name] = parts
    return out


# --------------------------------------------------------------------------- build

def build() -> Path:
    spec = mujoco.MjSpec.from_file(str(SRC_XML))

    # link -> (pos, quat) of the *body* frame, read from the existing mesh geoms.
    body_of_geom = {g.name: g.parent.name for g in spec.geoms if g.name}
    link_frame: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for geom in spec.geoms:
        name = geom.name or ""
        if "_collision_" not in name:
            continue
        link = name.rsplit("_collision_", 1)[0]
        link_frame[link] = (np.array(geom.pos, dtype=np.float64), np.array(geom.quat, dtype=np.float64))

    colliders = urdf_hand_colliders()
    print(f"URDF hand links with primitives: {len(colliders)}", flush=True)

    # --- drop every existing hand collision geom ---------------------------
    removed = 0
    for geom in list(spec.geoms):
        name = geom.name or ""
        if "_collision_" in name and any(k in name for k in HAND_KEYS):
            spec.delete(geom)
            removed += 1
    print(f"removed {removed} hand mesh colliders", flush=True)

    # --- add the URDF primitives in the owning body's frame ----------------
    added = 0
    for link, parts in sorted(colliders.items()):
        if link not in link_frame:
            raise KeyError(f"No body frame found for URDF link {link!r}.")
        if link not in body_of_geom.get(f"{link}_collision_1", ""):
            pass  # only used for the assertion above; body lookup is below
        body_name = body_of_geom.get(f"{link}_collision_1")
        if body_name is None:
            raise KeyError(f"No geom {link}_collision_1 to locate body {link!r}.")
        body = spec.body(body_name)
        pos_t, quat_t = link_frame[link]
        for index, part in enumerate(parts):
            q_local = quat_from_rpy(part["rpy"])
            geom = body.add_geom()
            if part["type"] == "box":
                sx, sy, sz = part["size"]
                geom.type = mujoco.mjtGeom.mjGEOM_BOX
                geom.size = [sx / 2.0, sy / 2.0, sz / 2.0]
                suffix = "box"
            elif part["type"] == "cylinder":
                geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
                geom.size = [part["radius"], part["length"] / 2.0, 0.0]
                suffix = "cyl"
            else:
                geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
                geom.size = [part["radius"], 0.0, 0.0]
                suffix = "sph"
            geom.pos = list(pos_t + quat_rotate(quat_t, part["xyz"]))
            geom.quat = list(quat_mul(quat_t, q_local))
            geom.name = f"{link}_{suffix}_{index}"
            # Same contact policy as the rest of the file: no self-collision,
            # contacts terrain and the free object.
            geom.contype = 0
            geom.conaffinity = 1
            geom.condim = 6
            geom.priority = 1
            geom.friction = [1.0, 5e-3, 5e-4]
            geom.solref = [0.01, 1.0]
            geom.group = 3
            geom.rgba = [0.055, 0.055, 0.063, 1.0]
            added += 1
    print(f"added {added} primitive hand colliders", flush=True)

    spec.modelname = "g1_29dof_rev_1_0_xhand_simple"
    xml = spec.to_xml().replace(str(MESH_DIR.resolve()), "meshes")
    OUT_XML.write_text(xml)
    print(f"wrote {OUT_XML} ({OUT_XML.stat().st_size} bytes)", flush=True)
    return OUT_XML


if __name__ == "__main__":
    build()
