"""Viser playback for :class:`~supermimic.motions.base.Motion` datasets."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import viser
from mjviser import ViserMujocoScene

if TYPE_CHECKING:
    from supermimic.motions.base import Motion
    from .vis_types import VisFrame, VisSession

_GROUND_Z = -0.02


def _geom_group_kinds(model: mujoco.MjModel) -> tuple[set[int], set[int]]:
    """Visual and collision geom groups for the display toggles.

    A group is collision if any of its geoms has nonzero contype/conaffinity,
    otherwise visual. Mixed groups (XHand puts active hand/foot colliders and
    disabled body ``*_collision_*`` meshes in group 3) are treated as collision
    so the "Collision geoms" checkbox can show them; mjviser only toggles by
    group, not per geom.
    """
    visual: set[int] = set()
    collision: set[int] = set()
    for geom_id in range(model.ngeom):
        group = int(model.geom_group[geom_id])
        if group >= 6:
            continue
        if int(model.geom_contype[geom_id]) or int(model.geom_conaffinity[geom_id]):
            collision.add(group)
        else:
            visual.add(group)
    mixed = visual & collision
    return visual - mixed, collision


def _qpos_addresses(model: mujoco.MjModel, joint_names: tuple[str, ...]) -> list[int]:
    adr: dict[str, int] = {}
    for jid in range(model.njnt):
        if int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        adr[name] = int(model.jnt_qposadr[jid])
    missing = [name for name in joint_names if name not in adr]
    if missing:
        raise KeyError(f"MuJoCo model is missing joints: {missing}")
    return [adr[name] for name in joint_names]


def _root_qposadr(model: mujoco.MjModel, free_joint: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, free_joint)
    if jid < 0:
        for candidate in range(model.njnt):
            if int(model.jnt_type[candidate]) == int(mujoco.mjtJoint.mjJNT_FREE):
                return int(model.jnt_qposadr[candidate])
        raise KeyError(f"Free joint {free_joint!r} not found in the vis model.")
    return int(model.jnt_qposadr[jid])


def write_robot_pose(
    data: mujoco.MjData,
    root_adr: int,
    joint_adr: list[int],
    frame: VisFrame,
) -> None:
    data.qpos[root_adr : root_adr + 3] = frame.root_pos
    data.qpos[root_adr + 3 : root_adr + 7] = frame.root_quat
    data.qpos[joint_adr] = frame.joint_q


def forward_scene(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    scene: ViserMujocoScene,
    motion: type[Motion],
    clip: int,
    frame: VisFrame,
    *,
    root_adr: int,
    joint_adr: list[int],
) -> None:
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    write_robot_pose(data, root_adr, joint_adr, frame)
    motion.apply_vis_frame(data, model, clip, frame)
    mujoco.mj_forward(model, data)
    if model.nmocap > 0:
        scene.update_from_mjdata(data)
    else:
        xpos = data.xpos.copy()
        xmat = data.xmat.reshape(-1, 3, 3).copy()
        scene.update_from_arrays(xpos[None], xmat[None])


def _hand_contact_positions(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    frame: VisFrame,
) -> np.ndarray:
    """World positions of bodies marked in contact on ``frame``, shape ``(K, 3)``."""
    names = frame.hand_contact_body_names
    bits = frame.hand_contact
    if names is None or bits is None:
        return np.zeros((0, 3), dtype=np.float64)
    bits = np.asarray(bits, dtype=np.float64).reshape(-1)
    if bits.shape[0] != len(names):
        raise ValueError(
            f"hand_contact length {bits.shape[0]} does not match "
            f"hand_contact_body_names ({len(names)})."
        )
    positions = []
    for name, bit in zip(names, bits):
        if bit <= 0.0:
            continue
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            continue
        positions.append(np.asarray(data.xpos[body_id], dtype=np.float64))
    if not positions:
        return np.zeros((0, 3), dtype=np.float64)
    return np.stack(positions, axis=0)


def play_viser(
    motion_cls: type[Motion],
    *,
    port: int = 8080,
    hz: float = 30.0,
    session_kwargs: dict | None = None,
) -> None:
    session_kwargs = dict(session_kwargs or {})
    session: VisSession = motion_cls.open_vis_session(**session_kwargs)
    model_cache: dict[str, mujoco.MjModel] = {}
    layout_cache: dict[str, tuple[mujoco.MjData, int, list[int], ViserMujocoScene | None]] = {}

    def ensure_scene(clip: int) -> tuple[mujoco.MjModel, mujoco.MjData, int, list[int], ViserMujocoScene]:
        key = session.vis_model_key(clip)
        if key not in model_cache:
            model_cache[key] = motion_cls.compile_vis_model(clip, session)
        model = model_cache[key]
        if key not in layout_cache:
            data = mujoco.MjData(model)
            root_adr = _root_qposadr(model, motion_cls.free_joint_name())
            joint_adr = _qpos_addresses(model, motion_cls.model_joint_names())
            layout_cache[key] = (data, root_adr, joint_adr, None)
        data, root_adr, joint_adr, _ = layout_cache[key]
        return model, data, root_adr, joint_adr, key

    server = viser.ViserServer(port=port, label=f"SuperMimic · {motion_cls.name}")
    server.scene.set_up_direction("+z")
    server.scene.add_grid(
        "/ground",
        plane="xy",
        cell_size=0.5,
        section_size=2.0,
        infinite_grid=True,
        fade_distance=40.0,
        shadow_opacity=0.15,
        plane_opacity=0.25,
        position=(0.0, 0.0, _GROUND_Z),
    )

    clip_labels = [session.clip_info(c).name for c in range(session.num_clips)]
    fps = float(session.fps)
    state = {"clip": 0, "frame": 0, "scene_key": None}

    with server.gui.add_folder("Clip"):
        clip_box = server.gui.add_dropdown(
            "Motion",
            options=tuple(clip_labels),
            initial_value=clip_labels[0] if clip_labels else "",
        )
        info = server.gui.add_markdown("")
    with server.gui.add_folder("Playback"):
        playing = server.gui.add_checkbox("Play", initial_value=True)
        loop = server.gui.add_checkbox("Loop", initial_value=True)
        speed = server.gui.add_slider("Speed", min=0.1, max=4.0, step=0.1, initial_value=1.0)
        max_frame = max(session.clip_info(0).length - 1, 0)
        frame_slider = server.gui.add_slider("Frame", min=0, max=max_frame, step=1, initial_value=0)
    with server.gui.add_folder("Display"):
        show_visual = server.gui.add_checkbox("Visual geoms", initial_value=True)
        show_collision = server.gui.add_checkbox("Collision geoms", initial_value=False)
        show_hand_contact = server.gui.add_checkbox(
            "Hand–object contact", initial_value=True
        )

    scene: ViserMujocoScene | None = None
    active_key: str | None = None
    contact_points = server.scene.add_point_cloud(
        "/hand_contact_points",
        points=np.zeros((0, 3), dtype=np.float64),
        colors=(240, 60, 40),
        point_size=0.022,
        point_shape="circle",
    )

    def apply_geom_visibility(vis_scene: ViserMujocoScene, model: mujoco.MjModel) -> None:
        visual_groups, collision_groups = _geom_group_kinds(model)
        for group in visual_groups:
            vis_scene.geom_groups_visible[group] = bool(show_visual.value)
        for group in collision_groups:
            vis_scene.geom_groups_visible[group] = bool(show_collision.value)
        vis_scene._sync_visibilities()

    def attach_scene(clip: int) -> ViserMujocoScene:
        nonlocal scene, active_key
        model, _data, _root, _joint, key = ensure_scene(clip)
        if active_key != key or scene is None:
            scene = ViserMujocoScene(server, model, num_envs=1)
            scene.camera_tracking_enabled = False
            apply_geom_visibility(scene, model)
            layout_cache[key] = (layout_cache[key][0], layout_cache[key][1], layout_cache[key][2], scene)
            active_key = key
        return scene

    def _refresh_geoms(_event: viser.GuiEvent) -> None:
        if scene is not None and active_key is not None:
            apply_geom_visibility(scene, model_cache[active_key])

    show_visual.on_update(_refresh_geoms)
    show_collision.on_update(_refresh_geoms)

    def refresh_info(clip: int) -> None:
        meta = session.clip_info(clip)
        tag_line = " · ".join(f"{k}={v}" for k, v in meta.tags.items())
        extra = f"  \n{tag_line}" if tag_line else ""
        info.content = (
            f"**{meta.name}**  \n"
            f"clip {clip} · {meta.length} frames · {meta.length / fps:.2f} s{extra}"
        )

    def publish(clip: int, local: int, *, force: bool = False) -> None:
        vis_scene = attach_scene(clip)
        model, data, root_adr, joint_adr, _ = ensure_scene(clip)
        frame = session.frame(clip, local)
        forward_scene(
            model,
            data,
            vis_scene,
            motion_cls,
            clip,
            frame,
            root_adr=root_adr,
            joint_adr=joint_adr,
        )
        if show_hand_contact.value and frame.hand_contact is not None:
            pts = _hand_contact_positions(model, data, frame)
        else:
            pts = np.zeros((0, 3), dtype=np.float64)
        contact_points.points = pts
        contact_points.visible = bool(show_hand_contact.value) and pts.shape[0] > 0
        if force or int(frame_slider.value) != local:
            frame_slider.value = local
        start = frame.root_pos
        server.initial_camera.position = start + np.array([2.5, -2.5, 1.5])
        server.initial_camera.look_at = start

    @show_hand_contact.on_update
    def _(_event: viser.GuiEvent) -> None:
        publish(state["clip"], state["frame"], force=True)

    @clip_box.on_update
    def _(_event: viser.GuiEvent) -> None:
        clip = clip_labels.index(clip_box.value)
        state["clip"] = clip
        state["frame"] = 0
        length = session.clip_info(clip).length
        frame_slider.max = max(length - 1, 0)
        frame_slider.value = 0
        refresh_info(clip)
        publish(clip, 0, force=True)

    refresh_info(0)
    publish(0, 0, force=True)
    print(f"Viewer at http://localhost:{port} ({hz:g} Hz)", flush=True)

    if hz <= 0:
        raise ValueError(f"hz must be positive, got {hz}.")
    view_dt = 1.0 / hz
    view_acc = 0.0
    sim_acc = 0.0
    last = time.perf_counter()
    shown: tuple[int, int] | None = None

    try:
        while True:
            now = time.perf_counter()
            dt = now - last
            last = now
            clip = state["clip"]
            length = session.clip_info(clip).length
            view_acc += dt
            draw = view_acc >= view_dt
            if draw:
                view_acc -= view_dt
                if view_acc > view_dt:
                    view_acc = 0.0

            if playing.value and length > 1:
                sim_acc += dt * float(speed.value) * fps
                steps = int(sim_acc)
                if steps > 0:
                    sim_acc -= steps
                    nxt = int(state["frame"]) + steps
                    last_frame = length - 1
                    if nxt > last_frame:
                        if loop.value:
                            nxt %= length
                        else:
                            nxt = last_frame
                            playing.value = False
                    state["frame"] = nxt
            else:
                sim_acc = 0.0
                state["frame"] = int(frame_slider.value)

            local = int(state["frame"])
            key = (clip, local)
            if draw and (shown != key):
                publish(clip, local)
                shown = key

            time.sleep(min(0.02, 0.5 * view_dt))
    except KeyboardInterrupt:
        return
    finally:
        server.stop()
