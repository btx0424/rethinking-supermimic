"""Live Viser playback of tracking clips, with optional policy-vs-reference overlay.

Works for LAFAN1 and Weave (object HOI). With ``--ckpt``, the task comes from the
checkpoint config; otherwise pass ``--task``.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
import viser
from mjviser import ViserMujocoScene
from mjviser.conversions import get_body_name, is_fixed_body, merge_geoms
from omegaconf import OmegaConf

from supermimic.utils.kinematics import make_kinematics
from supermimic.mdp import GOAL_FRAMES, HIST, PAPER_OBS_TERMS, WorldModelState
from supermimic.motion import MotionLib
from supermimic.supermimic.network import InputNorms, build_policy
from supermimic.tasks import TaskSpec, make_task, make_task_from_config

# Solid policy from mjviser (model materials). Reference ghost is blue + translucent.
_REF_RGB = (40, 120, 255)
_REF_ALPHA = 0.45
_OBJ_POLICY_RGB = (220, 120, 40)
_OBJ_REF_RGB = (40, 120, 255)
_GROUND_Z = -0.02  # feet sit near z=0; keep the plane just under the soles
# mjlab G1 visuals are group 2; the inspire-hand asset uses group 1.
_VISUAL_GEOM_GROUPS = (1, 2)


def _qpos_addresses(model: mujoco.MjModel, joint_names: tuple[str, ...]) -> list[int]:
    adr = {}
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        adr[name] = int(model.jnt_qposadr[jid])
    return [adr[name] for name in joint_names]


def _write_pose(
    data: mujoco.MjData,
    root_p: np.ndarray,
    root_quat: np.ndarray,
    q: np.ndarray,
    qpos_adr: list[int],
) -> None:
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[0:3] = root_p
    data.qpos[3:7] = root_quat
    data.qpos[qpos_adr] = q


def _fk(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    root_p: np.ndarray,
    root_quat: np.ndarray,
    q: np.ndarray,
    qpos_adr: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    _write_pose(data, root_p, root_quat, q, qpos_adr)
    mujoco.mj_forward(model, data)
    return data.xpos.copy(), data.xquat.copy()


def _clip_length(motion: MotionLib, clip: int) -> int:
    return int(motion.clip_end[clip] - motion.clip_start[clip])


def _wm_state(sim) -> WorldModelState:
    """Robot (+ object) state for analytic observations."""
    p, quat, v, w, q, qd = sim.state()
    bodies = sim.tracked_bodies()
    if sim.task.has_object:
        return WorldModelState(p, quat, v, w, q, qd, bodies[0], bodies[1], *sim.object_state())
    return WorldModelState(p, quat, v, w, q, qd, bodies[0], bodies[1])


def _resolve_task(ckpt: Path | None, task_name: str | None) -> tuple[TaskSpec, dict]:
    """Task for the viewer. Checkpoint config wins when a ckpt is loaded."""
    saved_cfg: dict = {}
    if ckpt is not None:
        saved = torch.load(ckpt, map_location="cpu", weights_only=False)
        saved_cfg = dict(saved.get("cfg", {}))
        node = saved.get("task_cfg")
        if node is None:
            node = {
                "name": saved_cfg.get("task", task_name or "lafan1"),
                "object_name": saved_cfg.get("object_name"),
                "split": saved_cfg.get("split", "train"),
                "simple_hands": saved_cfg.get("simple_hands", True),
                "has_object": saved_cfg.get("has_object"),
            }
        saved_cfg["task_cfg"] = node
        return make_task_from_config(OmegaConf.create(dict(node))), saved_cfg
    return make_task(str(task_name or "lafan1")), saved_cfg


@dataclass
class PolicyOverlay:
    algo: str
    sim: object
    reset: Callable[[int], None]
    step: Callable[[], bool]
    pose: Callable[[], tuple[np.ndarray, np.ndarray, np.ndarray]]
    object_pose: Callable[[], tuple[np.ndarray, np.ndarray] | None]


def _tracking_sim(task: TaskSpec, saved_cfg: dict, device: str):
    from supermimic.sim import TrackingSim

    model, joint_names = task.compile_model()
    kin = make_kinematics(
        model,
        joint_names,
        str(saved_cfg.get("fk", "warp")),
        device,
        hand_bodies=task.hand_bodies,
    )
    kin.set_anchor(str(saved_cfg.get("anchor", "torso")))
    motion_dev = task.load_motion(device)
    sim = TrackingSim(
        1,
        motion_dev,
        kin,
        device,
        eval_mode=True,
        reset_noise=False,
        anchor=str(saved_cfg.get("anchor", "torso")),
        obs_terms=tuple(saved_cfg.get("obs_terms", PAPER_OBS_TERMS)),
        reward_style=str(saved_cfg.get("reward_style", "gaussian")),
        task=task,
    )
    return sim


def _load_policy_overlay(ckpt: Path, task: TaskSpec, saved_cfg: dict, device: str) -> PolicyOverlay:
    saved = torch.load(ckpt, map_location=device, weights_only=False)
    sim = _tracking_sim(task, saved_cfg, device)

    def pose_np() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p, quat, _v, _w, q, _qd = sim.state()
        return (
            p[0].detach().cpu().numpy(),
            quat[0].detach().cpu().numpy(),
            q[0].detach().cpu().numpy(),
        )

    def object_pose_np() -> tuple[np.ndarray, np.ndarray] | None:
        if not sim.task.has_object:
            return None
        obj_p, obj_quat, _v, _w = sim.object_state()
        return obj_p[0].detach().cpu().numpy(), obj_quat[0].detach().cpu().numpy()

    def reset_policy(clip: int) -> None:
        start_f = sim.motion.clip_start[clip].to(device).view(1)
        sim.reset(torch.zeros(1, dtype=torch.long, device=device), start_f)

    if "pi" in saved:
        policy = build_policy(
            sim.mdp.obs_dim, sim.mdp.task.goal_dim, HIST, saved_cfg, sim.mdp.task.num_joints
        ).to(device)
        policy.load_state_dict(saved["pi"])
        policy.eval()
        norms = InputNorms(sim.mdp.obs_dim, sim.mdp.task).to(device)
        norms.load_state_dict(saved["norms"])
        obs_hist = torch.zeros(1, HIST, sim.mdp.obs_dim, device=device)

        def reset(clip: int) -> None:
            reset_policy(clip)
            obs_hist.zero_()

        def step() -> bool:
            state = _wm_state(sim)
            obs = sim.mdp.obs(state, sim.a_prev, sim.xi, sim.f)
            fresh = sim.ep_step == 0
            obs_hist[:] = torch.where(
                fresh[:, None, None],
                obs[:, None, :].expand(-1, HIST, -1),
                torch.cat((obs_hist[:, 1:], obs[:, None, :]), dim=1),
            )
            goal = sim.motion.goal(sim.f, GOAL_FRAMES)
            with torch.no_grad():
                action = policy(obs_hist, goal, norms, deterministic=True)
            out = sim.step(action, auto_reset=False)
            # Playback stops at the end of the episode, whether the policy fell
            # (``terminated``) or the clip ran out (``truncated``).
            return not bool(out["done"].any())

        return PolicyOverlay("supermimic", sim, reset, step, pose_np, object_pose_np)

    if "actor" in saved:
        from supermimic.sac.network import NormalActor
        from supermimic.utils.nn import ObsVecNorm

        policy_obs_dim = sim.mdp.obs_dim + sim.mdp.task.goal_dim
        actor = NormalActor(
            policy_obs_dim,
            sim.mdp.task.num_joints,
            action_init=str(saved_cfg.get("actor_init", "zeros")),
        ).to(device)
        actor.load_state_dict(saved["actor"])
        actor.eval()
        use_vecnorm = bool(saved_cfg.get("vecnorm", True))
        vecnorm = ObsVecNorm(policy_obs_dim).to(device) if use_vecnorm else None
        if vecnorm is not None and "vecnorm" in saved:
            vecnorm.load_state_dict(saved["vecnorm"])
        if vecnorm is not None:
            vecnorm.eval()

        def reset(clip: int) -> None:
            reset_policy(clip)

        def step() -> bool:
            state = _wm_state(sim)
            obs = sim.mdp.obs(state, sim.a_prev, sim.xi, sim.f)
            goal = sim.motion.goal(sim.f, GOAL_FRAMES)
            policy_obs = torch.cat([obs, goal], dim=-1)
            with torch.no_grad():
                obs_n = vecnorm(policy_obs) if vecnorm is not None else policy_obs
                loc, _scale = actor(obs_n)
                action = loc
            out = sim.step(action, auto_reset=False)
            # Playback stops at the end of the episode, whether the policy fell
            # (``terminated``) or the clip ran out (``truncated``).
            return not bool(out["done"].any())

        return PolicyOverlay("sac", sim, reset, step, pose_np, object_pose_np)

    keys = ", ".join(sorted(saved.keys()))
    raise ValueError(f"Checkpoint is not SuperMimic (pi) or SAC (actor). Keys: {keys}")


class GhostRobot:
    """Blue translucent robot meshes. Batched GLB from mjviser ignores per-env colors."""

    def __init__(
        self,
        server: viser.ViserServer,
        model: mujoco.MjModel,
        rgb: tuple[int, int, int] = _REF_RGB,
        opacity: float = _REF_ALPHA,
        root: str = "/reference",
        visual_groups: tuple[int, ...] = _VISUAL_GEOM_GROUPS,
    ):
        self.handles: list[tuple[int, object]] = []
        color = np.array(rgb, dtype=np.uint8)
        alpha = np.float32(opacity)
        for body_id in range(model.nbody):
            if is_fixed_body(model, body_id):
                continue
            geom_ids = [
                i
                for i in range(model.ngeom)
                if int(model.geom_bodyid[i]) == body_id
                and int(model.geom_group[i]) in visual_groups
                and float(model.geom_rgba[i, 3]) > 0.0
            ]
            if not geom_ids:
                continue
            mesh = merge_geoms(model, geom_ids)
            handle = server.scene.add_batched_meshes_simple(
                f"{root}/{get_body_name(model, body_id)}",
                mesh.vertices.astype(np.float32),
                mesh.faces.astype(np.int32),
                batched_wxyzs=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                batched_positions=np.zeros((1, 3), dtype=np.float32),
                batched_colors=color[None],
                batched_opacities=np.array([alpha], dtype=np.float32),
                cast_shadow=False,
                receive_shadow=False,
                lod="off",
            )
            self.handles.append((body_id, handle))
        self.server = server

    def update(self, xpos: np.ndarray, xquat: np.ndarray) -> None:
        # One websocket payload for every body. Per-handle writes each become
        # their own packet, which stalls an SSH-forwarded viewer.
        with self.server.atomic():
            for body_id, handle in self.handles:
                handle.batched_positions = xpos[body_id : body_id + 1]
                handle.batched_wxyzs = xquat[body_id : body_id + 1]
            self.server.flush()


class ObjectMesh:
    """Single rigid object mesh with an updatable world pose."""

    def __init__(
        self,
        server: viser.ViserServer,
        mesh_path: Path,
        *,
        name: str,
        rgb: tuple[int, int, int],
        opacity: float | None = None,
    ):
        import trimesh

        mesh = trimesh.load(mesh_path, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = mesh.dump(concatenate=True)
        self.handle = server.scene.add_mesh_simple(
            name,
            np.asarray(mesh.vertices, dtype=np.float32),
            np.asarray(mesh.faces, dtype=np.int32),
            color=rgb,
            opacity=opacity,
            cast_shadow=False,
            receive_shadow=False,
        )

    def update(self, pos: np.ndarray, quat_wxyz: np.ndarray) -> None:
        self.handle.position = tuple(float(x) for x in pos)
        self.handle.wxyz = tuple(float(x) for x in quat_wxyz)


def play(
    port: int = 8080,
    ckpt: Path | None = None,
    device: str = "cuda",
    hz: float = 30.0,
    task_name: str | None = None,
) -> None:
    task, saved_cfg = _resolve_task(ckpt, task_name)
    motion = task.load_motion("cpu")
    model, joint_names = task.compile_model()
    data = mujoco.MjData(model)
    qpos_adr = _qpos_addresses(model, joint_names)
    fps = float(motion.fps)
    clip_names = list(motion.clip_names)
    overlay = ckpt is not None
    has_object = task.has_object and getattr(motion, "object_p", None) is not None

    overlay_policy: PolicyOverlay | None = None
    if overlay:
        assert ckpt is not None
        overlay_policy = _load_policy_overlay(ckpt, task, saved_cfg, device)

    server = viser.ViserServer(port=port, label=f"SuperMimic play ({task.name})")
    # Solid robot: policy when overlaying, otherwise the reference.
    scene = ViserMujocoScene(server, model, num_envs=1)
    # Keep world frames so the solid robot, blue ghost, and ground share one origin.
    # (mjviser's camera-tracking offset would yank only the solid robot off the plane.)
    scene.camera_tracking_enabled = False
    ghost = GhostRobot(server, model) if overlay else None
    obj_ref: ObjectMesh | None = None
    obj_policy: ObjectMesh | None = None
    if has_object:
        assert task.object_name is not None
        mesh_path = task.object_mesh_path()
        obj_ref = ObjectMesh(
            server,
            mesh_path,
            name="/reference_object",
            rgb=_OBJ_REF_RGB,
            opacity=_REF_ALPHA if overlay else None,
        )
        if overlay:
            obj_policy = ObjectMesh(
                server,
                mesh_path,
                name="/policy_object",
                rgb=_OBJ_POLICY_RGB,
                opacity=None,
            )
    start = motion.root_p[int(motion.clip_start[0])].numpy()
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
    server.initial_camera.position = start + np.array([2.5, -2.5, 1.5])
    server.initial_camera.look_at = start

    with server.gui.add_folder("Clip"):
        clip_box = server.gui.add_dropdown(
            "Motion", options=tuple(clip_names), initial_value=clip_names[0]
        )
        info = server.gui.add_markdown("")
    with server.gui.add_folder("Playback"):
        playing = server.gui.add_checkbox("Play", initial_value=True)
        loop = server.gui.add_checkbox("Loop", initial_value=True)
        speed = server.gui.add_slider("Speed", min=0.1, max=4.0, step=0.1, initial_value=1.0)
        frame_slider = server.gui.add_slider(
            "Frame",
            min=0,
            max=max(_clip_length(motion, 0) - 1, 0),
            step=1,
            initial_value=0,
        )
        legend = server.gui.add_markdown("")

    state = {"clip": 0, "need_reset": True, "frame": 0}

    def refresh_info(clip: int) -> None:
        length = _clip_length(motion, clip)
        info.content = (
            f"**{clip_names[clip]}**  \n"
            f"task `{task.name}` · clip {clip} · {length} frames · {length / fps:.2f} s"
        )
        if overlay:
            legend.content = (
                "solid = **policy**, blue ghost = **reference**"
                + (", orange / blue meshes = **objects**" if has_object else "")
            )
        else:
            legend.content = (
                "**reference** only (pass `--ckpt` to overlay tracking)"
                + ("; object mesh follows the clip" if has_object else "")
            )

    def reference_pose(clip: int, local: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        frame = int(motion.clip_start[clip]) + local
        return (
            motion.root_p[frame].numpy(),
            motion.root_quat[frame].numpy(),
            motion.q[frame].numpy(),
        )

    def reference_object_pose(clip: int, local: int) -> tuple[np.ndarray, np.ndarray] | None:
        if not has_object:
            return None
        frame = int(motion.clip_start[clip]) + local
        return motion.object_p[frame].numpy(), motion.object_quat[frame].numpy()

    def push_scene(
        solid_pose: tuple[np.ndarray, np.ndarray, np.ndarray],
        ghost_pose: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
        solid_obj: tuple[np.ndarray, np.ndarray] | None = None,
        ghost_obj: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        xpos, _xquat = _fk(model, data, *solid_pose, qpos_adr)
        xmat = data.xmat.reshape(-1, 3, 3).copy()
        scene.update_from_arrays(xpos[None], xmat[None])
        if ghost is not None and ghost_pose is not None:
            gx, gq = _fk(model, data, *ghost_pose, qpos_adr)
            ghost.update(gx, gq)
        with server.atomic():
            if obj_policy is not None and solid_obj is not None:
                obj_policy.update(*solid_obj)
            if obj_ref is not None and ghost_obj is not None:
                obj_ref.update(*ghost_obj)
            elif obj_ref is not None and solid_obj is not None and not overlay:
                # Reference-only: one object mesh follows the clip.
                obj_ref.update(*solid_obj)
            server.flush()

    @clip_box.on_update
    def _(_event: viser.GuiEvent) -> None:
        clip = clip_names.index(clip_box.value)
        state["clip"] = clip
        state["need_reset"] = True
        length = _clip_length(motion, clip)
        frame_slider.value = 0
        frame_slider.max = max(length - 1, 0)
        refresh_info(clip)

    refresh_info(0)
    ref0 = reference_pose(0, 0)
    push_scene(ref0, None, reference_object_pose(0, 0), None)
    print(f"Viewer at http://localhost:{port} ({hz:g} Hz) · task={task.name}", flush=True)
    if overlay_policy is not None:
        print(
            f"Policy checkpoint ({overlay_policy.algo}): {ckpt}",
            flush=True,
        )

    if hz <= 0:
        raise ValueError(f"hz must be positive, got {hz}.")
    view_dt = 1.0 / hz
    accumulator = 0.0
    view_acc = 0.0
    last = time.perf_counter()
    # Republishing a held pose every loop tick is what saturates an SSH tunnel.
    shown: dict[str, tuple[int, int] | None] = {"key": None}

    def publish(
        clip_i: int,
        local: int,
        solid,
        ghost_pose,
        solid_obj=None,
        ghost_obj=None,
        force: bool = False,
    ) -> None:
        key = (clip_i, local)
        if not force and shown["key"] == key:
            return
        if int(frame_slider.value) != local:
            frame_slider.value = local
        push_scene(solid, ghost_pose, solid_obj, ghost_obj)
        shown["key"] = key

    try:
        while True:
            now = time.perf_counter()
            dt = now - last
            last = now
            clip = state["clip"]
            length = _clip_length(motion, clip)

            view_acc += dt
            draw = view_acc >= view_dt
            if draw:
                view_acc -= view_dt
                if view_acc > view_dt:
                    view_acc = 0.0

            if state["need_reset"]:
                state["need_reset"] = False
                state["frame"] = 0
                frame_slider.value = 0
                ref = reference_pose(clip, 0)
                ref_obj = reference_object_pose(clip, 0)
                if overlay_policy is not None:
                    overlay_policy.reset(clip)
                    publish(
                        clip,
                        0,
                        overlay_policy.pose(),
                        ref,
                        overlay_policy.object_pose(),
                        ref_obj,
                        force=True,
                    )
                else:
                    publish(clip, 0, ref, None, ref_obj, None, force=True)
                accumulator = 0.0
                view_acc = 0.0
                draw = False

            if playing.value and length > 1:
                # Policy and the clip stay on the 50 Hz motion clock. Only the viewer is capped.
                accumulator += dt * float(speed.value) * fps
                steps = int(accumulator)
                if steps > 0:
                    accumulator -= steps
                    if overlay_policy is not None:
                        alive = True
                        for _ in range(steps):
                            alive = overlay_policy.step()
                            if not alive:
                                break
                        sim = overlay_policy.sim
                        state["frame"] = min(int(sim.ep_step[0].item()), length - 1)  # type: ignore[union-attr]
                        if not alive:
                            if loop.value:
                                state["need_reset"] = True
                            else:
                                playing.value = False
                    else:
                        nxt = int(state["frame"]) + steps
                        last_frame = length - 1
                        if nxt > last_frame:
                            if loop.value:
                                nxt %= length
                            else:
                                nxt = last_frame
                                playing.value = False
                        state["frame"] = nxt
                if draw:
                    local = int(state["frame"])
                    ref = reference_pose(clip, local)
                    ref_obj = reference_object_pose(clip, local)
                    if overlay_policy is not None:
                        publish(
                            clip,
                            local,
                            overlay_policy.pose(),
                            ref,
                            overlay_policy.object_pose(),
                            ref_obj,
                        )
                    else:
                        publish(clip, local, ref, None, ref_obj, None)
            else:
                accumulator = 0.0
                state["frame"] = int(frame_slider.value)
                if draw:
                    local = int(state["frame"])
                    ref = reference_pose(clip, local)
                    ref_obj = reference_object_pose(clip, local)
                    if overlay_policy is not None:
                        publish(
                            clip,
                            local,
                            overlay_policy.pose(),
                            ref,
                            overlay_policy.object_pose(),
                            ref_obj,
                        )
                    else:
                        publish(clip, local, ref, None, ref_obj, None)

            time.sleep(min(0.02, 0.5 * view_dt))
    except KeyboardInterrupt:
        return
    finally:
        server.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="SuperMimic (pi) or SAC (actor) checkpoint for overlay.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task when no checkpoint is given (lafan1 | weave). Ignored if --ckpt sets task.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--hz",
        type=float,
        default=30.0,
        help="Viewer update rate. The clip and policy still step at 50 Hz.",
    )
    args = parser.parse_args()
    play(port=args.port, ckpt=args.ckpt, device=args.device, hz=args.hz, task_name=args.task)


if __name__ == "__main__":
    main()
