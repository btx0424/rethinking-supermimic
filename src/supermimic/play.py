"""Live Viser playback of LAFAN1 clips, with optional policy-vs-reference overlay."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
import viser
from mjviser import ViserMujocoScene
from mjviser.conversions import get_body_name, is_fixed_body, merge_geoms

from .kinematics import TorchKinematics, compile_g1
from .mdp import GOAL_DIM, GOAL_FRAMES, HIST, PAPER_OBS_TERMS
from .motion import MotionLib
from .network import InputNorms, Policy

# Solid policy from mjviser (model materials). Reference ghost is blue + translucent.
_REF_RGB = (40, 120, 255)
_REF_ALPHA = 0.45
_GROUND_Z = -0.02  # feet sit near z=0; keep the plane just under the soles


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


class GhostRobot:
    """Blue translucent robot meshes. Batched GLB from mjviser ignores per-env colors."""

    def __init__(
        self,
        server: viser.ViserServer,
        model: mujoco.MjModel,
        rgb: tuple[int, int, int] = _REF_RGB,
        opacity: float = _REF_ALPHA,
        root: str = "/reference",
    ):
        self.handles: list[tuple[int, object]] = []
        color = np.array(rgb, dtype=np.uint8)
        alpha = np.float32(opacity)
        for body_id in range(model.nbody):
            if is_fixed_body(model, body_id):
                continue
            # Visual meshes only (group 2 on the mjlab G1); skip collision geoms.
            geom_ids = [
                i
                for i in range(model.ngeom)
                if int(model.geom_bodyid[i]) == body_id
                and int(model.geom_group[i]) == 2
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

    def update(self, xpos: np.ndarray, xquat: np.ndarray) -> None:
        for body_id, handle in self.handles:
            handle.batched_positions = xpos[body_id : body_id + 1]
            handle.batched_wxyzs = xquat[body_id : body_id + 1]


def play(
    port: int = 8080,
    ckpt: Path | None = None,
    device: str = "cuda",
) -> None:
    motion = MotionLib.load(device="cpu")
    model, joint_names = compile_g1()
    data = mujoco.MjData(model)
    qpos_adr = _qpos_addresses(model, joint_names)
    fps = float(motion.fps)
    clip_names = list(motion.clip_names)
    overlay = ckpt is not None

    policy = norms = sim = obs_hist = None
    if overlay:
        from .sim import TrackingSim

        saved = torch.load(ckpt, map_location=device, weights_only=False)
        saved_cfg = saved.get("cfg", {})
        kin = TorchKinematics(model, joint_names)
        kin.set_anchor(str(saved_cfg.get("anchor", "torso")))
        motion_dev = MotionLib.load(device=device)
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
        )
        policy = Policy(
            sim.mdp.obs_dim,
            GOAL_DIM,
            HIST,
            stochastic=bool(saved_cfg.get("stochastic_actor", False)),
        ).to(device)
        policy.load_state_dict(saved["pi"])
        policy.eval()
        norms = InputNorms(sim.mdp.obs_dim).to(device)
        norms.load_state_dict(saved["norms"])
        obs_hist = torch.zeros(1, HIST, sim.mdp.obs_dim, device=device)

    server = viser.ViserServer(port=port, label="SuperMimic play")
    # Solid robot: policy when overlaying, otherwise the reference.
    scene = ViserMujocoScene(server, model, num_envs=1)
    # Keep world frames so the solid robot, blue ghost, and ground share one origin.
    # (mjviser's camera-tracking offset would yank only the solid robot off the plane.)
    scene.camera_tracking_enabled = False
    ghost = GhostRobot(server, model) if overlay else None
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
        legend = server.gui.add_markdown(
            "solid gray = **policy**, blue ghost = **reference**"
            if overlay
            else "**reference** only (pass `--ckpt` to overlay tracking)"
        )

    state = {"clip": 0, "need_reset": True}

    def refresh_info(clip: int) -> None:
        length = _clip_length(motion, clip)
        info.content = (
            f"**{clip_names[clip]}**  \nclip {clip} · {length} frames · {length / fps:.2f} s"
        )
        legend.content = (
            "solid gray = **policy**, blue ghost = **reference**"
            if overlay
            else "**reference** only (pass `--ckpt` to overlay tracking)"
        )

    def reference_pose(clip: int, local: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        frame = int(motion.clip_start[clip]) + local
        return (
            motion.root_p[frame].numpy(),
            motion.root_quat[frame].numpy(),
            motion.q[frame].numpy(),
        )

    def push_scene(
        solid_pose: tuple[np.ndarray, np.ndarray, np.ndarray],
        ghost_pose: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    ) -> None:
        xpos, _xquat = _fk(model, data, *solid_pose, qpos_adr)
        xmat = data.xmat.reshape(-1, 3, 3).copy()
        scene.update_from_arrays(xpos[None], xmat[None])
        if ghost is not None and ghost_pose is not None:
            gx, gq = _fk(model, data, *ghost_pose, qpos_adr)
            ghost.update(gx, gq)

    def policy_pose_np() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert sim is not None
        p, quat, _v, _w, q, _qd = sim.state()
        return (
            p[0].detach().cpu().numpy(),
            quat[0].detach().cpu().numpy(),
            q[0].detach().cpu().numpy(),
        )

    def reset_policy(clip: int) -> None:
        assert sim is not None and obs_hist is not None
        start_f = motion.clip_start[clip].to(device).view(1)
        sim.reset(torch.zeros(1, dtype=torch.long, device=device), start_f)
        obs_hist.zero_()

    def step_policy() -> bool:
        assert sim is not None and policy is not None and norms is not None and obs_hist is not None
        p, quat, v, w, q, qd = sim.state()
        obs = sim.mdp.obs(p, quat, v, w, q, qd, sim.a_prev, sim.xi, sim.f)
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
        return not bool(out["done"].any())

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
    push_scene(reference_pose(0, 0), None)
    print(f"Viewer at http://localhost:{port}", flush=True)
    if ckpt is not None:
        print(f"Policy checkpoint: {ckpt}", flush=True)

    accumulator = 0.0
    last = time.perf_counter()
    try:
        while True:
            now = time.perf_counter()
            dt = now - last
            last = now
            clip = state["clip"]
            length = _clip_length(motion, clip)

            if state["need_reset"]:
                state["need_reset"] = False
                frame_slider.value = 0
                ref = reference_pose(clip, 0)
                if overlay:
                    reset_policy(clip)
                    push_scene(policy_pose_np(), ref)
                else:
                    push_scene(ref, None)
                accumulator = 0.0

            if playing.value and length > 1:
                accumulator += dt * float(speed.value) * fps
                steps = int(accumulator)
                if steps > 0:
                    accumulator -= steps
                    if overlay:
                        alive = True
                        for _ in range(steps):
                            alive = step_policy()
                            if not alive:
                                break
                        local = min(int(sim.ep_step[0].item()), length - 1)  # type: ignore[union-attr]
                        frame_slider.value = local
                        push_scene(policy_pose_np(), reference_pose(clip, local))
                        if not alive:
                            if loop.value:
                                state["need_reset"] = True
                            else:
                                playing.value = False
                    else:
                        nxt = int(frame_slider.value) + steps
                        last_frame = length - 1
                        if nxt > last_frame:
                            if loop.value:
                                nxt %= length
                            else:
                                nxt = last_frame
                                playing.value = False
                        frame_slider.value = nxt
                        push_scene(reference_pose(clip, int(frame_slider.value)), None)
            else:
                accumulator = 0.0
                local = int(frame_slider.value)
                ref = reference_pose(clip, local)
                if overlay and sim is not None and not state["need_reset"]:
                    push_scene(policy_pose_np(), ref)
                else:
                    push_scene(ref, None)

            time.sleep(0.005)
    except KeyboardInterrupt:
        return
    finally:
        server.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--ckpt", type=Path, default=None, help="Policy checkpoint for overlay.")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    play(port=args.port, ckpt=args.ckpt, device=args.device)


if __name__ == "__main__":
    main()
