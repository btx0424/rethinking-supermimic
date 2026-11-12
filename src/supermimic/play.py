"""Play a trained policy in the simulator or in its own world model, with viser.

Both backends share the policy, task terms and reference command. ``sim`` steps
MuJoCo-Warp with evaluation settings (nominal context, hard height/tilt
termination, no root-position termination). ``wm`` rolls the checkpoint's world
model open loop from the reference state; it never resets and reports the soft
survival weight instead of terminating.

The world model only carries the tracked bodies. For display, a full G1 pose is
fitted to those bodies by damped least squares; the raw predicted body points
are drawn as well, so the fit never hides a prediction error.
"""

import argparse
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch
from torch import nn

from supermimic.algo.supermimic.normalization import VecNorm
from supermimic.algo.supermimic.policy.actor import Actor
from supermimic.algo.supermimic.wm import build_world_model
from supermimic.algo.supermimic.wm.env import WorldModelEnv
from supermimic.dataset.motion import MotionDataset
from supermimic.env.mdp import CONTROL_DT
from supermimic.env.mdp.command import CommandState
from supermimic.env.mdp.config import TaskConfig
from supermimic.env.mdp.geometry.representation import MaxCoordRepresentation
from supermimic.env.mdp.geometry.state import build_maxcoord_full_state
from supermimic.env.mdp.observation import concatenate
from supermimic.env.mdp.reference import TrackingReference
from supermimic.env.mdp.tracking import TrackingTask
from supermimic.env.rollout.interface import PolicyObservation
from supermimic.utils.math import mat_to_6d, quat_apply_inverse, quat_inv, quat_mul, quat_to_mat

FORMAT = "g1_lafan_v1"
# Goal root-position convention per checkpoint protocol.
GOAL_BY_PROTOCOL = {
    "task_terms_command_state_all_collision_friction_v1": "reference_root",
    "task_terms_command_state_all_collision_friction_robot_root_goal_v2": "robot_root",
}
DEFAULT_MOTION = Path(__file__).resolve().parents[2] / "data/lafan1_g1.npz"


def reference_root_goal(motion, frame, root_pos, root_quat, horizon=10):
    """Goal of v1 checkpoints: future reference root displacement in the current reference frame."""
    del root_pos
    future = motion.future_frames(frame, horizon)
    ref_quat = motion.root_quat[frame, None].expand(*future.shape, 4)
    dp = quat_apply_inverse(ref_quat, motion.root_p[future] - motion.root_p[frame, None])
    relative = quat_mul(quat_inv(root_quat).unsqueeze(-2), motion.root_quat[future])
    return torch.cat((dp, mat_to_6d(quat_to_mat(relative)), motion.q[future]), -1).flatten(-2)


class Policy:
    """Checkpoint policy, world model and task, without trainer or optimizer state."""

    def __init__(self, path, motion_path=None, device="cuda"):
        data = torch.load(path, map_location=device, weights_only=False)
        protocol = data.get("protocol")
        if data.get("format") != FORMAT or protocol not in GOAL_BY_PROTOCOL:
            raise ValueError(
                f"Unsupported checkpoint format/protocol: {data.get('format')}, {protocol}"
            )
        self.path = Path(path)
        self.iteration = data["iteration"]
        self.protocol = protocol
        self.goal_kind = GOAL_BY_PROTOCOL[protocol]
        cfg = data["config"]
        self.cfg = cfg
        self.device = device
        task_cfg = TaskConfig(**{**cfg["task"], "motion": str(motion_path or DEFAULT_MOTION)})
        task_cfg.validate()
        self.motion = TrackingReference(
            MotionDataset(task_cfg.motion, device), task_cfg.tracking_bodies
        )
        if self.motion.body_names != tuple(data["body_names"]) or self.motion.joint_names != tuple(
            data["joint_names"]
        ):
            raise ValueError("Checkpoint body/joint names differ from the motion data")
        rep = MaxCoordRepresentation(len(self.motion.body_names)).to(device)
        self.task = TrackingTask(self.motion, rep, task_cfg)
        if self.goal_kind == "reference_root":
            self._use_goal(reference_root_goal)
        self.vecnorms = nn.ModuleDict()
        for name in sorted({key.split(".")[0] for key in data["vecnorms"]}):
            width = data["vecnorms"][f"{name}.sum"].numel()
            self.vecnorms[name] = VecNorm(width, cfg["vecnorm_decay"])
        self.vecnorms.load_state_dict(data["vecnorms"])
        self.vecnorms.to(device).eval()
        num_joints = len(self.motion.joint_names)
        self.actor = Actor(self.vecnorms["actor"].sum.numel(), num_joints, cfg["actor_hidden"])
        self.actor.load_state_dict(data["actor"])
        self.actor.to(device).eval()
        self.wm = build_world_model(SimpleNamespace(**cfg), rep, num_joints)
        self.wm.load_state_dict(data["wm"])
        self.wm.to(device).eval()
        self.wm.input_transform = lambda name, x: self.vecnorms[name](x)
        self.actor_observations = tuple(task_cfg.actor_observations)

    def _use_goal(self, goal_fn):
        task = self.task

        def observe(state):
            physical = state.physical
            return PolicyObservation(
                physical.f,
                goal_fn(
                    task.motion,
                    task.command.frame(state.command),
                    physical.aux[..., :3],
                    physical.aux[..., 3:7],
                    task.goal_frames,
                ),
                physical.xi_n,
                state.full,
            )

        task.observe = observe

    def __call__(self, observation):
        inputs = concatenate(observation, self.actor_observations)
        return self.actor(self.vecnorms["actor"](inputs))


def nominal_context(num_joints, device):
    xi = torch.zeros(1, num_joints + 4, device=device)
    xi[:, 0] = 1.0  # evaluation friction, zero encoder bias and COM offset
    return xi


class SimBackend:
    name = "sim"

    def __init__(self, policy):
        from supermimic.env.sim.env import SimulationEnv

        self.policy = policy
        self.env = SimulationEnv(policy.task, 1, policy.device, eval_mode=True, auto_reset=False)

    def reset(self, frame):
        return self.env.reset(frames=torch.tensor([frame], device=self.policy.device))

    def step(self, state, action):
        return self.env.step(state, action)

    def qpos(self):
        robot = self.env.robot
        root = robot.data.root_link_pose_w[0].clone()
        root[:3] -= self.env.scene.env_origins[0]
        return torch.cat((root, robot.data.joint_pos[0])).cpu().numpy()


class WMBackend:
    name = "wm"

    def __init__(self, policy):
        self.policy = policy
        self.env = WorldModelEnv(policy.wm, policy.task)

    def reset(self, frame):
        motion = self.policy.motion
        f = torch.tensor([frame], device=self.policy.device)
        full = build_maxcoord_full_state(
            motion.body_p[f], motion.body_quat[f], motion.body_v[f], motion.body_w[f]
        )
        command = self.policy.task.command.from_frame(f)
        num_joints = len(motion.joint_names)
        return self.env.reset(
            full,
            CommandState(command.motion_id, command.motion_t),
            torch.zeros(1, num_joints, device=self.policy.device),
            nominal_context(num_joints, self.policy.device),
            torch.zeros(1, dtype=torch.long, device=self.policy.device),
        )

    def step(self, state, action):
        return self.env.step(state, action)

    def qpos(self):
        return None


BACKENDS = {"sim": SimBackend, "wm": WMBackend}


def tracked_bodies(policy, state):
    """World positions (B, 3) and rotation matrices (B, 3, 3) of the tracked bodies."""
    rep = policy.task.representation
    pos, mat, _, _ = rep.world_frame_bodies(state.physical.aux, state.physical.f)
    return pos[0].cpu().numpy(), mat[0].cpu().numpy()


class G1Pose:
    """CPU G1 model for display: reference FK and fitting a pose to tracked bodies."""

    def __init__(self, motion):
        from mjlab.scene import Scene, SceneCfg
        from mjlab.terrains import TerrainEntityCfg

        from supermimic.robots.g1 import robot_config

        scene = Scene(
            SceneCfg(
                num_envs=1,
                env_spacing=2.0,
                terrain=TerrainEntityCfg(terrain_type="plane"),
                entities={"robot": robot_config()},
            ),
            "cpu",
        )
        self.model = scene.spec.compile()
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.body_ids = np.array([m.body(f"robot/{n}").id for n in motion.body_names])
        root_joint = m.joint("robot/floating_base_joint")
        self.root_adr = int(root_joint.qposadr[0])
        self.joint_adr = np.array(
            [int(m.joint(f"robot/{n}").qposadr[0]) for n in motion.joint_names]
        )
        self.qpos = m.qpos0.copy()

    def from_root_joints(self, root_p, root_quat, q):
        qpos = self.model.qpos0.copy()
        qpos[self.root_adr : self.root_adr + 3] = root_p
        qpos[self.root_adr + 3 : self.root_adr + 7] = root_quat
        qpos[self.joint_adr] = q
        return qpos

    def kinematics(self, qpos):
        self.data.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, self.data)
        return self.data.xpos.copy(), self.data.xmat.reshape(-1, 3, 3).copy()

    def fit(self, pos, mat, iterations=12, damping=1e-4, rot_weight=0.3):
        """Fit qpos to tracked body poses; returns qpos and RMS position residual (m)."""
        m, d = self.model, self.data
        qpos = self.qpos.copy()
        qpos[self.root_adr : self.root_adr + 3] = pos[0]
        root_quat = np.empty(4)
        mujoco.mju_mat2Quat(root_quat, mat[0].reshape(-1))
        qpos[self.root_adr + 3 : self.root_adr + 7] = root_quat
        target_quat = np.empty((len(self.body_ids), 4))
        for k in range(len(self.body_ids)):
            mujoco.mju_mat2Quat(target_quat[k], mat[k].reshape(-1))
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        rows = 6 * len(self.body_ids)
        for _ in range(iterations):
            d.qpos[:] = qpos
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            residual = np.empty(rows)
            jacobian = np.empty((rows, m.nv))
            for k, body in enumerate(self.body_ids):
                residual[6 * k : 6 * k + 3] = pos[k] - d.xpos[body]
                error = np.empty(3)
                # subQuat gives the error in the current body frame; Jacobian is world frame.
                mujoco.mju_subQuat(error, target_quat[k], d.xquat[body])
                residual[6 * k + 3 : 6 * k + 6] = rot_weight * d.xmat[body].reshape(3, 3) @ error
                mujoco.mj_jacBody(m, d, jacp, jacr, body)
                jacobian[6 * k : 6 * k + 3] = jacp
                jacobian[6 * k + 3 : 6 * k + 6] = rot_weight * jacr
            step = np.linalg.solve(
                jacobian.T @ jacobian + damping * np.eye(m.nv), jacobian.T @ residual
            )
            mujoco.mj_integratePos(m, qpos, step, 1.0)
        d.qpos[:] = qpos
        mujoco.mj_kinematics(m, d)
        error = pos - d.xpos[self.body_ids]
        self.qpos = qpos
        return qpos, float(np.sqrt((error**2).sum(-1).mean()))


class Session:
    """One rollout of one backend; shared by the viewer and headless checks."""

    def __init__(self, policy, pose):
        self.policy = policy
        self.pose = pose
        self.backends = {}
        self.backend = None
        self.clip = 0
        self.state = None
        self.done = True
        self.info = {}

    def select(self, name):
        if name not in self.backends:
            self.backends[name] = BACKENDS[name](self.policy)
        self.backend = self.backends[name]

    def frame(self):
        return int(self.policy.task.command.frame(self.state.command)[0])

    @torch.no_grad()
    def reset(self, clip=None):
        if clip is not None:
            self.clip = clip
        motion = self.policy.motion
        start = int(motion.clip_start[self.clip])
        self.state = self.backend.reset(start)
        self.pose.qpos = self.pose.from_root_joints(
            motion.root_p[start].cpu().numpy(),
            motion.root_quat[start].cpu().numpy(),
            motion.q[start].cpu().numpy(),
        )
        self.done = False
        self.info = {"step": 0, "return": 0.0, "terms": {}, "status": "running"}
        self._measure(None)

    @torch.no_grad()
    def step(self):
        if self.done:
            return
        observation = self.backend.env.observe(self.state)
        action = self.policy(observation)
        result = self.backend.step(self.state, action)
        self.state = result.state
        self.info["step"] += 1
        self.info["return"] += float(result.reward[0])
        self.info["terms"] = {k: float(v[0]) / CONTROL_DT for k, v in result.terms.items()}
        motion = self.policy.motion
        at_end = self.frame() >= int(motion.clip_end[self.clip]) - 1
        if self.backend.name == "sim":
            terminated = bool(result.terminated[0])
            self.info["status"] = (
                "terminated" if terminated else ("clip end" if at_end else "running")
            )
            self.done = terminated or at_end
        else:
            self.info["survival"] = self.info.get("survival", 1.0) * (
                1 - float(result.terminated[0])
            )
            self.info["status"] = "clip end" if at_end else "running"
            self.done = at_end
        self._measure(result)

    def _measure(self, result):
        motion = self.policy.motion
        frame = self.frame()
        pos, mat = tracked_bodies(self.policy, self.state)
        ref = motion.body_p[frame].cpu().numpy()
        self.info["frame"] = frame - int(motion.clip_start[self.clip])
        self.info["length"] = int(motion.clip_end[self.clip] - motion.clip_start[self.clip])
        self.info["root_xy_error"] = float(np.linalg.norm(pos[0, :2] - ref[0, :2]))
        self.info["root_z_error"] = float(abs(pos[0, 2] - ref[0, 2]))
        # Body positions after aligning each pelvis, matching local tracking error.
        self.info["local_body_error"] = float(
            np.linalg.norm((pos - pos[:1]) - (ref - ref[:1]), axis=-1).mean()
        )
        self.tracked = (pos, mat)
        qpos = self.backend.qpos()
        if qpos is None:
            qpos, residual = self.pose.fit(pos, mat)
            self.info["fit_rms"] = residual
        else:
            self.pose.qpos = qpos
            self.info.pop("fit_rms", None)
        self.robot_qpos = qpos
        self.reference_qpos = self.pose.from_root_joints(
            motion.root_p[frame].cpu().numpy(),
            motion.root_quat[frame].cpu().numpy(),
            motion.q[frame].cpu().numpy(),
        )
        self.reference_points = ref


def headless(session, backend, clips, max_steps):
    session.select(backend)
    rows = []
    for clip in clips:
        session.reset(clip)
        fits = []
        while not session.done and session.info["step"] < max_steps:
            session.step()
            if "fit_rms" in session.info:
                fits.append(session.info["fit_rms"])
        row = {
            "backend": backend,
            "clip": session.policy.motion.clip_names[clip],
            "steps": session.info["step"],
            "status": session.info["status"],
            "return": round(session.info["return"], 3),
            "root_xy_error": round(session.info["root_xy_error"], 3),
            "local_body_error": round(session.info["local_body_error"], 3),
        }
        if fits:
            row["fit_rms_max"] = round(max(fits), 4)
        if "survival" in session.info:
            row["survival"] = round(session.info["survival"], 4)
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def view(session, backend, port):
    import viser
    from mjviser import ViserMujocoScene

    policy = session.policy
    motion = policy.motion
    server = viser.ViserServer(port=port, label=f"supermimic play: {policy.path.name}")
    scene = ViserMujocoScene(server, session.pose.model, num_envs=2)
    tabs = scene.create_visualization_gui()
    lock = threading.Lock()
    control = {"playing": False, "step": False, "reset": True, "backend": backend}
    with tabs.add_tab("Play", icon=viser.Icon.PLAYER_PLAY):
        backend_dropdown = server.gui.add_dropdown(
            "Simulator", ("sim", "wm"), initial_value=backend, hint="sim: MuJoCo; wm: world model"
        )
        clip_dropdown = server.gui.add_dropdown("Motion", motion.clip_names)
        with server.gui.add_folder("Playback"):
            play = server.gui.add_button("Play", icon=viser.Icon.PLAYER_PLAY)
            pause = server.gui.add_button("Pause", icon=viser.Icon.PLAYER_PAUSE, visible=False)
            step = server.gui.add_button("Step", icon=viser.Icon.PLAYER_TRACK_NEXT)
            restart = server.gui.add_button("Restart", icon=viser.Icon.REFRESH)
            speed = server.gui.add_slider("Speed", 0.1, 2.0, 0.1, 1.0)
            loop = server.gui.add_checkbox("Restart at end", True)
        with server.gui.add_folder("Reference"):
            offset = server.gui.add_slider("Side offset (m)", 0.0, 2.0, 0.05, 1.0)
            show_points = server.gui.add_checkbox("Tracked bodies", True)
        status = server.gui.add_markdown("")

    @play.on_click
    def _(_):
        control["playing"] = True
        play.visible, pause.visible = False, True

    @pause.on_click
    def _(_):
        control["playing"] = False
        play.visible, pause.visible = True, False

    @step.on_click
    def _(_):
        control["step"] = True

    @restart.on_click
    def _(_):
        control["reset"] = True

    @clip_dropdown.on_update
    def _(_):
        control["reset"] = True

    @backend_dropdown.on_update
    def _(_):
        control["backend"] = backend_dropdown.value
        control["reset"] = True

    robot_points = server.scene.add_point_cloud(
        "/tracked/robot", np.zeros((1, 3)), np.array([[0, 170, 255]]), point_size=0.05
    )
    reference_points = server.scene.add_point_cloud(
        "/tracked/reference", np.zeros((1, 3)), np.array([[255, 140, 0]]), point_size=0.05
    )
    pelvis = session.pose.body_ids[0]

    def render():
        robot_xpos, robot_xmat = session.pose.kinematics(session.robot_qpos)
        ref_xpos, ref_xmat = session.pose.kinematics(session.reference_qpos)
        shift = np.array([0.0, offset.value, 0.0])
        ref_xpos = ref_xpos + shift
        scene.update_from_arrays(np.stack((robot_xpos, ref_xpos)), np.stack((robot_xmat, ref_xmat)))
        origin = -robot_xpos[pelvis] if scene.camera_tracking_enabled else np.zeros(3)
        robot_points.points = (session.tracked[0] + origin).astype(np.float32)
        reference_points.points = (session.reference_points + origin).astype(np.float32)
        robot_points.visible = reference_points.visible = show_points.value
        info = session.info
        lines = [
            f"**{session.backend.name}** · iteration {policy.iteration} · goal `{policy.goal_kind}`",
            f"frame {info['frame']} / {info['length'] - 1} · {info['status']}",
            f"return {info['return']:.3f}",
            f"root XY error {info['root_xy_error']:.3f} m · root Z error {info['root_z_error']:.3f} m",
            f"local body error {info['local_body_error']:.3f} m",
        ]
        if "survival" in info:
            lines.append(f"WM survival {info['survival']:.3f}")
        if "fit_rms" in info:
            lines.append(f"display fit RMS {info['fit_rms'] * 100:.1f} cm")
        lines.append("")
        lines.append("| reward term | value |")
        lines.append("|---|---:|")
        lines.extend(f"| {k} | {v:.3f} |" for k, v in info["terms"].items())
        status.content = "\n".join(lines)

    print(f"viser: http://localhost:{port}", flush=True)
    while True:
        start = time.perf_counter()
        with lock:
            if control["reset"]:
                session.select(control["backend"])
                session.reset(motion.clip_names.index(clip_dropdown.value))
                control["reset"] = False
                render()
            elif control["playing"] or control["step"]:
                control["step"] = False
                if session.done:
                    if control["playing"] and loop.value:
                        session.reset()
                else:
                    session.step()
                render()
        time.sleep(max(0.0, CONTROL_DT / speed.value - (time.perf_counter() - start)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backend", choices=tuple(BACKENDS), default="sim")
    parser.add_argument("--motion", help="Prepared motion npz; defaults to data/lafan1_g1.npz")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--headless", action="store_true", help="Print per-clip metrics, no viewer")
    parser.add_argument("--clips", type=int, nargs="*", help="Clip indices for --headless")
    parser.add_argument("--max-steps", type=int, default=10**9)
    args = parser.parse_args()
    torch.set_num_threads(8)
    policy = Policy(args.checkpoint, args.motion, args.device)
    print(
        f"loaded {policy.path} iteration {policy.iteration} wm {policy.cfg['wm_kind']} "
        f"goal {policy.goal_kind}",
        flush=True,
    )
    session = Session(policy, G1Pose(policy.motion))
    if args.headless:
        clips = args.clips if args.clips else range(policy.motion.num_clips)
        headless(session, args.backend, clips, args.max_steps)
    else:
        view(session, args.backend, args.port)


if __name__ == "__main__":
    main()
