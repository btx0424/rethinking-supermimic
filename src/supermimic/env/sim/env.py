"""G1 MuJoCo environment. Hard episode endings reset only the affected environments."""

import torch
import warp as wp
from mjlab.managers.event_manager import RecomputeLevel
from mjlab.scene import Scene, SceneCfg
from mjlab.sim import MujocoCfg, Simulation, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from supermimic.robots.g1 import (
    robot_config,
    action_scale,
    normalize_context,
    ROOT_BODY,
)
from supermimic.env.mdp import CONTROL_DT, DECIMATION
from supermimic.utils.math import quat_from_euler_xyz, quat_mul
from supermimic.env.mdp.state import PhysicalState
from supermimic.env.mdp.geometry.state import build_maxcoord_full_state
from supermimic.env.rollout.interface import RolloutEnv, TrackingState, Transition


class SimulationEnv(RolloutEnv):
    def __init__(self, task, num_envs, device="cuda", gamma=0.99, eval_mode=False, auto_reset=True):
        self.task = task
        self.motion = task.motion
        self.device = device
        self.num_envs = num_envs
        self.gamma = gamma
        self.eval_mode = eval_mode
        self.auto_reset = auto_reset
        self.max_steps = 10**9 if eval_mode else task.cfg.max_episode_steps
        if task.motion.body_names[0] != ROOT_BODY:
            raise ValueError("The first tracked body must be the G1 root")
        wp.set_mempool_release_threshold(device, 1.0)
        self.scene = Scene(
            SceneCfg(
                num_envs=num_envs,
                env_spacing=2.0,
                terrain=TerrainEntityCfg(terrain_type="plane"),
                entities={"robot": robot_config()},
            ),
            device,
        )
        self.sim = Simulation(
            num_envs=num_envs,
            cfg=SimulationCfg(
                nconmax=128,
                njmax=500,
                mujoco=MujocoCfg(timestep=CONTROL_DT / DECIMATION, iterations=10, ls_iterations=20),
            ),
            spec=self.scene.spec,
            device=device,
        )
        self.scene.initialize(self.sim.mj_model, self.sim.model, self.sim.data)
        self.robot = self.scene["robot"]
        if tuple(self.robot.joint_names) != self.motion.joint_names:
            raise ValueError("Motion/G1 joint order differs")
        default = self.robot.data.default_joint_pos
        self.default_q = (default[0] if default.ndim == 2 else default).detach().clone()
        limits = self.robot.data.soft_joint_pos_limits
        if limits.ndim == 3:
            limits = limits[0]
        self.soft_lo, self.soft_hi = limits[:, 0].clone(), limits[:, 1].clone()
        self.action_scale = torch.tensor(action_scale(self.motion.joint_names), device=device)
        missing = set(self.motion.body_names) - set(self.robot.body_names)
        if missing:
            raise ValueError(f"Dataset bodies absent from robot: {missing}")
        self.tracked_ids = torch.tensor(
            [self.robot.body_names.index(n) for n in self.motion.body_names], device=device
        )
        self.torso_id = int(
            self.robot.indexing.body_ids[self.robot.body_names.index(task.cfg.torso_body)]
        )
        # Generic collision mask. No foot/hand names or semantic body groups.
        ids = self.robot.indexing.geom_ids
        mj = self.sim.mj_model
        collision = torch.as_tensor(
            (mj.geom_contype != 0) | (mj.geom_conaffinity != 0), device=device
        )
        self.collision_ids = ids[collision[ids]]
        if self.collision_ids.numel() == 0:
            raise ValueError("G1 has no collision geoms")
        self.terrain_ids = self.scene["terrain"].indexing.geom_ids
        self.sim.expand_model_fields(("geom_friction", "body_ipos"))
        self.default_ipos = self.sim.get_default_field("body_ipos").clone()
        self.sim.model.geom_friction[:, self.terrain_ids, 0] = 1.0 if eval_mode else 0.3
        n = num_envs
        d = device
        self.frame = torch.zeros(n, dtype=torch.long, device=d)
        self.ep_step = torch.zeros_like(self.frame)
        self.lifetime = torch.zeros_like(self.frame)
        self.ep_id = torch.zeros_like(self.frame)
        self.start_bin = torch.zeros_like(self.frame)
        self.previous = torch.zeros(n, task.num_joints, device=d)
        self.xi = torch.zeros(n, task.num_joints + 4, device=d)
        self.ep_return = torch.zeros(n, device=d)
        self.ep_discounted = torch.zeros_like(self.ep_return)
        self.bin_return = torch.zeros(self.motion.num_bins, device=d)
        self.bin_average = torch.full_like(self.bin_return, 1.0 / self.motion.num_bins)
        self.max_return = task.max_step_reward / (1 - gamma)
        self.episode_counter = 0

    def read_state(self):
        pose = self.robot.data.body_link_pose_w[:, self.tracked_ids]
        velocity = self.robot.data.body_link_vel_w[:, self.tracked_ids]
        full = build_maxcoord_full_state(
            pose[..., :3] - self.scene.env_origins[:, None, :],
            pose[..., 3:7],
            velocity[..., :3],
            velocity[..., 3:],
        )
        rep = self.task.representation
        xi = self.xi.clone()
        physical = PhysicalState(
            rep.extract_aux_state(full), rep.represent(full), normalize_context(xi)
        )
        return TrackingState(
            physical,
            self.task.command.from_frame(self.frame),
            self.previous.clone(),
            xi,
            self.ep_step.clone(),
            full,
        )

    def sample_frames(self, n):
        soft = torch.softmax(-25 * self.bin_return / self.max_return, 0)
        self.bin_average.mul_(0.999).add_(soft, alpha=0.001)
        probs = (
            0.5 / self.motion.num_bins
            + 0.4 * soft
            + 0.1 * self.bin_average / self.bin_average.sum()
        )
        return self.task.command.frame(self.task.command.sample(n, probs))

    def reset(self, env_ids=None, frames=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if env_ids.numel() == 0:
            return self.read_state()
        if frames is None:
            frames = (
                self.motion.clip_start[env_ids % self.motion.num_clips]
                if self.eval_mode
                else self.sample_frames(env_ids.numel())
            )
        started = env_ids[self.ep_step[env_ids] > 0]
        self.bin_return[self.start_bin[started]] = (
            0.8 * self.bin_return[self.start_bin[started]] + 0.2 * self.ep_discounted[started]
        )
        self.sim.reset(env_ids)
        n = env_ids.numel()
        j = self.task.num_joints
        if self.eval_mode:
            self.xi[env_ids, 0] = 1.0
            self.xi[env_ids, 1:] = 0
        else:
            self.xi[env_ids, 0] = torch.empty(n, device=self.device).uniform_(0.3, 1.2)
            self.xi[env_ids, 1 : 1 + j] = torch.empty(n, j, device=self.device).uniform_(
                -0.01, 0.01
            )
            for k, scale in enumerate((0.025, 0.05, 0.05)):
                self.xi[env_ids, 1 + j + k] = torch.empty(n, device=self.device).uniform_(
                    -scale, scale
                )
        self.sim.model.geom_friction[env_ids[:, None], self.collision_ids, 0] = self.xi[env_ids, :1]
        self.robot.data.encoder_bias[env_ids] = self.xi[env_ids, 1 : 1 + j]
        base = self.default_ipos
        ipos = (
            base.unsqueeze(0).expand(n, -1, -1).clone() if base.ndim == 2 else base[env_ids].clone()
        )
        ipos[:, self.torso_id] += self.xi[env_ids, 1 + j :]
        self.sim.model.body_ipos[env_ids] = ipos
        self.sim.recompute_constants(RecomputeLevel.set_const)
        motion = self.motion
        p = motion.root_p[frames].clone()
        q = motion.root_quat[frames].clone()
        v = motion.root_v[frames].clone()
        w = motion.root_w[frames].clone()
        joints = motion.q[frames].clone()
        joint_vel = motion.qd[frames].clone()
        if not self.eval_mode:
            pose = torch.empty(n, 6, device=self.device).uniform_(-1, 1) * p.new_tensor(
                [0.05, 0.05, 0.01, 0.1, 0.1, 0.2]
            )
            velocity = torch.empty(n, 6, device=self.device).uniform_(-1, 1) * p.new_tensor(
                [0.5, 0.5, 0.2, 0.52, 0.52, 0.78]
            )
            p += pose[:, :3]
            q = quat_mul(quat_from_euler_xyz(pose[:, 3], pose[:, 4], pose[:, 5]), q)
            v += velocity[:, :3]
            w += velocity[:, 3:]
            joints += torch.empty_like(joints).uniform_(-0.1, 0.1)
        joints = joints.clamp(self.soft_lo, self.soft_hi)
        self.robot.write_root_link_pose_to_sim(
            torch.cat((p + self.scene.env_origins[env_ids], q), -1), env_ids=env_ids
        )
        self.robot.write_root_link_velocity_to_sim(torch.cat((v, w), -1), env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joints, joint_vel, env_ids=env_ids)
        self.robot.set_joint_position_target(joints, env_ids=env_ids)
        self.frame[env_ids] = frames
        self.ep_step[env_ids] = 0
        self.previous[env_ids] = 0
        self.start_bin[env_ids] = motion.frame_bin[frames]
        self.ep_return[env_ids] = 0
        self.ep_discounted[env_ids] = 0
        self.episode_counter += 1
        self.ep_id[env_ids] = self.episode_counter
        self.scene.write_data_to_sim()
        self.sim.forward()
        return self.read_state()

    def step(self, state, action):
        episode = self.ep_id.clone()
        step = self.lifetime.clone()
        self.robot.set_joint_position_target(self.default_q + self.action_scale * action)
        for _ in range(DECIMATION):
            self.scene.write_data_to_sim()
            self.sim.step()
        self.sim.forward()
        self.frame = self.task.command.frame(self.task.command.advance(state.command))
        self.previous = action.detach().clone()
        self.ep_step = state.episode_step + 1
        self.lifetime += 1
        successor = self.read_state()
        reward, terms = self.task.reward(
            successor.physical, self.frame, action, state.previous_action
        )
        terminated = self.task.terminated(successor.physical, self.frame)
        truncated = (self.ep_step >= self.max_steps) | (
            self.frame >= self.motion.last_frame(self.frame)
        )
        done = terminated | truncated
        self.ep_return += reward
        self.ep_discounted += self.gamma ** (self.ep_step - 1).float() * reward
        info = {
            "episode": episode,
            "step": step,
            "episode_length": self.ep_step.clone(),
            "episode_return": self.ep_return.clone(),
        }
        continuation = (
            self.reset(done.nonzero().flatten())
            if self.auto_reset and bool(done.any())
            else successor
        )
        return Transition(successor, continuation, reward, terminated, truncated, terms, info)
