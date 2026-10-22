"""Thin mjlab collector. The simulator logs transitions; it is not the learner."""

from __future__ import annotations

import torch

from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.managers.event_manager import RecomputeLevel
from mjlab.scene import Scene, SceneCfg
from mjlab.sim import MujocoCfg, Simulation, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.string import resolve_expr

from .kinematics import TorchKinematics
from .math_utils import quat_from_euler_xyz, quat_mul
from .mdp import MAX_STEP_REWARD, MDP, flatten_state
from .motion import BIN_FRAMES, MotionLib

DECIMATION = 4
MAX_EPISODE_STEPS = 500  # 10 s at 50 Hz.


class TrackingSim:
    def __init__(
        self,
        num_envs: int,
        motion: MotionLib,
        kin: TorchKinematics,
        device: str,
        gamma: float = 0.99,
        eval_mode: bool = False,
        reset_noise: bool = True,
    ):
        self.device = device
        self.motion = motion
        self.num_envs = num_envs
        self.gamma = gamma
        self.eval_mode = eval_mode
        self.reset_noise = reset_noise and not eval_mode
        self.max_steps = 10**9 if eval_mode else MAX_EPISODE_STEPS
        scene_cfg = SceneCfg(
            num_envs=num_envs,
            env_spacing=2.0,
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": get_g1_robot_cfg()},
        )
        self.scene = Scene(scene_cfg, device)
        sim_cfg = SimulationCfg(
            nconmax=128,
            njmax=500,
            mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
        )
        self.sim = Simulation(
            num_envs=num_envs,
            cfg=sim_cfg,
            spec=self.scene.spec,
            device=device,
        )
        self.scene.initialize(self.sim.mj_model, self.sim.model, self.sim.data)
        self.robot = self.scene["robot"]
        if tuple(self.robot.joint_names) != motion.joint_names:
            raise RuntimeError(
                "Motion joint order does not match the mjlab G1: "
                f"{motion.joint_names} vs {tuple(self.robot.joint_names)}"
            )
        default = self.robot.data.default_joint_pos
        self.default_q = (default[0] if default.ndim == 2 else default).detach().clone()
        limits = self.robot.data.soft_joint_pos_limits
        if limits.ndim == 3:
            limits = limits[0]
        self.mdp = MDP(kin, motion, self.default_q, limits[:, 0].clone(), limits[:, 1].clone())
        scale = resolve_expr(G1_ACTION_SCALE, tuple(self.robot.joint_names))
        if any(value is None for value in scale):
            missing = [name for name, value in zip(self.robot.joint_names, scale) if value is None]
            raise KeyError(f"No action scale for joints {missing}")
        self.action_scale = torch.tensor(scale, device=device, dtype=torch.float32)
        foot_local, foot_names = self.robot.find_geoms(r"^(left|right)_foot[1-7]_collision$")
        if not foot_names:
            raise RuntimeError("The G1 model has no foot collision geoms.")
        self._foot_ids = self.robot.indexing.geom_ids[foot_local]
        torso_local = self.robot.body_names.index("torso_link")
        self._torso_id = int(self.robot.indexing.body_ids[torso_local])
        self._terrain_ids = self.scene["terrain"].indexing.geom_ids
        self.sim.expand_model_fields(("geom_friction", "body_ipos"))
        self._default_ipos = self.sim.get_default_field("body_ipos").clone()
        self.G_max = MAX_STEP_REWARD / (1.0 - gamma)
        self.bin_G = torch.zeros(motion.num_bins, device=device)
        self.bin_w_avg = torch.full((motion.num_bins,), 1.0 / motion.num_bins, device=device)
        n = num_envs
        self.f = torch.zeros(n, dtype=torch.long, device=device)
        self.ep_step = torch.zeros(n, dtype=torch.long, device=device)
        self.a_prev = torch.zeros(n, self.default_q.shape[0], device=device)
        self.xi = torch.zeros(n, 1 + self.default_q.shape[0] + 3, device=device)
        self.ep_id = torch.zeros(n, dtype=torch.long, device=device)
        self.lifetime = torch.zeros(n, dtype=torch.long, device=device)
        self.start_bin = torch.zeros(n, dtype=torch.long, device=device)
        self.ep_disc = torch.zeros(n, device=device)
        self.ep_ret = torch.zeros(n, device=device)
        self._ep_counter = 0
        ground = 1.0 if eval_mode else 0.3
        self.sim.model.geom_friction[:, self._terrain_ids, 0] = ground
        env_ids = torch.arange(n, device=device)
        if eval_mode:
            frames = motion.clip_start[torch.arange(n, device=device) % motion.num_clips]
            self.reset(env_ids, frames)
        else:
            self.reset(env_ids)

    def state(self):
        data = self.robot.data
        pose = data.root_link_pose_w
        p = pose[:, :3] - self.scene.env_origins
        quat = pose[:, 3:7]
        vel = data.root_link_vel_w
        return p, quat, vel[:, :3], vel[:, 3:6], data.joint_pos, data.joint_vel

    def reset(self, env_ids: torch.Tensor, frames: torch.Tensor | None = None) -> None:
        if env_ids.numel() == 0:
            return
        if frames is None:
            frames, bins = self._sample_frames(env_ids.numel())
        else:
            bins = self.motion.frame_bin[frames]
        self._close_bins(env_ids)
        self.sim.reset(env_ids)
        self._sample_xi(env_ids)
        self._apply_xi(env_ids)
        self._write_reference(env_ids, frames)
        self.f[env_ids] = frames
        self.ep_step[env_ids] = 0
        self.a_prev[env_ids] = 0
        self.start_bin[env_ids] = bins
        self.ep_disc[env_ids] = 0
        self.ep_ret[env_ids] = 0
        self._ep_counter += 1
        self.ep_id[env_ids] = self._ep_counter
        self.scene.write_data_to_sim()
        self.sim.forward()

    def step(self, action: torch.Tensor, auto_reset: bool = True) -> dict[str, torch.Tensor]:
        p, quat, v, w, q, qd = self.state()
        s = flatten_state(p, quat, v, w, q, qd)
        a_prev = self.a_prev
        f = self.f
        xi = self.xi
        ep = self.ep_id.clone()
        step_id = self.lifetime.clone()
        target = self.default_q + self.action_scale * action
        self.robot.set_joint_position_target(target)
        for _ in range(DECIMATION):
            self.scene.write_data_to_sim()
            self.sim.step()
        self.sim.forward()
        p1, quat1, v1, w1, q1, qd1 = self.state()
        last = self.motion.clip_end[self.motion.clip_id[f]] - 1
        f_next = torch.minimum(f + 1, last)
        reward = self.mdp.reward(p1, quat1, v1, w1, q1, qd1, f_next, action, a_prev)
        hard = self.mdp.termination(p1, quat1, f_next)
        self.ep_step += 1
        self.lifetime += 1
        timeout = self.ep_step >= self.max_steps
        clip_end = f_next >= last
        self.ep_disc = self.ep_disc + torch.pow(self.gamma, (self.ep_step - 1).float()) * reward
        self.ep_ret = self.ep_ret + reward
        self.f = f_next
        self.a_prev = action.detach().clone()
        done = hard | timeout | clip_end
        out = {
            "s": s,
            "s1": flatten_state(p1, quat1, v1, w1, q1, qd1),
            "a": action.detach(),
            "a_prev": a_prev.detach().clone(),
            "xi": xi.clone(),
            "f": f.clone(),
            "ep": ep,
            "step": step_id,
            "r": reward,
            "d": hard,
            "timeout": timeout,
            "clip_end": clip_end,
            "done": done,
            "ep_ret": self.ep_ret.clone(),
            "ep_len": self.ep_step.clone(),
        }
        if auto_reset and bool(done.any()):
            self.reset(done.nonzero(as_tuple=False).flatten())
        return out

    def _sample_frames(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        motion = self.motion
        if self.eval_mode:
            raise RuntimeError("Eval resets must pass explicit frames.")
        soft = torch.softmax(-25.0 * self.bin_G / self.G_max, dim=0)
        self.bin_w_avg.mul_(0.999).add_(soft, alpha=0.001)
        probs = 0.5 / motion.num_bins + 0.4 * soft + 0.1 * self.bin_w_avg / self.bin_w_avg.sum()
        bins = torch.multinomial(probs, n, replacement=True)
        start = motion.bin_start[bins]
        clip = motion.bin_clip[bins]
        hi = torch.minimum(start + BIN_FRAMES, motion.clip_end[clip] - 1)
        hi = torch.maximum(hi, start + 1)
        frames = start + (torch.rand(n, device=self.device) * (hi - start).float()).long()
        return frames, bins

    def _sample_xi(self, env_ids: torch.Tensor) -> None:
        n = env_ids.numel()
        dev = self.device
        if self.eval_mode:
            self.xi[env_ids, 0] = 1.0
            self.xi[env_ids, 1:] = 0
            return
        self.xi[env_ids, 0] = torch.empty(n, device=dev).uniform_(0.3, 1.2)
        self.xi[env_ids, 1:30] = torch.empty(n, 29, device=dev).uniform_(-0.01, 0.01)
        self.xi[env_ids, 30] = torch.empty(n, device=dev).uniform_(-0.025, 0.025)
        self.xi[env_ids, 31] = torch.empty(n, device=dev).uniform_(-0.05, 0.05)
        self.xi[env_ids, 32] = torch.empty(n, device=dev).uniform_(-0.05, 0.05)

    def _apply_xi(self, env_ids: torch.Tensor) -> None:
        mu = self.xi[env_ids, 0]
        self.sim.model.geom_friction[env_ids[:, None], self._foot_ids, 0] = mu[:, None]
        self.robot.data.encoder_bias[env_ids] = self.xi[env_ids, 1:30]
        base = self._default_ipos
        if base.ndim == 2:
            ipos = base.unsqueeze(0).expand(env_ids.numel(), -1, -1).clone()
        else:
            ipos = base[env_ids].clone()
        ipos[:, self._torso_id] = ipos[:, self._torso_id] + self.xi[env_ids, 30:33]
        self.sim.model.body_ipos[env_ids] = ipos
        self.sim.recompute_constants(RecomputeLevel.set_const)

    def _write_reference(self, env_ids: torch.Tensor, frames: torch.Tensor) -> None:
        motion = self.motion
        p = motion.root_p[frames].clone()
        quat = motion.root_quat[frames].clone()
        v = motion.root_v[frames].clone()
        w = motion.root_w[frames].clone()
        q = motion.q[frames].clone()
        qd = motion.qd[frames].clone()
        if self.reset_noise:
            n = env_ids.numel()
            dev = self.device
            pose = torch.empty(n, 6, device=dev)
            pose[:, 0].uniform_(-0.05, 0.05)
            pose[:, 1].uniform_(-0.05, 0.05)
            pose[:, 2].uniform_(-0.01, 0.01)
            pose[:, 3].uniform_(-0.1, 0.1)
            pose[:, 4].uniform_(-0.1, 0.1)
            pose[:, 5].uniform_(-0.2, 0.2)
            vel = torch.empty(n, 6, device=dev)
            vel[:, 0].uniform_(-0.5, 0.5)
            vel[:, 1].uniform_(-0.5, 0.5)
            vel[:, 2].uniform_(-0.2, 0.2)
            vel[:, 3].uniform_(-0.52, 0.52)
            vel[:, 4].uniform_(-0.52, 0.52)
            vel[:, 5].uniform_(-0.78, 0.78)
            p = p + pose[:, 0:3]
            quat = quat_mul(quat_from_euler_xyz(pose[:, 3], pose[:, 4], pose[:, 5]), quat)
            v = v + vel[:, 0:3]
            w = w + vel[:, 3:6]
            q = q + torch.empty_like(q).uniform_(-0.1, 0.1)
        q = torch.maximum(torch.minimum(q, self.mdp.soft_hi), self.mdp.soft_lo)
        origin = self.scene.env_origins[env_ids]
        self.robot.write_root_link_pose_to_sim(torch.cat((p + origin, quat), dim=-1), env_ids=env_ids)
        self.robot.write_root_link_velocity_to_sim(torch.cat((v, w), dim=-1), env_ids=env_ids)
        self.robot.write_joint_state_to_sim(q, qd, env_ids=env_ids)
        self.robot.set_joint_position_target(q, env_ids=env_ids)

    def _close_bins(self, env_ids: torch.Tensor) -> None:
        # The first reset has no return to record. ep_step stays 0 until a step lands.
        started = self.ep_step[env_ids] > 0
        if not bool(started.any()):
            return
        ids = env_ids[started]
        updated = 0.8 * self.bin_G[self.start_bin[ids]] + 0.2 * self.ep_disc[ids]
        self.bin_G[self.start_bin[ids]] = updated
