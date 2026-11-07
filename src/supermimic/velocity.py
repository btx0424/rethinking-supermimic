"""Flat-ground velocity tasks, beside the motion-tracking simulator.

The scene, twist command, sensors, and rewards are mjlab's
``unitree_g1_flat_env_cfg`` or ``unitree_go1_flat_env_cfg``. This wrapper only
exposes the step dict the SAC loop already consumes. It does not use a motion
library or the tracking MDP.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from supermimic.utils.episode import done_from


_FLAT_CFG = {
    "g1": ("g1", "unitree_g1_flat_env_cfg"),
    "go1": ("go1", "unitree_go1_flat_env_cfg"),
}


def _flat_env_cfg(robot: str):
    """mjlab's flat velocity config, without importing the task registry.

    ``mjlab.tasks`` walks every task on import and pulls in the on-policy runner.
    The env config module does not need that, so the package init is skipped.
    """
    import importlib.util
    import sys
    import types
    from pathlib import Path

    import mjlab

    try:
        folder, function = _FLAT_CFG[robot]
    except KeyError as exc:
        known = ", ".join(sorted(_FLAT_CFG))
        raise KeyError(f"Unknown velocity robot {robot!r}. Known: {known}.") from exc
    tasks_root = Path(mjlab.__file__).resolve().parent / "tasks"
    if "mjlab.tasks" not in sys.modules:
        package = types.ModuleType("mjlab.tasks")
        package.__path__ = [str(tasks_root)]
        package.__package__ = "mjlab.tasks"
        sys.modules["mjlab.tasks"] = package
    spec = importlib.util.spec_from_file_location(
        f"supermimic._{robot}_flat_env_cfgs",
        tasks_root / "velocity" / "config" / folder / "env_cfgs.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {robot} flat env config from {tasks_root}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, function)(play=False)


class IdentityCriticNorms(nn.Module):
    """Critic features are already the policy observation, so they are not split."""

    def normalize_critic_features(self, flat: torch.Tensor) -> torch.Tensor:
        return flat


class CommandGoals:
    """Twist command indexed like a motion frame.

    Each environment owns a stride of the integer frame space. Within that stride
    a ring holds the command for every live replay frame plus one episode of
    future, so ``goal(frame + 1)`` during imagination still sees the twist.
    """

    def __init__(self, num_envs: int, ring: int, device: str) -> None:
        self.num_envs = int(num_envs)
        self.ring = int(ring)
        self.stride = 1 << 40
        dev = device
        self.command = torch.zeros(self.num_envs, self.ring, 3, device=dev)
        self.last = torch.zeros(self.num_envs, self.ring, dtype=torch.long, device=dev)
        self.counter = torch.zeros(self.num_envs, dtype=torch.long, device=dev)

    def frame(self) -> torch.Tensor:
        env = torch.arange(self.num_envs, device=self.command.device)
        return env * self.stride + self.counter

    def fill(self, env_ids: torch.Tensor, command: torch.Tensor, remaining: torch.Tensor) -> None:
        """Write ``command`` from the current frame through ``remaining`` steps."""
        if env_ids.numel() == 0:
            return
        remaining = remaining.to(dtype=torch.long, device=self.command.device).reshape(-1)
        command = command.to(dtype=self.command.dtype, device=self.command.device).reshape(-1, 3)
        span = int(remaining.max().item())
        if span <= 0:
            return
        steps = torch.arange(span, device=self.command.device)
        base = self.counter[env_ids].unsqueeze(1) + steps.unsqueeze(0)
        slot = base % self.ring
        mask = steps.unsqueeze(0) < remaining.unsqueeze(1)
        env_grid = env_ids.reshape(-1, 1).expand_as(slot)
        cmd = command.unsqueeze(1).expand(-1, span, 3)
        self.command[env_grid[mask], slot[mask]] = cmd[mask]
        end = env_ids.reshape(-1) * self.stride + self.counter[env_ids] + remaining - 1
        self.last[env_grid[mask], slot[mask]] = end.unsqueeze(1).expand_as(slot)[mask]

    def goal(self, frame: torch.Tensor, horizon: int) -> torch.Tensor:
        env = torch.div(frame, self.stride, rounding_mode="floor")
        slot = (frame % self.stride) % self.ring
        cmd = self.command[env, slot]
        return cmd.unsqueeze(1).expand(frame.shape[0], horizon, 3).reshape(frame.shape[0], horizon * 3)

    def episode_last(self, frame: torch.Tensor) -> torch.Tensor:
        env = torch.div(frame, self.stride, rounding_mode="floor")
        slot = (frame % self.stride) % self.ring
        return self.last[env, slot]

    def advance(self) -> None:
        self.counter += 1


class VelocityMDP:
    """Goals are the sampled twist. Rewards use the root and the joints only.

    Foot-contact costs need sensors the world model does not predict, so they
    stay out of this analytic reward. The same terms score real transitions and
    imagined ones.
    """

    reward_terms_compiled = False
    object_reward_surrogate = False

    def __init__(
        self,
        task,
        goals: CommandGoals,
        robot,
        weights: dict[str, float],
        pose,
        walking_threshold: float,
        running_threshold: float,
    ) -> None:
        from supermimic.mdp import VELOCITY_REWARD_TERMS

        self.task = task
        self.goals = goals
        self.term_names = VELOCITY_REWARD_TERMS
        self.weights = {name: float(weights.get(name, 0.0)) for name in self.term_names}
        self.default_q = robot.data.default_joint_pos[0].detach().clone()
        limits = robot.data.soft_joint_pos_limits
        self.soft_lo = limits[0, :, 0].detach().clone()
        self.soft_hi = limits[0, :, 1].detach().clone()
        self.std_standing = pose.std_standing.detach().clone()
        self.std_walking = pose.std_walking.detach().clone()
        self.std_running = pose.std_running.detach().clone()
        self.walk = float(walking_threshold)
        self.run = float(running_threshold)
        self.fall_angle = float(torch.deg2rad(torch.tensor(70.0)))
        self.gravity = self.default_q.new_tensor([0.0, 0.0, -1.0])
        self.lin_std = 0.5  # sqrt(0.25), mjlab's flat linear-velocity std
        self.ang_std = 0.5 ** 0.5
        self.upright_std = 0.2 ** 0.5
        positive = sum(max(weight, 0.0) for weight in self.weights.values())
        # No control-dt factor: training already normalizes the return.
        self.max_step_reward = positive
        # Proprioception, matching the flat actor group without the command.
        # The command is the goal, repeated across the horizon.
        n = int(self.default_q.shape[0])
        self.obs_dim = 9 + 3 * n
        noise = self.default_q.new_zeros(self.obs_dim)
        # lin, ang, gravity, joints, joint vel, last action.
        noise[0:3] = 0.5
        noise[3:6] = 0.2
        noise[6:9] = 0.05
        noise[9 : 9 + n] = 0.01
        noise[9 + n : 9 + 2 * n] = 1.5
        self.noise = noise

    def goal(self, frame: torch.Tensor, horizon: int) -> torch.Tensor:
        return self.goals.goal(frame, horizon)

    def episode_last(self, frame: torch.Tensor) -> torch.Tensor:
        return self.goals.episode_last(frame)

    def add_noise(self, obs: torch.Tensor) -> torch.Tensor:
        return obs + (2.0 * torch.rand_like(obs) - 1.0) * self.noise

    def bodies(self, p, quat, v, w, q, qd):
        return self.kin.forward(p, quat, v, w, q, qd)

    def heading_bodies(self, p, quat, v, w, q, qd, body_p=None) -> torch.Tensor:
        from supermimic.mdp import heading_body_pos

        del v, w, q, qd
        if body_p is None:
            raise ValueError("Velocity observations pass body positions in.")
        return heading_body_pos(p, quat, body_p)

    def privileged(self, p, quat, v, w, q, qd, frame, object_state=None) -> torch.Tensor:
        del p, quat, v, w, frame, object_state
        err = q - self.default_q
        pad = q.new_zeros(q.shape[0], self.task.priv_dim - err.shape[-1])
        return torch.cat((err, pad), dim=-1)

    def obs(self, state, a_prev, xi, frame) -> torch.Tensor:
        del xi
        from supermimic.utils.math import quat_apply_inverse

        cmd = self.goals.goal(frame, 1)[:, :3]
        del cmd
        lin = quat_apply_inverse(state.quat, state.v)
        ang = quat_apply_inverse(state.quat, state.w)
        gravity = quat_apply_inverse(state.quat, self.gravity.expand_as(state.v))
        return torch.cat(
            (lin, ang, gravity, state.q - self.default_q, state.qd, a_prev),
            dim=-1,
        )

    def reward(self, state, frame, action, a_prev, terms: bool = False, *, surrogate=None):
        del surrogate
        from supermimic.utils.math import quat_apply_inverse

        cmd = self.goals.goal(frame, 1)[:, :3]
        lin = quat_apply_inverse(state.quat, state.v)
        ang = quat_apply_inverse(state.quat, state.w)
        gravity = quat_apply_inverse(state.quat, self.gravity.expand_as(state.v))
        xy = (cmd[:, :2] - lin[:, :2]).square().sum(-1) + lin[:, 2].square()
        track_lin = torch.exp(-xy / self.lin_std**2)
        ang_err = (cmd[:, 2] - ang[:, 2]).square() + ang[:, :2].square().sum(-1)
        track_ang = torch.exp(-ang_err / self.ang_std**2)
        upright = torch.exp(-gravity[:, :2].square().sum(-1) / self.upright_std**2)
        speed = cmd[:, :2].norm(dim=-1) + cmd[:, 2].abs()
        stand = (speed < self.walk).float()
        walk = ((speed >= self.walk) & (speed < self.run)).float()
        run = (speed >= self.run).float()
        std = (
            self.std_standing * stand.unsqueeze(1)
            + self.std_walking * walk.unsqueeze(1)
            + self.std_running * run.unsqueeze(1)
        )
        pose = torch.exp(-((state.q - self.default_q).square() / std.square()).mean(-1))
        body_ang = state.w[:, :2].square().sum(-1)
        over = (self.soft_lo - state.q).clamp(min=0) + (state.q - self.soft_hi).clamp(min=0)
        limits = over.sum(-1)
        rate = (action - a_prev).square().sum(-1)
        raw = {
            "track_linear_velocity": track_lin,
            "track_angular_velocity": track_ang,
            "upright": upright,
            "pose": pose,
            "body_ang_vel": body_ang,
            "dof_pos_limits": limits,
            "action_rate_l2": rate,
        }
        parts = {name: raw[name] * self.weights[name] for name in self.term_names}
        total = parts[self.term_names[0]]
        for name in self.term_names[1:]:
            total = total + parts[name]
        if terms:
            return total, parts
        return total

    def termination(self, p, quat, frame) -> torch.Tensor:
        del p, frame
        return self._tilt(quat) > self.fall_angle

    def soft_termination(self, p, quat, frame, height_scale: float = 0.02, tilt_scale: float = 0.04) -> torch.Tensor:
        del p, frame, height_scale
        return torch.sigmoid((self._tilt(quat) - self.fall_angle) / tilt_scale)

    def _tilt(self, quat: torch.Tensor) -> torch.Tensor:
        from supermimic.utils.math import quat_apply_inverse

        gravity = quat_apply_inverse(quat, self.gravity.expand(quat.shape[0], 3))
        return torch.acos(torch.clamp(-gravity[:, 2], -1.0, 1.0)).abs()

    def compile_heads(self, enabled: bool = True) -> None:
        del enabled
        self.reward_terms_compiled = False


def _pose_term(env):
    for term in env.reward_manager._term_cfgs:
        if hasattr(term.func, "std_standing"):
            return term.func
    raise RuntimeError("Flat velocity config has no variable_posture reward.")


class VelocitySim:
    """mjlab humanoid or quadruped on a plane, tracking a sampled twist.

    ``robot`` is ``g1`` or ``go1``. Each uses that robot's flat locomotion config.
    """

    def __init__(
        self,
        num_envs: int,
        device: str,
        *,
        robot: str = "g1",
        seed: int = 0,
        history: int | None = None,
        task=None,
    ) -> None:
        from mjlab.envs import ManagerBasedRlEnv

        from supermimic.tasks import make_task

        self.robot_name = robot
        self.task = task if task is not None else make_task("go1_flat" if robot == "go1" else "g1_flat")
        cfg = _flat_env_cfg(robot)
        cfg.scene.num_envs = int(num_envs)
        cfg.seed = int(seed)
        # The caller snapshots the post-step state before the reset. mjlab's
        # auto-reset would replace that state with the next episode.
        cfg.auto_reset = False
        self.env = ManagerBasedRlEnv(cfg, device=device)
        self.num_envs = int(num_envs)
        self.device = device
        self.robot = self.env.scene["robot"]
        obs, _ = self.env.reset(seed=int(seed))
        actor = obs["actor"]
        if not isinstance(actor, torch.Tensor):
            raise TypeError(f"Flat {robot} actor observations are expected to be one tensor.")
        self._obs = actor
        self.obs_dim = int(actor.shape[-1])
        self.act_dim = int(self.env.action_manager.total_action_dim)
        dev = actor.device
        ring = int(history or self.env.max_episode_length) + int(self.env.max_episode_length)
        self.goals = CommandGoals(self.num_envs, ring, device)
        pose_params = self.env.cfg.rewards["pose"].params
        from supermimic.utils.kinematics import TorchKinematics

        _model, joint_names = self.task.compile_model()
        self.kin = TorchKinematics(_model, joint_names, body_names=self.task.tracked_bodies)
        with torch.no_grad():
            zeros = torch.zeros(1, 3, device=self.device)
            joints = torch.zeros(1, self.task.num_joints, device=self.device)
            quat0 = torch.zeros(1, 4, device=self.device)
            quat0[:, 0] = 1
            self.kin.forward(zeros, quat0, zeros, zeros, joints, joints)
        self.mdp = VelocityMDP(
            self.task,
            self.goals,
            self.robot,
            {name: float(term.weight) for name, term in self.env.cfg.rewards.items()},
            _pose_term(self.env),
            float(pose_params.get("walking_threshold", 0.05)),
            float(pose_params.get("running_threshold", 1.5)),
        )
        self.mdp.kin = self.kin
        self.term_names = self.mdp.term_names
        self.r_max = self.mdp.max_step_reward
        scale = self.env.action_manager.get_term("joint_pos").scale
        if isinstance(scale, torch.Tensor):
            self.action_scale = scale[0].detach().clone()
        else:
            self.action_scale = torch.full((self.act_dim,), float(scale), device=dev)
        self.default_q = self.mdp.default_q
        self.ep_step = torch.zeros(self.num_envs, dtype=torch.long, device=dev)
        self.lifetime = torch.zeros(self.num_envs, dtype=torch.long, device=dev)
        self.ep_id = torch.zeros(self.num_envs, dtype=torch.long, device=dev)
        self._ep_counter = 0
        self.a_prev = torch.zeros(self.num_envs, self.act_dim, device=dev)
        self.xi = torch.zeros(self.num_envs, self.task.xi_dim, device=dev)
        self.ep_ret = torch.zeros(self.num_envs, device=dev)
        self.ep_term = {name: torch.zeros(self.num_envs, device=dev) for name in self.term_names}
        names = list(self.robot.body_names)
        self._tracked_local = torch.tensor(
            [names.index(name) for name in self.task.tracked_bodies],
            dtype=torch.long,
            device=dev,
        )
        self._publish()

    @property
    def f(self) -> torch.Tensor:
        return self.goals.frame()

    def actor_obs(self) -> torch.Tensor:
        return self._obs

    def state(self):
        data = self.robot.data
        pose = data.root_link_pose_w
        origin = self.env.scene.env_origins
        p = pose[:, :3] - origin
        quat = pose[:, 3:7]
        vel = data.root_link_vel_w
        return p, quat, vel[:, :3], vel[:, 3:6], data.joint_pos, data.joint_vel

    def tracked_bodies(self):
        idx = self._tracked_local
        pose = self.robot.data.body_link_pose_w[:, idx]
        origin = self.env.scene.env_origins
        pos = pose[..., :3] - origin[:, None, :]
        quat = pose[..., 3:7]
        vel = self.robot.data.body_link_vel_w[:, idx]
        return pos, quat, vel[..., :3], vel[..., 3:6]

    def step(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        from supermimic.mdp import WorldModelState, flatten_state

        p, quat, v, w, q, qd = self.state()
        s = flatten_state(p, quat, v, w, q, qd)
        body_pos, body_quat, _, _ = self.tracked_bodies()
        a_prev = self.a_prev
        frame = self.f
        ep = self.ep_id.clone()
        step_id = self.lifetime.clone()
        _obs, _reward, terminated, timeout, _extras = self.env.step(action)
        p1, quat1, v1, w1, q1, qd1 = self.state()
        body1 = self.tracked_bodies()
        contact = p.new_zeros(self.num_envs, self.task.n_contact)
        reward_state = WorldModelState(p1, quat1, v1, w1, q1, qd1, body1[0], body1[1])
        reward, terms = self.mdp.reward(reward_state, frame, action, a_prev, terms=True)
        terminated = terminated.bool().reshape(self.num_envs)
        truncated = timeout.bool().reshape(self.num_envs)
        done = done_from(terminated, truncated)
        self.goals.advance()
        self.ep_step += 1
        self.lifetime += 1
        self.ep_ret = self.ep_ret + reward
        for name, value in terms.items():
            self.ep_term[name] = self.ep_term[name] + value
        out = {
            "s": s,
            "s1": flatten_state(p1, quat1, v1, w1, q1, qd1),
            "body_pos": body_pos.detach(),
            "body_quat": body_quat.detach(),
            "a": action.detach(),
            "a_prev": a_prev.detach().clone(),
            "xi": self.xi.clone(),
            "c": contact,
            "c1": contact,
            "f": frame,
            "ep": ep,
            "step": step_id,
            "r": reward,
            "terms": terms,
            "terminated": terminated,
            "truncated": truncated,
            "done": done,
            "ep_ret": self.ep_ret.clone(),
            "ep_term": {name: value.clone() for name, value in self.ep_term.items()},
            "ep_len": self.ep_step.clone(),
        }
        self.a_prev = action.detach().clone()
        if bool(done.any()):
            ids = done.nonzero(as_tuple=False).flatten()
            self.env.reset(env_ids=ids)
            self._ep_counter += 1
            self.ep_id[ids] = self._ep_counter
            self.ep_step[ids] = 0
            self.a_prev[ids] = 0
            self.ep_ret[ids] = 0
            for value in self.ep_term.values():
                value[ids] = 0
        actor = self.env.obs_buf["actor"]
        if not isinstance(actor, torch.Tensor):
            raise TypeError(f"Flat {self.robot_name} actor observations are expected to be one tensor.")
        self._obs = actor
        self._publish()
        return out

    def _command(self) -> torch.Tensor:
        command = self.env.command_manager.get_term("twist").command
        return command[:, :3]

    def _publish(self) -> None:
        env_ids = torch.arange(self.num_envs, device=self.device)
        remaining = (self.env.max_episode_length - self.ep_step).clamp(min=1)
        self.goals.fill(env_ids, self._command(), remaining)
