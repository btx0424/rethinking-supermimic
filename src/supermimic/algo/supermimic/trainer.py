"""One tracking algo, two interchangeable WMs, one explicit actor."""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import torch

from supermimic.algo.supermimic.policy.actor import Actor
from supermimic.algo.supermimic.policy.critic import Critic
from supermimic.algo.supermimic.wm import MaxCoordRepresentation, build_world_model
from supermimic.algo.supermimic.wm.env import WorldModelEnv
from supermimic.dataset.motion import MotionDataset
from supermimic.env.mdp import CONTROL_DT
from supermimic.env.mdp.observation import concatenate, observation_width
from supermimic.env.mdp.reference import TrackingReference
from supermimic.env.mdp.tracking import TrackingTask
from supermimic.env.rollout.policy import rollout
from supermimic.env.sim.env import SimulationEnv
from supermimic.nn.optim import build_optimizer
from supermimic.utils.logging import Logger
from supermimic.utils.timing import IterationTimer

from . import checkpoint
from .normalization import build_vecnorms
from .replay import ReplayBuffer
from .reward_norm import RewardNormalizer
from .updates import Updates


class Trainer:
    def __init__(self, cfg):
        cfg.validate()
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        name = (
            f"{cfg.wm_kind}-{datetime.now(ZoneInfo('Asia/Taipei')):%Y%m%d-%H%M%S}-{uuid4().hex[:6]}"
        )
        self.log_dir = Path(cfg.run_dir or f"logs/{name}").resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if (self.log_dir / "train_log.csv").exists():
            raise FileExistsError("Use a new run_dir; training logs are never overwritten")
        (self.log_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
        self.motion = TrackingReference(
            MotionDataset(cfg.task.motion, cfg.device), cfg.task.tracking_bodies
        )
        rep = MaxCoordRepresentation(len(self.motion.body_names)).to(cfg.device)
        self.task = TrackingTask(self.motion, rep, cfg.task)
        self.real = SimulationEnv(self.task, cfg.num_envs, cfg.device, cfg.gamma)
        self.wm = build_world_model(cfg, rep, self.task.num_joints).to(cfg.device)
        actor_dim = observation_width(
            cfg.task.actor_observations,
            rep.rep_state_dim,
            cfg.task.goal_frames,
            self.task.num_joints,
        )
        critic_dim = observation_width(
            cfg.task.critic_observations,
            rep.rep_state_dim,
            cfg.task.goal_frames,
            self.task.num_joints,
        )
        self.actor = Actor(actor_dim, self.task.num_joints, cfg.actor_hidden).to(cfg.device)
        self.critic = Critic(critic_dim, cfg.critic_heads).to(cfg.device)
        self.vecnorms = build_vecnorms(cfg, rep.rep_state_dim, self.task.num_joints).to(cfg.device)
        self.wm.input_transform = lambda name, x: self.vecnorms[name](x)
        self.imagined = WorldModelEnv(self.wm, self.task)
        self.opt_wm = build_optimizer(self.wm, cfg.wm_lr, cfg.optim)
        self.opt_pi = build_optimizer(self.actor, cfg.pi_lr, cfg.optim, adam_betas=(0.9, 0.95))
        self.opt_v = build_optimizer(self.critic, cfg.v_lr, cfg.critic_optim)
        self.buffer = ReplayBuffer(
            cfg.buffer_steps, cfg.num_envs, rep.full_state_dim, self.task.num_joints, cfg.device
        )
        self.reward_norm = RewardNormalizer(
            cfg.num_envs, cfg.gamma, self.task.max_step_reward / (1 - cfg.gamma), cfg.device
        )
        self.start_iteration = checkpoint.resume(self, cfg.resume) if cfg.resume else 0
        self.updates = Updates(self)
        self.collection_state = self.real.reset()

    def actor_input(self, observation):
        return concatenate(observation, self.cfg.task.actor_observations)

    def raw_critic_input(self, observation):
        return concatenate(observation, self.cfg.task.critic_observations)

    def critic_input(self, observation):
        return self.vecnorms["critic"](self.raw_critic_input(observation))

    def policy(self, observation):
        return self.actor(self.vecnorms["actor"](self.actor_input(observation)))

    def collect_policy(self, observation):
        if observation.full is None:
            raise ValueError("Collect requires physical simulator samples")
        for key, values in self.wm.observation_network_inputs(observation.full).items():
            self.vecnorms[key].update(values)
        wm_state = self.wm.initialize(observation.full, observation.xi_n)
        self.vecnorms["actor"].update(self.actor_input(observation))
        self.vecnorms["critic"].update(self.raw_critic_input(observation))
        action = self.policy(observation)
        action = action + self.cfg.action_noise * torch.randn_like(action)
        for key, values in self.wm.transition_network_inputs(wm_state, action).items():
            self.vecnorms[key].update(values)
        return action

    @torch.no_grad()
    def collect(self, steps):
        rewards = []
        terms = {key: [] for key in self.task.term_names}
        terminations = []
        lengths = []
        for record in rollout(self.real, self.collection_state, self.collect_policy, steps):
            self.buffer.add(record)
            result = record.transition
            done = result.terminated | result.truncated
            self.reward_norm.update(result.reward, done)
            rewards.append(result.reward.mean())
            terminations.append(result.terminated.float().mean())
            lengths.extend(result.info["episode_length"][done].unbind())
            for key, values in terms.items():
                values.append(result.terms[key].mean())
            self.collection_state = result.continuation
        return {
            "SimReward/total": torch.stack(rewards).mean() / CONTROL_DT,
            "SimReward/terminated_fraction": torch.stack(terminations).mean(),
            "SimReward/episode_length": torch.stack(lengths).float().mean()
            if lengths
            else rewards[0].new_zeros(()),
            **{f"SimReward/{k}": torch.stack(v).mean() / CONTROL_DT for k, v in terms.items()},
        }

    def fit(self):
        cfg = self.cfg
        logger = Logger(cfg.wandb, self.log_dir, asdict(cfg))
        timer = IterationTimer(cfg.device)
        try:
            self.collect(max(2 * cfg.wm_window, 32))
            for iteration in range(self.start_iteration, cfg.iterations):
                timer.start()
                metrics = self.collect(cfg.collect_steps)
                timer.collected()
                self.buffer.prepare(cfg.wm_window, cfg.horizon, self.motion)
                metrics.update(self.updates.fit_world_model())
                metrics.update(self.updates.fit_policy(iteration))
                timing = timer.finish()
                emit = (iteration + 1) % cfg.log_every == 0 or iteration == self.start_iteration
                logger.log(iteration + 1, timing, metrics if emit else None)
                if (iteration + 1) % cfg.ckpt_every == 0:
                    checkpoint.save(self, iteration + 1)
            checkpoint.save(self, cfg.iterations)
        finally:
            logger.close()
