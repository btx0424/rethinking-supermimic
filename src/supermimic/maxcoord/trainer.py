"""Independent MaxCoord trainer: simulator replay, WM fitting, actor BPTT, critic."""
from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from supermimic.mdp import normalize_xi
from supermimic.motion import CONTROL_DT
from supermimic.sim import DECIMATION, TrackingSim
from supermimic.tasks import make_task_from_config
from supermimic.supermimic.buffer import ReplayBuffer
from supermimic.supermimic.network import ScalarCritic, critic_forward
from supermimic.supermimic.reward_norm import RewardNormalizer
from supermimic.utils.kinematics import make_kinematics
from supermimic.utils.nn import hold_out_net
from supermimic.utils.optim import build_optimizer
from .interface import WMBatch
from .timing import IterationTimer
from .normalization import build_vecnorms, bind_vecnorms
from .mdp import configure_sim, full_from_flat, goal, reward_state
from .models import actor_features, build_models
from .state import build_maxcoord_full_state
from .logging import ExperimentLogger, grouped_metrics


@dataclass
class MaxCoordConfig:
    algo: str = "supermimic_maxcoord"
    wm_kind: str = "explicit"
    actor_kind: str = "explicit"
    latent_dim: int = 256
    wm_xi: bool = True
    simnorm_dim: int = 8
    collect_steps: int = 4
    wm_hidden: tuple[int, ...] = (512, 512, 512)
    actor_hidden: tuple[int, ...] = (512, 512, 512)
    consistency_weight: float = 1.0
    goal_frames: int = 10
    num_envs: int = 4096
    iterations: int = 40000
    buffer_steps: int = 1000
    utd: float = 0.4  # Legacy shared Hydra field; only default accepted, unused.
    wm_window: int = 8
    wm_batches: int = 10
    wm_batch: int = 2048
    pi_batch: int = 4096
    horizon: int = 32
    pi_warmup: int = 50
    wm_lr: float = 2e-4
    pi_lr: float = 5e-4
    v_lr: float = 5e-4
    gamma: float = 0.99
    lam: float = 0.95
    action_noise: float = 0.1
    grad_clip_norm: float = 1.0
    critic_ensemble: int = 2
    critic_xi: bool = True
    input_normalization: str = "vecnorm"
    vecnorm_decay: float = 0.9999
    critic_epochs: int = 5
    critic_minibatches: int = 4
    normalize_reward: bool = True
    optim: str = "muon"
    compile: bool = False
    compile_mdp: bool = False
    compile_updates: bool = False
    cuda_graphs: bool = False
    anchor: str = "pelvis"
    reward_style: str = "gaussian"
    obs_terms: tuple[str, ...] = ("joint_pos", "joint_vel", "base_ang_vel", "projected_gravity")
    fk: str = "torch"
    device: str = "cuda"
    seed: int = 0
    log_every: int = 50
    ckpt_every: int = 500
    save_optim: bool = True
    resume: str | None = None
    log_backend: str = "swanlab"
    wandb_project: str = "rethinking-supermimic"
    wandb_mode: str = "online"
    wandb_experiment: str = "maxcoord"
    swanlab_project: str = "rethinking-supermimic"
    swanlab_mode: str = "online"
    swanlab_experiment: str = "maxcoord"


def optimize(loss, module, optimizer, max_norm):
    torch._assert_async(torch.isfinite(loss), "Non-finite learner loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm, error_if_nonfinite=False)
    torch._assert_async(torch.isfinite(norm), "Non-finite learner gradient norm")
    optimizer.step()
    return norm.detach()


class Trainer:
    def __init__(self, cfg: MaxCoordConfig, task_cfg, log_dir):
        self.cfg = cfg
        self.task_cfg = task_cfg
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if cfg.buffer_steps < cfg.wm_window:
            raise ValueError("buffer_steps must cover wm_window")
        if cfg.utd != 0.4:
            raise ValueError("MaxCoord uses collect_steps; utd is unused and must retain its legacy default 0.4")
        for key in ("iterations", "goal_frames", "wm_window", "wm_batches", "wm_batch", "pi_batch", "horizon", "num_envs",
                    "critic_epochs", "critic_minibatches", "log_every", "ckpt_every", "collect_steps"):
            if getattr(cfg, key) < 1:
                raise ValueError(f"{key} must be positive")
        if not 0 <= cfg.gamma < 1 or not 0 <= cfg.lam <= 1:
            raise ValueError("Require 0 <= gamma < 1 and 0 <= lam <= 1")
        if cfg.consistency_weight < 0 or cfg.grad_clip_norm <= 0:
            raise ValueError("consistency_weight must be nonnegative and grad_clip_norm positive")
        if cfg.input_normalization not in ("vecnorm", "legacy"):
            raise ValueError("input_normalization must be vecnorm or legacy")
        if cfg.anchor != "pelvis":
            raise ValueError("MaxCoord goal uses the pelvis anchor")
        torch.manual_seed(cfg.seed)
        self.task = make_task_from_config(task_cfg)
        if self.task.name != "lafan1" or self.task.has_object:
            raise ValueError("MaxCoord currently supports LAFAN1 G1 only")
        motion = self.task.load_motion(cfg.device)
        model, names = self.task.compile_model()
        kin = make_kinematics(model, names, cfg.fk, cfg.device)
        self.sim = TrackingSim(cfg.num_envs, motion, kin, cfg.device, gamma=cfg.gamma,
                               anchor=cfg.anchor, obs_terms=cfg.obs_terms,
                               reward_style=cfg.reward_style, task=self.task)
        configure_sim(self.sim)
        self.mdp = self.sim.mdp
        self.wm, self.actor = build_models(cfg, self.task.n_body, self.task.num_joints,
                                           DECIMATION, CONTROL_DT)
        self.wm.to(cfg.device)
        self.actor.to(cfg.device)
        input_dim = self.actor.input_dim
        self.critic = ScalarCritic(input_dim + (self.task.num_joints + 4 if cfg.critic_xi else 0), ensemble=cfg.critic_ensemble).to(cfg.device)
        self.vecnorms = None
        if cfg.input_normalization == "vecnorm":
            self.vecnorms = build_vecnorms(cfg, self.wm.representation.rep_state_dim, self.task.num_joints).to(cfg.device)
            bind_vecnorms(self.wm, self.actor, self.vecnorms)
        self.opt_wm = build_optimizer(self.wm, cfg.wm_lr, cfg.optim)
        self.opt_pi = build_optimizer(self.actor, cfg.pi_lr, cfg.optim, adam_betas=(0.9, 0.95))
        self.opt_v = build_optimizer(self.critic, cfg.v_lr, "adam")
        self.buffer = ReplayBuffer(cfg.buffer_steps, cfg.num_envs, cfg.device, self.task)
        self.reward_norm = RewardNormalizer(cfg.num_envs, cfg.gamma,
                                            self.mdp.max_step_reward / (1 - cfg.gamma), cfg.device)
        # Compile only the differentiable transition. Keeping target construction,
        # loss and detached diagnostics outside avoids mixed-grad output aliasing
        # observed with whole-loss Inductor compilation on Torch 2.11.
        self.wm_step = torch.compile(self.wm.forward) if cfg.compile else self.wm
        self.reward_fn = torch.compile(self.mdp.reward) if cfg.compile_mdp else self.mdp.reward
        self.start_iteration = 0
        if cfg.resume:
            self.load(cfg.resume)
        self.accelerated = None
        if cfg.compile_updates or cfg.cuda_graphs:
            from .accelerated import AcceleratedUpdates

            self.accelerated = AcceleratedUpdates(self)

    @torch.no_grad()
    def collect(self, steps):
        rewards, terminations, lengths = [], [], []
        reward_terms = {name: [] for name in self.mdp.term_names}
        for _ in range(steps):
            full = build_maxcoord_full_state(*self.sim.tracked_bodies())
            if self.vecnorms is not None:
                for name, values in self.wm.observation_network_inputs(full).items():
                    self.vecnorms[name].update(values)
            state = self.wm.initialize(full, normalize_xi(self.sim.xi))
            features = actor_features(state, self.cfg.actor_kind)
            target = goal(self.sim.motion, self.sim.f, state.aux[..., 3:7], self.cfg.goal_frames)
            if self.vecnorms is not None:
                self.vecnorms["actor"].update(self.actor.inputs(features, target))
                self.vecnorms["critic"].update(self.raw_critic_inputs(state, target))
            action = self.actor(features, target)
            action = action + self.cfg.action_noise * torch.randn_like(action)
            if self.vecnorms is not None:
                for name, values in self.wm.transition_network_inputs(state, action).items():
                    self.vecnorms[name].update(values)
            out = self.sim.step(action)
            self.buffer.add(out)
            self.reward_norm.update(out["r"], out["done"])
            rewards.append(out["r"].mean())
            for name in reward_terms:
                reward_terms[name].append(out["terms"][name].mean())
            terminations.append(out["terminated"].float().mean())
            lengths.extend(out["ep_len"][out["done"]].unbind())
        return {**{f"sim_term_{k}": torch.stack(v).mean() / CONTROL_DT for k, v in reward_terms.items()},
                "sim_reward": torch.stack(rewards).mean() / CONTROL_DT,
                "sim_terminated": torch.stack(terminations).mean(),
                "episode_length": torch.stack(lengths).float().mean() if lengths else rewards[0].new_zeros(())}

    def fit_world_model(self):
        if self.accelerated is not None:
            return self.accelerated.fit_wm()
        cfg = self.cfg
        records = []
        for _ in range(cfg.wm_batches):
            batch = self.buffer.sample_windows(cfg.wm_batch, cfg.wm_window)
            with torch.no_grad():
                full = full_from_flat(self.mdp, batch["s"])
                successors = full_from_flat(self.mdp, batch["s1"])
                if self.vecnorms is None:
                    self.wm.update_normalization(full, batch["a"])
            result = self.wm.compute_loss(WMBatch(full[:, 0], batch["a"], successors,
                                                normalize_xi(batch["xi"][:, 0])),
                                          cfg.consistency_weight, step=self.wm_step)
            loss, stats = result.loss, {k: v.detach() for k, v in result.metrics.items()}
            norm = optimize(loss, self.wm, self.opt_wm, cfg.grad_clip_norm)
            records.append({**stats, "wm_loss": loss.detach(), "grad_wm": norm})
        return {key: torch.stack([r[key] for r in records]).mean() for key in records[0]}

    def raw_critic_inputs(self, state, target):
        inputs = self.actor.inputs(actor_features(state, self.cfg.actor_kind), target)
        if self.cfg.critic_xi:
            if state.xi_n is None or state.xi_n.shape[-1] != self.task.num_joints + 4:
                raise ValueError("Critic requires normalized simulator context")
            inputs = torch.cat((inputs, state.xi_n), -1)
        return inputs

    def critic_inputs(self, state, target):
        inputs = self.raw_critic_inputs(state, target)
        return self.vecnorms["critic"](inputs) if self.vecnorms is not None else inputs

    def train_actor_critic(self, iteration):
        if self.accelerated is not None:
            return self.accelerated.fit_policy(iteration)
        cfg = self.cfg
        batch = self.buffer.sample_starts(cfg.pi_batch, 1)
        with torch.no_grad():
            state = self.wm.initialize(full_from_flat(self.mdp, batch["s"]), normalize_xi(batch["xi"]))
            g = goal(self.sim.motion, batch["f"], state.aux[..., 3:7], cfg.goal_frames)
            if self.vecnorms is None:
                self.actor.state_norm.update(actor_features(state, cfg.actor_kind))
                self.actor.goal_norm.update(g)
        inputs, rewards, terminated = [], [], []
        reward_terms = {name: [] for name in self.mdp.term_names}
        frame, previous = batch["f"], batch["a_prev"]
        returns = torch.zeros(cfg.pi_batch, device=cfg.device)
        alive = torch.ones_like(returns)
        scale = self.reward_norm.scale() if cfg.normalize_reward else 1.0
        # WM/critic weights are held fixed; derivatives through state and action remain.
        with hold_out_net(self.wm), hold_out_net(self.critic):
            for t in range(cfg.horizon):
                features = actor_features(state, cfg.actor_kind)
                g = goal(self.sim.motion, frame, state.aux[..., 3:7], cfg.goal_frames)
                inputs.append(self.critic_inputs(state, g).detach())
                action = self.actor(features, g)
                state = self.wm_step(state, action)
                frame = frame + 1
                rs = reward_state(self.wm.representation, state, self.task.num_joints)
                raw_reward, parts = self.reward_fn(rs, frame, action, previous, terms=True)
                reward = raw_reward / scale
                for name in reward_terms:
                    reward_terms[name].append(parts[name].detach().mean())
                term = self.mdp.soft_termination(rs.p, rs.quat, frame)
                returns = returns + cfg.gamma**t * alive * reward
                alive = alive * (1 - term)
                rewards.append(reward.detach())
                terminated.append(term.detach())
                previous = action
            last_g = goal(self.sim.motion, frame, state.aux[..., 3:7], cfg.goal_frames)
            last_inputs = self.critic_inputs(state, last_g)
            inputs.append(last_inputs.detach())
            value = self.critic.expected_values(critic_forward(self.critic, last_inputs)).mean(-1)
            pi_loss = -(returns + cfg.gamma**cfg.horizon * alive * value).mean()
            lr_scale = min(1.0, iteration / max(cfg.pi_warmup, 1)) if cfg.pi_warmup else 1.0
            for group in self.opt_pi.param_groups:
                group["lr"] = cfg.pi_lr * lr_scale
            grad_pi = optimize(pi_loss, self.actor, self.opt_pi, cfg.grad_clip_norm)

        features = torch.stack(inputs)
        reward, term = torch.stack(rewards), torch.stack(terminated)
        with torch.no_grad():
            values = self.critic.expected_values(critic_forward(self.critic, features.flatten(0, 1)))
            values = values.reshape(cfg.horizon + 1, cfg.pi_batch, cfg.critic_ensemble)
            target = values[-1]
            targets = []
            for t in reversed(range(cfg.horizon)):
                target = reward[t, :, None] + cfg.gamma * (1 - term[t, :, None]) * (
                    (1 - cfg.lam) * values[t + 1] + cfg.lam * target)
                targets.append(target)
            targets = torch.stack(targets[::-1]).flatten(0, 1)
        x = features[:-1].flatten(0, 1)
        losses, critic_norms = [], []
        for _ in range(cfg.critic_epochs):
            order = torch.randperm(x.shape[0], device=x.device)
            for idx in order.tensor_split(min(cfg.critic_minibatches, x.shape[0])):
                loss = self.critic.compute_loss(critic_forward(self.critic, x[idx]), targets[idx])
                critic_norms.append(optimize(loss, self.critic, self.opt_v, cfg.grad_clip_norm))
                losses.append(loss.detach())
        return {**{f"imag_term_{k}": torch.stack(v).mean() / CONTROL_DT for k, v in reward_terms.items()},
                "imag_raw_reward": reward.mean() * scale / CONTROL_DT, "grad_v": torch.stack(critic_norms).mean(),
                "pi_loss": pi_loss.detach(), "v_loss": torch.stack(losses).mean(),
                "grad_pi": grad_pi, "imag_reward": reward.mean() / CONTROL_DT, "imag_survival": alive.detach().mean()}

    def save(self, iteration):
        saved = {"format": "maxcoord_v4", "cfg": asdict(self.cfg),
                 "task_cfg": OmegaConf.to_container(self.task_cfg, resolve=True),
                 "iteration": iteration, "wm": self.wm.state_dict(), "pi": self.actor.state_dict(),
                 "critic": self.critic.state_dict(), "reward_norm": self.reward_norm.state_dict()}
        if self.vecnorms is not None:
            saved["vecnorms"] = self.vecnorms.state_dict()
        if self.cfg.save_optim:
            saved.update(opt_wm=self.opt_wm.state_dict(), opt_pi=self.opt_pi.state_dict(),
                         opt_v=self.opt_v.state_dict())
        path = self.log_dir / f"ckpt_{iteration:06d}.pt"
        torch.save(saved, path)
        temporary = self.log_dir / "last.pt.tmp"
        torch.save(saved, temporary)
        temporary.replace(self.log_dir / "last.pt")

    def load(self, path):
        saved = torch.load(path, map_location=self.cfg.device, weights_only=False)
        if saved.get("format") not in ("maxcoord_v2", "maxcoord_v3", "maxcoord_v4"):
            raise ValueError("Resume requires maxcoord_v2/v3/v4; legacy v1 remains supported for eval/play only. Start a fresh run for the new protocol.")
        if saved["cfg"].get("input_normalization", "legacy") != self.cfg.input_normalization:
            raise ValueError("Resume input_normalization differs: legacy checkpoints require input_normalization=legacy")
        if self.vecnorms is not None:
            if saved["cfg"]["vecnorm_decay"] != self.cfg.vecnorm_decay:
                raise ValueError("Resume vecnorm_decay differs")
            self.vecnorms.load_state_dict(saved["vecnorms"])
        if saved["cfg"].get("critic_xi", False) != self.cfg.critic_xi:
            raise ValueError("Resume critic_xi differs: legacy v2 needs critic_xi=false; start fresh to add context")
        for key in ("wm_kind", "actor_kind", "goal_frames", "latent_dim", "wm_xi", "simnorm_dim", "collect_steps"):
            if saved["cfg"][key] != getattr(self.cfg, key):
                raise ValueError(f"Resume configuration differs: {key}")
        for key, module in (("wm", self.wm), ("pi", self.actor), ("critic", self.critic)):
            module.load_state_dict(saved[key])
        for key, opt in (("opt_wm", self.opt_wm), ("opt_pi", self.opt_pi), ("opt_v", self.opt_v)):
            if key in saved:
                opt.load_state_dict(saved[key])
        self.reward_norm.load_state_dict(saved["reward_norm"])
        self.start_iteration = saved["iteration"]
        print("Resumed weights/optimizers; collecting fresh replay (simulator state is not restored)", flush=True)

    def fit(self):
        cfg = self.cfg
        backend = cfg.log_backend
        logger = ExperimentLogger(backend, getattr(cfg, f"{backend}_mode"), getattr(cfg, f"{backend}_project"),
                                  getattr(cfg, f"{backend}_experiment"), asdict(cfg), self.log_dir)
        started = time.monotonic()
        try:
            with SummaryWriter(str(self.log_dir / "tb")) as tb, (self.log_dir / "train_log.csv").open("a") as output, (self.log_dir / "performance.csv").open("a") as perf_output:
                writer = None
                perf_writer = csv.DictWriter(perf_output, fieldnames=["iteration", "collect_time", "train_time", "compute_time"])
                if perf_output.tell() == 0:
                    perf_writer.writeheader()
                timer = IterationTimer(cfg.device)
                # Enough transitions for eight-action windows, even with short episodes.
                self.collect(max(cfg.wm_window * 2, 32))
                for iteration in range(self.start_iteration, cfg.iterations):
                    steps = cfg.collect_steps
                    timer.start()
                    stats = self.collect(steps)
                    timer.collected()
                    self.buffer.prepare(cfg.wm_window, cfg.horizon, self.mdp.episode_last(self.buffer.f))
                    stats.update(self.fit_world_model())
                    stats.update(self.train_actor_critic(iteration))
                    timing = timer.finish()
                    perf_row = {"iteration": iteration + 1, **timing}
                    perf_writer.writerow(perf_row)
                    perf_output.flush()
                    payload = grouped_metrics(perf_row)
                    if (iteration + 1) % cfg.log_every == 0 or iteration == self.start_iteration:
                        row = {"iteration": iteration + 1, "seconds": time.monotonic() - started,
                               **{key: float(value) for key, value in stats.items()}}
                        if writer is None:
                            writer = csv.DictWriter(output, fieldnames=list(row))
                            if output.tell() == 0:
                                writer.writeheader()
                        writer.writerow(row)
                        output.flush()
                        payload.update(grouped_metrics(row))
                        print(json.dumps(row), flush=True)
                    for key, value in payload.items():
                        tb.add_scalar(key, value, iteration + 1)
                    logger.log(payload, step=iteration + 1)
                    if (iteration + 1) % cfg.ckpt_every == 0:
                        self.save(iteration + 1)
                self.save(cfg.iterations)
        finally:
            logger.finish()


def run(cfg, task_cfg, log_dir):
    Trainer(cfg, task_cfg, log_dir).fit()
