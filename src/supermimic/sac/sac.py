"""Standalone SAC trainer (sac_rlpd without RLPD, symmetry, tensordict, or DDP)."""

from __future__ import annotations

import copy
import math
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import einops
import swanlab
import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

from supermimic.mdp import (
    CONTROL_DT,
    GOAL_DIM,
    GOAL_FRAMES,
    INV_DIM,
    NUM_J,
    PRIV_DIM,
    REWARD_TERM_NAMES,
    REWARD_TERM_WEIGHT,
    XI_DIM,
    invariant_features,
    normalize_xi,
    resolve_obs_terms,
)
from supermimic.mdp import WorldModelState
from supermimic.sac.network import AlphaModule, MultiStepReturn, NormalActor, make_critic
from supermimic.sac.replay_buffer import ReplayBuffer
from supermimic.sac.util import (
    IndependentNormal,
    ObsVecNorm,
    autocast,
    gaussian_target_entropy,
    hold_out_net,
    soft_copy_,
    diagonal_gaussian_entropy,
)
from supermimic.supermimic.network import InputNorms
from supermimic.supermimic.reward_norm import RewardNormalizer

CRITIC_FEAT_DIM = INV_DIM + GOAL_DIM + XI_DIM + PRIV_DIM


@contextmanager
def timed(device: str):
    """Wall time of a block, after the GPU has finished the work inside it."""
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        yield
    finally:
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        timed.last = time.perf_counter() - start


timed.last = 0.0


def _mean(xs: list) -> float:
    """Host mean. CUDA tensors are reduced on-device and synced once here (log time)."""
    if not xs:
        return 0.0
    if isinstance(xs[0], torch.Tensor):
        return float(torch.stack([x.detach().reshape(()) for x in xs]).mean())
    return sum(xs) / len(xs)


def _term_score(
    name: str, value: torch.Tensor | float, style: str = "gaussian"
) -> torch.Tensor | float:
    """Normalize a reward term for logging (same scaling as SuperMimic ``_log``)."""
    if name in ("action_rate", "joint_limit"):
        return value / CONTROL_DT
    return value / (REWARD_TERM_WEIGHT[style][name] * CONTROL_DT)


def actor_lr_at(train_op: int, base: float, warmup_ops: int) -> float:
    """Linear actor LR warmup over the first ``warmup_ops`` :meth:`SACAgent.train_op` calls."""
    if warmup_ops <= 0:
        return base
    step = train_op + 1
    if step < warmup_ops:
        return base * step / warmup_ops
    return base


@dataclass
class TrainConfig:
    num_envs: int = 2048
    iterations: int = 320_000
    buffer_steps: int = 1000
    utd: float = 0.4

    actor_tau: float = 0.02
    critic_tau: float = 0.02

    critic_kind: str = "c51"
    critic_ensemble: int = 2
    critic_atoms: int = 51
    critic_v_min: float = -2.0
    critic_v_max: float = 15.0
    critic_sigma_ratio: float = 0.75
    normalize_reward: bool = True

    gamma: float = 0.99
    n_steps: int = 3
    lr: float = 5e-4
    lr_alpha: float = 5e-4
    optim: str = "muon"
    weight_decay: float = 0.02
    warm_up_steps: int = 200
    actor_lr_warmup_ops: int = 100
    train_every: int = 4
    utd_ratio: int = 4
    critic_batch_size: int = 2048
    actor_batch_size: int = 2048
    alpha_init: float = 4e-3
    target_entropy_sigma: float | None = 0.15
    entropy_bonus: float = 1.0
    target_action_noise: float = 0.01
    max_grad_norm: float = 1.0
    soft_bound: float = 3.0 * math.pi
    actor_init: str = "zeros"
    vecnorm: bool = True
    use_amp: bool = True

    anchor: str = "pelvis"
    obs_terms: tuple[str, ...] = ()
    reward_style: str = "gaussian"
    device: str = "cuda:0"
    seed: int = 0
    log_every: int = 50
    ckpt_every: int = 1000
    # Polyak EMA decay β: θ_ema ← β θ_ema + (1-β) θ_actor after each actor step.
    pi_ema: float | None = 0.995
    swanlab_project: str = "rethinking-supermimic"
    swanlab_mode: str = "online"
    swanlab_experiment: str = "lafan1"

    # note that we do not need RLPD and symmetry in this version
    # we also do not need distributed training


class SACAgent:
    """Soft actor-critic update logic on dict batches."""

    def __init__(
        self,
        cfg: TrainConfig,
        policy_obs_dim: int,
        critic_dim: int,
        act_dim: int,
        device: torch.device,
        *,
        r_max: float,
    ):
        self.cfg = cfg
        self.device = device
        self.policy_obs_dim = policy_obs_dim
        self.critic_dim = critic_dim
        self.act_dim = act_dim
        self.entropy_scale = 1.0 / math.sqrt(act_dim)

        self.vecnorm = ObsVecNorm(policy_obs_dim).to(device) if cfg.vecnorm else None
        g_max = r_max / (1.0 - cfg.gamma)
        if cfg.critic_kind == "classification" and cfg.normalize_reward:
            v_min, v_max = -g_max, g_max
        else:
            v_min, v_max = cfg.critic_v_min, cfg.critic_v_max

        self.Q = make_critic(
            cfg.critic_kind,
            critic_dim,
            act_dim,
            ensemble=cfg.critic_ensemble,
            num_atoms=cfg.critic_atoms,
            v_min=v_min,
            v_max=v_max,
            sigma_ratio=cfg.critic_sigma_ratio,
        ).to(device)
        self.actor = NormalActor(
            policy_obs_dim,
            act_dim,
            std_max=1.0,
            std_min=0.001,
            action_init=cfg.actor_init,
        ).to(device)
        self.Q_target = copy.deepcopy(self.Q).to(device)
        self.actor_target = copy.deepcopy(self.actor).to(device)
        ema_tau = self._actor_ema_tau()
        if ema_tau is not None:
            self.actor_ema = copy.deepcopy(self.actor).to(device)
            self.actor_ema.requires_grad_(False)
            self.actor_ema.eval()
        else:
            self.actor_ema = None
        for mod in (self.Q_target, self.actor_target):
            mod.requires_grad_(False)
            mod.eval()

        if cfg.target_entropy_sigma is not None:
            self.target_entropy = gaussian_target_entropy(act_dim, cfg.target_entropy_sigma)
        else:
            # Match E[-log pi] scale used historically when tuning alpha without sigma.
            self.target_entropy = 0.0
        self.alpha = AlphaModule(cfg.alpha_init).to(device)

        from supermimic.optim import build_optimizer

        self.opt_actor = build_optimizer(self.actor, cfg.lr, cfg.optim)
        self.opt_Q = build_optimizer(self.Q, cfg.lr, cfg.optim)
        self.opt_alpha = torch.optim.Adam(self.alpha.parameters(), lr=cfg.lr_alpha)

        self.msr = MultiStepReturn(cfg.gamma, cfg.n_steps).to(device) if cfg.n_steps > 1 else None
        self.reward_norm = (
            RewardNormalizer(cfg.num_envs, cfg.gamma, g_max=g_max, device=device)
            if cfg.normalize_reward
            else None
        )

        self.global_step = 0
        self._train_op_idx = 0
        self._amp = bool(cfg.use_amp and device.type == "cuda")
        self._scaler_q = GradScaler(device.type, enabled=self._amp)
        self._scaler_pi = GradScaler(device.type, enabled=self._amp)

    def _actor_ema_tau(self) -> float | None:
        beta = self.cfg.pi_ema
        if beta is None:
            return None
        if not 0.0 < beta < 1.0:
            raise ValueError(f"pi_ema must be in (0, 1), got {beta!r}")
        return 1.0 - beta

    def _norm_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if self.vecnorm is None:
            return obs
        return self.vecnorm(obs)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, *, deterministic: bool = False) -> torch.Tensor:
        obs_n = self._norm_obs(obs)
        loc, scale = self.actor(obs_n)
        if deterministic:
            return loc
        return IndependentNormal(loc, scale).sample()

    def _apply_actor_lr(self) -> float:
        lr = actor_lr_at(self._train_op_idx, self.cfg.lr, self.cfg.actor_lr_warmup_ops)
        for group in self.opt_actor.param_groups:
            group["lr"] = lr
        return lr

    def observe_transition(self, batch: dict[str, torch.Tensor], rb: ReplayBuffer) -> dict:
        if self.vecnorm is not None:
            self.vecnorm.update(batch["obs"])
        if self.reward_norm is not None:
            self.reward_norm.update(batch["reward"].reshape(-1), batch["done"].reshape(-1))
        rb.push(batch)
        self.global_step += 1
        if self.global_step <= self.cfg.warm_up_steps:
            return {}
        if self.global_step % self.cfg.train_every != 0:
            return {}
        return self.train_op(rb)

    def train_op(self, rb: ReplayBuffer) -> dict:
        actor_lr = self._apply_actor_lr()
        infos: dict = {"rb_size": len(rb), "actor/lr": actor_lr}
        critic_iters = self.cfg.train_every * self.cfg.utd_ratio
        last_info: dict = {}
        for i in range(critic_iters):
            batch = rb.sample(
                self.cfg.critic_batch_size,
                steps=self.cfg.n_steps,
                next_obs=True,
            )
            last_info = self.train_critic(batch, diagnostics=i == critic_iters - 1)
        infos.update(last_info)
        for j in range(self.cfg.train_every):
            batch = rb.sample(self.cfg.actor_batch_size, steps=1, next_obs=False)
            last_info = self.train_actor(batch, diagnostics=j == self.cfg.train_every - 1)
        self._train_op_idx += 1

        infos.update(last_info)
        return infos

    def _prepare_critic_batch(self, batch: dict[str, torch.Tensor]) -> tuple:
        reward = batch["reward"]
        if self.reward_norm is not None:
            if reward.dim() == 3:
                flat = reward.reshape(-1, reward.shape[-1])
                flat = self.reward_norm.normalize(flat)
                reward = flat.reshape(reward.shape)
            else:
                reward = self.reward_norm.normalize(reward)
        else:
            reward = reward * (1.0 - self.cfg.gamma)

        obs_raw = batch["critic"]
        next_q_raw = batch["next_critic"]
        next_policy_raw = batch["next_obs"]
        if self.cfg.n_steps == 1:
            q = obs_raw
            next_q = next_q_raw
            next_policy = next_policy_raw
            act = batch["action"]
            term = batch["terminated"].float()
            discount = self.cfg.gamma * (1.0 - term)
            is_init = batch["is_init"]
            terminated = batch["terminated"].bool().squeeze(-1)
        else:
            assert self.msr is not None
            rewards_msr = reward
            act_n, next_q, reward, discount, terminated = self.msr(
                actions=batch["action"],
                next_observations=next_q_raw,
                rewards=rewards_msr,
                terminated=batch["terminated"],
                done=batch["done"],
                env_discount=None,
            )
            _, next_policy, _, _, _ = self.msr(
                actions=batch["action"],
                next_observations=next_policy_raw,
                rewards=rewards_msr,
                terminated=batch["terminated"],
                done=batch["done"],
                env_discount=None,
            )
            q = obs_raw[0]
            act = act_n[:, 0]
            is_init = batch["is_init"][0]
        return q, act, next_q, next_policy, reward, discount, terminated, is_init

    def train_critic(self, batch: dict[str, torch.Tensor], *, diagnostics: bool) -> dict:
        self.Q.train()
        q, act, next_q, next_policy, reward, discount, terminated, is_init = self._prepare_critic_batch(batch)
        with autocast(self.device, self._amp):
            q_target = self._compute_target(next_q, next_policy, reward, discount)
            pred = self.Q(q, act)
            per_loss = self.Q.compute_loss(pred, q_target)
            valid = (1.0 - is_init.float()).reshape_as(per_loss)
            q_loss = (per_loss * valid).sum() / valid.sum().clamp_min(1e-8)

        self.opt_Q.zero_grad(set_to_none=True)
        if self._amp:
            self._scaler_q.scale(q_loss).backward()
            self._scaler_q.unscale_(self.opt_Q)
            gn = nn.utils.clip_grad_norm_(self.Q.parameters(), self.cfg.max_grad_norm)
            self._scaler_q.step(self.opt_Q)
            self._scaler_q.update()
        else:
            q_loss.backward()
            gn = nn.utils.clip_grad_norm_(self.Q.parameters(), self.cfg.max_grad_norm)
            self.opt_Q.step()
        soft_copy_(self.Q, self.Q_target, self.cfg.critic_tau)

        if not diagnostics:
            return {"critic/q_loss": q_loss.detach().item()}
        with torch.no_grad():
            q_pred = self.Q.get_values(q, act).min(dim=-1).values
            q_val = q_pred.mean().item()
            out: dict[str, float] = {
                "critic/q_loss": q_loss.item(),
                "critic/grad_norm": float(gn),
                "critic/q_value": q_val,
            }
            term_mask = terminated.reshape(q_pred.shape[0]).bool()
            if bool(term_mask.any().item()):
                out["critic/q_value_terminated"] = q_pred[term_mask].mean().item()
                term_loss = per_loss.reshape(q_pred.shape[0], -1).mean(dim=-1)
                out["critic/q_loss_terminated"] = term_loss[term_mask].mean().item()
            return out

    @torch.no_grad()
    def _compute_target(
        self,
        next_q: torch.Tensor,
        next_policy_obs: torch.Tensor,
        reward: torch.Tensor,
        discount: torch.Tensor,
    ) -> torch.Tensor:
        loc, scale = self.actor_target(self._norm_obs(next_policy_obs))
        dist = IndependentNormal(loc, scale)
        next_action = dist.sample()
        next_log_prob = dist.log_prob(next_action).unsqueeze(-1)
        alpha = self.alpha()
        entropy_bonus = (-alpha * next_log_prob) * self.entropy_scale
        adjusted = reward + discount * self.cfg.entropy_bonus * entropy_bonus
        noise = torch.randn_like(next_action).clamp(-3.0, 3.0)
        return self.Q_target.compute_target(
            next_q,
            next_action + noise * self.cfg.target_action_noise,
            adjusted,
            discount,
        )

    def train_actor(self, batch: dict[str, torch.Tensor], *, diagnostics: bool) -> dict:
        policy_obs = self._norm_obs(batch["obs"])
        q_state = batch["critic"]
        is_init = batch["is_init"]
        with hold_out_net(self.Q), autocast(self.device, self._amp):
            loc, scale = self.actor(policy_obs)
            dist = IndependentNormal(loc, scale)
            action_update = dist.rsample((4,))
            q_heads = self.Q.get_values(
                q_state,
                einops.rearrange(action_update, "k n d -> n k d"),
            )
            q = q_heads.mean(dim=-1)
            policy_term = -q.mean(dim=1)

        # Differential entropy (sum over action dims). Do not use -log_prob(samples): E[-log p] = H + d/2.
        entropy_est = diagonal_gaussian_entropy(scale.float())
        alpha = self.alpha()
        actor_loss = policy_term + alpha.detach() * (
            -entropy_est.reshape_as(policy_term) * self.entropy_scale
        ) + 0.01 * ((loc / self.cfg.soft_bound) ** 6).sum(-1).reshape_as(policy_term)
        valid = (1.0 - is_init.float()).reshape_as(actor_loss)
        actor_loss = (actor_loss * valid).sum() / valid.sum().clamp_min(1.0)

        alpha_loss = -(alpha * (-entropy_est.detach() + self.target_entropy)).mean()
        self.opt_alpha.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.opt_alpha.step()

        self.opt_actor.zero_grad(set_to_none=True)
        if self._amp:
            self._scaler_pi.scale(actor_loss).backward()
            self._scaler_pi.unscale_(self.opt_actor)
            gn_pi = nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
            self._scaler_pi.step(self.opt_actor)
            self._scaler_pi.update()
        else:
            actor_loss.backward()
            gn_pi = nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
            self.opt_actor.step()
        soft_copy_(self.actor, self.actor_target, self.cfg.actor_tau)
        tau = self._actor_ema_tau()
        if tau is not None and self.actor_ema is not None:
            soft_copy_(self.actor, self.actor_ema, tau)

        if not diagnostics:
            return {"actor/loss": actor_loss.detach().item()}
        return {
            "actor/loss": actor_loss.item(),
            "actor/grad_norm": float(gn_pi),
            "actor/alpha": alpha.detach().item(),
            "actor/entropy": entropy_est.mean().item(),
            "actor/mean_scale": scale.detach().float().mean().item(),
        }


class Trainer:
    """Environment loop + :class:`SACAgent` (tracking sim, dict transitions)."""

    def __init__(self, cfg: TrainConfig, log_dir: Path | str):
        from supermimic.kinematics import compile_g1, make_kinematics
        from supermimic.motion import MotionLib
        from supermimic.sim import TrackingSim

        self.cfg = cfg
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(cfg.device)
        torch.manual_seed(cfg.seed)

        self.motion = MotionLib.load(device=str(self.device))
        model, names = compile_g1()
        self.kin = make_kinematics(model, names, "torch", str(self.device))
        obs_terms = cfg.obs_terms or resolve_obs_terms(())
        self.sim = TrackingSim(
            cfg.num_envs,
            self.motion,
            self.kin,
            str(self.device),
            gamma=cfg.gamma,
            anchor=cfg.anchor,
            obs_terms=obs_terms,
            reward_style=cfg.reward_style,
            survival_bonus=1.0,
        )
        self.r_max = self.sim.r_max
        self.policy_obs_dim = self.sim.mdp.obs_dim + GOAL_DIM
        assert CRITIC_FEAT_DIM == INV_DIM + GOAL_DIM + XI_DIM + PRIV_DIM
        self.norms = InputNorms(self.sim.mdp.obs_dim).to(self.device)
        self.agent = SACAgent(
            cfg,
            self.policy_obs_dim,
            CRITIC_FEAT_DIM,
            NUM_J,
            self.device,
            r_max=self.r_max,
        )
        self.buffer = ReplayBuffer(
            cfg.buffer_steps,
            cfg.num_envs,
            self.policy_obs_dim,
            CRITIC_FEAT_DIM,
            NUM_J,
            self.device,
        )
        self._step_acc = 0.0
        self.stats = {
            "r": [],
            "ep": [],
            "term": [],
            "done_n": [],
            "sim_terms": {name: [] for name in REWARD_TERM_NAMES},
            "ep_terms": {name: [] for name in REWARD_TERM_NAMES},
        }
        self.train_agg: dict[str, list] = defaultdict(list)
        self.timing = {"collect": []}
        self.started = 0.0
        self.logged = 0.0
        self._log_f = None
        self._tb: SummaryWriter | None = None

    def _record_sim_stats(self, out: dict[str, torch.Tensor]) -> None:
        style = self.cfg.reward_style
        self.stats["r"].append((out["r"].mean() / self.r_max).detach())
        for name, value in out["terms"].items():
            self.stats["sim_terms"][name].append(_term_score(name, value.mean().detach(), style))
        done = out["done"]
        if bool(done.any().item()):
            length = out["ep_len"][done].float().clamp(min=1)
            self.stats["ep"].append((out["ep_ret"][done] / (self.r_max * length)).mean().detach())
            for name, value in out["ep_term"].items():
                self.stats["ep_terms"][name].append(
                    _term_score(name, (value[done] / length).mean().detach(), style)
                )
            self.stats["term"].append(float(out["d"][done].float().sum()))
            self.stats["done_n"].append(float(done.sum()))

    def _accum_train_info(self, info: dict) -> None:
        for key, value in info.items():
            if isinstance(value, (int, float)):
                self.train_agg[key].append(float(value))

    def _policy_features(self) -> torch.Tensor:
        p, quat, v, w, q, qd = self.sim.state()
        bodies = self.sim.tracked_bodies()
        state = WorldModelState(p, quat, v, w, q, qd, bodies[0], bodies[1])
        obs = self.sim.mdp.add_noise(self.sim.mdp.obs(state, self.sim.a_prev, self.sim.xi, self.sim.f))
        goal = self.motion.goal(self.sim.f, GOAL_FRAMES)
        return torch.cat([obs, goal], dim=-1)

    def _critic_features(self) -> torch.Tensor:
        p, quat, v, w, q, qd = self.sim.state()
        inv = invariant_features(p, quat, v, w, q, qd)
        goal = self.motion.goal(self.sim.f, GOAL_FRAMES)
        xi_n = normalize_xi(self.sim.xi)
        priv = self.sim.mdp.privileged(p, quat, v, w, q, qd, self.sim.f)
        return self.norms.critic(inv, goal, xi_n, priv)

    def _update_norms(self) -> None:
        p, quat, v, w, q, qd = self.sim.state()
        inv = invariant_features(p, quat, v, w, q, qd)
        bodies = self.sim.tracked_bodies()
        state = WorldModelState(p, quat, v, w, q, qd, bodies[0], bodies[1])
        obs = self.sim.mdp.obs(state, self.sim.a_prev, self.sim.xi, self.sim.f)
        goal = self.motion.goal(self.sim.f, GOAL_FRAMES)
        xi_n = normalize_xi(self.sim.xi)
        priv = self.sim.mdp.privileged(p, quat, v, w, q, qd, self.sim.f)
        self.norms.state.update(inv)
        self.norms.goal.update(goal)
        self.norms.xi.update(xi_n)
        self.norms.priv.update(priv)
        self.norms.obs.update(obs)

    def _collect_step(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        self._update_norms()
        policy_obs = self._policy_features()
        critic = self._critic_features()
        fresh = (self.sim.ep_step == 0).float().unsqueeze(-1)
        with torch.no_grad():
            action = self.agent.act(policy_obs)
        out = self.sim.step(action)
        transition = {
            "obs": policy_obs,
            "critic": critic,
            "action": action,
            "reward": out["r"].unsqueeze(-1),
            "terminated": out["d"].unsqueeze(-1).float(),
            "done": out["done"].unsqueeze(-1).float(),
            "is_init": fresh,
        }
        return transition, out

    def _collect(self, n_steps: int) -> None:
        for _ in range(n_steps):
            transition, out = self._collect_step()
            self._record_sim_stats(out)
            info = self.agent.observe_transition(transition, self.buffer)
            self._accum_train_info(info)

    def _log(self, iteration: int) -> None:
        cfg = self.cfg
        now = time.time()
        term = sum(self.stats["term"]) / max(sum(self.stats["done_n"]), 1.0)
        seconds = {key: _mean(self.timing[key]) for key in self.timing}
        sim_score = _mean(self.stats["r"])
        ep_score = _mean(self.stats["ep"])
        sim_terms = {name: _mean(self.stats["sim_terms"][name]) for name in REWARD_TERM_NAMES}
        ep_terms = {name: _mean(self.stats["ep_terms"][name]) for name in REWARD_TERM_NAMES}

        def train_metric(key: str, default: float = 0.0) -> float:
            return _mean(self.train_agg[key]) if self.train_agg[key] else default

        critic_loss = train_metric("critic/q_loss")
        actor_loss = train_metric("actor/loss")
        q_value = train_metric("critic/q_value")
        q_term = train_metric("critic/q_value_terminated", float("nan"))
        alpha = train_metric("actor/alpha")
        entropy = train_metric("actor/entropy")
        gn_critic = train_metric("critic/grad_norm")
        gn_actor = train_metric("actor/grad_norm")
        rb_size = train_metric("rb_size", float(len(self.buffer)))

        lr = train_metric("actor/lr", float(cfg.lr))
        it_per_s = cfg.log_every / (now - self.logged)

        row = [
            iteration,
            (now - self.started) / 3600,
            sim_score,
            ep_score,
            term,
            rb_size,
            critic_loss,
            actor_loss,
            q_value,
            q_term,
            alpha,
            entropy,
            gn_critic,
            gn_actor,
            it_per_s,
            lr,
            seconds["collect"],
        ]
        for name in REWARD_TERM_NAMES:
            row.append(sim_terms[name])
        for name in REWARD_TERM_NAMES:
            row.append(ep_terms[name])

        assert self._log_f is not None
        self._log_f.write(",".join(f"{x:.6g}" for x in row) + "\n")
        self._log_f.flush()

        metrics = {
            "train/sim_score": sim_score,
            "train/ep_score": ep_score,
            "train/term_frac": term,
            "train/rb_size": rb_size,
            "train/critic_loss": critic_loss,
            "train/actor_loss": actor_loss,
            "train/q_value": q_value,
            "train/alpha": alpha,
            "train/entropy": entropy,
            "train/it_per_s": it_per_s,
            "train/lr": lr,
            "grad/critic": gn_critic,
            "grad/actor": gn_actor,
            "time/collect_s": seconds["collect"],
        }
        for name in REWARD_TERM_NAMES:
            metrics[f"score/sim_{name}"] = sim_terms[name]
            metrics[f"score/ep_{name}"] = ep_terms[name]
        if math.isfinite(q_term):
            metrics["train/q_value_terminated"] = q_term
        if self.agent.reward_norm is not None:
            metrics["train/reward_scale"] = float(self.agent.reward_norm.scale())

        swanlab.log(metrics, step=iteration)
        if self._tb is not None:
            for key, value in metrics.items():
                if math.isfinite(value):
                    self._tb.add_scalar(key, value, iteration)
            self._tb.flush()

        track = " ".join(
            f"{n[0]}{sim_terms[n]:.2f}"
            for n in ("body_pos", "body_ori", "body_lin_vel", "body_ang_vel", "root_pos", "root_ori")
        )
        q_term_s = f" Q_term {q_term:.3g}" if math.isfinite(q_term) else ""
        print(
            f"it {iteration} | {row[1]:.3f} h | sim {sim_score:.3f} ep {ep_score:.3f} term {term:.2f} | "
            f"[{track}] rate {sim_terms['action_rate']:.3g} lim {sim_terms['joint_limit']:.3g} | "
            f"critic {critic_loss:.4f} actor {actor_loss:.3f} Q {q_value:.3f}{q_term_s} α {alpha:.4g} H {entropy:.3f} | "
            f"rb {rb_size:.0f} {it_per_s:.2f} it/s lr {lr:.2e} | "
            f"gn Q {gn_critic:.3g} π {gn_actor:.3g} | collect {seconds['collect']:.3f}s",
            flush=True,
        )

        self.logged = now
        self.stats = {
            "r": [],
            "ep": [],
            "term": [],
            "done_n": [],
            "sim_terms": {name: [] for name in REWARD_TERM_NAMES},
            "ep_terms": {name: [] for name in REWARD_TERM_NAMES},
        }
        self.train_agg = defaultdict(list)
        self.timing = {"collect": []}

    def save_checkpoint(self, iteration: int) -> None:
        if self.agent.actor_ema is not None:
            actor_state = self.agent.actor_ema.state_dict()
            actor_live_state = self.agent.actor.state_dict()
        else:
            actor_state = self.agent.actor.state_dict()
            actor_live_state = None
        ckpt = {
            "actor": actor_state,
            "Q": self.agent.Q.state_dict(),
            "Q_target": self.agent.Q_target.state_dict(),
            "actor_target": self.agent.actor_target.state_dict(),
            "alpha": self.agent.alpha.state_dict(),
            "iteration": iteration,
            "global_step": self.agent.global_step,
            "cfg": self.cfg.__dict__,
        }
        if actor_live_state is not None:
            ckpt["actor_live"] = actor_live_state
        if self.agent.vecnorm is not None:
            ckpt["vecnorm"] = self.agent.vecnorm.state_dict()
        if self.agent.reward_norm is not None:
            ckpt["reward_norm"] = self.agent.reward_norm.state_dict()
        ckpt["norms"] = self.norms.state_dict()
        path = self.log_dir / f"ckpt_{iteration:06d}.pt"
        torch.save(ckpt, path)
        torch.save(ckpt, self.log_dir / "last.pt")

    def fit(self) -> None:
        cfg = self.cfg
        logged_cfg = {
            key: list(value) if isinstance(value, tuple) else value for key, value in cfg.__dict__.items()
        }
        swanlab.init(
            project=cfg.swanlab_project,
            name=cfg.swanlab_experiment or None,
            config={
                **logged_cfg,
                "algo": "sac",
                "policy_obs_dim": self.policy_obs_dim,
                "critic_feat_dim": CRITIC_FEAT_DIM,
                "obs_dim": self.sim.mdp.obs_dim,
                "anchor_body": self.kin.anchor_name,
            },
            mode=cfg.swanlab_mode,
        )
        tb_dir = self.log_dir / "tb"
        self._tb = SummaryWriter(log_dir=tb_dir)
        print(f"tensorboard={tb_dir}", flush=True)
        term_cols = ",".join(f"sim_{n}" for n in REWARD_TERM_NAMES)
        ep_cols = ",".join(f"ep_{n}" for n in REWARD_TERM_NAMES)
        self._log_f = (self.log_dir / "train_log.csv").open("w")
        self._log_f.write(
            "iteration,wall_h,sim_score,ep_score,term_frac,rb_size,critic_loss,actor_loss,q_value,q_value_terminated,alpha,entropy,"
            f"grad_critic,grad_actor,it_per_s,lr,t_collect,{term_cols},{ep_cols}\n"
        )
        self.started = time.time()
        self.logged = self.started
        try:
            for iteration in range(cfg.iterations):
                self._step_acc += 1.0 / cfg.utd
                n_collect = int(self._step_acc)
                self._step_acc -= n_collect
                if n_collect < 1:
                    n_collect = 1
                with timed(str(self.device)):
                    self._collect(n_collect)
                self.timing["collect"].append(timed.last)

                if (iteration + 1) % cfg.log_every == 0:
                    self._log(iteration + 1)
                if (iteration + 1) % cfg.ckpt_every == 0 or iteration + 1 == cfg.iterations:
                    self.save_checkpoint(iteration + 1)
        finally:
            swanlab.finish()
            if self._tb is not None:
                self._tb.close()
            if self._log_f is not None:
                self._log_f.close()


def run(cfg: TrainConfig, log_dir: Path | str) -> None:
    Trainer(cfg, log_dir).fit()
