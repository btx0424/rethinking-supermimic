"""ScoreMimic policy improvement on top of SuperMimic's world-model stack.

The actor is a diagonal Gaussian. A deterministic mean stays on a steep reward,
so each update explores, proposes a nearby improvement, and re-scores it:

1. From each start, sample ``score_samples`` action sequences from π and roll
   them out in the world model. Record the return of each sequence. The
   standard deviation is fixed at ``score_std``.
2. Differentiate each sequence's imagined return with respect to its own
   actions. Step the sampled action by NablaMimic's RMS setpoint ``score_delta``
   along that gradient.
3. Roll the world model out again on those target sequences and record their
   returns. Both sets are kept, so a step that does not raise the return is
   down-weighted rather than fitted.
4. Fit the mean by advantage-weighted log-likelihood of all 2K sequences. The
   weight of an action is ``exp(A / τ)`` with A the per-step λ-advantage, and
   the exponent is clipped. A KL penalty keeps the mean near the policy that
   produced the candidates. ``σ`` is not trained.

The critic is SuperMimic's λ-return update on every sequence. Low-return
rollouts are not removed from the value loss. It is not on the action-gradient
tape: the target step is the world-model return only, and the value is read
under ``no_grad``.

The imagination batch is ``score_samples * pi_batch`` (default ``pi_batch`` is
1024). Both rollouts are concatenated and consumed whole; nothing is chunked
further.
"""

from __future__ import annotations

import math
from dataclasses import field
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import DictConfig

from supermimic.algos.network import (
    InputNorms,
    Policy,
    WorldModel,
    WorldModelState,
    critic_forward,
)
from supermimic.mdp import (
    GOAL_FRAMES,
    HIST,
    invariant_features,
    normalize_xi,
    object_features,
    unflatten_object,
    unflatten_state,
)
from supermimic.utils.nn import soft_copy_
from supermimic.utils.optim import build_optimizer
from supermimic.supermimic.nablamimic import nabla_wm_step
from supermimic.supermimic.supermimic import (
    Trainer,
    _clip_grads,
    _flatten_object_time,
    _flatten_time,
    _init_wm_state,
    _stack,
    _term_score,
    _unflatten_time,
    standalone_config,
    timed,
)


ScoreTrainConfig = standalone_config(
    "ScoreTrainConfig",
    "scoremimic",
    overrides={
        "stochastic_actor": True,
        "pi_lr": 6e-4,
        "pi_batch": 1024,
        # Action gradient is the world-model return. The critic stays off that tape.
        "policy_bootstrap": False,
    },
    extra=[
        ("score_samples", int, field(default=1)),
        ("score_std", float, field(default=0.1)),
        # Same RMS action step as NablaMimic (``action_target_delta``).
        ("score_delta", float, field(default=0.03)),
        ("score_delta_floor", float, field(default=0.3)),
        ("score_act_clip", float, field(default=3.0)),
        ("score_temperature", float, field(default=0.1)),
        ("score_adv_clip", float, field(default=5.0)),
        ("score_kl", float, field(default=0.1)),
        ("score_steps", int, field(default=4)),
        ("score_lr", float, field(default=6e-4)),
    ],
)


def _repeat_start(start: dict[str, torch.Tensor], k: int) -> dict[str, torch.Tensor]:
    """Repeat each start ``k`` times along the batch axis."""
    if k == 1:
        return start
    return {key: value.repeat_interleave(k, dim=0) for key, value in start.items()}


def _cat_batch(a: torch.Tensor, b: torch.Tensor, steps: int) -> torch.Tensor:
    """Concatenate two time-major batches along the environment axis."""
    ax = a.view(steps, -1, *a.shape[1:])
    bx = b.view(steps, -1, *b.shape[1:])
    return torch.cat((ax, bx), dim=1).reshape(-1, *a.shape[1:])


def _nabla_target(
    action: torch.Tensor,
    grad: torch.Tensor,
    action_scale: torch.Tensor,
    delta: float,
    cot_clip: float,
) -> torch.Tensor:
    """``action + η ∂J/∂a`` with NablaMimic's median clip and RMS step ``delta``."""
    g = torch.where(torch.isfinite(grad), grad, torch.zeros_like(grad))
    nrm = g.norm(dim=-1)
    live = nrm > 0
    if cot_clip is not None and bool(live.any()):
        med = nrm[live].median()
        thr = cot_clip * med
        scale = torch.clamp(thr / (nrm + 1e-30), max=1.0)
        scale = torch.where(live, scale, torch.ones_like(scale))
        g = g * scale.unsqueeze(-1)
    live_f = live.to(g.dtype)
    ss = (g.square() * action_scale.square() * live_f.unsqueeze(-1)).sum()
    cnt = live_f.sum().clamp_min(1.0) * float(g.shape[-1])
    eta = delta / (ss / cnt).clamp_min(1e-30).sqrt()
    return action + eta * g


def _gaussian_log_prob(mean: torch.Tensor, log_std: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Diagonal-Gaussian log density, summed over the action."""
    z = (action - mean) * (-log_std).exp()
    return -0.5 * (z.square() + 2.0 * log_std + math.log(2.0 * math.pi)).sum(dim=-1)


def _gaussian_kl(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    mean_old: torch.Tensor,
    log_std_old: torch.Tensor,
) -> torch.Tensor:
    """KL(π_new || π_old) for two diagonal Gaussians, summed over the action."""
    var_ratio = (log_std - log_std_old).exp().square()
    deviation = ((mean - mean_old) * (-log_std_old).exp()).square()
    return (log_std_old - log_std + 0.5 * (var_ratio + deviation - 1.0)).sum(dim=-1)


def score_horizon(
    policy: Policy,
    wm: WorldModel,
    critic: nn.Module,
    norms: InputNorms,
    mdp,
    hist_s: torch.Tensor,
    hist_a_prev: torch.Tensor,
    hist_xi: torch.Tensor,
    hist_f: torch.Tensor,
    hist_body_pos: torch.Tensor,
    hist_body_quat: torch.Tensor,
    s: torch.Tensor,
    a_prev: torch.Tensor,
    xi: torch.Tensor,
    frame: torch.Tensor,
    contact: torch.Tensor,
    body_pos: torch.Tensor,
    body_quat: torch.Tensor,
    reward_scale: torch.Tensor,
    horizon: int,
    gamma: float,
    wm_step=nabla_wm_step,
    bootstrap_grad: bool = True,
    hist_obj: torch.Tensor | None = None,
    obj: torch.Tensor | None = None,
    actions_in: torch.Tensor | None = None,
) -> dict:
    """H-step world-model rollout.

    With ``actions_in is None`` the stochastic policy samples every action and
    each sample is re-wrapped as a free leaf, so ``autograd.grad`` yields
    ∂J/∂a through the world model only. Otherwise ``actions_in`` is
    ``(H, B, A)`` and is applied open-loop.
    """
    batch = s.shape[0]
    device = s.device
    if actions_in is not None and actions_in.shape[:2] != (horizon, batch):
        raise RuntimeError(
            f"Target actions must be {(horizon, batch, 'A')}, got {tuple(actions_in.shape)}."
        )
    term_names = mdp.term_names
    p, quat, v, w, q, qd = unflatten_state(hist_s)
    hist_n = p.shape[0] * p.shape[1]
    hist_obj_fields = (None, None, None, None)
    if hist_obj is not None:
        op, oq, ov, ow = unflatten_object(hist_obj)
        hist_obj_fields = (
            op.reshape(hist_n, 3),
            oq.reshape(hist_n, 4),
            ov.reshape(hist_n, 3),
            ow.reshape(hist_n, 3),
        )
    hist_state = WorldModelState(
        p.reshape(hist_n, 3),
        quat.reshape(hist_n, 4),
        v.reshape(hist_n, 3),
        w.reshape(hist_n, 3),
        q.reshape(hist_n, -1),
        qd.reshape(hist_n, -1),
        hist_body_pos.reshape(hist_n, hist_body_pos.shape[-2], 3),
        hist_body_quat.reshape(hist_n, hist_body_quat.shape[-2], 4),
        *hist_obj_fields,
    )
    obs = mdp.obs(
        hist_state,
        hist_a_prev.reshape(hist_n, -1),
        hist_xi.reshape(hist_n, -1),
        hist_f.reshape(hist_n),
    )
    obs_hist = mdp.add_noise(obs).view(batch, HIST, -1).detach()
    state = _init_wm_state(s, body_pos, body_quat, obj)
    xi_n = normalize_xi(xi)
    goal = mdp.goal(frame, GOAL_FRAMES)
    states = [state]
    frames = [frame]
    goals = [goal]
    actions: list[torch.Tensor] = []
    prev_actions: list[torch.Tensor] = []
    obs_ins: list[torch.Tensor] = []
    goal_ins: list[torch.Tensor] = []
    carried = contact
    for t in range(horizon):
        obs_in = obs_hist.detach()
        goal_in = goal.detach()
        obs_ins.append(obs_in)
        goal_ins.append(goal_in)
        prev_actions.append(a_prev)
        if actions_in is None:
            with torch.no_grad():
                sampled = policy(obs_in, goal_in, norms, deterministic=False)
            action = sampled.detach().requires_grad_(True)
        else:
            action = actions_in[t]
        state, frame, goal, obs, carried = wm_step(
            wm, norms, mdp, state, action, xi_n, xi, frame, carried
        )
        obs_hist = torch.cat((obs_in[:, 1:], obs.detach().unsqueeze(1)), dim=1)
        actions.append(action)
        a_prev = action
        states.append(state)
        frames.append(frame)
        goals.append(goal)

    flat = _flatten_time(states[1:])
    object_state = _flatten_object_time(states[1:]) if mdp.task.has_object else None
    obj_fields = object_state if object_state is not None else (None, None, None, None)
    obj_p_prev = obj_quat_prev = None
    if mdp.task.has_object:
        prev_object = _flatten_object_time(states[:-1])
        obj_p_prev, obj_quat_prev = prev_object[0], prev_object[1]
    reward_state = WorldModelState(
        *flat, None, None, *obj_fields, None, None, None, obj_p_prev, obj_quat_prev
    )
    reward, term_parts = mdp.reward(
        reward_state,
        _stack(frames[1:]),
        _stack(actions),
        _stack(prev_actions),
        terms=True,
        surrogate=True,
    )
    terminated = mdp.soft_termination(flat[0], flat[1], _stack(frames[1:]))
    reward = reward.view(horizon, batch)
    terminated = terminated.view(horizon, batch)
    term_raw = torch.stack([term_parts[name] for name in term_names]).view(
        len(term_names), horizon, batch
    )
    term_mean = term_raw.detach().mean(dim=(1, 2))
    reward_mean = reward.detach().mean()
    reward = reward / reward_scale
    alive = torch.cumprod(1.0 - terminated, dim=0)
    survival = torch.cat((torch.ones(1, batch, device=device), alive[:-1]), dim=0)
    discount = gamma ** torch.arange(horizon, device=device, dtype=torch.float32)
    weighted = discount[:, None] * survival
    step_return = weighted * reward
    returns = step_return.sum(0)
    last = states[-1]
    last_obj = last.object() if mdp.task.has_object else None
    last_obj_feat = object_features(*last.root()[:2], *last_obj) if last_obj is not None else None
    head_values = critic.expected_values(
        critic_forward(
            critic,
            norms.critic(
                invariant_features(*last.root()),
                goals[-1],
                xi_n,
                mdp.privileged(*last.root(), frames[-1], object_state=last_obj),
                last_obj_feat,
            ),
        )
    )
    bootstrap = gamma**horizon * alive[-1] * head_values.mean(-1)
    if not bootstrap_grad:
        bootstrap = bootstrap.detach()
    policy_loss = -(returns + bootstrap).mean()
    critic_states = tuple(t.detach() for t in _flatten_time(states))
    critic_objects = (
        tuple(t.detach() for t in _flatten_object_time(states)) if mdp.task.has_object else None
    )
    return {
        "policy_loss": policy_loss,
        "reward": reward.detach(),
        "terminated": terminated.detach(),
        "alive_last": alive[-1].detach(),
        "reward_mean": reward_mean.detach(),
        "term_mean": term_mean,
        "critic_states": critic_states,
        "critic_objects": critic_objects,
        "frames": _stack(frames).detach(),
        "goals": _stack(goals).detach(),
        "xi": xi.detach(),
        "actions": actions,
        "obs_ins": torch.stack(obs_ins),
        "goal_ins": torch.stack(goal_ins),
    }


class ScoreTrainer(Trainer):
    """SuperMimic trainer with stochastic, advantage-weighted action targets."""

    cfg: ScoreTrainConfig

    def __init__(self, cfg: ScoreTrainConfig, task_cfg: DictConfig, log_dir: Path | str):
        if cfg.score_samples < 1:
            raise ValueError(f"score_samples must be >= 1, got {cfg.score_samples}.")
        if cfg.score_std <= 0:
            raise ValueError(f"score_std must be positive, got {cfg.score_std}.")
        if cfg.score_delta <= 0:
            raise ValueError(f"score_delta must be positive, got {cfg.score_delta}.")
        if cfg.score_temperature <= 0:
            raise ValueError(f"score_temperature must be positive, got {cfg.score_temperature}.")
        cfg.stochastic_actor = True
        cfg.pi_lr = cfg.score_lr
        super().__init__(cfg, task_cfg, log_dir)
        if self.policy.log_std is None:
            raise RuntimeError("ScoreMimic requires a stochastic policy.")
        # Fixed exploration noise. The likelihood fits the mean only.
        log_std = math.log(float(cfg.score_std))
        with torch.no_grad():
            self.policy.log_std.fill_(log_std)
            if self.policy_ema is not None:
                self.policy_ema.log_std.fill_(log_std)
        self.policy.log_std.requires_grad_(False)
        self.opt_pi = build_optimizer(
            self.policy, cfg.score_lr, cfg.optim, adam_betas=(0.9, 0.95)
        )
        self._score_wm_step = (
            torch.compile(nabla_wm_step, mode="default", dynamic=False)
            if cfg.compile
            else nabla_wm_step
        )

    def _rollout(self, start: dict[str, torch.Tensor], actions_in: torch.Tensor | None) -> dict:
        cfg = self.cfg
        scale = (
            self.reward_norm.scale().detach()
            if self.reward_norm is not None
            else start["s"].new_ones(())
        )
        return score_horizon(
            self.policy,
            self.policy_wm(),
            self.critic,
            self.norms,
            self.sim.mdp,
            start["hist_s"],
            start["hist_a_prev"],
            start["hist_xi"],
            start["hist_f"],
            start["hist_body_pos"],
            start["hist_body_quat"],
            start["s"],
            start["a_prev"],
            start["xi"],
            start["f"],
            start["c"],
            start["body_pos"],
            start["body_quat"],
            scale,
            self.horizon_pi,
            cfg.gamma,
            self._score_wm_step,
            cfg.policy_bootstrap,
            hist_obj=start.get("hist_obj"),
            obj=start.get("obj"),
            actions_in=actions_in,
        )

    def _critic_batch(self, sampled: dict, rescored: dict) -> dict:
        """Both rollouts, concatenated along the environment axis."""
        steps = self.horizon_pi + 1
        objects = None
        if sampled["critic_objects"] is not None and rescored["critic_objects"] is not None:
            objects = tuple(
                _cat_batch(a, b, steps)
                for a, b in zip(sampled["critic_objects"], rescored["critic_objects"])
            )
        xi = torch.cat((sampled["xi"], rescored["xi"]), dim=0)
        return {
            "states": tuple(
                _cat_batch(a, b, steps)
                for a, b in zip(sampled["critic_states"], rescored["critic_states"])
            ),
            "objects": objects,
            "frames": _cat_batch(sampled["frames"], rescored["frames"], steps),
            "goals": _cat_batch(sampled["goals"], rescored["goals"], steps),
            "xi": xi.unsqueeze(0).expand(steps, -1, -1).reshape(-1, xi.shape[-1]),
            "reward": torch.cat((sampled["reward"], rescored["reward"]), dim=1),
            "terminated": torch.cat((sampled["terminated"], rescored["terminated"]), dim=1),
        }

    def _advantages(self, batch: dict) -> torch.Tensor:
        """Per-step λ-advantage, mean over critic heads. Shape ``(H, N)``."""
        cfg = self.cfg
        horizon, n = batch["reward"].shape
        flat = _unflatten_time(batch["states"], horizon + 1)
        obj_flat = None
        obj_feat = None
        if batch.get("objects") is not None:
            obj_flat = _unflatten_time(batch["objects"], horizon + 1)
            obj_feat = object_features(flat[0], flat[1], *obj_flat)
        priv = self.sim.mdp.privileged(*flat, batch["frames"], object_state=obj_flat)
        features = self.norms.critic(
            invariant_features(*flat),
            batch["goals"],
            normalize_xi(batch["xi"]),
            priv,
            obj_feat,
        )
        with torch.no_grad():
            values = self.critic.expected_values(self._critic_fwd(self.critic, features)).view(
                horizon + 1, n, self.critic.ensemble
            )
            value = values.mean(-1)
            target = value[-1]
            targets = []
            reward = batch["reward"]
            cont = 1.0 - batch["terminated"]
            for t in range(horizon - 1, -1, -1):
                nxt = (1.0 - cfg.lam) * value[t + 1] + cfg.lam * target
                target = reward[t] + cfg.gamma * cont[t] * nxt
                targets.append(target)
        return torch.stack(targets[::-1]) - value[:-1]

    def _score_delta(self, iteration: int) -> float:
        """NablaMimic's cosine decay from ``score_delta`` down to ``score_delta_floor``."""
        cfg = self.cfg
        delta = cfg.score_delta
        floor = cfg.score_delta_floor
        if floor >= 1.0 or cfg.lr_decay_iters <= 0:
            return delta
        frac = min(iteration / cfg.lr_decay_iters, 1.0)
        return delta * (floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * frac)))

    def improve_policy(
        self,
        start: dict[str, torch.Tensor],
        diagnose_reward: bool = False,
        iteration: int = 0,
    ) -> tuple:
        """Sample K sequences, take NablaMimic's step, re-score, then weighted NLL."""
        del diagnose_reward
        cfg = self.cfg
        repeated = _repeat_start(start, cfg.score_samples)
        with timed(self.device):
            sampled = self._rollout(repeated, None)
            leaves = sampled["actions"]
            grads = torch.autograd.grad(
                sampled["policy_loss"], leaves, retain_graph=False, allow_unused=False
            )
            grad = -torch.stack(grads)
            taken = torch.stack([leaf.detach() for leaf in leaves])
            del grads, leaves, sampled["policy_loss"], sampled["actions"]
            targets = _nabla_target(
                taken,
                grad,
                self.sim.action_scale,
                self._score_delta(iteration),
                cfg.score_act_clip,
            )
            del grad
            with torch.no_grad():
                rescored = self._rollout(repeated, targets)
            del rescored["policy_loss"], rescored["actions"]
        self.timing["imagine"].append(timed.last)

        with timed(self.device):
            critic_batch = self._critic_batch(sampled, rescored)
            advantage = self._advantages(critic_batch)
            obs = torch.cat((sampled["obs_ins"], rescored["obs_ins"]), dim=1)
            goal = torch.cat((sampled["goal_ins"], rescored["goal_ins"]), dim=1)
            action = torch.cat((taken, targets), dim=1)
            pi_loss, gn_pi = self._fit_policy(obs, goal, action, advantage)
        self.timing["policy"].append(timed.last)

        if self.pi_ema_tau is not None and self.policy_ema is not None:
            soft_copy_(self.policy, self.policy_ema, self.pi_ema_tau)

        term_mean = 0.5 * (sampled["term_mean"] + rescored["term_mean"])
        imag_terms = {
            name: _term_score(name, term_mean[i], cfg.reward_style)
            for i, name in enumerate(self.term_names)
        }
        reward_mean = 0.5 * (sampled["reward_mean"] + rescored["reward_mean"])
        survival = 0.5 * (sampled["alive_last"].mean() + rescored["alive_last"].mean())
        return (
            pi_loss,
            critic_batch,
            reward_mean / self.r_max,
            survival,
            imag_terms,
            gn_pi,
        )

    def _fit_policy(
        self,
        obs: torch.Tensor,
        goal: torch.Tensor,
        action: torch.Tensor,
        advantage: torch.Tensor,
    ) -> tuple[torch.Tensor, float | torch.Tensor]:
        """Advantage-weighted NLL plus a KL penalty, on the whole batch."""
        cfg = self.cfg
        n = advantage.shape[0] * advantage.shape[1]
        obs_f = obs.reshape(n, *obs.shape[2:])
        goal_f = goal.reshape(n, goal.shape[-1])
        action_f = action.reshape(n, action.shape[-1])
        exponent = (advantage / cfg.score_temperature).clamp(-cfg.score_adv_clip, cfg.score_adv_clip)
        weight = exponent.exp().reshape(n)
        log_std_old = self.policy.log_std.detach().clone()
        with torch.no_grad():
            mean_old = self.policy(obs_f, goal_f, self.norms, deterministic=True)

        params = [p for p in self.policy.parameters() if p.requires_grad]
        last_loss = obs.new_zeros(())
        last_gn: float | torch.Tensor = 0.0
        for _ in range(cfg.score_steps):
            mean = self.policy(obs_f, goal_f, self.norms, deterministic=True)
            log_std = self.policy.log_std
            logp = _gaussian_log_prob(mean, log_std, action_f)
            nll = -(weight * logp).sum() / weight.sum().clamp_min(1e-8)
            kl = _gaussian_kl(mean, log_std, mean_old, log_std_old).mean()
            loss = nll + cfg.score_kl * kl
            self.opt_pi.zero_grad(set_to_none=True)
            loss.backward()
            last_gn = _clip_grads(
                params, cfg.grad_clip_norm, cfg.grad_clip_value, no_sync=cfg.no_host_sync
            )
            self.opt_pi.step()
            last_loss = loss.detach()
        return last_loss, last_gn


def run(cfg: ScoreTrainConfig, task_cfg: DictConfig, log_dir) -> None:
    ScoreTrainer(cfg, task_cfg, log_dir).fit()
