"""NablaMimic policy improvement on top of SuperMimic's world-model stack.

Replaces SuperMimic's direct BPTT actor step with ∇Mimic / PODS action targets:
clip per-sample ∂J/∂a, form ``y = a + η g`` with RMS setpoint step ``δ``, then
take several Adam regression steps of ``||π(o) - y||²`` (in joint-setpoint radians).

Imagination is separate from SuperMimic's: π runs under ``no_grad``, actions are free
leaves (``detach().requires_grad_(True)``), and only the WM / reward / critic bootstrap
enter the autograd tape used for ``∂J/∂a``.
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
from supermimic.utils.optim import build_optimizer
from supermimic.utils.nn import soft_copy_
from supermimic.supermimic.supermimic import (
    Trainer,
    _clip_grads,
    _flatten_object_time,
    _flatten_time,
    _init_wm_state,
    _stack,
    _term_score,
    standalone_config,
    timed,
)


NablaTrainConfig = standalone_config(
    "NablaTrainConfig",
    "nablamimic",
    overrides={
        "pi_lr": 6e-4,
        "grad_clip_value": 0.1,
        "normalize_reward": False,
    },
    extra=[
        ("act_cot_clip", float | None, field(default=3.0)),
        ("action_target_steps", int, field(default=8)),
        ("action_target_windows", int, field(default=1)),
        ("action_target_delta", float, field(default=0.03)),
        ("action_target_delta_floor", float, field(default=0.3)),
        ("action_target_batch", int, field(default=16384)),
        ("action_target_lr", float, field(default=6e-4)),
    ],
)


def nabla_wm_step(
    wm: WorldModel,
    norms: InputNorms,
    mdp,
    state: WorldModelState,
    action: torch.Tensor,
    xi_n: torch.Tensor,
    xi: torch.Tensor,
    frame: torch.Tensor,
    contact: torch.Tensor,
):
    """WM transition + next (obs, goal). Action is an external free variable."""
    state, logit = wm.step(state, action, xi_n, norms, contact, mdp)
    frame = frame + 1
    goal = mdp.goal(frame, GOAL_FRAMES)
    obs = mdp.add_noise(mdp.obs(state, action, xi, frame))
    return state, frame, goal, obs, logit.sigmoid()


def nabla_imagine_horizon(
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
    deterministic: bool,
    wm_step=nabla_wm_step,
    bootstrap_grad: bool = True,
    hist_obj: torch.Tensor | None = None,
    obj: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | list[torch.Tensor] | tuple]:
    """H-step imagination for PODS: free actions, no π tape.

    π proposes actions under ``torch.no_grad``; each action is re-wrapped as a leaf
    so ``autograd.grad`` yields ``∂J/∂a`` through the WM only.
    """
    batch = s.shape[0]
    device = s.device
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
    a_pis: list[torch.Tensor] = []
    prev_actions: list[torch.Tensor] = []
    obs_ins: list[torch.Tensor] = []
    goal_ins: list[torch.Tensor] = []
    carried = contact
    for _ in range(horizon):
        obs_in = obs_hist.detach()
        goal_in = goal.detach()
        obs_ins.append(obs_in)
        goal_ins.append(goal_in)
        prev_actions.append(a_prev)
        with torch.no_grad():
            a_pi = policy(obs_in, goal_in, norms, deterministic=deterministic)
        action = a_pi.detach().requires_grad_(True)
        state, frame, goal, obs, carried = wm_step(
            wm, norms, mdp, state, action, xi_n, xi, frame, carried
        )
        # Detach policy inputs; keep ``state`` on the WM tape for rewards / bootstrap.
        obs_hist = torch.cat((obs_in[:, 1:], obs.detach().unsqueeze(1)), dim=1)
        actions.append(action)
        a_pis.append(a_pi.detach())
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
    # Imagined rollouts only ever *terminate*: the horizon cut is not a truncation,
    # it is handled by the bootstrap value at the end of the horizon.
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
        "a_pi": torch.stack(a_pis),
        "obs_ins": torch.stack(obs_ins),
        "goal_ins": torch.stack(goal_ins),
        "survival": survival.detach().mean(-1),
    }


class NablaTrainer(Trainer):
    """SuperMimic trainer with PODS / ∇Mimic policy improvement."""

    cfg: NablaTrainConfig

    def __init__(self, cfg: NablaTrainConfig, task_cfg: DictConfig, log_dir: Path | str):
        # Schedule / logging read cfg.pi_lr; keep it aligned with the regression LR.
        cfg.pi_lr = cfg.action_target_lr
        super().__init__(cfg, task_cfg, log_dir)
        self.opt_pi = build_optimizer(
            self.policy, cfg.action_target_lr, cfg.optim, adam_betas=(0.9, 0.95)
        )
        self._nabla_wm_step = (
            torch.compile(nabla_wm_step, mode="default", dynamic=False)
            if cfg.compile
            else nabla_wm_step
        )
        self._at_count = 0
        self._at_buf: dict[str, torch.Tensor] | None = None

    def _ensure_at_buf(
        self, obs_ins: torch.Tensor, goal_ins: torch.Tensor, actions: torch.Tensor
    ) -> None:
        if self._at_buf is not None:
            return
        cfg = self.cfg
        k = cfg.action_target_windows
        h, b, hist, obs_dim = obs_ins.shape
        goal_dim = goal_ins.shape[-1]
        act_dim = actions.shape[-1]
        dev = obs_ins.device
        self._at_buf = {
            "hist": torch.zeros(k, h, b, hist, obs_dim, device=dev),
            "goal": torch.zeros(k, h, b, goal_dim, device=dev),
            "tgt": torch.zeros(k, h, b, act_dim, device=dev),
            "w": torch.zeros(k, h, b, device=dev),
        }

    def _action_target_delta(self, iteration: int) -> float:
        cfg = self.cfg
        delta = cfg.action_target_delta
        floor = cfg.action_target_delta_floor
        if floor >= 1.0 or cfg.lr_decay_iters <= 0:
            return delta
        frac = min(iteration / cfg.lr_decay_iters, 1.0)
        return delta * (floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * frac)))

    def _nabla_imagine(self, start: dict[str, torch.Tensor]) -> dict:
        cfg = self.cfg
        scale = (
            self.reward_norm.scale().detach()
            if self.reward_norm is not None
            else start["s"].new_ones(())
        )
        raw = nabla_imagine_horizon(
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
            cfg.horizon,
            cfg.gamma,
            not cfg.stochastic_actor,
            self._nabla_wm_step,
            cfg.policy_bootstrap,
            hist_obj=start.get("hist_obj"),
            obj=start.get("obj"),
        )
        imag_terms = {
            name: _term_score(name, raw["term_mean"][i], cfg.reward_style)
            for i, name in enumerate(self.term_names)
        }
        xi = raw["xi"]
        critic_batch = {
            "states": raw["critic_states"],
            "objects": raw["critic_objects"],
            "frames": raw["frames"],
            "goals": raw["goals"],
            "xi": xi.unsqueeze(0).expand(cfg.horizon + 1, -1, -1).reshape(-1, xi.shape[-1]),
            "reward": raw["reward"],
            # Imagined rollouts only ever *terminate*; the horizon cut is handled
            # by the bootstrap term, not by truncation.
            "terminated": raw["terminated"],
        }
        return {
            "policy_loss": raw["policy_loss"],
            "critic_batch": critic_batch,
            "raw_score": raw["reward_mean"] / self.r_max,
            "survival_last": raw["alive_last"].mean(),
            "imag_terms": imag_terms,
            "actions": raw["actions"],
            "a_pi": raw["a_pi"],
            "obs_ins": raw["obs_ins"],
            "goal_ins": raw["goal_ins"],
            "horizon": cfg.horizon,
        }

    def improve_policy(
        self,
        start: dict[str, torch.Tensor],
        diagnose_reward: bool = False,
        iteration: int = 0,
    ) -> tuple:
        """WM imagination (free actions) → clipped targets → multistep regression."""
        del diagnose_reward
        with timed(self.device):
            g = self._nabla_imagine(start)
        self.timing["imagine"].append(timed.last)

        with timed(self.device):
            pi_loss, gn_pi = self._action_target_update(g, iteration)
        self.timing["policy"].append(timed.last)

        if self.pi_ema_tau is not None and self.policy_ema is not None:
            soft_copy_(self.policy, self.policy_ema, self.pi_ema_tau)

        return (
            pi_loss,
            g["critic_batch"],
            g["raw_score"],
            g["survival_last"],
            g["imag_terms"],
            gn_pi,
        )

    def _action_target_update(
        self, g: dict, iteration: int
    ) -> tuple[torch.Tensor, float | torch.Tensor]:
        """Form PODS targets from ∂J/∂a and regress π for ``action_target_steps``."""
        cfg = self.cfg
        actions: list[torch.Tensor] = g["actions"]
        # policy_loss = -J; ascent direction g = ∂J/∂a = -∂L/∂a.
        grads = torch.autograd.grad(
            g["policy_loss"], actions, retain_graph=False, allow_unused=False
        )
        grad = -torch.stack(grads)  # (H, B, A)
        if not torch.isfinite(grad).all():
            grad = torch.where(torch.isfinite(grad), grad, torch.zeros_like(grad))
        nrm = grad.norm(dim=-1)
        live = nrm > 0
        if cfg.act_cot_clip is not None and live.any():
            med = nrm[live].median()
            thr = cfg.act_cot_clip * med
            scale = torch.clamp(thr / (nrm + 1e-30), max=1.0)
            scale = torch.where(live, scale, torch.ones_like(scale))
            grad = grad * scale.unsqueeze(-1)

        action_scale = self.sim.action_scale
        scale2 = action_scale.square()
        live_f = live.to(grad.dtype)
        ss = (grad.square() * scale2 * live_f.unsqueeze(-1)).sum()
        cnt = live_f.sum().clamp_min(1.0) * float(grad.shape[-1])
        delta = self._action_target_delta(iteration)
        eta = delta / (ss / cnt).clamp_min(1e-30).sqrt()
        a_pi = g["a_pi"]
        tgt = a_pi + eta * grad.detach()

        obs_ins = g["obs_ins"]
        goal_ins = g["goal_ins"]
        self._ensure_at_buf(obs_ins, goal_ins, a_pi)
        assert self._at_buf is not None
        k = cfg.action_target_windows
        j = self._at_count % k
        self._at_buf["hist"][j] = obs_ins
        self._at_buf["goal"][j] = goal_ins
        self._at_buf["tgt"][j] = tgt
        self._at_buf["w"][j] = live_f
        self._at_count += 1
        nfill = min(self._at_count, k)

        h, b = obs_ins.shape[:2]
        mb = min(cfg.action_target_batch, nfill * h * b)
        w_err = action_scale
        last_loss = obs_ins.new_zeros(())
        last_gn: float | torch.Tensor = 0.0
        params = [p for p in self.policy.parameters() if p.requires_grad]
        for _ in range(cfg.action_target_steps):
            si = torch.randint(nfill, (mb,), device=obs_ins.device)
            ti = torch.randint(h, (mb,), device=obs_ins.device)
            ei = torch.randint(b, (mb,), device=obs_ins.device)
            hb = self._at_buf["hist"][si, ti, ei]
            gb = self._at_buf["goal"][si, ti, ei]
            yb = self._at_buf["tgt"][si, ti, ei]
            wb = self._at_buf["w"][si, ti, ei]
            pred = self.policy(hb, gb, self.norms, deterministic=True)
            err = (pred - yb) * w_err
            denom = wb.sum().clamp_min(1.0)
            loss = (wb * err.square().mean(dim=-1)).sum() / denom
            self.opt_pi.zero_grad(set_to_none=True)
            loss.backward()
            gn = _clip_grads(
                params, cfg.grad_clip_norm, cfg.grad_clip_value, no_sync=cfg.no_host_sync
            )
            # A non-finite gradient was zeroed on the device, so the step is safe.
            self.opt_pi.step()
            last_loss = loss.detach()
            last_gn = gn
        return last_loss, last_gn


def run(cfg: NablaTrainConfig, task_cfg: DictConfig, log_dir) -> None:
    NablaTrainer(cfg, task_cfg, log_dir).fit()
