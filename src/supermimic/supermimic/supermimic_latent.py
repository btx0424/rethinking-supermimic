"""SuperMimic with a TD-MPC2-style implicit world model.

An encoder maps a real state to a latent and the dynamics are trained to match
the encoding of the real next state. The actor still differentiates through that
latent rollout, but the return is the analytic reward and the soft height/tilt
termination of the decoded pose, not the learned heads. Those heads stay
auxiliary world-model losses. The base trainer is not modified.
"""

from __future__ import annotations

import copy
from dataclasses import field

import torch
import torch.nn as nn
from omegaconf import DictConfig

from supermimic.mdp import (
    GOAL_FRAMES,
    WorldModelState,
    invariant_features,
    normalize_xi,
    object_features,
    pelvis_body_pos,
    pelvis_features,
    unflatten_object,
    unflatten_state,
)
from supermimic.motion import CONTROL_DT
from supermimic.utils.math import (
    mat_to_quat,
    quat_apply,
    quat_inv,
    quat_mul,
    quat_normalize,
    rotvec_to_quat,
    yaw_quat,
)
from supermimic.supermimic.network import MLP, ScalarCritic
from supermimic.supermimic.latent_wm import LatentWM, build_latent_wm
from supermimic.supermimic.supermimic import (
    HIST,
    Trainer,
    _clip_grads,
    _mean,
    _term_score,
    standalone_config,
)
from supermimic.tasks import TaskSpec
from supermimic.utils.nn import soft_copy_
from supermimic.utils.optim import build_optimizer


LatentTrainConfig = standalone_config(
    "LatentTrainConfig",
    "supermimic_latent",
    overrides={"imagine_wm": "live"},
    extra=[
        ("latent_dim", int, field(default=256)),
        ("consistency_coef", float, field(default=1.0)),
        ("termination_coef", float, field(default=1.0)),
        ("rho", float, field(default=1.0)),
        ("value_coef", float, field(default=0.0)),
        ("decode_coef", float, field(default=1.0)),
        ("simnorm_dim", int, field(default=8)),
        # absolute: next simplex. logit: residual in logit space. residual: Euclidean z + Δ, SimNorm off.
        ("wm_step", str, field(default="logit")),
        # tdmpc: the SimNorm dynamics above. lewm: next-embedding prediction + SIGReg.
        ("wm_arch", str, field(default="tdmpc")),
        ("sigreg_coef", float, field(default=0.09)),
        # Off unless set. LeWM's embedding loss does not need them.
        ("lewm_reward_coef", float, field(default=1.0)),
        ("lewm_contact_coef", float, field(default=1.0)),
        # decode: analytic reward of the decoded pose. head: the reward prediction head.
        ("rollout_reward", str, field(default="decode")),
    ],
)


class LatentActor(nn.Module):
    """Deterministic policy on ``(z, goal)``. Same call shape as :class:`Policy`."""

    def __init__(
        self,
        latent_dim: int,
        goal_dim: int,
        action_dim: int,
        hidden: tuple[int, ...],
        stochastic: bool,
        ref_residual: float,
    ):
        super().__init__()
        self.stochastic = stochastic
        self.ref_residual = float(ref_residual)
        self.net = MLP(latent_dim + goal_dim, list(hidden), action_dim)
        nn.init.zeros_(self.net.head.bias)
        with torch.no_grad():
            self.net.head.weight.mul_(0.01)
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.0)) if stochastic else None
        if self.ref_residual > 0:
            self.register_buffer("q_default", torch.zeros(action_dim))
            self.register_buffer("alpha", torch.ones(action_dim))

    def forward(self, z: torch.Tensor, goal: torch.Tensor, norms, deterministic: bool = True) -> torch.Tensor:
        mean = self.net(torch.cat((z, norms.goal(goal)), dim=-1))
        if self.log_std is not None and not deterministic:
            mean = mean + self.log_std.exp() * torch.randn_like(mean)
        if self.ref_residual > 0:
            action_dim = self.q_default.shape[0]
            q_next = goal[..., 15 : 15 + action_dim]
            mean = mean + self.ref_residual * (q_next - self.q_default) / self.alpha
        return mean

    def entropy(self) -> torch.Tensor:
        if self.log_std is None:
            raise RuntimeError("Entropy is only defined for the stochastic actor.")
        return (self.log_std + 0.5 * (1.0 + torch.log(torch.tensor(2.0 * torch.pi, device=self.log_std.device)))).sum()

    @torch.no_grad()
    def set_reference_action(self, q_default: torch.Tensor, alpha: torch.Tensor) -> None:
        if self.ref_residual <= 0:
            return
        self.q_default.copy_(q_default.to(self.q_default))
        self.alpha.copy_(alpha.to(self.alpha))


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def _rot6d_to_mat(r: torch.Tensor) -> torch.Tensor:
    """Inverse of ``mat_to_6d``: the first two columns, row-major in groups of two."""
    cols = r.reshape(*r.shape[:-1], 3, 2)
    b1 = _unit(cols[..., 0])
    a2 = cols[..., 1]
    b2 = _unit(a2 - (b1 * a2).sum(-1, keepdim=True) * b1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def _tilt_from_gravity(g_body: torch.Tensor) -> torch.Tensor:
    """Yaw-free quaternion whose body-frame gravity is ``g_body``.

    World gravity is ``-Z``. Yaw is removed so the caller can attach the yaw it
    integrated from the angular velocity.
    """
    g_world = g_body.new_tensor((0.0, 0.0, -1.0))
    g_b = _unit(g_body)
    cross = torch.cross(g_b, g_world.expand_as(g_b), dim=-1)
    dot = (g_b * g_world).sum(-1, keepdim=True).clamp(-1.0, 1.0)
    quat = quat_normalize(torch.cat((1.0 + dot, cross), dim=-1))
    # The cross product vanishes when the robot is upside down.
    anti = dot.squeeze(-1) < -0.999
    fallback = g_body.new_zeros(g_body.shape[:-1] + (4,))
    fallback[..., 1] = 1.0
    quat = torch.where(anti[..., None], fallback, quat)
    return quat_mul(quat_inv(yaw_quat(quat)), quat)


def _stack_states(states: list[WorldModelState]) -> WorldModelState:
    def column(name: str):
        values = [getattr(state, name) for state in states]
        if values[0] is None:
            return None
        return torch.stack(values, 0)

    return WorldModelState(*(column(name) for name in WorldModelState._fields))


class LatentTrainer(Trainer):
    """SuperMimic trainer whose world model is latent consistency, not state MSE."""

    def __init__(self, cfg: LatentTrainConfig, task_cfg: DictConfig, log_dir):
        if float(cfg.value_coef) != 0.0:
            raise ValueError(
                "supermimic_latent trains the critic with its own λ-return. "
                f"value_coef must be 0, got {cfg.value_coef}."
            )
        if cfg.critic_source not in ("imagined", "sim"):
            raise ValueError(
                f"critic_source={cfg.critic_source!r}. Expected 'imagined' or 'sim'."
            )
        resume = cfg.resume
        cfg.resume = None
        super().__init__(cfg, task_cfg, log_dir)
        cfg.resume = resume
        self._install_latent()
        self._check_rollout_reward()
        # Replaced each iteration by prepare_replay with the length the buffer can supervise.
        self._open_loop = int(self.horizon_pi)
        if resume:
            self.load_checkpoint(resume)

    def _install_latent(self) -> None:
        cfg = self.cfg
        goal_dim = self.task.goal_dim
        self.wm = build_latent_wm(cfg, self.task, goal_dim).to(self.device)
        wm = self.wm
        if self.wm_ema_tau is not None:
            self.wm_ema = copy.deepcopy(self.wm)
            self.wm_ema.requires_grad_(False)
            self.wm_ema.eval()
        self.opt_wm = build_optimizer(self.wm, cfg.wm_lr, cfg.optim)
        actor = LatentActor(
            int(cfg.latent_dim),
            goal_dim,
            self.task.num_joints,
            cfg.policy_trunk,
            cfg.stochastic_actor,
            cfg.ref_residual,
        ).to(self.device)
        if cfg.ref_residual > 0:
            actor.set_reference_action(self.sim.default_q, self.sim.action_scale)
        self.policy = actor
        if self.pi_ema_tau is not None:
            self.policy_ema = copy.deepcopy(self.policy)
            self.policy_ema.requires_grad_(False)
            self.policy_ema.eval()
        self.opt_pi = build_optimizer(self.policy, cfg.pi_lr, cfg.optim, adam_betas=(0.9, 0.95))
        # V(z, goal). Privileged tracking error needs a decoded state, which this
        # model does not produce.
        self.critic = ScalarCritic(int(cfg.latent_dim) + goal_dim, ensemble=cfg.critic_ensemble).to(self.device)
        self.opt_v = torch.optim.Adam(self.critic.parameters(), lr=cfg.v_lr)

    def _check_rollout_reward(self) -> None:
        mode = str(self.cfg.rollout_reward)
        if mode == "decode":
            if float(self.cfg.decode_coef) == 0.0:
                raise ValueError("rollout_reward='decode' requires decode_coef > 0.")
            return
        if mode == "head":
            coef = self.cfg.lewm_reward_coef if self.wm.arch == "lewm" else self.cfg.reward_coef
            name = "lewm_reward_coef" if self.wm.arch == "lewm" else "reward_coef"
            if float(coef) == 0.0:
                raise ValueError(f"rollout_reward='head' requires {name} > 0.")
            return
        raise ValueError(f"rollout_reward must be 'decode' or 'head', got {mode!r}.")

    # ------------------------------------------------------------------ features

    def _encode(self, wm: LatentWM, p, quat, v, w, q, qd, xi, contact, obj) -> torch.Tensor:
        """``h(s)``. ``xi`` is raw; contact is the recorded per-body bit."""
        return wm.encode(self.norms, self.sim.mdp, p, quat, v, w, q, qd, xi, contact, obj)

    def _recon_scale(self) -> torch.Tensor:
        """Per-component divisors matching the explicit world-model MSE."""
        return self.wm.feature_scale(self.device)

    def _recon_target(self, state: torch.Tensor, obj: torch.Tensor | None) -> torch.Tensor:
        """Normalized pose features of a real state. No gradient into the target."""
        return self.wm.pose_target(None, self.sim.mdp, state, obj)

    def _encode_flat(self, wm: LatentWM, state: torch.Tensor, xi, contact, obj) -> torch.Tensor:
        p, quat, v, w, q, qd = unflatten_state(state)
        obj_t = unflatten_object(obj) if obj is not None else None
        return self._encode(wm, p, quat, v, w, q, qd, xi, contact, obj_t)

    # ------------------------------------------------------------------ model fit

    def _transition_targets(self, batch: dict[str, torch.Tensor]):
        """Analytic reward and hard termination of each stored simulator transition.

        Both are what :meth:`TrackingSim.step` computed: the reward is scored on
        the post-step state at the next reference frame, and truncation is not a
        termination.
        """
        s1 = batch["s1"]
        b, k, _ = s1.shape
        flat = b * k
        p, quat, v, w, q, qd = unflatten_state(s1.reshape(flat, -1))
        frame = batch["f"]
        last = self.sim.mdp.episode_last(frame)
        f_next = torch.minimum(frame + 1, last)
        obj = unflatten_object(batch["obj1"].reshape(flat, -1)) if "obj1" in batch else None
        if obj is None:
            state = WorldModelState(p, quat, v, w, q, qd, None, None)
        else:
            state = WorldModelState(p, quat, v, w, q, qd, None, None, *obj)
        reward = self.sim.mdp.reward(
            state,
            f_next.reshape(flat),
            batch["a"].reshape(flat, -1),
            batch["a_prev"].reshape(flat, -1),
        ).view(b, k)
        terminated = self.sim.mdp.termination(p, quat, f_next.reshape(flat)).view(b, k).float()
        goal = self.sim.mdp.goal(f_next.reshape(flat), GOAL_FRAMES).view(b, k, -1)
        return reward, terminated, goal

    def update_world_model(self, log_now: bool = False):
        """One world-model step. The loss is whatever ``wm.compute_loss`` defines."""
        del log_now
        cfg = self.cfg
        totals, errs, norms = [], [], []
        buckets: dict[str, list] = {}
        for _ in range(cfg.wm_batches):
            batch = self.buffer.sample_windows(cfg.wm_batch, self._open_loop)
            loss, stats = self.wm.compute_loss(batch, self.norms, self.sim.mdp)
            self.opt_wm.zero_grad(set_to_none=True)
            if self.wm_frozen:
                norms.append(0.0)
            else:
                loss.backward()
                norms.append(_clip_grads(self.wm.parameters(), cfg.grad_clip_norm, no_sync=cfg.no_host_sync))
                self.opt_wm.step()
                if self.wm_ema_tau is not None and self.wm_ema is not None:
                    soft_copy_(self.wm, self.wm_ema, self.wm_ema_tau)
            totals.append(loss.detach())
            errs.append(stats["mse"].detach())
            for key, value in stats.items():
                buckets.setdefault(key, []).append(value.detach())
        disc_stats = {key: torch.stack(values).mean() for key, values in buckets.items()}
        return torch.stack(totals).mean(), torch.stack(errs).mean(), _mean(norms), disc_stats

    def _integrate_decode(self, raw: torch.Tensor, p_prev, quat_prev, obj_prev):
        """World state from one decoded pose, carrying xy and yaw from ``p_prev``.

        The decoder emits the pelvis-frame (or heading-frame) pose. Height, tilt,
        and the joints come from that pose. Horizontal position and yaw are not in
        the code, so they are integrated from the decoded body-frame twist and the
        real root this rollout started from. Body poses are left empty; the reward
        fills them with FK of this root.
        """
        task = self.task
        nj = task.num_joints
        feat_dim = task.pelvis_feat_dim if self.wm.frame == "pelvis" else task.inv_dim
        feat = raw[..., :feat_dim]
        if self.wm.frame == "pelvis":
            height, gravity, v_body, w_body, q, qd = feat.split((1, 3, 3, 3, nj, nj), dim=-1)
            tilt = _tilt_from_gravity(gravity)
        else:
            height, rot6, v_body, w_body, q, qd = feat.split((1, 6, 3, 3, nj, nj), dim=-1)
            tilt = mat_to_quat(_rot6d_to_mat(rot6))
            tilt = quat_mul(quat_inv(yaw_quat(tilt)), tilt)
        quat_step = quat_normalize(quat_mul(quat_prev, rotvec_to_quat(w_body * CONTROL_DT)))
        quat = quat_normalize(quat_mul(yaw_quat(quat_step), tilt))
        v_step = quat_apply(quat_prev, v_body)
        p = torch.cat((p_prev[..., :2] + v_step[..., :2] * CONTROL_DT, height), dim=-1)
        v = quat_apply(quat, v_body)
        w = quat_apply(quat, w_body)
        obj_next = None
        obj_kwargs: dict = {}
        if task.has_object:
            if obj_prev is None:
                raise ValueError("Object features are required for this decoder.")
            obj_raw = raw[..., feat_dim + task.body_pos_dim :]
            dp, rot6, ov, ow = obj_raw.split((3, 6, 3, 3), dim=-1)
            yaw = yaw_quat(quat)
            obj_quat = quat_normalize(quat_mul(yaw, mat_to_quat(_rot6d_to_mat(rot6))))
            obj_p = p + quat_apply(yaw, dp)
            obj_prev_p, obj_prev_quat = obj_prev
            obj_kwargs = {
                "obj_p": obj_p,
                "obj_quat": obj_quat,
                "obj_v": quat_apply(obj_quat, ov),
                "obj_w": quat_apply(obj_quat, ow),
                "obj_p_prev": obj_prev_p,
                "obj_quat_prev": obj_prev_quat,
            }
            obj_next = (obj_p, obj_quat)
        state = WorldModelState(p, quat, v, w, q, qd, **obj_kwargs)
        return state, p, quat, obj_next

    def _latent_rollout(self, start: dict[str, torch.Tensor], horizon: int, wm: LatentWM):
        """H-step latent rollout.

        ``rollout_reward='decode'`` scores the analytic reward of the decoded pose.
        ``rollout_reward='head'`` uses the reward prediction head and does not decode.
        The encoder of the real start is detached.
        """
        z = self._encode_flat(
            wm,
            start["s"],
            start["xi"],
            start["c"],
            start.get("obj"),
        ).detach()
        frame = start["f"]
        a_prev = start["a_prev"]
        zs = [z]
        goals = [self.sim.mdp.goal(frame, GOAL_FRAMES)]
        actions = []
        prevs = []
        if str(self.cfg.rollout_reward) == "head":
            rewards = []
            terminateds = []
            for _ in range(horizon):
                action = self.policy(z, goals[-1], self.norms, deterministic=not self.cfg.stochastic_actor)
                action_n = self.norms.action(action)
                last = self.sim.mdp.episode_last(frame)
                f_next = torch.minimum(frame + 1, last)
                goal_next = self.sim.mdp.goal(f_next, GOAL_FRAMES)
                rewards.append(wm.reward(z, action_n, self.norms.goal(goal_next)))
                z = wm.step(z, action_n)
                if hasattr(wm, "termination_logit"):
                    terminateds.append(wm.termination_logit(z, self.norms.goal(goal_next)).sigmoid())
                else:
                    terminateds.append(torch.zeros(action.shape[0], device=action.device))
                actions.append(action)
                prevs.append(a_prev)
                a_prev = action
                frame = f_next
                zs.append(z)
                goals.append(goal_next)
            reward = torch.stack(rewards, 0)
            terminated = torch.stack(terminateds, 0)
            parts = {name: torch.zeros_like(reward) for name in self.term_names}
            return {
                "z": torch.stack(zs),
                "goals": torch.stack(goals),
                "reward": reward,
                "parts": parts,
                "terminated": terminated,
                "actions": actions,
            }
        p, quat, _v, _w, _q, _qd = unflatten_state(start["s"])
        p = p.detach()
        quat = quat.detach()
        obj_prev = None
        if self.task.has_object:
            obj_p, obj_quat, _ov, _ow = unflatten_object(start["obj"])
            obj_prev = (obj_p.detach(), obj_quat.detach())
        scale = self._recon_scale()
        states: list[WorldModelState] = []
        frames = []
        for _ in range(horizon):
            action = self.policy(z, goals[-1], self.norms, deterministic=not self.cfg.stochastic_actor)
            z = wm.step(z, self.norms.action(action))
            last = self.sim.mdp.episode_last(frame)
            f_next = torch.minimum(frame + 1, last)
            state, p, quat, obj_prev = self._integrate_decode(wm.decoder(z) * scale, p, quat, obj_prev)
            states.append(state)
            frames.append(f_next)
            actions.append(action)
            prevs.append(a_prev)
            a_prev = action
            frame = f_next
            zs.append(z)
            goals.append(self.sim.mdp.goal(f_next, GOAL_FRAMES))
        stacked = _stack_states(states)
        frame_t = torch.stack(frames, 0)
        action_t = torch.stack(actions, 0)
        prev_t = torch.stack(prevs, 0)
        mdp = self.sim.mdp
        prev_flag = mdp.object_reward_surrogate
        mdp.object_reward_surrogate = True
        try:
            if getattr(mdp, "reward_terms_compiled", False):
                reward, parts = mdp.reward_flat(stacked, frame_t, action_t, prev_t)
            else:
                reward, parts = mdp.reward(
                    stacked, frame_t, action_t, prev_t, terms=True, surrogate=True
                )
        finally:
            mdp.object_reward_surrogate = prev_flag
        terminated = mdp.soft_termination(stacked.p, stacked.quat, frame_t)
        return {
            "z": torch.stack(zs),
            "goals": torch.stack(goals),
            "reward": reward,
            "parts": parts,
            "terminated": terminated,
            "actions": actions,
        }

    def prepare_replay(self, clip_last: torch.Tensor) -> None:
        """Fit and imagine for the same length.

        The actor used to unroll ``horizon`` steps through a model trained for
        ``wm_window``. The open-loop tail was unsupervised, and the decoder
        scored it as upright. Both lengths are the longest in-episode window
        the buffer can support, capped at ``horizon_pi``.
        """
        if self.cfg.critic_source == "imagined" and self.horizon_pi != self.horizon_v:
            raise ValueError(
                f"horizon_pi={self.horizon_pi} and horizon_v={self.horizon_v} differ. "
                "supermimic_latent fits the critic on the actor's latent rollout. "
                "Use critic_source=sim to give the critic its own horizon."
            )
        desired = int(self.horizon_pi)
        chosen = 0
        for length in range(desired, 0, -1):
            self.buffer.prepare(length, start_horizon=length, clip_last=clip_last)
            windows = self.buffer._window_candidates
            starts = self.buffer._start_candidates
            if windows is not None and windows.numel() > 0 and starts is not None and starts.numel() > 0:
                chosen = length
                break
        if chosen < 1:
            raise RuntimeError(
                "No in-episode window for the latent rollout. Collect more steps before training."
            )
        self._open_loop = chosen
        if chosen != getattr(self, "_logged_open_loop", None):
            print(f"latent rollout length {chosen} (horizon {desired})", flush=True)
            self._logged_open_loop = chosen

    def _imagine_graph(self, start: dict[str, torch.Tensor], horizon: int | None = None) -> dict:
        cfg = self.cfg
        # ``prepare_replay`` set ``_open_loop`` to the length the model was just fit for.
        horizon = self._open_loop if horizon is None else int(horizon)
        rollout = self._latent_rollout(start, horizon, self.policy_wm())
        reward = rollout["reward"]
        terminated = rollout["terminated"]
        zs = rollout["z"]
        goals = rollout["goals"]
        actions = rollout["actions"]
        scale = self.reward_norm.scale().detach() if self.reward_norm is not None else reward.new_ones(())
        reward_n = reward / scale
        alive = torch.cumprod(1.0 - terminated, dim=0)
        survival = torch.cat((torch.ones(1, alive.shape[1], device=alive.device), alive[:-1]), dim=0)
        discount = cfg.gamma ** torch.arange(horizon, device=reward.device, dtype=torch.float32)
        weighted = discount[:, None] * survival
        returns = (weighted * reward_n).sum(0)
        goal_n = self.norms.goal(goals[-1])
        values = self.critic.expected_values(
            self._critic_fwd(self.critic, torch.cat((zs[-1], goal_n), dim=-1))
        )
        bootstrap = cfg.gamma**horizon * alive[-1] * values.mean(-1)
        if not cfg.policy_bootstrap:
            bootstrap = bootstrap.detach()
        policy_loss = -(returns + bootstrap).mean()
        for action in actions:
            if action.requires_grad:
                action.retain_grad()
        term_raw = torch.stack([rollout["parts"][name] for name in self.term_names])
        term_r = term_raw / scale
        term_loss = -(weighted * term_r).sum(dim=1).mean(dim=-1)
        term_mean = term_raw.detach().mean(dim=(1, 2))
        term_losses = {name: term_loss[i] for i, name in enumerate(self.term_names)}
        term_losses["bootstrap"] = -bootstrap.mean()
        imag_terms = {
            name: _term_score(name, term_mean[i], cfg.reward_style) for i, name in enumerate(self.term_names)
        }
        return {
            "policy_loss": policy_loss,
            "critic_batch": {
                "z": zs.detach(),
                "goals": goals.detach(),
                "reward": reward_n.detach(),
                "terminated": terminated.detach(),
            },
            "raw_score": reward.detach().mean() / self.r_max,
            "survival_last": alive[-1].detach().mean(),
            "imag_terms": imag_terms,
            "term_losses": term_losses,
            "step_return": (weighted * reward_n).detach(),
            "actions": actions,
            "horizon": horizon,
        }

    def update_critic(self, batch: dict):
        """λ-return of the latent rollout. Features are ``(z, goal)``."""
        cfg = self.cfg
        z = batch["z"]
        goals = batch["goals"]
        horizon, n = batch["reward"].shape
        flat_z = z.reshape((horizon + 1) * n, -1)
        flat_goal = goals.reshape((horizon + 1) * n, -1)
        features = torch.cat((flat_z, self.norms.goal(flat_goal)), dim=-1)
        ensemble = self.critic.ensemble
        with torch.no_grad():
            values = self.critic.expected_values(self._critic_fwd(self.critic, features)).view(
                horizon + 1, n, ensemble
            )
            value_mean = values.mean()
            target = values[-1]
            targets = []
            reward = batch["reward"].unsqueeze(-1)
            cont = (1.0 - batch["terminated"]).unsqueeze(-1)
            for t in range(horizon - 1, -1, -1):
                nxt = (1.0 - cfg.lam) * values[t + 1] + cfg.lam * target
                target = reward[t] + cfg.gamma * cont[t] * nxt
                targets.append(target)
            targets = torch.stack(targets[::-1])
        pred_in = features.view(horizon + 1, n, -1)[:-1].reshape(-1, features.shape[-1])
        target_flat = targets.reshape(-1, ensemble)
        count = pred_in.shape[0]
        minibatch = max(count // cfg.critic_minibatches, 1)
        last = pred_in.new_zeros(())
        norms: list[float] = []
        for _ in range(cfg.critic_epochs):
            order = torch.randperm(count, device=pred_in.device)
            for i in range(cfg.critic_minibatches):
                idx = order[i * minibatch : (i + 1) * minibatch]
                if idx.numel() == 0:
                    continue
                pred = self._critic_fwd(self.critic, pred_in[idx])
                last = self.critic.compute_loss(pred, target_flat[idx])
                self.opt_v.zero_grad(set_to_none=True)
                last.backward()
                head_norms = [
                    _clip_grads(head.parameters(), cfg.grad_clip_norm) for head in self.critic.heads
                ]
                norms.append(sum(head_norms) / len(head_norms))
                self.opt_v.step()
        return last.detach(), _mean(norms), value_mean.detach()

    def sim_critic_batch(self) -> dict | None:
        """λ-return on real transitions, in the latent the actor bootstraps.

        Same windows as :meth:`Trainer.sim_critic_batch`: the latest ``horizon_v``
        steps of each environment, reward and hard termination from the analytic
        MDP. Each state is encoded with the actor's selected world model and the
        code is detached, so the critic and actor use the same latent coordinates.
        """
        horizon = self.horizon_v
        buf = self.buffer
        if buf.filled < horizon or horizon < 1:
            return None
        device = buf.device
        if buf.filled < buf.capacity:
            last = buf.filled - 1
            rows = last - torch.arange(horizon - 1, -1, -1, device=device)
        else:
            last = (buf.ptr - 1) % buf.capacity
            rows = (last - torch.arange(horizon - 1, -1, -1, device=device)) % buf.capacity
        rows = rows[:, None].expand(horizon, buf.num_envs).contiguous()
        env = torch.arange(buf.num_envs, device=device)
        ep = buf.ep[rows, env]
        step = buf.step[rows, env]
        valid = (ep == ep[:1]).all(0) & ((step[1:] - step[:-1]) == 1).all(0)
        if horizon == 1:
            valid = (ep == ep[:1]).all(0)
        if not bool(valid.any()):
            return None
        rows, env = rows[:, valid], env[valid]
        s = buf.s[rows, env]
        s1 = buf.s1[rows, env]
        action = buf.a[rows, env]
        a_prev = buf.a_prev[rows, env]
        frame = buf.f[rows, env]
        xi = buf.xi[rows, env]
        contact = buf.c[rows, env]
        contact1 = buf.c1[rows, env]
        s_seq = torch.cat((s, s1[-1:]), dim=0)
        c_seq = torch.cat((contact, contact1[-1:]), dim=0)
        xi_seq = torch.cat((xi, xi[-1:]), dim=0)
        clip_last = self.sim.mdp.episode_last(frame)
        f_next = torch.minimum(frame + 1, clip_last)
        f_seq = torch.cat((frame, f_next[-1:]), dim=0)
        steps, n = action.shape[:2]
        flat_n = steps * n
        seq_n = (steps + 1) * n
        p1, quat1, _v1, _w1, _q1, _qd1 = unflatten_state(s1.reshape(flat_n, -1))
        obj_seq = None
        if buf.obj is not None:
            obj = buf.obj[rows, env]
            obj1 = buf.obj1[rows, env]
            op, oq, ov, ow = unflatten_object(obj1.reshape(flat_n, -1))
            successor = WorldModelState(p1, quat1, _v1, _w1, _q1, _qd1, None, None, op, oq, ov, ow)
            obj_seq = torch.cat((obj, obj1[-1:]), dim=0)
        else:
            successor = WorldModelState(p1, quat1, _v1, _w1, _q1, _qd1, None, None)
        mdp = self.sim.mdp
        reward = mdp.reward(
            successor, f_next.reshape(flat_n), action.reshape(flat_n, -1), a_prev.reshape(flat_n, -1)
        ).view(steps, n)
        terminated = mdp.termination(p1, quat1, f_next.reshape(flat_n)).view(steps, n)
        scale = self.reward_norm.scale().detach() if self.reward_norm is not None else reward.new_ones(())
        with torch.no_grad():
            z = self._encode_flat(
                self.policy_wm(),
                s_seq.reshape(seq_n, -1),
                xi_seq.reshape(seq_n, -1),
                c_seq.reshape(seq_n, -1),
                None if obj_seq is None else obj_seq.reshape(seq_n, -1),
            ).view(steps + 1, n, -1)
        goals = mdp.goal(f_seq.reshape(-1), GOAL_FRAMES).view(steps + 1, n, -1)
        return {
            "z": z.detach(),
            "goals": goals.detach(),
            "reward": (reward / scale).detach(),
            "terminated": terminated.detach().float(),
        }

    def collect(self, n_steps: int) -> None:
        """Act from the latent of the real state. The simulator is unchanged."""
        cfg = self.cfg
        for _ in range(n_steps):
            p, quat, v, w, q, qd = self.sim.state()
            bodies = self.sim.tracked_bodies()
            obj = self.sim.object_state() if self.task.has_object else None
            state = WorldModelState(
                p,
                quat,
                v,
                w,
                q,
                qd,
                bodies[0],
                bodies[1],
                *(obj if obj is not None else (None, None, None, None)),
            )
            obs = self.sim.mdp.add_noise(self.sim.mdp.obs(state, self.sim.a_prev, self.sim.xi, self.sim.f))
            fresh = self.sim.ep_step == 0
            self.obs_hist = torch.where(
                fresh[:, None, None],
                obs[:, None, :].expand(-1, HIST, -1),
                torch.cat((self.obs_hist[:, 1:], obs[:, None, :]), dim=1),
            )
            goal = self.sim.mdp.goal(self.sim.f, GOAL_FRAMES)
            xi_n = normalize_xi(self.sim.xi)
            self.norms.obs.update(obs)
            self.norms.goal.update(goal)
            if not self.wm_frozen:
                self.norms.state.update(invariant_features(p, quat, v, w, q, qd))
                self.norms.xi.update(xi_n)
                if self.wm.frame == "pelvis":
                    self.norms.dyn.update(pelvis_features(p, quat, v, w, q, qd))
                    self.norms.body.update(pelvis_body_pos(p, quat, bodies[0]))
                else:
                    self.norms.body.update(self.sim.mdp.heading_bodies(p, quat, v, w, q, qd, bodies[0]))
                if obj is not None and self.norms.obj is not None:
                    self.norms.obj.update(object_features(p, quat, *obj))
            with torch.no_grad():
                z = self._encode(self.policy_wm(), p, quat, v, w, q, qd, self.sim.xi, self.sim.ground_contact(), obj)
                action = self.policy(z, goal, self.norms, deterministic=not cfg.stochastic_actor)
            executed = action + cfg.action_noise * torch.randn_like(action)
            if not self.wm_frozen:
                self.norms.action.update(executed)
            out = self.sim.step(executed)
            self.buffer.add(out)
            if self.reward_norm is not None:
                self.reward_norm.update(out["r"], out["done"])
            style = self.cfg.reward_style
            self.stats["r"].append((out["r"].mean() / self.r_max).detach())
            for name, value in out["terms"].items():
                self.stats["sim_terms"][name].append(_term_score(name, value.mean().detach(), style))
            done = out["done"]
            if bool(done.any().item()):
                length = out["ep_len"][done].float().clamp(min=1)
                self.stats["ep"].append((out["ep_ret"][done] / (self.r_max * length)).mean().detach())
                self.stats["ep_len"].append(length.sum().detach())
                for name, value in out["ep_term"].items():
                    self.stats["ep_terms"][name].append(
                        _term_score(name, (value[done] / length).mean().detach(), style)
                    )
                self.stats["term"].append(out["terminated"][done].float().sum())
                self.stats["trunc"].append(out["truncated"][done].float().sum())
                self.stats["done_n"].append(done.float().sum())


def run(cfg: LatentTrainConfig, task_cfg: DictConfig, log_dir) -> None:
    LatentTrainer(cfg, task_cfg, log_dir).fit()
