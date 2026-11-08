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
import torch.nn.functional as F
from omegaconf import DictConfig

from supermimic.mdp import (
    GOAL_FRAMES,
    WorldModelState,
    heading_body_pos,
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
from supermimic.supermimic.supermimic import (
    HIST,
    Trainer,
    _clip_grads,
    _mean,
    _term_score,
    contact_bce,
    contact_precision_recall,
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
    ],
)

# Floor for ``log z`` in the logit residual. Softmax coordinates can sit near 0.
_LOGIT_EPS = 1e-8
_WM_STEPS = ("absolute", "logit", "residual")


class SimNorm(nn.Module):
    """Split the latent into groups and softmax each group into a simplex.

    The same map TD-MPC2 puts on the encoder and the dynamics. A group size of
    0 is the identity, so the latent stays an unbounded vector.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim <= 0:
            return x
        groups = x.view(*x.shape[:-1], -1, self.dim)
        return F.softmax(groups, dim=-1).view(x.shape)


class LatentWorldModel(nn.Module):
    """Encoder, latent step, reward head, and termination head. No state decoder."""

    def __init__(
        self,
        task: TaskSpec,
        latent_dim: int,
        goal_dim: int,
        norm: str,
        frame: str,
        simnorm_dim: int = 8,
        wm_step: str = "absolute",
    ):
        super().__init__()
        if frame not in ("heading", "pelvis"):
            raise ValueError(f"wm frame must be 'heading' or 'pelvis', got {frame!r}.")
        if wm_step not in _WM_STEPS:
            raise ValueError(f"wm_step must be one of {_WM_STEPS}, got {wm_step!r}.")
        self.task = task
        self.frame = frame
        self.wm_step = wm_step
        self.latent_dim = int(latent_dim)
        # The Euclidean residual lives in an unbounded code, so SimNorm is off.
        self.simnorm_dim = 0 if wm_step == "residual" else int(simnorm_dim)
        if wm_step == "logit" and self.simnorm_dim <= 0:
            raise ValueError("wm_step='logit' needs simnorm_dim > 0.")
        if self.simnorm_dim < 0:
            raise ValueError(f"simnorm_dim must be >= 0, got {self.simnorm_dim}.")
        if self.simnorm_dim and self.latent_dim % self.simnorm_dim:
            raise ValueError(
                f"latent_dim={self.latent_dim} must be divisible by simnorm_dim={self.simnorm_dim}."
            )
        feat_dim = task.pelvis_feat_dim if frame == "pelvis" else task.inv_dim
        enc_in = feat_dim + task.xi_dim + task.body_pos_dim + task.n_contact + task.object_feat_dim
        hidden = [512, 512]
        self.encoder = MLP(enc_in, hidden, self.latent_dim, norm)
        self.dynamics = MLP(self.latent_dim + task.num_joints, hidden, self.latent_dim, norm)
        self.reward_head = MLP(self.latent_dim + task.num_joints + goal_dim, hidden, 1, norm)
        self.termination_head = MLP(self.latent_dim + goal_dim, hidden, 1, norm)
        # Same contents the encoder reads, minus ξ and contact: the pose features,
        # tracked-body positions, and the object block.
        self.recon_dim = feat_dim + task.body_pos_dim + task.object_feat_dim
        self.decoder = MLP(self.latent_dim, hidden, self.recon_dim, norm)
        self.contact_head = MLP(self.latent_dim, hidden, task.n_contact, norm)
        self.simnorm = SimNorm(self.simnorm_dim)
        nn.init.zeros_(self.reward_head.head.weight)
        nn.init.zeros_(self.reward_head.head.bias)
        nn.init.zeros_(self.termination_head.head.weight)
        nn.init.zeros_(self.termination_head.head.bias)
        nn.init.zeros_(self.decoder.head.weight)
        nn.init.zeros_(self.decoder.head.bias)
        nn.init.zeros_(self.contact_head.head.weight)
        nn.init.zeros_(self.contact_head.head.bias)
        # Δ = 0 leaves z unchanged for the logit and Euclidean residuals.
        if wm_step != "absolute":
            nn.init.zeros_(self.dynamics.head.weight)
            nn.init.zeros_(self.dynamics.head.bias)

    def step(self, z: torch.Tensor, action_n: torch.Tensor) -> torch.Tensor:
        """One latent transition.

        ``absolute`` predicts the next code and applies SimNorm. ``logit`` adds a
        delta in logit space, ``softmax(log z + Δ)``. ``residual`` adds the delta
        in the Euclidean code, with SimNorm disabled.
        """
        delta = self.dynamics(torch.cat((z, action_n), dim=-1))
        if self.wm_step == "residual":
            return z + delta
        if self.wm_step == "logit":
            return self.simnorm(z.clamp(min=_LOGIT_EPS).log() + delta)
        return self.simnorm(delta)

    def reward(self, z: torch.Tensor, action_n: torch.Tensor, goal_n: torch.Tensor) -> torch.Tensor:
        return self.reward_head(torch.cat((z, action_n, goal_n), dim=-1)).squeeze(-1)

    def termination_logit(self, z: torch.Tensor, goal_n: torch.Tensor) -> torch.Tensor:
        return self.termination_head(torch.cat((z, goal_n), dim=-1)).squeeze(-1)

    def encode(self, norms, mdp, p, quat, v, w, q, qd, xi, contact, obj) -> torch.Tensor:
        """``h(s)``. ``xi`` is raw; ``contact`` is the per-body ground-contact bit."""
        body_pos = mdp.bodies(p, quat, v, w, q, qd)[0]
        if self.frame == "pelvis":
            state_n = norms.dyn(pelvis_features(p, quat, v, w, q, qd))
            body = pelvis_body_pos(p, quat, body_pos)
        else:
            state_n = norms.state(invariant_features(p, quat, v, w, q, qd))
            body = heading_body_pos(p, quat, body_pos)
        parts = [state_n, norms.xi(normalize_xi(xi)), norms.body(body), contact]
        if self.task.has_object:
            if obj is None:
                raise ValueError("Object features are required for this encoder.")
            parts.append(norms.obj(object_features(p, quat, *obj)))
        return self.simnorm(self.encoder(torch.cat(parts, dim=-1)))


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
        # Replaced each iteration by prepare_replay with the length the buffer can supervise.
        self._open_loop = int(self.horizon_pi)
        if resume:
            self.load_checkpoint(resume)

    def _install_latent(self) -> None:
        cfg = self.cfg
        goal_dim = self.task.goal_dim
        wm = LatentWorldModel(
            self.task,
            int(cfg.latent_dim),
            goal_dim,
            cfg.wm_norm,
            cfg.wm_frame,
            simnorm_dim=int(cfg.simnorm_dim),
            wm_step=str(cfg.wm_step),
        ).to(self.device)
        self.wm = wm
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

    # ------------------------------------------------------------------ features

    def _encode(self, wm: LatentWorldModel, p, quat, v, w, q, qd, xi, contact, obj) -> torch.Tensor:
        """``h(s)``. ``xi`` is raw; contact is the recorded per-body bit."""
        return wm.encode(self.norms, self.sim.mdp, p, quat, v, w, q, qd, xi, contact, obj)

    def _recon_scale(self) -> torch.Tensor:
        """Per-component divisors matching the explicit world-model MSE.

        The explicit loss divides ``local_state_error`` by nominal scales of
        0.25 m, 0.5 rad, 1 m/s, 3.14 rad/s, 0.5 rad and 5 rad/s. The decoder
        target is the absolute invariant pose (global xy and yaw are not in the
        latent), each block divided by the scale of that quantity. Gravity and
        6D rotation are already dimensionless, so their scale is 1.
        """
        nj = self.task.num_joints
        if self.wm.frame == "pelvis":
            groups = [(1, 0.25), (3, 1.0), (3, 1.0), (3, 3.14), (nj, 0.5), (nj, 5.0)]
        else:
            groups = [(1, 0.25), (6, 1.0), (3, 1.0), (3, 3.14), (nj, 0.5), (nj, 5.0)]
        groups.append((self.task.body_pos_dim, 0.25))
        if self.task.has_object:
            groups.extend(((3, 0.25), (6, 1.0), (3, 1.0), (3, 3.14)))
        return torch.cat(
            [torch.full((n,), scale, device=self.device) for n, scale in groups]
        )

    def _recon_target(self, state: torch.Tensor, obj: torch.Tensor | None) -> torch.Tensor:
        """Normalized pose features of a real state. No gradient into the target."""
        p, quat, v, w, q, qd = unflatten_state(state)
        body_pos = self.sim.mdp.bodies(p, quat, v, w, q, qd)[0]
        if self.wm.frame == "pelvis":
            feat = pelvis_features(p, quat, v, w, q, qd)
            body = pelvis_body_pos(p, quat, body_pos)
        else:
            feat = invariant_features(p, quat, v, w, q, qd)
            body = heading_body_pos(p, quat, body_pos)
        parts = [feat, body]
        if self.task.has_object:
            if obj is None:
                raise ValueError("Object features are required for this decoder.")
            parts.append(object_features(p, quat, *unflatten_object(obj)))
        raw = torch.cat(parts, dim=-1)
        return (raw / self._recon_scale()).detach()

    def _encode_flat(self, wm: LatentWorldModel, state: torch.Tensor, xi, contact, obj) -> torch.Tensor:
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
        """Consistency against encoded real states, plus reward and termination heads."""
        del log_now
        cfg = self.cfg
        zero = torch.zeros((), device=self.device)
        cons_all, rew_all, term_all, dec_all, contact_all = [], [], [], [], []
        prec_all, recall_all, norms = [], [], []
        rho = float(cfg.rho)
        for _ in range(cfg.wm_batches):
            batch = self.buffer.sample_windows(cfg.wm_batch, self._open_loop)
            reward_t, terminated_t, goal = self._transition_targets(batch)
            b, k, _ = batch["s"].shape
            obj0 = None if "obj" not in batch else batch["obj"][:, 0]
            obj1 = None if "obj1" not in batch else batch["obj1"]
            # Live encodings. The consistency target detaches them so the dynamics
            # cannot collapse by dragging the encoder onto a constant code. The
            # decoder is what forces those codes to keep the pose.
            z = self._encode_flat(self.wm, batch["s"][:, 0], batch["xi"][:, 0], batch["c"][:, 0], obj0)
            z_next = self._encode_flat(
                self.wm,
                batch["s1"].reshape(b * k, -1),
                batch["xi"].reshape(b * k, -1),
                batch["c1"].reshape(b * k, -1),
                None if obj1 is None else obj1.reshape(b * k, -1),
            ).view(b, k, -1)
            pose0 = self._recon_target(batch["s"][:, 0], obj0)
            pose1 = self._recon_target(
                batch["s1"].reshape(b * k, -1),
                None if obj1 is None else obj1.reshape(b * k, -1),
            ).view(b, k, -1)
            dec = (self.wm.decoder(z) - pose0).square().mean()
            contact_logit = [self.wm.contact_head(z)]
            contact_true = [batch["c"][:, 0]]
            cons = zero
            rew = zero
            term_logits = []
            for t in range(k):
                action_n = self.norms.action(batch["a"][:, t])
                goal_n = self.norms.goal(goal[:, t])
                weight = rho**t
                rew = rew + weight * (self.wm.reward(z, action_n, goal_n) - reward_t[:, t]).square().mean()
                z = self.wm.step(z, action_n)
                cons = cons + weight * (z - z_next[:, t].detach()).square().mean()
                dec = dec + weight * (self.wm.decoder(z) - pose1[:, t]).square().mean()
                dec = dec + weight * (self.wm.decoder(z_next[:, t]) - pose1[:, t]).square().mean()
                # Contact of the open-loop latent and of the real next encoding.
                contact_logit.append(self.wm.contact_head(z))
                contact_logit.append(self.wm.contact_head(z_next[:, t]))
                contact_true.append(batch["c1"][:, t])
                contact_true.append(batch["c1"][:, t])
                term_logits.append(self.wm.termination_logit(z, goal_n))
            cons = cons / k
            rew = rew / k
            dec = dec / (2 * k + 1)
            # ``(steps, batch, bodies)`` so the shared contact metrics apply.
            contact_logit_t = torch.stack(contact_logit, dim=0)
            contact_true_t = torch.stack(contact_true, dim=0)
            contact = contact_bce(contact_logit_t, contact_true_t)
            contact_prec, contact_recall = contact_precision_recall(
                contact_logit_t.detach(), contact_true_t
            )
            term_logit = torch.stack(term_logits, dim=1)
            term = F.binary_cross_entropy_with_logits(term_logit, terminated_t)
            loss = (
                cfg.consistency_coef * cons
                + cfg.reward_coef * rew
                + cfg.termination_coef * term
                + cfg.decode_coef * (dec + contact)
            )
            self.opt_wm.zero_grad(set_to_none=True)
            if self.wm_frozen:
                norms.append(0.0)
            else:
                loss.backward()
                norms.append(_clip_grads(self.wm.parameters(), cfg.grad_clip_norm, no_sync=cfg.no_host_sync))
                self.opt_wm.step()
                if self.wm_ema_tau is not None and self.wm_ema is not None:
                    soft_copy_(self.wm, self.wm_ema, self.wm_ema_tau)
            cons_all.append(cons.detach())
            rew_all.append(rew.detach())
            term_all.append(term.detach())
            dec_all.append(dec.detach())
            contact_all.append(contact.detach())
            prec_all.append(contact_prec.detach())
            recall_all.append(contact_recall.detach())
        consistency = torch.stack(cons_all).mean()
        reward = torch.stack(rew_all).mean()
        termination = torch.stack(term_all).mean()
        decode = torch.stack(dec_all).mean()
        decode_contact = torch.stack(contact_all).mean()
        total = (
            cfg.consistency_coef * consistency
            + cfg.reward_coef * reward
            + cfg.termination_coef * termination
            + cfg.decode_coef * (decode + decode_contact)
        )
        disc_stats = {
            "mse": consistency,
            "reward": reward,
            "contact_prec": torch.stack(prec_all).mean(),
            "contact_recall": torch.stack(recall_all).mean(),
            "termination": termination,
            "decode": decode,
            "decode_contact": decode_contact,
        }
        return total, consistency, _mean(norms), disc_stats

    # ------------------------------------------------------------------ imagination

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

    def _latent_rollout(self, start: dict[str, torch.Tensor], horizon: int, wm: LatentWorldModel):
        """H-step latent rollout scored by the analytic reward of the decoded pose.

        The encoder of the real start is detached. Reward and termination are the
        same functions the explicit model uses, evaluated on the pose decoded from
        each latent, so the actor gradient is a tracking gradient.
        """
        z = self._encode_flat(
            wm,
            start["s"],
            start["xi"],
            start["c"],
            start.get("obj"),
        ).detach()
        p, quat, _v, _w, _q, _qd = unflatten_state(start["s"])
        p = p.detach()
        quat = quat.detach()
        obj_prev = None
        if self.task.has_object:
            obj_p, obj_quat, _ov, _ow = unflatten_object(start["obj"])
            obj_prev = (obj_p.detach(), obj_quat.detach())
        frame = start["f"]
        a_prev = start["a_prev"]
        scale = self._recon_scale()
        zs = [z]
        goals = [self.sim.mdp.goal(frame, GOAL_FRAMES)]
        states: list[WorldModelState] = []
        frames = []
        actions = []
        prevs = []
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
                for name, value in out["ep_term"].items():
                    self.stats["ep_terms"][name].append(
                        _term_score(name, (value[done] / length).mean().detach(), style)
                    )
                self.stats["term"].append(out["terminated"][done].float().sum())
                self.stats["trunc"].append(out["truncated"][done].float().sum())
                self.stats["done_n"].append(done.float().sum())


def run(cfg: LatentTrainConfig, task_cfg: DictConfig, log_dir) -> None:
    LatentTrainer(cfg, task_cfg, log_dir).fit()
