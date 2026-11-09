"""Latent world models behind one interface.

Both expose ``encode``, ``step``, and ``compute_loss``. The trainer optimizes
whatever ``compute_loss`` returns and rolls the actor out with ``step``. Loss
weights are fixed when the module is built. The pose decoder is created only
when ``decode_coef`` is positive.

TD-MPC predicts the next code with SimNorm and trains it against a detached
encoding, plus reward, termination, and contact heads. LeWM predicts the next
embedding with an MLP, compares it to the live encoding (the target is not
detached), and adds the sketch isotropic Gaussian regularizer.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

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
from supermimic.supermimic.network import MLP

_LOGIT_EPS = 1e-8
_WM_STEPS = ("absolute", "logit", "residual")
_ARCHES = ("tdmpc", "lewm")


class SimNorm(nn.Module):
    """Split the latent into groups and softmax each group into a simplex.

    A group size of 0 is the identity, so the latent stays an unbounded vector.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim <= 0:
            return x
        groups = x.view(*x.shape[:-1], -1, self.dim)
        return F.softmax(groups, dim=-1).view(x.shape)


class SIGReg(nn.Module):
    """Sketch isotropic Gaussian regularizer."""

    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        self.num_proj = int(num_proj)
        t = torch.linspace(0, 3, int(knots), dtype=torch.float32)
        dt = 3 / (int(knots) - 1)
        weights = torch.full((int(knots),), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """``proj`` is ``(T, B, D)``."""
        direction = torch.randn(proj.size(-1), self.num_proj, device=proj.device, dtype=proj.dtype)
        direction = direction.div_(direction.norm(p=2, dim=0).clamp(min=1e-8))
        x_t = (proj @ direction).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


class LatentWM(nn.Module):
    """Shared proprioceptive encoder and one-step dynamics.

    The pose decoder exists only when ``decode_coef`` is positive. That weight,
    like the other loss weights, is fixed when the module is built.
    """

    arch = ""

    def __init__(self, task, latent_dim: int, norm: str, frame: str, decode_coef: float = 1.0):
        super().__init__()
        if frame not in ("heading", "pelvis"):
            raise ValueError(f"wm frame must be 'heading' or 'pelvis', got {frame!r}.")
        self.task = task
        self.frame = frame
        self.latent_dim = int(latent_dim)
        self.decode_coef = float(decode_coef)
        feat_dim = task.pelvis_feat_dim if frame == "pelvis" else task.inv_dim
        enc_in = feat_dim + task.xi_dim + task.body_pos_dim + task.n_contact + task.object_feat_dim
        hidden = [512, 512]
        self.encoder = MLP(enc_in, hidden, self.latent_dim, norm)
        self.dynamics = MLP(self.latent_dim + task.num_joints, hidden, self.latent_dim, norm)
        self.recon_dim = feat_dim + task.body_pos_dim + task.object_feat_dim
        if self.decode_coef > 0.0:
            self.decoder = MLP(self.latent_dim, hidden, self.recon_dim, norm)
            nn.init.zeros_(self.decoder.head.weight)
            nn.init.zeros_(self.decoder.head.bias)
        else:
            self.decoder = None

    def encode(self, norms, mdp, p, quat, v, w, q, qd, xi, contact, obj) -> torch.Tensor:
        raise NotImplementedError

    def step(self, z: torch.Tensor, action_n: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def compute_loss(self, batch, norms, mdp) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Scalar loss and detached terms. Weights were fixed in ``__init__``."""
        raise NotImplementedError

    def _features(self, norms, mdp, p, quat, v, w, q, qd, xi, contact, obj) -> torch.Tensor:
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
        return torch.cat(parts, dim=-1)

    def encode_flat(self, norms, mdp, state, xi, contact, obj) -> torch.Tensor:
        p, quat, v, w, q, qd = unflatten_state(state)
        obj_t = unflatten_object(obj) if obj is not None else None
        return self.encode(norms, mdp, p, quat, v, w, q, qd, xi, contact, obj_t)

    def feature_scale(self, device: torch.device) -> torch.Tensor:
        nj = self.task.num_joints
        if self.frame == "pelvis":
            groups = [(1, 0.25), (3, 1.0), (3, 1.0), (3, 3.14), (nj, 0.5), (nj, 5.0)]
        else:
            groups = [(1, 0.25), (6, 1.0), (3, 1.0), (3, 3.14), (nj, 0.5), (nj, 5.0)]
        groups.append((self.task.body_pos_dim, 0.25))
        if self.task.has_object:
            groups.extend(((3, 0.25), (6, 1.0), (3, 1.0), (3, 3.14)))
        return torch.cat([torch.full((n,), scale, device=device) for n, scale in groups])

    def pose_target(self, norms_unused, mdp, state: torch.Tensor, obj: torch.Tensor | None) -> torch.Tensor:
        """Normalized pose the decoder is trained to reconstruct. Detached."""
        del norms_unused
        p, quat, v, w, q, qd = unflatten_state(state)
        body_pos = mdp.bodies(p, quat, v, w, q, qd)[0]
        if self.frame == "pelvis":
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
        return (raw / self.feature_scale(raw.device)).detach()


class TDMPCWorldModel(LatentWM):
    """SimNorm latent dynamics. Consistency targets are detached."""

    arch = "tdmpc"

    def __init__(
        self,
        task,
        latent_dim: int,
        goal_dim: int,
        norm: str,
        frame: str,
        simnorm_dim: int = 8,
        wm_step: str = "absolute",
        decode_coef: float = 1.0,
        consistency_coef: float = 1.0,
        reward_coef: float = 1.0,
        termination_coef: float = 1.0,
        rho: float = 1.0,
    ):
        super().__init__(task, latent_dim, norm, frame, decode_coef=decode_coef)
        self.consistency_coef = float(consistency_coef)
        self.reward_coef = float(reward_coef)
        self.termination_coef = float(termination_coef)
        self.rho = float(rho)
        if wm_step not in _WM_STEPS:
            raise ValueError(f"wm_step must be one of {_WM_STEPS}, got {wm_step!r}.")
        self.wm_step = wm_step
        self.simnorm_dim = 0 if wm_step == "residual" else int(simnorm_dim)
        if wm_step == "logit" and self.simnorm_dim <= 0:
            raise ValueError("wm_step='logit' needs simnorm_dim > 0.")
        if self.simnorm_dim < 0:
            raise ValueError(f"simnorm_dim must be >= 0, got {self.simnorm_dim}.")
        if self.simnorm_dim and self.latent_dim % self.simnorm_dim:
            raise ValueError(
                f"latent_dim={self.latent_dim} must be divisible by simnorm_dim={self.simnorm_dim}."
            )
        hidden = [512, 512]
        self.reward_head = MLP(self.latent_dim + task.num_joints + goal_dim, hidden, 1, norm)
        self.termination_head = MLP(self.latent_dim + goal_dim, hidden, 1, norm)
        self.contact_head = MLP(self.latent_dim, hidden, task.n_contact, norm)
        self.simnorm = SimNorm(self.simnorm_dim)
        nn.init.zeros_(self.reward_head.head.weight)
        nn.init.zeros_(self.reward_head.head.bias)
        nn.init.zeros_(self.termination_head.head.weight)
        nn.init.zeros_(self.termination_head.head.bias)
        nn.init.zeros_(self.contact_head.head.weight)
        nn.init.zeros_(self.contact_head.head.bias)
        if wm_step != "absolute":
            nn.init.zeros_(self.dynamics.head.weight)
            nn.init.zeros_(self.dynamics.head.bias)

    def encode(self, norms, mdp, p, quat, v, w, q, qd, xi, contact, obj) -> torch.Tensor:
        return self.simnorm(self.encoder(self._features(norms, mdp, p, quat, v, w, q, qd, xi, contact, obj)))

    def step(self, z: torch.Tensor, action_n: torch.Tensor) -> torch.Tensor:
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

    def compute_loss(self, batch, norms, mdp) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        from supermimic.supermimic.supermimic import contact_bce, contact_precision_recall

        rho = self.rho
        b, k, _ = batch["s"].shape
        obj0 = None if "obj" not in batch else batch["obj"][:, 0]
        obj1 = None if "obj1" not in batch else batch["obj1"]
        reward_t, terminated_t, goal = _transition_targets(mdp, batch)
        z = self.encode_flat(norms, mdp, batch["s"][:, 0], batch["xi"][:, 0], batch["c"][:, 0], obj0)
        z_next = self.encode_flat(
            norms,
            mdp,
            batch["s1"].reshape(b * k, -1),
            batch["xi"].reshape(b * k, -1),
            batch["c1"].reshape(b * k, -1),
            None if obj1 is None else obj1.reshape(b * k, -1),
        ).view(b, k, -1)
        contact_logit = [self.contact_head(z)]
        contact_true = [batch["c"][:, 0]]
        cons = z.new_zeros(())
        rew = z.new_zeros(())
        term_logits = []
        dec = z.new_zeros(())
        pose0 = pose1 = None
        if self.decoder is not None:
            pose0 = self.pose_target(None, mdp, batch["s"][:, 0], obj0)
            pose1 = self.pose_target(
                None,
                mdp,
                batch["s1"].reshape(b * k, -1),
                None if obj1 is None else obj1.reshape(b * k, -1),
            ).view(b, k, -1)
            dec = (self.decoder(z) - pose0).square().mean()
        for t in range(k):
            action_n = norms.action(batch["a"][:, t])
            goal_n = norms.goal(goal[:, t])
            weight = rho**t
            rew = rew + weight * (self.reward(z, action_n, goal_n) - reward_t[:, t]).square().mean()
            z = self.step(z, action_n)
            cons = cons + weight * (z - z_next[:, t].detach()).square().mean()
            if self.decoder is not None:
                dec = dec + weight * (self.decoder(z) - pose1[:, t]).square().mean()
                dec = dec + weight * (self.decoder(z_next[:, t]) - pose1[:, t]).square().mean()
            contact_logit.append(self.contact_head(z))
            contact_logit.append(self.contact_head(z_next[:, t]))
            contact_true.append(batch["c1"][:, t])
            contact_true.append(batch["c1"][:, t])
            term_logits.append(self.termination_logit(z, goal_n))
        cons = cons / k
        rew = rew / k
        contact_logit_t = torch.stack(contact_logit, dim=0)
        contact_true_t = torch.stack(contact_true, dim=0)
        contact = contact_bce(contact_logit_t, contact_true_t)
        contact_prec, contact_recall = contact_precision_recall(contact_logit_t.detach(), contact_true_t)
        term = F.binary_cross_entropy_with_logits(torch.stack(term_logits, dim=1), terminated_t)
        loss = self.consistency_coef * cons + self.reward_coef * rew + self.termination_coef * term
        stats = {
            "mse": cons.detach(),
            "reward": rew.detach(),
            "contact_prec": contact_prec.detach(),
            "contact_recall": contact_recall.detach(),
            "termination": term.detach(),
        }
        if self.decoder is not None:
            dec = dec / (2 * k + 1)
            loss = loss + self.decode_coef * (dec + contact)
            stats["decode"] = dec.detach()
            stats["decode_contact"] = contact.detach()
        return loss, stats


class LeWM(LatentWM):
    """Next-embedding prediction without a stop-gradient, plus SIGReg.

    The dynamics are an MLP on ``(z, a)``. There is no SimNorm: the regularizer
    is what keeps the embedding from collapsing. The reward head, contact head,
    and pose decoder are created only when their coefficients are positive. The
    termination head is always created and trained against the real fall.
    """

    arch = "lewm"

    def __init__(
        self,
        task,
        latent_dim: int,
        goal_dim: int,
        norm: str,
        frame: str,
        sigreg_knots: int = 17,
        sigreg_proj: int = 1024,
        decode_coef: float = 1.0,
        sigreg_coef: float = 0.09,
        reward_coef: float = 0.0,
        contact_coef: float = 0.0,
        termination_coef: float = 1.0,
    ):
        super().__init__(task, latent_dim, norm, frame, decode_coef=decode_coef)
        hidden = [512, 512]
        self.sigreg_coef = float(sigreg_coef)
        self.reward_coef = float(reward_coef)
        self.contact_coef = float(contact_coef)
        self.termination_coef = float(termination_coef)
        self.sigreg = SIGReg(sigreg_knots, sigreg_proj)
        self.reward_head = None
        self.contact_head = None
        self.termination_head = MLP(self.latent_dim + goal_dim, hidden, 1, norm)
        nn.init.zeros_(self.termination_head.head.weight)
        nn.init.zeros_(self.termination_head.head.bias)
        if self.reward_coef > 0.0:
            self.reward_head = MLP(self.latent_dim + task.num_joints + goal_dim, hidden, 1, norm)
            nn.init.zeros_(self.reward_head.head.weight)
            nn.init.zeros_(self.reward_head.head.bias)
        if self.contact_coef > 0.0:
            self.contact_head = MLP(self.latent_dim, hidden, task.n_contact, norm)
            nn.init.zeros_(self.contact_head.head.weight)
            nn.init.zeros_(self.contact_head.head.bias)
        nn.init.zeros_(self.dynamics.head.weight)
        nn.init.zeros_(self.dynamics.head.bias)

    def encode(self, norms, mdp, p, quat, v, w, q, qd, xi, contact, obj) -> torch.Tensor:
        return self.encoder(self._features(norms, mdp, p, quat, v, w, q, qd, xi, contact, obj))

    def step(self, z: torch.Tensor, action_n: torch.Tensor) -> torch.Tensor:
        return self.dynamics(torch.cat((z, action_n), dim=-1))

    def reward(self, z: torch.Tensor, action_n: torch.Tensor, goal_n: torch.Tensor) -> torch.Tensor:
        return self.reward_head(torch.cat((z, action_n, goal_n), dim=-1)).squeeze(-1)

    def termination_logit(self, z: torch.Tensor, goal_n: torch.Tensor) -> torch.Tensor:
        return self.termination_head(torch.cat((z, goal_n), dim=-1)).squeeze(-1)

    def compute_loss(self, batch, norms, mdp) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Open-loop next-embedding loss, SIGReg, and decode of the predicted codes.

        The rollout starts at ``h(s_0)`` and applies ``step`` with the stored
        actions, which is the chain the actor differentiates. Each predicted code
        is matched to ``h(s_t)`` (target not detached) and, when a decoder exists,
        decoded onto the real future pose.
        """
        b, k, _ = batch["s"].shape
        obj_seq = None if "obj" not in batch else torch.cat((batch["obj"][:, :1], batch["obj1"]), dim=1)
        s_seq = torch.cat((batch["s"][:, :1], batch["s1"]), dim=1)
        xi_seq = torch.cat((batch["xi"][:, :1], batch["xi"]), dim=1)
        c_seq = torch.cat((batch["c"][:, :1], batch["c1"]), dim=1)
        length = k + 1
        flat = b * length
        emb = self.encode_flat(
            norms,
            mdp,
            s_seq.reshape(flat, -1),
            xi_seq.reshape(flat, -1),
            c_seq.reshape(flat, -1),
            None if obj_seq is None else obj_seq.reshape(flat, -1),
        ).view(b, length, -1)
        action_n = norms.action(batch["a"])
        z = emb[:, 0]
        predicted = []
        for t in range(k):
            z = self.step(z, action_n[:, t])
            predicted.append(z)
        pred = torch.stack(predicted, dim=1)
        # The target embedding stays in the graph. SIGReg is the anti-collapse term.
        pred_loss = (pred - emb[:, 1:]).pow(2).mean()
        _reward_t, terminated_t, goal = _transition_targets(mdp, batch)
        term_logits = [
            self.termination_logit(pred[:, t], norms.goal(goal[:, t])) for t in range(k)
        ]
        term = F.binary_cross_entropy_with_logits(torch.stack(term_logits, dim=1), terminated_t)
        sigreg = self.sigreg(emb.transpose(0, 1))
        loss = pred_loss + self.sigreg_coef * sigreg + self.termination_coef * term
        stats = {"mse": pred_loss.detach(), "sigreg": sigreg.detach(), "termination": term.detach()}
        # Latents the actor will actually read: the real start, then each open-loop step.
        z_chain = torch.cat((emb[:, :1], pred), dim=1)
        if self.decoder is not None:
            poses = []
            for t in range(length):
                obj_t = None if obj_seq is None else obj_seq[:, t]
                poses.append(self.pose_target(None, mdp, s_seq[:, t], obj_t))
            pose = torch.stack(poses, dim=1)
            dec_real = (self.decoder(emb) - pose).square().mean()
            dec_open = (self.decoder(pred) - pose[:, 1:]).square().mean()
            decode = 0.5 * (dec_real + dec_open)
            loss = loss + self.decode_coef * decode
            stats["decode"] = decode.detach()
            stats["decode_open"] = dec_open.detach()
        if self.reward_head is not None:
            reward_t = _reward_t
            z_pre = z_chain[:, :-1].reshape(b * k, -1)
            rew = self.reward(z_pre, action_n.reshape(b * k, -1), norms.goal(goal).reshape(b * k, -1))
            rew = (rew.view(b, k) - reward_t).square().mean()
            loss = loss + self.reward_coef * rew
            stats["reward"] = rew.detach()
        if self.contact_head is not None:
            from supermimic.supermimic.supermimic import contact_bce, contact_precision_recall

            logits = self.contact_head(z_chain).transpose(0, 1)
            target = c_seq.transpose(0, 1)
            contact = contact_bce(logits, target)
            prec, recall = contact_precision_recall(logits.detach(), target)
            loss = loss + self.contact_coef * contact
            stats["decode_contact"] = contact.detach()
            stats["contact_prec"] = prec.detach()
            stats["contact_recall"] = recall.detach()
        return loss, stats


def build_latent_wm(cfg, task, goal_dim: int) -> LatentWM:
    """``cfg`` is the train config or a checkpoint dict."""
    def get(name, default=None):
        if isinstance(cfg, dict):
            return cfg.get(name, default)
        return getattr(cfg, name)

    arch = str(get("wm_arch", "tdmpc"))
    if arch == "tdmpc":
        return TDMPCWorldModel(
            task,
            int(get("latent_dim", 256)),
            goal_dim,
            str(get("wm_norm", "rms")),
            str(get("wm_frame", "pelvis")),
            simnorm_dim=int(get("simnorm_dim", 8)),
            wm_step=str(get("wm_step", "absolute")),
            decode_coef=float(get("decode_coef", 1.0)),
            consistency_coef=float(get("consistency_coef", 1.0)),
            reward_coef=float(get("reward_coef", 1.0)),
            termination_coef=float(get("termination_coef", 1.0)),
            rho=float(get("rho", 1.0)),
        )
    if arch == "lewm":
        return LeWM(
            task,
            int(get("latent_dim", 256)),
            goal_dim,
            str(get("wm_norm", "rms")),
            str(get("wm_frame", "pelvis")),
            decode_coef=float(get("decode_coef", 1.0)),
            sigreg_coef=float(get("sigreg_coef", 0.09)),
            reward_coef=float(get("lewm_reward_coef", 0.0)),
            contact_coef=float(get("lewm_contact_coef", 0.0)),
            termination_coef=float(get("termination_coef", 1.0)),
        )
    raise ValueError(f"wm_arch must be one of {_ARCHES}, got {arch!r}.")


def _transition_targets(mdp, batch):
    """Analytic reward, hard termination, and goal of each stored transition."""
    s1 = batch["s1"]
    b, k, _ = s1.shape
    flat = b * k
    p, quat, v, w, q, qd = unflatten_state(s1.reshape(flat, -1))
    frame = batch["f"]
    last = mdp.episode_last(frame)
    f_next = torch.minimum(frame + 1, last)
    obj = unflatten_object(batch["obj1"].reshape(flat, -1)) if "obj1" in batch else None
    if obj is None:
        state = WorldModelState(p, quat, v, w, q, qd, None, None)
    else:
        state = WorldModelState(p, quat, v, w, q, qd, None, None, *obj)
    reward = mdp.reward(
        state,
        f_next.reshape(flat),
        batch["a"].reshape(flat, -1),
        batch["a_prev"].reshape(flat, -1),
    ).view(b, k)
    terminated = mdp.termination(p, quat, f_next.reshape(flat)).view(b, k).float()
    goal = mdp.goal(f_next.reshape(flat), GOAL_FRAMES).view(b, k, -1)
    return reward, terminated, goal
