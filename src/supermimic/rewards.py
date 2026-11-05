"""Modular reward terms composed by ``MDP.reward``."""

from __future__ import annotations

import torch

from .utils.math import quat_conjugate, quat_error_angle, quat_apply_inverse, quat_mul, quat_to_rotvec
from .mdp import (
    CONTROL_DT,
    REWARD_TERM_SCALE,
    REWARD_TERM_WEIGHT,
    RewardTerm,
    WorldModelState,
    _anchor_relative,
    cauchy_kernel,
    gaussian_kernel,
    gaussian_linear_kernel,
    hand_obj_relative_pos,
    hard_joint_violation,
    previous_frame,
)

# Translation-equivalent length of a rotation (m / rad), shared by object_motion progress.
_ROTATION_LENGTH = 0.15


def _object_increment(p_prev, q_prev, p, q):
    """Position delta and world-frame rotation vector of ``q * q_prev^-1``."""
    return p - p_prev, quat_to_rotvec(quat_mul(q, quat_conjugate(q_prev)))


def _object_tracking_components(dp, dr, target_dp, target_dr, pos_err, rot_err, vel_err, ang_vel_err, surrogate: bool):
    """ScaleTrack ``tracking_components``: progress along the reference object increment.

    Sim (``surrogate=False``) gates on both reference and object motion so a resting
    object scores 0 with no gradient. Imagination (``surrogate=True``) keeps only the
    reference gate so a still predicted object still gets a gradient toward moving.
    The scalar is 0 in both cases when the object is still; only the gradient differs.
    """
    actual = torch.cat([dp, _ROTATION_LENGTH * dr], -1)
    target = torch.cat([target_dp, _ROTATION_LENGTH * target_dr], -1)
    energy = target.square().sum(-1)
    alignment = (actual * target).sum(-1)
    # Branch-free gate so ``torch.compile`` does not specialize on the bool.
    ref_active = energy > 1e-8
    obj_active = actual.square().sum(-1) > 1e-8
    surrogate_t = energy.new_tensor(1.0 if surrogate else 0.0)
    active = ref_active & (obj_active | (surrogate_t > 0.5))
    # Cap at 1 from above; zero negative progress with ``where`` so a resting object
    # (alignment == 0) still has ∂progress/∂actual under the surrogate. ``clamp(0, 1)``
    # would kill that boundary gradient in PyTorch.
    raw = alignment / energy.clamp_min(1e-12)
    capped = raw.clamp(max=1.0)
    nonneg = torch.where(raw >= 0, capped, torch.zeros_like(raw))
    progress = torch.where(active, nonneg, torch.zeros_like(energy))
    position = 0.5 * torch.exp(-(pos_err / 0.4).square()) + 0.5 * torch.exp(-(pos_err / 0.15).square())
    rotation = torch.exp(-(rot_err / 0.4).square())
    velocity = torch.exp(-(vel_err / 1.0).square())
    angular = torch.exp(-(ang_vel_err / 3.14).square())
    score = progress * position * (0.5 + 0.25 * rotation + 0.125 * velocity + 0.125 * angular)
    return score, progress, position


def _relation_error(body_pos, ref_body_pos, obj_p, obj_quat, ref_p, ref_quat) -> torch.Tensor:
    """Weighted body positions in the object frame vs the demo (ScaleTrack ``relation_error``)."""
    p = obj_p.unsqueeze(-2)
    q = obj_quat.unsqueeze(-2)
    rp = ref_p.unsqueeze(-2)
    rq = ref_quat.unsqueeze(-2)
    actual = quat_apply_inverse(q.expand(*body_pos.shape[:-1], 4), body_pos - p)
    reference = quat_apply_inverse(rq.expand(*ref_body_pos.shape[:-1], 4), ref_body_pos - rp)
    weight = torch.exp(-reference.square().sum(-1) / 0.4**2)
    return ((actual - reference).square().sum(-1) * weight).sum(-1) / weight.sum(-1).clamp_min(1e-8)


def _object_motion_parts(mdp, state: WorldModelState, frame: torch.Tensor):
    """Shared object_motion / object_relation pieces for one step ending at ``frame``."""
    if state.obj_p is None or state.obj_quat is None or state.obj_v is None or state.obj_w is None:
        raise ValueError("object_motion reward needs state.obj_*.")
    obj_p_prev, obj_quat_prev = state.object_prev()
    if state.body_pos is None:
        raise ValueError("object_relation reward needs state.body_pos.")
    motion = mdp.motion
    frame_prev = previous_frame(motion, frame)
    dp, dr = _object_increment(obj_p_prev, obj_quat_prev, state.obj_p, state.obj_quat)
    tdp, tdr = _object_increment(
        motion.object_p[frame_prev],
        motion.object_quat[frame_prev],
        motion.object_p[frame],
        motion.object_quat[frame],
    )
    pos_err = (state.obj_p - motion.object_p[frame]).norm(dim=-1)
    rot_err = quat_error_angle(state.obj_quat, motion.object_quat[frame])
    vel_err = (state.obj_v - motion.object_v[frame]).norm(dim=-1)
    ang_err = (state.obj_w - motion.object_w[frame]).norm(dim=-1)
    score, progress, position = _object_tracking_components(
        dp, dr, tdp, tdr, pos_err, rot_err, vel_err, ang_err, mdp.object_reward_surrogate
    )
    relation = progress * position * torch.exp(
        -_relation_error(
            state.body_pos,
            motion.body_p[frame],
            state.obj_p,
            state.obj_quat,
            motion.object_p[frame],
            motion.object_quat[frame],
        )
        / 0.15**2
    )
    return score, relation


def _kernel_fn(style: str):
    if style == "cauchy":
        return cauchy_kernel
    if style == "gaussian_linear":
        return gaussian_linear_kernel
    return gaussian_kernel


class _TrackingReward(RewardTerm):
    """Gaussian / cauchy / gaussian_linear kernel on a squared error."""

    penalty = False

    def error_sq(self, state: WorldModelState, action, a_prev, frame) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, state, action, a_prev, frame):
        assert self.scale is not None
        kernel = _kernel_fn(self.mdp.reward_style)
        return self.weight * kernel(self.error_sq(state, action, a_prev, frame), self.scale) * CONTROL_DT


class BodyPos(_TrackingReward):
    name = "body_pos"

    def error_sq(self, state, action, a_prev, frame):
        if state.body_pos is None or state.body_quat is None:
            raise ValueError("body_pos reward needs body poses on the state.")
        ref_pos, _ = _anchor_relative(
            state.body_pos,
            state.body_quat,
            self.mdp.motion.body_p[frame],
            self.mdp.motion.body_quat[frame],
            self.mdp.anchor,
        )
        return (ref_pos - state.body_pos).square().sum(-1).mean(-1)


class BodyOri(_TrackingReward):
    name = "body_ori"

    def error_sq(self, state, action, a_prev, frame):
        if state.body_pos is None or state.body_quat is None:
            raise ValueError("body_ori reward needs body poses on the state.")
        _, ref_ori = _anchor_relative(
            state.body_pos,
            state.body_quat,
            self.mdp.motion.body_p[frame],
            self.mdp.motion.body_quat[frame],
            self.mdp.anchor,
        )
        return quat_error_angle(ref_ori, state.body_quat).square().mean(-1)


class BodyLinVel(_TrackingReward):
    name = "body_lin_vel"

    def error_sq(self, state, action, a_prev, frame):
        if state.body_lin_vel is None:
            raise ValueError("body_lin_vel reward needs state.body_lin_vel.")
        return (self.mdp.motion.body_v[frame] - state.body_lin_vel).square().sum(-1).mean(-1)


class BodyAngVel(_TrackingReward):
    name = "body_ang_vel"

    def error_sq(self, state, action, a_prev, frame):
        if state.body_ang_vel is None:
            raise ValueError("body_ang_vel reward needs state.body_ang_vel.")
        return (self.mdp.motion.body_w[frame] - state.body_ang_vel).square().sum(-1).mean(-1)


class RootPos(_TrackingReward):
    name = "root_pos"

    def error_sq(self, state, action, a_prev, frame):
        return (self.mdp.motion.root_p[frame] - state.p).square().sum(-1)


class RootOri(_TrackingReward):
    name = "root_ori"

    def error_sq(self, state, action, a_prev, frame):
        return quat_error_angle(self.mdp.motion.root_quat[frame], state.quat).square()


class ObjectPos(_TrackingReward):
    name = "object_pos"

    def error_sq(self, state, action, a_prev, frame):
        if state.obj_p is None:
            raise ValueError("object_pos reward needs state.obj_p.")
        return (self.mdp.motion.object_p[frame] - state.obj_p).square().sum(-1)


class ObjectOri(_TrackingReward):
    name = "object_ori"

    def error_sq(self, state, action, a_prev, frame):
        if state.obj_quat is None:
            raise ValueError("object_ori reward needs state.obj_quat.")
        return quat_error_angle(self.mdp.motion.object_quat[frame], state.obj_quat).square()


class ObjectLinVel(_TrackingReward):
    name = "object_lin_vel"

    def error_sq(self, state, action, a_prev, frame):
        if state.obj_v is None:
            raise ValueError("object_lin_vel reward needs state.obj_v.")
        return (self.mdp.motion.object_v[frame] - state.obj_v).square().sum(-1)


class ObjectAngVel(_TrackingReward):
    name = "object_ang_vel"

    def error_sq(self, state, action, a_prev, frame):
        if state.obj_w is None:
            raise ValueError("object_ang_vel reward needs state.obj_w.")
        return (self.mdp.motion.object_w[frame] - state.obj_w).square().sum(-1)


class ObjectMotion(RewardTerm):
    """Increment-gated object tracking (ScaleTrack ``object_motion``)."""

    name = "object_motion"
    penalty = False
    scale = None

    def __call__(self, state, action, a_prev, frame):
        score, relation = _object_motion_parts(self.mdp, state, frame)
        # Stash for ``ObjectRelation`` in the same ``MDP.reward`` call.
        self.mdp._object_reward_parts = (score, relation)
        return self.weight * score * CONTROL_DT


class ObjectRelation(RewardTerm):
    """Body–object spatial relation gated by object progress (ScaleTrack ``object_relation``)."""

    name = "object_relation"
    penalty = False
    scale = None

    def __call__(self, state, action, a_prev, frame):
        parts = getattr(self.mdp, "_object_reward_parts", None)
        if parts is None:
            _, relation = _object_motion_parts(self.mdp, state, frame)
        else:
            _, relation = parts
        return self.weight * relation * CONTROL_DT


class HandObjPos(_TrackingReward):
    name = "hand_obj_pos"

    def error_sq(self, state, action, a_prev, frame):
        if state.obj_p is None or state.obj_quat is None:
            raise ValueError("hand_obj_pos reward needs object pose on the state.")
        hand_pos = state.hand_pos
        if hand_pos is None:
            hand_pos = self.mdp.kin.hands(*state.root())
        motion = self.mdp.motion
        ref_local = hand_obj_relative_pos(
            motion.hand_p[frame], motion.object_p[frame], motion.object_quat[frame]
        )
        sim_local = hand_obj_relative_pos(hand_pos, state.obj_p, state.obj_quat)
        return (ref_local - sim_local).square().sum(-1).mean(-1)


class ActionRate(RewardTerm):
    name = "action_rate"
    penalty = True
    scale = None

    def __call__(self, state, action, a_prev, frame):
        # Penalties always use the gaussian table weights (paper), independent of style.
        return -self.weight * (action - a_prev).square().sum(-1) * CONTROL_DT


class JointLimit(RewardTerm):
    name = "joint_limit"
    penalty = True
    scale = None

    def __call__(self, state, action, a_prev, frame):
        return (
            -self.weight
            * hard_joint_violation(state.q, self.mdp.soft_lo, self.mdp.soft_hi)
            * CONTROL_DT
        )


REWARD_TERM_CLASSES: dict[str, type[RewardTerm]] = {
    "body_pos": BodyPos,
    "body_ori": BodyOri,
    "body_lin_vel": BodyLinVel,
    "body_ang_vel": BodyAngVel,
    "root_pos": RootPos,
    "root_ori": RootOri,
    "object_pos": ObjectPos,
    "object_ori": ObjectOri,
    "object_lin_vel": ObjectLinVel,
    "object_ang_vel": ObjectAngVel,
    "object_motion": ObjectMotion,
    "object_relation": ObjectRelation,
    "hand_obj_pos": HandObjPos,
    "action_rate": ActionRate,
    "joint_limit": JointLimit,
}


def build_reward_terms(mdp, names: tuple[str, ...]) -> list[RewardTerm]:
    style_weights = REWARD_TERM_WEIGHT[mdp.reward_style]
    # Penalties always use the paper (gaussian) weight table.
    pen_weights = REWARD_TERM_WEIGHT["gaussian"]
    terms: list[RewardTerm] = []
    for name in names:
        cls = REWARD_TERM_CLASSES.get(name)
        if cls is None:
            raise KeyError(f"No reward term class for {name!r}.")
        if getattr(cls, "penalty", False) or name in ("object_motion", "object_relation"):
            weight = pen_weights[name] if getattr(cls, "penalty", False) else style_weights[name]
            terms.append(cls(mdp, weight=weight))
        else:
            terms.append(cls(mdp, weight=style_weights[name], scale=REWARD_TERM_SCALE[name]))
    return terms
