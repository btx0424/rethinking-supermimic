"""Modular reward terms composed by ``MDP.reward``."""

from __future__ import annotations

import torch

from .math_utils import quat_error_angle
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
)


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
        if getattr(cls, "penalty", False):
            terms.append(cls(mdp, weight=pen_weights[name]))
        else:
            terms.append(cls(mdp, weight=style_weights[name], scale=REWARD_TERM_SCALE[name]))
    return terms
