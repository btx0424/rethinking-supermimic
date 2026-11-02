"""Modular policy observation terms composed by ``MDP.obs``."""

from __future__ import annotations

import torch

from .math_utils import (
    mat_to_6d,
    quat_apply_inverse,
    quat_inv,
    quat_mul,
    quat_to_mat,
    yaw_quat,
)
from .mdp import ObservationTerm, WorldModelState, _anchor_relative


class JointPos(ObservationTerm):
    name = "joint_pos"
    noise = 0.01

    def __init__(self, mdp):
        super().__init__(mdp)
        self.dim = mdp.task.num_joints

    def __call__(self, state, a_prev, xi, frame=None):
        measured = state.q + xi[..., 1 : 1 + self.mdp.task.num_joints]
        return measured - self.mdp.default_q


class JointVel(ObservationTerm):
    name = "joint_vel"
    noise = 0.5

    def __init__(self, mdp):
        super().__init__(mdp)
        self.dim = mdp.task.num_joints

    def __call__(self, state, a_prev, xi, frame=None):
        return state.qd


class BaseAngVel(ObservationTerm):
    name = "base_ang_vel"
    dim = 3
    noise = 0.2

    def __call__(self, state, a_prev, xi, frame=None):
        return quat_apply_inverse(state.quat, state.w)


class BaseLinVel(ObservationTerm):
    name = "base_lin_vel"
    dim = 3
    noise = 0.2

    def __call__(self, state, a_prev, xi, frame=None):
        return quat_apply_inverse(state.quat, state.v)


class ProjectedGravity(ObservationTerm):
    name = "projected_gravity"
    dim = 3
    noise = 0.05

    def __call__(self, state, a_prev, xi, frame=None):
        return quat_apply_inverse(state.quat, self.mdp.gravity.expand_as(state.w))


class LastAction(ObservationTerm):
    name = "last_action"
    noise = 0.0

    def __init__(self, mdp):
        super().__init__(mdp)
        self.dim = mdp.task.num_joints

    def __call__(self, state, a_prev, xi, frame=None):
        return a_prev


class AnchorOri(ObservationTerm):
    name = "anchor_ori"
    dim = 6
    noise = 0.05

    def __call__(self, state, a_prev, xi, frame=None):
        if frame is None:
            raise ValueError("The anchor_ori observation needs the reference frame.")
        mdp = self.mdp
        if mdp.anchor == mdp.kin.pelvis_index:
            robot_quat = state.quat
        else:
            if state.body_quat is None:
                raise ValueError("anchor_ori needs state.body_quat.")
            robot_quat = state.body_quat[..., mdp.anchor, :]
        ref_quat = mdp.motion.body_quat[frame][..., mdp.anchor, :]
        relative = quat_mul(quat_inv(robot_quat), ref_quat)
        return mat_to_6d(quat_to_mat(relative))


class AnchorPos(ObservationTerm):
    name = "anchor_pos"
    dim = 3
    noise = 0.01

    def __call__(self, state, a_prev, xi, frame=None):
        if frame is None:
            raise ValueError("The anchor_pos observation needs the reference frame.")
        mdp = self.mdp
        if mdp.anchor == mdp.kin.pelvis_index:
            robot_p = state.p
        else:
            if state.body_pos is None:
                raise ValueError("anchor_pos needs state.body_pos.")
            robot_p = state.body_pos[..., mdp.anchor, :]
        ref_p = mdp.motion.body_p[frame][..., mdp.anchor, :]
        return quat_apply_inverse(yaw_quat(state.quat), ref_p - robot_p)


class BodyPos(ObservationTerm):
    name = "body_pos"
    noise = 0.01

    def __init__(self, mdp):
        super().__init__(mdp)
        self.dim = mdp.task.n_body * 3

    def __call__(self, state, a_prev, xi, frame=None):
        if frame is None:
            raise ValueError("The body_pos observation needs the reference frame.")
        if state.body_pos is None or state.body_quat is None:
            raise ValueError("body_pos needs state.body_pos and state.body_quat.")
        mdp = self.mdp
        ref_pos, _ = _anchor_relative(
            state.body_pos,
            state.body_quat,
            mdp.motion.body_p[frame],
            mdp.motion.body_quat[frame],
            mdp.anchor,
        )
        error = quat_apply_inverse(yaw_quat(state.quat).unsqueeze(-2), ref_pos - state.body_pos)
        return error.flatten(-2)


class BodyOri(ObservationTerm):
    name = "body_ori"
    noise = 0.05

    def __init__(self, mdp):
        super().__init__(mdp)
        self.dim = mdp.task.n_body * 6

    def __call__(self, state, a_prev, xi, frame=None):
        if frame is None:
            raise ValueError("The body_ori observation needs the reference frame.")
        if state.body_pos is None or state.body_quat is None:
            raise ValueError("body_ori needs state.body_pos and state.body_quat.")
        mdp = self.mdp
        _, ref_ori = _anchor_relative(
            state.body_pos,
            state.body_quat,
            mdp.motion.body_p[frame],
            mdp.motion.body_quat[frame],
            mdp.anchor,
        )
        relative = quat_mul(quat_inv(state.body_quat), ref_ori)
        return mat_to_6d(quat_to_mat(relative)).flatten(-2)


class ObjectPos(ObservationTerm):
    name = "object_pos"
    dim = 3
    noise = 0.01

    def __call__(self, state, a_prev, xi, frame=None):
        if state.obj_p is None:
            raise ValueError("object_pos needs state.obj_p.")
        yaw = yaw_quat(state.quat)
        return quat_apply_inverse(yaw, state.obj_p - state.p)


class ObjectOri(ObservationTerm):
    name = "object_ori"
    dim = 6
    noise = 0.05

    def __call__(self, state, a_prev, xi, frame=None):
        if state.obj_quat is None:
            raise ValueError("object_ori needs state.obj_quat.")
        relative = quat_mul(quat_inv(yaw_quat(state.quat)), state.obj_quat)
        return mat_to_6d(quat_to_mat(relative))


OBS_TERM_CLASSES: dict[str, type[ObservationTerm]] = {
    "joint_pos": JointPos,
    "joint_vel": JointVel,
    "base_ang_vel": BaseAngVel,
    "base_lin_vel": BaseLinVel,
    "projected_gravity": ProjectedGravity,
    "last_action": LastAction,
    "anchor_ori": AnchorOri,
    "anchor_pos": AnchorPos,
    "body_pos": BodyPos,
    "body_ori": BodyOri,
    "object_pos": ObjectPos,
    "object_ori": ObjectOri,
}


def build_obs_terms(mdp, names: tuple[str, ...]) -> list[ObservationTerm]:
    missing = [name for name in names if name not in OBS_TERM_CLASSES]
    if missing:
        raise KeyError(f"No observation term class for {missing}.")
    return [OBS_TERM_CLASSES[name](mdp) for name in names]
