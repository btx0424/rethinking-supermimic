"""Seven name-agnostic tracking terms, in physical units, multiplied by dt."""

import torch
from supermimic.utils.math import quat_apply, quat_inv, quat_mul, yaw_quat, quat_error_angle
from supermimic.env.mdp.geometry.math import matrix_to_quat
from . import CONTROL_DT


def _anchor_relative(robot_p, robot_quat, ref_p, ref_quat, anchor: int):
    """BeyondMimic anchor: keep the reference's height, take the robot's yaw and xy."""
    robot_anchor_p = robot_p[..., anchor, :]
    robot_anchor_q = robot_quat[..., anchor, :]
    ref_anchor_p = ref_p[..., anchor, :]
    ref_anchor_q = ref_quat[..., anchor, :]
    delta_pos = robot_anchor_p.clone()
    delta_pos = torch.cat((delta_pos[..., :2], ref_anchor_p[..., 2:3]), dim=-1)
    delta_ori = yaw_quat(quat_mul(robot_anchor_q, quat_inv(ref_anchor_q)))
    dpos = delta_pos.unsqueeze(-2)
    dori = delta_ori.unsqueeze(-2)
    ref_anchor = ref_anchor_p.unsqueeze(-2)
    pos = dpos + quat_apply(dori.expand(*ref_p.shape[:-1], 4), ref_p - ref_anchor)
    ori = quat_mul(dori.expand(*ref_quat.shape[:-1], 4), ref_quat)
    return pos, ori


TERM_NAMES = (
    "body_pos",
    "body_ori",
    "body_lin_vel",
    "body_ang_vel",
    "root_pos",
    "root_ori",
    "action_rate",
)


def reward_terms(representation, state, motion, frame, action, previous, cfg):
    p, rot, v, w = representation.world_frame_bodies(state.aux, state.f)
    q = matrix_to_quat(rot)
    ref_p = motion.body_p[frame]
    ref_q = motion.body_quat[frame]
    anchored_p, anchored_q = _anchor_relative(p, q, ref_p, ref_q, 0)
    errors = {
        "body_pos": (p - anchored_p).square().sum(-1).mean(-1),
        "body_ori": quat_error_angle(anchored_q, q).square().mean(-1),
        "body_lin_vel": (v - motion.body_v[frame]).square().sum(-1).mean(-1),
        "body_ang_vel": (w - motion.body_w[frame]).square().sum(-1).mean(-1),
        "root_pos": (state.aux[..., :3] - motion.root_p[frame]).square().sum(-1),
        "root_ori": quat_error_angle(motion.root_quat[frame], state.aux[..., 3:7]).square(),
    }
    terms = {
        key: cfg.reward_weights[key] * torch.exp(-err / cfg.reward_sigmas[key] ** 2) * CONTROL_DT
        for key, err in errors.items()
        if key in cfg.reward_weights
    }
    if "action_rate" in cfg.reward_weights:
        terms["action_rate"] = (
            cfg.reward_weights["action_rate"] * (action - previous).square().sum(-1) * CONTROL_DT
        )
    return terms
