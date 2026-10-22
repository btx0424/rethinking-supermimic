"""Analytic observation, reward, and termination shared by the simulator and the world model."""

from __future__ import annotations

import torch

from .kinematics import TorchKinematics
from .math_utils import (
    mat_to_6d,
    quat_apply,
    quat_apply_inverse,
    quat_error_angle,
    quat_inv,
    quat_mul,
    quat_to_mat,
    quat_to_rotvec,
    yaw_quat,
)
from .motion import CONTROL_DT, GOAL_PER_FRAME, MotionLib

NUM_J = 29
OBS_DIM = NUM_J + NUM_J + 3 + 3 + NUM_J
XI_DIM = 1 + NUM_J + 3
INV_DIM = 1 + 6 + 3 + 3 + NUM_J + NUM_J
STATE_DIM = 3 + 4 + 3 + 3 + NUM_J + NUM_J
HIST = 10
GOAL_FRAMES = 10
GOAL_DIM = GOAL_PER_FRAME * GOAL_FRAMES
GRAVITY = (0.0, 0.0, -1.0)
# Per-component scale of a state error. Eq. 1 is an unweighted square, which is
# dominated by whichever unit is largest; these nominal errors balance the six
# groups and are the normalizer for the logged 8-step prediction error.
NOMINAL_ERROR = (
    (3, 0.25),
    (3, 0.5),
    (3, 1.0),
    (3, 3.14),
    (NUM_J, 0.5),
    (NUM_J, 5.0),
)
MAX_STEP_REWARD = (4.0 * 1.0 + 2.0 * 0.5) * CONTROL_DT

OBS_NOISE = torch.cat(
    (
        torch.full((NUM_J,), 0.01),
        torch.full((NUM_J,), 0.5),
        torch.full((3,), 0.2),
        torch.full((3,), 0.05),
        torch.zeros(NUM_J),
    )
)


def nominal_error(device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.cat(
        [torch.full((n,), scale, device=device, dtype=dtype) for n, scale in NOMINAL_ERROR]
    )


def flatten_state(p, quat, v, w, q, qd) -> torch.Tensor:
    return torch.cat((p, quat, v, w, q, qd), dim=-1)


def unflatten_state(s: torch.Tensor):
    return (
        s[..., 0:3],
        s[..., 3:7],
        s[..., 7:10],
        s[..., 10:13],
        s[..., 13:42],
        s[..., 42:71],
    )


def invariant_features(p, quat, v, w, q, qd) -> torch.Tensor:
    """Heading-invariant state: height, tilt, body-frame twist, joints."""
    yaw = yaw_quat(quat)
    tilt = quat_mul(quat_inv(yaw), quat)
    return torch.cat(
        (
            p[..., 2:3],
            mat_to_6d(quat_to_mat(tilt)),
            quat_apply_inverse(quat, v),
            quat_apply_inverse(quat, w),
            q,
            qd,
        ),
        dim=-1,
    )


def normalize_xi(xi: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        ((xi[..., :1] - 0.75) / 0.45, xi[..., 1:30] / 0.01, xi[..., 30:33] / 0.05),
        dim=-1,
    )


def local_state_error(p, quat, v, w, q, qd, p_hat, quat_hat, v_hat, w_hat, q_hat, qd_hat) -> torch.Tensor:
    """State error of the hat quantities, expressed in the true state's frames."""
    yaw = yaw_quat(quat)
    return torch.cat(
        (
            quat_apply_inverse(yaw, p_hat - p),
            quat_to_rotvec(quat_mul(quat_inv(quat), quat_hat)),
            quat_apply_inverse(quat, v_hat - v),
            quat_apply_inverse(quat, w_hat - w),
            q_hat - q,
            qd_hat - qd,
        ),
        dim=-1,
    )


class MDP:
    def __init__(
        self,
        kin: TorchKinematics,
        motion: MotionLib,
        default_q: torch.Tensor,
        soft_lo: torch.Tensor,
        soft_hi: torch.Tensor,
    ):
        self.kin = kin
        self.motion = motion
        self.default_q = default_q
        self.soft_lo = soft_lo
        self.soft_hi = soft_hi
        self.anchor = kin.anchor_index
        self.noise = OBS_NOISE.to(default_q.device)
        self.gravity = default_q.new_tensor(GRAVITY)

    def obs(self, p, quat, v, w, q, qd, a_prev, xi) -> torch.Tensor:
        """Noise-free proprioception. ``xi`` supplies the encoder offset."""
        del v  # linear velocity is not in the observation.
        measured = q + xi[..., 1:30]
        omega = quat_apply_inverse(quat, w)
        gravity = quat_apply_inverse(quat, self.gravity.expand_as(w))
        return torch.cat((measured - self.default_q, qd, omega, gravity, a_prev), dim=-1)

    def add_noise(self, obs: torch.Tensor) -> torch.Tensor:
        return obs + (2.0 * torch.rand_like(obs) - 1.0) * self.noise

    def bodies(self, p, quat, v, w, q, qd):
        return self.kin.forward(p, quat, v, w, q, qd)

    def reward(self, p, quat, v, w, q, qd, frame, action, a_prev) -> torch.Tensor:
        body_p, body_quat, body_v, body_w = self.bodies(p, quat, v, w, q, qd)
        ref_p = self.motion.body_p[frame]
        ref_quat = self.motion.body_quat[frame]
        ref_pos, ref_ori = _anchor_relative(body_p, body_quat, ref_p, ref_quat, self.anchor)
        e_pos = (ref_pos - body_p).square().sum(-1).mean(-1)
        e_ori = quat_error_angle(ref_ori, body_quat).square().mean(-1)
        e_lv = (self.motion.body_v[frame] - body_v).square().sum(-1).mean(-1)
        e_av = (self.motion.body_w[frame] - body_w).square().sum(-1).mean(-1)
        e_rp = (self.motion.root_p[frame] - p).square().sum(-1)
        e_ro = quat_error_angle(self.motion.root_quat[frame], quat).square()
        tracking = (
            torch.exp(-e_pos / 0.3**2)
            + torch.exp(-e_ori / 0.4**2)
            + torch.exp(-e_lv / 1.0**2)
            + torch.exp(-e_av / 3.14**2)
            + 0.5 * torch.exp(-e_rp / 0.3**2)
            + 0.5 * torch.exp(-e_ro / 0.4**2)
        )
        action_rate = (action - a_prev).square().sum(-1)
        joint_limit = (torch.relu(q - self.soft_hi) + torch.relu(self.soft_lo - q)).sum(-1)
        return (tracking - 0.1 * action_rate - 2.0 * joint_limit) * CONTROL_DT

    def termination(self, p, quat, frame) -> torch.Tensor:
        height, tilt = self._errors(p, quat, frame)
        return (height > 0.25) | (tilt > 0.8)

    def soft_termination(self, p, quat, frame, height_scale: float = 0.02, tilt_scale: float = 0.04) -> torch.Tensor:
        """Differentiable termination probability around the hard thresholds."""
        height, tilt = self._errors(p, quat, frame)
        alive = (1.0 - torch.sigmoid((height - 0.25) / height_scale)) * (
            1.0 - torch.sigmoid((tilt - 0.8) / tilt_scale)
        )
        return 1.0 - alive

    def _errors(self, p, quat, frame):
        ref_p = self.motion.root_p[frame]
        ref_quat = self.motion.root_quat[frame]
        height = (ref_p[..., 2] - p[..., 2]).abs()
        g_ref = quat_apply_inverse(ref_quat, self.gravity.expand_as(ref_p))
        g_robot = quat_apply_inverse(quat, self.gravity.expand_as(p))
        tilt = (g_ref[..., 2] - g_robot[..., 2]).abs()
        return height, tilt

    def critic_input(self, p, quat, v, w, q, qd, goal, xi) -> torch.Tensor:
        return torch.cat((invariant_features(p, quat, v, w, q, qd), goal, normalize_xi(xi)), dim=-1)


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
