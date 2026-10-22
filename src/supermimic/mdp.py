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
XI_DIM = 1 + NUM_J + 3
INV_DIM = 1 + 6 + 3 + 3 + NUM_J + NUM_J
STATE_DIM = 3 + 4 + 3 + 3 + NUM_J + NUM_J
HIST = 10
GOAL_FRAMES = 10
GOAL_DIM = GOAL_PER_FRAME * GOAL_FRAMES
GRAVITY = (0.0, 0.0, -1.0)
# (name, dimension, uniform noise half-width). Order is the observation layout.
OBS_SPEC = (
    ("joint_pos", NUM_J, 0.01),
    ("joint_vel", NUM_J, 0.5),
    ("base_ang_vel", 3, 0.2),
    ("projected_gravity", 3, 0.05),
    ("last_action", NUM_J, 0.0),
    ("anchor_ori", 6, 0.05),
)
PAPER_OBS_TERMS = ("joint_pos", "joint_vel", "base_ang_vel", "projected_gravity", "last_action")
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
# Tracking kernels. Action-rate and joint-limit stay the paper penalties in every style:
# ``-w · feature · dt`` (feature is already a square / a ReLU sum, so no saturation).
# ``gaussian``: paper ``exp(-error²/σ²)``.
# ``cauchy``: ``1/(1+error²/σ²)``.
# ``gaussian_linear``: Gaussian near 0 plus a smooth linear-in-error tail (can go negative).
REWARD_STYLES = ("gaussian", "cauchy", "gaussian_linear")
# Asymptotic |∂k/∂x| = LINEAR_ALPHA / σ. The linear regime starts near LINEAR_WIDTH·σ.
LINEAR_ALPHA = 0.25
LINEAR_WIDTH = 4.0
REWARD_TERM_NAMES = (
    "body_pos",
    "body_ori",
    "body_lin_vel",
    "body_ang_vel",
    "root_pos",
    "root_ori",
    "action_rate",
    "joint_limit",
)
# Shared length-scales (σ). Tracking uses the paper values; regularizers are style-specific uses.
REWARD_TERM_SCALE = {
    "body_pos": 0.3,
    "body_ori": 0.4,
    "body_lin_vel": 1.0,
    "body_ang_vel": 3.14,
    "root_pos": 0.3,
    "root_ori": 0.4,
    "action_rate": 1.0,
    "joint_limit": 0.1,
}
# Peak |weight| before ×dt. Tracking weights match the paper; cauchy joint_limit is a bonus.
REWARD_TERM_WEIGHT = {
    "gaussian": {
        "body_pos": 1.0,
        "body_ori": 1.0,
        "body_lin_vel": 1.0,
        "body_ang_vel": 1.0,
        "root_pos": 0.5,
        "root_ori": 0.5,
        "action_rate": 0.1,
        "joint_limit": 2.0,
    },
    "cauchy": {
        "body_pos": 1.0,
        "body_ori": 1.0,
        "body_lin_vel": 1.0,
        "body_ang_vel": 1.0,
        "root_pos": 0.5,
        "root_ori": 0.5,
        "action_rate": 0.1,
        "joint_limit": 0.5,
    },
    "gaussian_linear": {
        "body_pos": 1.0,
        "body_ori": 1.0,
        "body_lin_vel": 1.0,
        "body_ang_vel": 1.0,
        "root_pos": 0.5,
        "root_ori": 0.5,
        "action_rate": 0.1,
        "joint_limit": 0.5,
    },
}


def resolve_reward_style(style: str) -> str:
    if style not in REWARD_STYLES:
        raise ValueError(f"Unknown reward style {style!r}. Known: {REWARD_STYLES}.")
    return style


def max_step_reward(style: str = "gaussian") -> float:
    """Upper bound on the positive tracking reward (penalties are ≤ 0 in every style)."""
    resolve_reward_style(style)
    weights = REWARD_TERM_WEIGHT[style]
    return sum(weights[name] for name in REWARD_TERM_NAMES if name not in ("action_rate", "joint_limit")) * CONTROL_DT


# Default = paper Gaussian (backward-compatible import site).
MAX_STEP_REWARD = max_step_reward("gaussian")


def gaussian_kernel(error_sq: torch.Tensor, scale: float) -> torch.Tensor:
    """Paper kernel ``exp(-error² / σ²)``. ``error_sq`` is already ``error²``."""
    return torch.exp(-error_sq / (scale * scale))


def cauchy_kernel(error_sq: torch.Tensor, scale: float) -> torch.Tensor:
    """Heavy-tailed kernel ``1 / (1 + error² / σ²)``. ``error_sq`` is already ``error²``.

    C∞ for ``error² ≥ 0``. Peak 1 at 0; algebraic tails (no exp dead-zone).
    ``|∂k/∂(error²)|`` peaks at 0 (``1/σ²``) and decays as ``O(error⁻⁴)``.
    """
    return 1.0 / (1.0 + error_sq / (scale * scale))


def gaussian_linear_kernel(error_sq: torch.Tensor, scale: float) -> torch.Tensor:
    """Gaussian plus a hyperbolic term that is linear in the raw error ``x = √(error²)``.

    ``k = exp(-x²/σ²) - β (√(1 + (x/δ)²) - 1)`` with ``δ = LINEAR_WIDTH·σ`` and
    ``β = LINEAR_ALPHA·LINEAR_WIDTH``. C∞, ``k(0) = 1``, ``k'(0) = 0`` (matches the
    Gaussian to first order). ``k'(x) → -LINEAR_ALPHA/σ``, so the tail does not
    saturate; it becomes negative past a finite error.
    """
    delta = LINEAR_WIDTH * scale
    beta = LINEAR_ALPHA * LINEAR_WIDTH
    gauss = torch.exp(-error_sq / (scale * scale))
    tail = beta * (torch.sqrt(1.0 + error_sq / (delta * delta)) - 1.0)
    return gauss - tail


def hard_joint_violation(q: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Paper joint-limit penalty feature ``(relu(q-hi)+relu(lo-q)).sum(-1)``."""
    return (torch.relu(q - hi) + torch.relu(lo - q)).sum(-1)


def resolve_obs_terms(terms: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    known = {name: (dim, noise) for name, dim, noise in OBS_SPEC}
    chosen = tuple(terms)
    unknown = [name for name in chosen if name not in known]
    if unknown:
        raise ValueError(f"Unknown observation terms {unknown}. Known: {tuple(known)}.")
    if not chosen:
        raise ValueError("At least one observation term is required.")
    # Keep the spec order so the layout does not depend on config key order.
    return tuple(name for name, _, _ in OBS_SPEC if name in chosen)


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
        obs_terms: tuple[str, ...] = PAPER_OBS_TERMS,
        reward_style: str = "gaussian",
    ):
        self.kin = kin
        self.motion = motion
        self.default_q = default_q
        self.soft_lo = soft_lo
        self.soft_hi = soft_hi
        self.anchor = kin.anchor_index
        self.obs_terms = resolve_obs_terms(obs_terms)
        self.reward_style = resolve_reward_style(reward_style)
        self.term_weight = REWARD_TERM_WEIGHT[self.reward_style]
        self.max_step_reward = max_step_reward(self.reward_style)
        spec = {name: (dim, noise) for name, dim, noise in OBS_SPEC}
        self.obs_dim = sum(spec[name][0] for name in self.obs_terms)
        self.noise = torch.cat(
            [torch.full((spec[name][0],), spec[name][1], device=default_q.device) for name in self.obs_terms]
        )
        self.gravity = default_q.new_tensor(GRAVITY)

    def obs(self, p, quat, v, w, q, qd, a_prev, xi, frame=None) -> torch.Tensor:
        """Noise-free proprioception. ``xi`` supplies the encoder offset."""
        measured = q + xi[..., 1:30]
        parts = {
            "joint_pos": measured - self.default_q,
            "joint_vel": qd,
            "base_ang_vel": quat_apply_inverse(quat, w),
            "projected_gravity": quat_apply_inverse(quat, self.gravity.expand_as(w)),
            "last_action": a_prev,
        }
        if "anchor_ori" in self.obs_terms:
            if frame is None:
                raise ValueError("The anchor_ori observation needs the reference frame.")
            # Pelvis is the free-joint state — no FK. Torso (default) needs one.
            if self.anchor == self.kin.pelvis_index:
                robot_quat = quat
            else:
                robot_quat = self.bodies(p, quat, v, w, q, qd)[1][..., self.anchor, :]
            ref_quat = self.motion.body_quat[frame][..., self.anchor, :]
            relative = quat_mul(quat_inv(robot_quat), ref_quat)
            parts["anchor_ori"] = mat_to_6d(quat_to_mat(relative))
        return torch.cat([parts[name] for name in self.obs_terms], dim=-1)

    def add_noise(self, obs: torch.Tensor) -> torch.Tensor:
        return obs + (2.0 * torch.rand_like(obs) - 1.0) * self.noise

    def bodies(self, p, quat, v, w, q, qd):
        return self.kin.forward(p, quat, v, w, q, qd)

    def reward(self, p, quat, v, w, q, qd, frame, action, a_prev, terms: bool = False):
        """Per-step reward (×dt). ``reward_style`` changes tracking kernels only.

        Action-rate and joint-limit are always the paper penalties
        ``-w · ||Δa||² · dt`` and ``-w · relu_violation · dt``.

        With ``terms=True``, also return a name→tensor dict that sums to the reward.
        """
        body_p, body_quat, body_v, body_w = self.bodies(p, quat, v, w, q, qd)
        ref_p = self.motion.body_p[frame]
        ref_quat = self.motion.body_quat[frame]
        ref_pos, ref_ori = _anchor_relative(body_p, body_quat, ref_p, ref_quat, self.anchor)
        if self.reward_style == "cauchy":
            kernel = cauchy_kernel
        elif self.reward_style == "gaussian_linear":
            kernel = gaussian_linear_kernel
        else:
            kernel = gaussian_kernel
        errors = {
            "body_pos": (ref_pos - body_p).square().sum(-1).mean(-1),
            "body_ori": quat_error_angle(ref_ori, body_quat).square().mean(-1),
            "body_lin_vel": (self.motion.body_v[frame] - body_v).square().sum(-1).mean(-1),
            "body_ang_vel": (self.motion.body_w[frame] - body_w).square().sum(-1).mean(-1),
            "root_pos": (self.motion.root_p[frame] - p).square().sum(-1),
            "root_ori": quat_error_angle(self.motion.root_quat[frame], quat).square(),
        }
        w = self.term_weight
        s = REWARD_TERM_SCALE
        pen = REWARD_TERM_WEIGHT["gaussian"]
        parts = {
            name: w[name] * kernel(errors[name], s[name]) * CONTROL_DT for name in errors
        }
        parts["action_rate"] = -pen["action_rate"] * (action - a_prev).square().sum(-1) * CONTROL_DT
        parts["joint_limit"] = -pen["joint_limit"] * hard_joint_violation(q, self.soft_lo, self.soft_hi) * CONTROL_DT
        reward = parts["body_pos"]
        for name in REWARD_TERM_NAMES[1:]:
            reward = reward + parts[name]
        if terms:
            return reward, parts
        return reward

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
