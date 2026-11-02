"""Analytic observation, reward, and termination shared by the simulator and the world model."""

from __future__ import annotations

from typing import NamedTuple

import torch

from .kinematics import TRACKED_BODIES, TorchKinematics
from .tasks import Lafan1Task, TaskSpec
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
from .motion import CONTROL_DT, MotionLib

# LAFAN1 widths. Training reads ``TaskSpec``; these aliases keep analysis scripts working.
_LAFAN1 = Lafan1Task()
NUM_J = _LAFAN1.num_joints
XI_DIM = _LAFAN1.xi_dim
INV_DIM = _LAFAN1.inv_dim
# Pelvis-frame world-model features: height, gravity in the pelvis, body-frame twist, joints.
# Three fewer than INV_DIM because gravity replaces the 6D yaw-removed tilt.
PELVIS_FEAT_DIM = _LAFAN1.pelvis_feat_dim
WM_FRAMES = ("heading", "pelvis")
STATE_DIM = _LAFAN1.state_dim
# Critic-only tracking errors: joint position, heading-frame anchored body position,
# heading-frame root position, 6D root orientation.
PRIV_DIM = _LAFAN1.priv_dim
# One ground-contact bit per tracked body. The world model reads it and predicts the next one.
N_CONTACT = _LAFAN1.n_contact
N_BODY = _LAFAN1.n_body
# Heading-frame body positions. The world model reads them and does not predict them.
BODY_POS_DIM = _LAFAN1.body_pos_dim
HIST = _LAFAN1.hist
GOAL_FRAMES = _LAFAN1.goal_frames
GOAL_DIM = _LAFAN1.goal_dim
GRAVITY = (0.0, 0.0, -1.0)
# Hard termination. Height is |root z − reference z|. Tilt is the projected-gravity z error.
TERM_HEIGHT = 0.25
TERM_TILT = 0.8
# (name, dimension, uniform noise half-width). Order is the observation layout.
OBS_SPEC = (
    ("joint_pos", NUM_J, 0.01),
    ("joint_vel", NUM_J, 0.5),
    ("base_ang_vel", 3, 0.2),
    ("base_lin_vel", 3, 0.2),
    ("projected_gravity", 3, 0.05),
    ("last_action", NUM_J, 0.0),
    ("anchor_ori", 6, 0.05),
    # Heading-frame position of the reference anchor relative to the robot anchor.
    # Not removed by the reward's xy alignment, so the policy can see drift.
    ("anchor_pos", 3, 0.01),
    # Tracking error of the reward bodies, not their world pose.
    ("body_pos", len(TRACKED_BODIES) * 3, 0.01),
    ("body_ori", len(TRACKED_BODIES) * 6, 0.05),
    # Object pose in the robot yaw frame (Weave HOI). Absolute relative state, not error.
    ("object_pos", 3, 0.01),
    ("object_ori", 6, 0.05),
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
# Object HOI extras. Same σ / weights as root pose and body twist in object_hoi.
OBJECT_REWARD_TERM_NAMES = (
    "object_pos",
    "object_ori",
    "object_lin_vel",
    "object_ang_vel",
    # Hand tips in the object frame vs the demo (object_hoi motion_hand_obj_relative_pos).
    "hand_obj_pos",
)
# Shared length-scales (σ). Tracking uses the paper values; regularizers are style-specific uses.
REWARD_TERM_SCALE = {
    "body_pos": 0.3,
    "body_ori": 0.4,
    "body_lin_vel": 1.0,
    "body_ang_vel": 3.14,
    "root_pos": 0.3,
    "root_ori": 0.4,
    "object_pos": 0.3,
    "object_ori": 0.4,
    "object_lin_vel": 1.0,
    "object_ang_vel": 3.14,
    "hand_obj_pos": 0.1,
    "action_rate": 1.0,
    "joint_limit": 0.1,
}
# Peak |weight| before ×dt. Tracking weights match the paper; cauchy joint_limit is a bonus.
_TRACKING_WEIGHTS = {
    "body_pos": 1.0,
    "body_ori": 1.0,
    "body_lin_vel": 1.0,
    "body_ang_vel": 1.0,
    "root_pos": 0.5,
    "root_ori": 0.5,
    "object_pos": 1.0,
    "object_ori": 1.0,
    "object_lin_vel": 1.0,
    "object_ang_vel": 1.0,
    "hand_obj_pos": 4.0,
}
REWARD_TERM_WEIGHT = {
    "gaussian": {**_TRACKING_WEIGHTS, "action_rate": 0.1, "joint_limit": 2.0},
    "cauchy": {**_TRACKING_WEIGHTS, "action_rate": 0.1, "joint_limit": 0.5},
    "gaussian_linear": {**_TRACKING_WEIGHTS, "action_rate": 0.1, "joint_limit": 0.5},
}


class ObservationTerm:
    """One policy observation feature. Composed by ``MDP.obs``."""

    name: str
    dim: int
    noise: float = 0.0

    def __init__(self, mdp: "MDP"):
        self.mdp = mdp

    def __call__(
        self,
        state: "WorldModelState",
        a_prev: torch.Tensor,
        xi: torch.Tensor,
        frame: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ``(..., dim)``."""
        raise NotImplementedError


class RewardTerm:
    """One reward contribution. Composed by ``MDP.reward``."""

    name: str
    weight: float
    scale: float | None = None
    penalty: bool = False

    def __init__(self, mdp: "MDP", weight: float, scale: float | None = None):
        self.mdp = mdp
        self.weight = float(weight)
        if scale is not None:
            self.scale = float(scale)

    def __call__(
        self,
        state: "WorldModelState",
        action: torch.Tensor,
        a_prev: torch.Tensor,
        frame: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``(...)`` per-env contribution (already × weight × kernel/penalty × dt)."""
        raise NotImplementedError


def resolve_reward_style(style: str) -> str:
    if style not in REWARD_STYLES:
        raise ValueError(f"Unknown reward style {style!r}. Known: {REWARD_STYLES}.")
    return style


def reward_term_names(task: TaskSpec | None = None) -> tuple[str, ...]:
    """Reward terms that sum to the step reward for this task."""
    if task is not None and task.has_object:
        # Object terms sit with the other tracking kernels, before the penalties.
        base = REWARD_TERM_NAMES[:-2]
        object_terms = OBJECT_REWARD_TERM_NAMES
        if not task.hand_bodies:
            object_terms = tuple(name for name in object_terms if name != "hand_obj_pos")
        return base + object_terms + REWARD_TERM_NAMES[-2:]
    return REWARD_TERM_NAMES


def hand_obj_relative_pos(hand_p, obj_p, obj_quat) -> torch.Tensor:
    """Finger/hand link positions in the object frame, ``(..., n_hand, 3)``."""
    return quat_apply_inverse(obj_quat.unsqueeze(-2), hand_p - obj_p.unsqueeze(-2))


def max_step_reward(style: str = "gaussian", task: TaskSpec | None = None) -> float:
    """Upper bound on the positive tracking reward (penalties are ≤ 0 in every style)."""
    resolve_reward_style(style)
    weights = REWARD_TERM_WEIGHT[style]
    names = reward_term_names(task)
    return sum(weights[name] for name in names if name not in ("action_rate", "joint_limit")) * CONTROL_DT


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


def nominal_error(
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    task: TaskSpec | None = None,
) -> torch.Tensor:
    groups = NOMINAL_ERROR if task is None else task.nominal_error_groups()
    return torch.cat(
        [torch.full((n,), scale, device=device, dtype=dtype) for n, scale in groups]
    )


def flatten_state(p, quat, v, w, q, qd) -> torch.Tensor:
    return torch.cat((p, quat, v, w, q, qd), dim=-1)


def flatten_object(obj_p, obj_quat, obj_v, obj_w) -> torch.Tensor:
    return torch.cat((obj_p, obj_quat, obj_v, obj_w), dim=-1)


def unflatten_state(s: torch.Tensor, num_joints: int | None = None):
    """Split a flat robot state. Joint count is inferred when the vector is robot-only."""
    if num_joints is None:
        extra = s.shape[-1] - 13
        if extra < 2 or extra % 2 != 0:
            raise ValueError(f"State width {s.shape[-1]} is not 13 + 2 * num_joints.")
        num_joints = extra // 2
    j0 = 13
    j1 = j0 + num_joints
    j2 = j1 + num_joints
    if s.shape[-1] < j2:
        raise ValueError(f"State width {s.shape[-1]} is shorter than {j2}.")
    return (
        s[..., 0:3],
        s[..., 3:7],
        s[..., 7:10],
        s[..., 10:13],
        s[..., j0:j1],
        s[..., j1:j2],
    )


def unflatten_object(obj: torch.Tensor):
    """Split a flat object state ``(p, quat, v, w)`` of width 13."""
    if obj.shape[-1] != 13:
        raise ValueError(f"Object state width {obj.shape[-1]} is not 13.")
    return obj[..., 0:3], obj[..., 3:7], obj[..., 7:10], obj[..., 10:13]


class WorldModelState(NamedTuple):
    """Root pose and twist in the world frame, plus hinge joints.

    ``body_pos`` / ``body_quat`` are tracked-body poses from forward kinematics
    (or from the simulator). ``MDP.obs`` reads them and does not run FK.
    ``WorldModel.step`` fills them on the returned next state.
    Object fields are set for HOI tasks; LAFAN1 leaves them ``None``.
    """

    p: torch.Tensor
    """World position, (..., 3)."""
    quat: torch.Tensor
    """World orientation as wxyz, (..., 4)."""
    v: torch.Tensor
    """World linear velocity, (..., 3)."""
    w: torch.Tensor
    """World angular velocity, (..., 3)."""
    q: torch.Tensor
    """Joint positions, (..., num_joints)."""
    qd: torch.Tensor
    """Joint velocities, (..., num_joints)."""
    body_pos: torch.Tensor | None = None
    """World-frame tracked-body positions, (..., n_body, 3)."""
    body_quat: torch.Tensor | None = None
    """World-frame tracked-body orientations, (..., n_body, 4)."""
    obj_p: torch.Tensor | None = None
    """World-frame object position, (..., 3)."""
    obj_quat: torch.Tensor | None = None
    """World-frame object orientation as wxyz, (..., 4)."""
    obj_v: torch.Tensor | None = None
    """World-frame object linear velocity, (..., 3)."""
    obj_w: torch.Tensor | None = None
    """World-frame object angular velocity, (..., 3)."""
    body_lin_vel: torch.Tensor | None = None
    """World-frame tracked-body linear velocity, (..., n_body, 3). Reward only."""
    body_ang_vel: torch.Tensor | None = None
    """World-frame tracked-body angular velocity, (..., n_body, 3). Reward only."""
    hand_pos: torch.Tensor | None = None
    """World-frame finger link positions, (..., n_hand, 3). HOI reward only."""

    def root(self) -> tuple[torch.Tensor, ...]:
        """The six root-and-joint tensors the buffer stores."""
        return (self.p, self.quat, self.v, self.w, self.q, self.qd)

    def object(self) -> tuple[torch.Tensor, ...]:
        """Object pose and twist. Raises if any object field is missing."""
        if self.obj_p is None or self.obj_quat is None or self.obj_v is None or self.obj_w is None:
            raise ValueError("WorldModelState is missing object fields.")
        return (self.obj_p, self.obj_quat, self.obj_v, self.obj_w)

    def flatten(self) -> torch.Tensor:
        return torch.cat(self.root(), dim=-1)

    def flatten_object(self) -> torch.Tensor:
        return flatten_object(*self.object())


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


def pelvis_features(p, quat, v, w, q, qd) -> torch.Tensor:
    """World-model features in the pelvis frame: height, gravity, body-frame twist, joints.

    Horizontal translation and yaw cancel. Gravity replaces the yaw-removed tilt, so
    the features stay continuous when the pelvis is horizontal.
    """
    gravity = quat.new_tensor(GRAVITY).expand(v.shape)
    return torch.cat(
        (
            p[..., 2:3],
            quat_apply_inverse(quat, gravity),
            quat_apply_inverse(quat, v),
            quat_apply_inverse(quat, w),
            q,
            qd,
        ),
        dim=-1,
    )


def pelvis_body_pos(p, quat, body_p) -> torch.Tensor:
    """Tracked-body positions in the pelvis frame, relative to the pelvis.

    ``R_pelvisᵀ (p_body − p_pelvis)``. Same invariance as ``heading_body_pos``, without
    the yaw cut.
    """
    local = quat_apply_inverse(quat.unsqueeze(-2), body_p - p.unsqueeze(-2))
    return local.flatten(-2)


def heading_body_pos(p, quat, body_p) -> torch.Tensor:
    """Tracked-body positions in the root heading frame, relative to the root.

    ``body_p`` is ``(..., n_body, 3)`` in the world. Yaw is removed and the root
    position is subtracted, so these features stay invariant with the rest of the
    world-model state. The z component is the height difference.
    """
    yaw = yaw_quat(quat).unsqueeze(-2)
    local = quat_apply_inverse(yaw, body_p - p.unsqueeze(-2))
    return local.flatten(-2)


def object_features(p, quat, obj_p, obj_quat, obj_v, obj_w) -> torch.Tensor:
    """Object pose in the robot yaw frame and twist in the object body frame.

    Matches the object block of ``MotionLib.goal`` so the critic / WM see the
    same layout as the future reference.
    """
    yaw = yaw_quat(quat)
    dp = quat_apply_inverse(yaw, obj_p - p)
    relative = quat_mul(quat_inv(yaw), obj_quat)
    return torch.cat(
        (
            dp,
            mat_to_6d(quat_to_mat(relative)),
            quat_apply_inverse(obj_quat, obj_v),
            quat_apply_inverse(obj_quat, obj_w),
        ),
        dim=-1,
    )


def local_object_error(
    obj_p,
    obj_quat,
    obj_v,
    obj_w,
    obj_p_hat,
    obj_quat_hat,
    obj_v_hat,
    obj_w_hat,
    root_quat,
    frame: str = "heading",
) -> torch.Tensor:
    """Object-state error of the hat quantities in the true robot's frames."""
    pos_frame = root_quat if frame == "pelvis" else yaw_quat(root_quat)
    return torch.cat(
        (
            quat_apply_inverse(pos_frame, obj_p_hat - obj_p),
            quat_to_rotvec(quat_mul(quat_inv(obj_quat), obj_quat_hat)),
            quat_apply_inverse(obj_quat, obj_v_hat - obj_v),
            quat_apply_inverse(obj_quat, obj_w_hat - obj_w),
        ),
        dim=-1,
    )


def normalize_xi(xi: torch.Tensor) -> torch.Tensor:
    """Friction, encoder bias, torso COM. Joint count is ``xi.shape[-1] - 4``."""
    num_joints = xi.shape[-1] - 4
    bias = slice(1, 1 + num_joints)
    com = slice(1 + num_joints, None)
    return torch.cat(
        ((xi[..., :1] - 0.75) / 0.45, xi[..., bias] / 0.01, xi[..., com] / 0.05),
        dim=-1,
    )


def local_state_error(p, quat, v, w, q, qd, p_hat, quat_hat, v_hat, w_hat, q_hat, qd_hat, frame: str = "heading") -> torch.Tensor:
    """State error of the hat quantities, expressed in the true state's frames.

    ``frame="heading"`` writes the position error in the yaw frame. ``frame="pelvis"``
    writes it in the full pelvis frame, matching that integration.
    """
    pos_frame = quat if frame == "pelvis" else yaw_quat(quat)
    return torch.cat(
        (
            quat_apply_inverse(pos_frame, p_hat - p),
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
        task: TaskSpec | None = None,
    ):
        self.task = _LAFAN1 if task is None else task
        self.kin = kin
        self.motion = motion
        self.default_q = default_q
        self.soft_lo = soft_lo
        self.soft_hi = soft_hi
        self.anchor = kin.anchor_index
        self.obs_terms = resolve_obs_terms(obs_terms)
        self.reward_style = resolve_reward_style(reward_style)
        self.term_weight = REWARD_TERM_WEIGHT[self.reward_style]
        self.term_names = reward_term_names(self.task)
        self.max_step_reward = max_step_reward(self.reward_style, self.task)
        self.gravity = default_q.new_tensor(GRAVITY)
        from .observations import build_obs_terms
        from .rewards import build_reward_terms

        self._obs_terms = build_obs_terms(self, self.obs_terms)
        self._reward_terms = build_reward_terms(self, self.term_names)
        self.obs_dim = sum(term.dim for term in self._obs_terms)
        self.noise = torch.cat(
            [
                torch.full((term.dim,), term.noise, device=default_q.device)
                for term in self._obs_terms
            ]
        )

    def needs_body_pose(self) -> bool:
        """Whether observation terms read tracked-body poses from the state."""
        return "body_pos" in self.obs_terms or "body_ori" in self.obs_terms or (
            ("anchor_ori" in self.obs_terms or "anchor_pos" in self.obs_terms)
            and self.anchor != self.kin.pelvis_index
        )

    def needs_object(self) -> bool:
        """Whether observation terms or the task require object pose."""
        return self.task.has_object or "object_pos" in self.obs_terms or "object_ori" in self.obs_terms

    def obs(self, state: WorldModelState, a_prev, xi, frame=None) -> torch.Tensor:
        """Noise-free proprioception. ``xi`` supplies the encoder offset.

        Body-dependent terms read ``state.body_pos`` / ``state.body_quat``. Callers
        must fill those (simulator or ``WorldModel.step``); this method does not run FK.
        """
        if self.needs_body_pose() and (state.body_pos is None or state.body_quat is None):
            raise ValueError(
                "Observation terms need state.body_pos and state.body_quat; fill them before obs."
            )
        if self.needs_object() and (
            state.obj_p is None or state.obj_quat is None or state.obj_v is None or state.obj_w is None
        ):
            raise ValueError("Object observation terms need state.obj_*; fill them before obs.")
        parts = [term(state, a_prev, xi, frame) for term in self._obs_terms]
        return torch.cat(parts, dim=-1)

    def add_noise(self, obs: torch.Tensor) -> torch.Tensor:
        return obs + (2.0 * torch.rand_like(obs) - 1.0) * self.noise

    def bodies(self, p, quat, v, w, q, qd):
        return self.kin.forward(p, quat, v, w, q, qd)

    def heading_bodies(self, p, quat, v, w, q, qd, body_p=None) -> torch.Tensor:
        """Heading-frame tracked-body positions, ``(..., BODY_POS_DIM)``. Not a prediction target."""
        if body_p is None:
            body_p = self.bodies(p, quat, v, w, q, qd)[0]
        return heading_body_pos(p, quat, body_p)

    def _complete_reward_state(self, state: WorldModelState) -> WorldModelState:
        """Fill body pose/twist via FK when the caller only packed root/joints."""
        if self.task.has_object and (
            state.obj_p is None or state.obj_quat is None or state.obj_v is None or state.obj_w is None
        ):
            raise ValueError("Object reward needs state.obj_* packed on WorldModelState.")
        need_bodies = (
            state.body_pos is None
            or state.body_quat is None
            or state.body_lin_vel is None
            or state.body_ang_vel is None
        )
        if not need_bodies:
            return state
        body_p, body_quat, body_v, body_w = self.bodies(*state.root())
        return state._replace(
            body_pos=body_p if state.body_pos is None else state.body_pos,
            body_quat=body_quat if state.body_quat is None else state.body_quat,
            body_lin_vel=body_v if state.body_lin_vel is None else state.body_lin_vel,
            body_ang_vel=body_w if state.body_ang_vel is None else state.body_ang_vel,
        )

    def reward(
        self,
        state: WorldModelState,
        frame,
        action,
        a_prev,
        terms: bool = False,
    ):
        """Per-step reward (×dt). ``reward_style`` changes tracking kernels only.

        ``state`` is packed like ``obs`` (root, joints, bodies, optional object/hands).
        Missing body pose/twist are filled with FK. Action-rate and joint-limit stay
        the paper penalties ``-w · feature · dt``.

        With ``terms=True``, also return a name→tensor dict that sums to the reward.
        """
        state = self._complete_reward_state(state)
        parts = {term.name: term(state, action, a_prev, frame) for term in self._reward_terms}
        names = self.term_names
        reward = parts[names[0]]
        for name in names[1:]:
            reward = reward + parts[name]
        if terms:
            return reward, parts
        return reward

    def termination(self, p, quat, frame) -> torch.Tensor:
        height, tilt = self._errors(p, quat, frame)
        return (height > TERM_HEIGHT) | (tilt > TERM_TILT)

    def soft_termination(self, p, quat, frame, height_scale: float = 0.02, tilt_scale: float = 0.04) -> torch.Tensor:
        """Differentiable termination probability around the hard thresholds."""
        height, tilt = self._errors(p, quat, frame)
        alive = (1.0 - torch.sigmoid((height - TERM_HEIGHT) / height_scale)) * (
            1.0 - torch.sigmoid((tilt - TERM_TILT) / tilt_scale)
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

    def privileged(self, p, quat, v, w, q, qd, frame, object_state=None) -> torch.Tensor:
        """Tracking errors the policy does not observe.

        Joint position error, anchored body-position error and root-position error
        in the robot heading frame, and the 6D reference-root orientation in the
        robot root frame. The anchor matches the reward. Object tasks append
        heading-frame object position error and 6D object orientation error.
        """
        body_p, body_quat, _, _ = self.bodies(p, quat, v, w, q, qd)
        anchored_p, _ = _anchor_relative(
            body_p, body_quat, self.motion.body_p[frame], self.motion.body_quat[frame], self.anchor
        )
        yaw = yaw_quat(quat)
        body_error = quat_apply_inverse(yaw.unsqueeze(-2), anchored_p - body_p).flatten(-2)
        root_error = quat_apply_inverse(yaw, self.motion.root_p[frame] - p)
        relative = quat_mul(quat_inv(quat), self.motion.root_quat[frame])
        parts = [
            self.motion.q[frame] - q,
            body_error,
            root_error,
            mat_to_6d(quat_to_mat(relative)),
        ]
        if self.task.has_object:
            if object_state is None:
                raise ValueError("Privileged object features need object_state.")
            obj_p, obj_quat, _, _ = object_state
            parts.append(quat_apply_inverse(yaw, self.motion.object_p[frame] - obj_p))
            obj_rel = quat_mul(quat_inv(obj_quat), self.motion.object_quat[frame])
            parts.append(mat_to_6d(quat_to_mat(obj_rel)))
        return torch.cat(parts, dim=-1)

    def critic_input(self, p, quat, v, w, q, qd, goal, xi, frame, object_state=None) -> torch.Tensor:
        parts = [
            invariant_features(p, quat, v, w, q, qd),
            goal,
            normalize_xi(xi),
            self.privileged(p, quat, v, w, q, qd, frame, object_state=object_state),
        ]
        if self.task.has_object:
            if object_state is None:
                raise ValueError("Critic object features need object_state.")
            parts.insert(1, object_features(p, quat, *object_state))
        return torch.cat(parts, dim=-1)


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

