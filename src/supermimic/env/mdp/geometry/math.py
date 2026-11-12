# Adapted from HybridRobotics/supermimic, commit 5efbf114 (Apache-2.0).
# Modifications: standalone imports/API; see THIRD_PARTY.md at repository root.
"""MaxCoord math: wxyz quaternions, row-interleaved rot6d."""

import torch
from supermimic.utils.math import quat_to_mat as matrix_from_quat, quat_mul


def matrix_to_quat(rot_mat: torch.Tensor) -> torch.Tensor:
    """Rotation matrix to quaternion — compile-safe (no boolean indexing).

    Same algorithm as pytorch3d/mjlab ``quat_from_matrix`` but uses
    ``torch.clamp`` + ``torch.gather`` instead of boolean advanced indexing,
    so it produces only static-shaped tensors and works inside
    ``torch.compile(mode="reduce-overhead")``.

    Args:
      rot_mat: (..., 3, 3) rotation matrix

    Returns:
      (..., 4) quaternion (w, x, y, z)
    """
    m00, m01, m02 = rot_mat[..., 0, 0], rot_mat[..., 0, 1], rot_mat[..., 0, 2]
    m10, m11, m12 = rot_mat[..., 1, 0], rot_mat[..., 1, 1], rot_mat[..., 1, 2]
    m20, m21, m22 = rot_mat[..., 2, 0], rot_mat[..., 2, 1], rot_mat[..., 2, 2]

    q_abs = torch.sqrt(
        torch.clamp(
            torch.stack(
                [
                    1.0 + m00 + m11 + m22,
                    1.0 + m00 - m11 - m22,
                    1.0 - m00 + m11 - m22,
                    1.0 - m00 - m11 + m22,
                ],
                dim=-1,
            ),
            min=1e-10,
        )
    )

    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )  # (..., 4, 4)

    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].clamp(min=0.1))

    best = q_abs.argmax(dim=-1)  # (...)
    best_expanded = best[..., None, None].expand(*best.shape, 1, 4)
    return quat_candidates.gather(-2, best_expanded).squeeze(-2)


def small_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a @ b`` without TF32 rounding, fusable by ``torch.compile``."""
    return (a.unsqueeze(-1) * b.unsqueeze(-3)).sum(-2)


def rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to 3x3 rotation matrix via Gram-Schmidt.

    Args:
      rot6d: (..., 6) first two columns of rotation matrix, flattened

    Returns:
      (..., 3, 3) rotation matrix
    """
    cols = rot6d.unflatten(-1, (3, 2))
    a1 = cols[..., 0]
    a2 = cols[..., 1]

    b1 = torch.nn.functional.normalize(a1, dim=-1)
    dot = (b1 * a2).sum(dim=-1, keepdim=True)
    b2 = torch.nn.functional.normalize(a2 - dot * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack([b1, b2, b3], dim=-1)


def quat_to_rot6d(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternions to 6D rotation representation (first two columns of rotation matrix)."""
    mat = matrix_from_quat(quat)  # (..., 3, 3)
    return mat[..., :2].flatten(-2, -1)


def rodrigues_precompute(
    omega: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decompose axis-angle into (unit_axis, cos_angle, sin_angle).

    Args:
      omega: (..., 3) axis-angle rotation (axis * angle)

    Returns:
      Tuple of (k, cos_a, sin_a) where k is the unit axis.
    """
    angle = torch.sqrt((omega * omega).sum(-1, keepdim=True) + 1e-8)
    k = omega / angle
    return k, torch.cos(angle), torch.sin(angle)


def rodrigues_apply(
    v: torch.Tensor,
    k: torch.Tensor,
    cos_a: torch.Tensor,
    sin_a: torch.Tensor,
) -> torch.Tensor:
    """Rodrigues rotation with precomputed trig.

    Args:
      v: (..., 3) vectors to rotate
      k: (..., 3) unit rotation axis
      cos_a: (..., 1) cosine of angle
      sin_a: (..., 1) sine of angle

    Returns:
      (..., 3) rotated vectors
    """
    k_cross_v = torch.linalg.cross(k, v, dim=-1)
    k_dot_v = (k * v).sum(-1, keepdim=True)
    return cos_a * v + sin_a * k_cross_v + (1.0 - cos_a) * k_dot_v * k


def quat_integrate(quat: torch.Tensor, ang_vel: torch.Tensor, dt: float) -> torch.Tensor:
    """Integrate quaternion by angular velocity over dt (axis-angle method).

    Args:
      quat: (..., 4) current quaternion (w, x, y, z)
      ang_vel: (..., 3) angular velocity in world frame
      dt: time step

    Returns:
      (..., 4) next quaternion, normalized
    """
    omega = ang_vel * dt
    norm_sq = (omega * omega).sum(dim=-1, keepdim=True)
    norm_safe = torch.sqrt(norm_sq + 1e-8)
    half_angle = norm_safe / 2
    axis = omega / norm_safe
    delta_q = torch.cat([torch.cos(half_angle), torch.sin(half_angle) * axis], dim=-1)
    q_next = quat_mul(delta_q, quat)
    return q_next / torch.norm(q_next, dim=-1, keepdim=True)
