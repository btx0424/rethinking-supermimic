"""Quaternion helpers; wxyz convention throughout."""

from __future__ import annotations
import torch


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product. Applies ``q2`` first, then ``q1``."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def quat_inv(q: torch.Tensor) -> torch.Tensor:
    return quat_conjugate(q) / q.square().sum(-1, keepdim=True).clamp(min=1e-8)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by ``q``. ``v_world = R(q) @ v_body`` when ``q`` is body-to-world."""
    q_vec = q[..., 1:]
    t = 2.0 * torch.cross(q_vec, v, dim=-1)
    return v + q[..., :1] * t + torch.cross(q_vec, t, dim=-1)


def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return quat_apply(quat_conjugate(q), v)


def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    """Rotation matrix, body vectors to world."""
    w, x, y, z = q.unbind(-1)
    two = 2.0
    return torch.stack(
        (
            torch.stack(
                (1 - two * (y * y + z * z), two * (x * y - z * w), two * (x * z + y * w)), -1
            ),
            torch.stack(
                (two * (x * y + z * w), 1 - two * (x * x + z * z), two * (y * z - x * w)), -1
            ),
            torch.stack(
                (two * (x * z - y * w), two * (y * z + x * w), 1 - two * (x * x + y * y)), -1
            ),
        ),
        -2,
    )


def mat_to_6d(m: torch.Tensor) -> torch.Tensor:
    """First two columns, flattened. ``m`` is (..., 3, 3)."""
    return m[..., :, :2].reshape(*m.shape[:-2], 6)


def yaw_quat(q: torch.Tensor) -> torch.Tensor:
    """Yaw-only quaternion, matching the XYZ yaw extraction used by mjlab."""
    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    half = 0.5 * yaw
    out = torch.zeros_like(q)
    out[..., 0] = half.cos()
    out[..., 3] = half.sin()
    return quat_normalize(out)


def quat_to_rotvec(q: torch.Tensor) -> torch.Tensor:
    q = quat_normalize(q)
    q = torch.where(q[..., :1] < 0, -q, q)
    w = q[..., :1].clamp(-1.0, 1.0)
    xyz = q[..., 1:]
    sin_half = xyz.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, w)
    small = sin_half < 1e-8
    scale = torch.where(small, 2.0 + angle.square() / 12.0, angle / sin_half.clamp(min=1e-8))
    return xyz * scale


def quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """XYZ Euler angles to a quaternion. Angles are (...,)."""
    cy, sy = (yaw * 0.5).cos(), (yaw * 0.5).sin()
    cr, sr = (roll * 0.5).cos(), (roll * 0.5).sin()
    cp, sp = (pitch * 0.5).cos(), (pitch * 0.5).sin()
    return torch.stack(
        (
            cy * cr * cp + sy * sr * sp,
            cy * sr * cp - sy * cr * sp,
            cy * cr * sp + sy * sr * cp,
            sy * cr * cp - cy * sr * sp,
        ),
        dim=-1,
    )


def quat_error_angle(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Rotation angle in radians between two quaternions."""
    return quat_to_rotvec(quat_mul(q1, quat_inv(q2))).norm(dim=-1)
