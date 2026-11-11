# Adapted from HybridRobotics/supermimic, commit 5efbf114 (Apache-2.0).
# Modifications: standalone imports/API; see THIRD_PARTY.md at repository root.
"""Independent MaxCoord representation and Hybrid-compatible integration."""
import math
import torch
from torch import nn
from supermimic.utils.math import quat_to_mat as matrix_from_quat, quat_apply, quat_apply_inverse
from .math import quat_integrate, rodrigues_apply, rodrigues_precompute
from .state import (
    compute_maxcoord_rep_state, decompose_maxcoord_rep_state,
    reconstruct_full_from_aux_and_rep, world_frame_bodies,
)


def _root_loss_blocks(aux_state):
    rot = matrix_from_quat(aux_state[..., 3:7]).flatten(-2, -1)
    return {"root_pos": aux_state[..., :3], "root_ori": rot * math.sqrt(0.5)}


class MaxCoordRepresentation(nn.Module):
  def __init__(self, num_bodies: int, decimation: int = 4):
    super().__init__()
    if num_bodies < 2 or decimation < 1:
      raise ValueError("Need at least two bodies and positive decimation")
    self.num_bodies = num_bodies
    self.full_state_dim = num_bodies * 13
    self.rep_state_dim = num_bodies * 15 + 4
    self.prediction_dim = num_bodies * 6
    self.substep_coeff = (decimation + 1) / (2 * decimation)

  def represent(self, full_state: torch.Tensor) -> torch.Tensor:
    flat = full_state.reshape(-1, self.full_state_dim)
    rep = compute_maxcoord_rep_state(flat, self.num_bodies)
    return rep.reshape(*full_state.shape[:-1], -1)

  def extract_aux_state(self, full_state: torch.Tensor) -> torch.Tensor:
    B = self.num_bodies
    return torch.cat([full_state[..., :3], full_state[..., B * 3 : B * 3 + 4]], dim=-1)

  def apply_prediction(
    self,
    aux_state: torch.Tensor,
    curr_rep_state: torch.Tensor,
    prediction: torch.Tensor,
    dt: float,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrate state by prediction deltas in root body frame.

    Integrates positions and rotations in the current root frame,
    then applies a frame-change rotation to the new root frame. Pose
    updates weight the delta by ``substep_coeff`` to match how the sim
    composes its substeps; velocity states advance by the full delta.
    """
    B = self.num_bodies
    c = self.substep_coeff
    root_pos = aux_state[..., :3]
    root_quat = aux_state[..., 3:7]

    local_pos, local_rot_6d, local_vel, local_avel, _, _ = decompose_maxcoord_rep_state(
      curr_rep_state, B
    )

    # Prediction deltas (in root body frame)
    dv_local = prediction[..., : B * 3].unflatten(-1, (B, 3))
    dw_local = prediction[..., B * 3 : B * 6].unflatten(-1, (B, 3))

    # --- Step 1: Apply deltas in local frame ---
    next_vel = local_vel + dv_local
    next_avel = local_avel + dw_local
    pose_vel = local_vel + c * dv_local
    pose_avel = local_avel + c * dw_local

    # --- Step 2: Integrate in current root frame ---
    # Positions (with drift correction: subtract root_pos_local)
    root_pos_local = local_pos[..., 0:1, :]
    rel_vel = pose_vel - pose_vel[..., 0:1, :]
    next_pos = (local_pos - root_pos_local) + rel_vel * dt
    # Rotations (per-body omega, shared trig for 2 rot6d cols)
    k_body, cos_body, sin_body = rodrigues_precompute(pose_avel * dt)
    cols = local_rot_6d.unflatten(-1, (3, 2))
    c1_mid = rodrigues_apply(cols[..., 0], k_body, cos_body, sin_body)
    c2_mid = rodrigues_apply(cols[..., 1], k_body, cos_body, sin_body)

    # --- Step 3: Root pose in world frame (with drift correction) ---
    next_root_pos = root_pos + quat_apply(
      root_quat,
      root_pos_local.squeeze(-2) + pose_vel[..., 0, :] * dt,
    )
    next_root_quat = quat_integrate(
      root_quat, quat_apply(root_quat, pose_avel[..., 0, :]), dt
    )

    # --- Step 4: Frame change (old root → new root) ---
    frame_omega = -pose_avel[..., 0:1, :] * dt
    k_f, cos_f, sin_f = rodrigues_precompute(frame_omega)

    next_local_pos = rodrigues_apply(next_pos, k_f, cos_f, sin_f)
    next_local_vel = rodrigues_apply(next_vel, k_f, cos_f, sin_f)
    next_local_avel = rodrigues_apply(next_avel, k_f, cos_f, sin_f)
    next_local_rot_6d = torch.stack(
      [
        rodrigues_apply(c1_mid, k_f, cos_f, sin_f),
        rodrigues_apply(c2_mid, k_f, cos_f, sin_f),
      ],
      dim=-1,
    ).flatten(-2, -1)

    # --- Step 5: Aux state & trailing fields ---
    world_down = torch.zeros_like(next_root_pos)
    world_down[..., 2] = -1.0
    next_up_dir = quat_apply_inverse(next_root_quat, world_down)

    next_aux = torch.cat([next_root_pos, next_root_quat], dim=-1)
    next_rep = torch.cat(
      [
        next_local_pos.flatten(-2, -1),
        next_local_rot_6d.flatten(-2, -1),
        next_local_vel.flatten(-2, -1),
        next_local_avel.flatten(-2, -1),
        next_root_pos[..., 2:3],  # root_height
        next_up_dir,
      ],
      dim=-1,
    )
    return next_aux, next_rep

  def reconstruct_full_state(
    self, aux_state: torch.Tensor, rep_state: torch.Tensor
  ) -> torch.Tensor:
    return reconstruct_full_from_aux_and_rep(aux_state, rep_state, self.num_bodies)

  def loss_blocks(
    self, rep_state: torch.Tensor, aux_state: torch.Tensor
  ) -> dict[str, torch.Tensor]:
    # Body 0 is the root: its pose is identically zero/identity and its twist is
    # the root twist. root_height/up_dir also follow from the root pose. All of
    # them are scored by root_pos/root_ori instead.
    pos, rot_6d, vel, avel, _, _ = decompose_maxcoord_rep_state(
      rep_state, self.num_bodies
    )
    return {
      **_root_loss_blocks(aux_state),
      "body_pos": pos[..., 1:, :].flatten(-2, -1),
      "body_rot": rot_6d[..., 1:, :].flatten(-2, -1),
      "body_vel": vel[..., 1:, :].flatten(-2, -1),
      "body_avel": avel[..., 1:, :].flatten(-2, -1),
    }

  def world_frame_bodies(
    self, aux_state: torch.Tensor, rep_state: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return world_frame_bodies(aux_state, rep_state, self.num_bodies)
