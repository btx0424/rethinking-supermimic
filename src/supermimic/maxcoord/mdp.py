"""MaxCoord-specific goal and reward adapters; existing MDPs stay unchanged."""
import torch

from supermimic.mdp import WorldModelState, unflatten_state
from supermimic.utils.math import quat_apply_inverse, quat_inv, quat_mul, quat_to_mat, mat_to_6d
from .math import matrix_to_quat
from .state import build_maxcoord_full_state


def full_from_flat(mdp, flat):
    """Reconstruct all body twists from the existing generalized-coordinate replay."""
    shape = flat.shape[:-1]
    bodies = mdp.bodies(*unflatten_state(flat.reshape(-1, flat.shape[-1]), mdp.task.num_joints))
    return build_maxcoord_full_state(*bodies).reshape(*shape, -1)


def goal(motion, frame, root_quat, horizon=10):
    """Reference-only displacement in current reference pelvis axes.

    Position is future reference minus current reference (no robot position).
    Orientation is future reference relative to the current robot pelvis.
    """
    future = motion.future_frames(frame, horizon)
    ref_quat = motion.root_quat[frame, None].expand(*future.shape, 4)
    dp = quat_apply_inverse(ref_quat, motion.root_p[future] - motion.root_p[frame, None])
    relative = quat_mul(quat_inv(root_quat).unsqueeze(-2), motion.root_quat[future])
    rot6 = mat_to_6d(quat_to_mat(relative))
    return torch.cat((dp, rot6, motion.q[future]), -1).flatten(-2)


def configure_sim(sim):
    """Remove joint_limit on this sim instance in collect, eval and imagination."""
    if sim.task.has_object or sim.task.name != "lafan1":
        raise ValueError("MaxCoord currently supports the LAFAN1 G1 task only")
    mdp = sim.mdp
    mdp._reward_terms = [term for term in mdp._reward_terms if term.name != "joint_limit"]
    mdp.term_names = tuple(term.name for term in mdp._reward_terms)
    sim.ep_term.pop("joint_limit", None)
    # joint_limit is a penalty in all current styles, so the maximum is unchanged.


def reward_state(rep, state, num_joints):
    p, rot, v, w = rep.world_frame_bodies(state.aux, state.f)
    quat = matrix_to_quat(rot)
    # Joints are deliberately unused: no joint_limit and no FK on imagined bodies.
    unused = p.new_zeros(*p.shape[:-2], num_joints)
    return WorldModelState(state.aux[..., :3], state.aux[..., 3:7], v[..., 0, :], w[..., 0, :],
                           unused, unused, p, quat, body_lin_vel=v, body_ang_vel=w)
