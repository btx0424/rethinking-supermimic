import torch
from supermimic.utils.math import quat_apply_inverse, quat_inv, quat_mul, quat_to_mat, mat_to_6d


def goal(motion, frame, root_quat, horizon=10):
    future = motion.future_frames(frame, horizon)
    ref_quat = motion.root_quat[frame, None].expand(*future.shape, 4)
    dp = quat_apply_inverse(ref_quat, motion.root_p[future] - motion.root_p[frame, None])
    relative = quat_mul(quat_inv(root_quat).unsqueeze(-2), motion.root_quat[future])
    return torch.cat((dp, mat_to_6d(quat_to_mat(relative)), motion.q[future]), -1).flatten(-2)


OBSERVATION_TERMS = ("features", "goal", "xi_n")


def observation_width(names, feature_dim, goal_frames, action_dim):
    widths = {
        "features": feature_dim,
        "goal": goal_frames * (9 + action_dim),
        "xi_n": action_dim + 4,
    }
    return sum(widths[name] for name in names)


def concatenate(observation, names):
    return torch.cat([getattr(observation, name) for name in names], -1)
