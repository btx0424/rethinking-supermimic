"""G1 asset adapter. Algorithm code has no body-role/limb matching."""

import torch

ROOT_BODY = "pelvis"
NUM_JOINTS = 29


def robot_config():
    from mjlab.asset_zoo.robots import get_g1_robot_cfg

    return get_g1_robot_cfg()


def action_scale(joint_names):
    from mjlab.asset_zoo.robots import G1_ACTION_SCALE
    from mjlab.utils.string import resolve_expr

    values = list(resolve_expr(G1_ACTION_SCALE, joint_names))
    if any(x is None for x in values):
        raise ValueError("G1 actuator scales missing for dataset joint order")
    return [float(x) for x in values]


def normalize_context(xi):
    if xi.shape[-1] != NUM_JOINTS + 4:
        raise ValueError("Invalid G1 simulator context")
    return torch.cat(
        (
            (xi[..., :1] - 0.75) / 0.45,
            xi[..., 1 : 1 + NUM_JOINTS] / 0.01,
            xi[..., 1 + NUM_JOINTS :] / 0.05,
        ),
        -1,
    )
