"""Feature construction shared by the SAC trainer and the rollout collector.

Both need to build the identical policy observation and critic input from a
:class:`~supermimic.sim.TrackingSim`, including the object block for HOI tasks.
Keeping it in one place is what stops a prior archive from being collected with a
different layout than the run that consumes it.
"""

from __future__ import annotations

import torch

from supermimic.mdp import (
    GOAL_FRAMES,
    WorldModelState,
    invariant_features,
    normalize_xi,
    object_features,
)


def wm_state(sim) -> WorldModelState:
    """Current robot (and object) state packed for :meth:`MDP.obs` / ``reward``."""
    p, quat, v, w, q, qd = sim.state()
    bodies = sim.tracked_bodies()
    obj = sim.object_state() if sim.task.has_object else None
    return WorldModelState(
        p,
        quat,
        v,
        w,
        q,
        qd,
        bodies[0],
        bodies[1],
        *(obj if obj is not None else (None, None, None, None)),
    )


def policy_obs(sim, *, noise: bool = True) -> torch.Tensor:
    """Policy input: sensor observation concatenated with the goal."""
    state = wm_state(sim)
    obs = sim.mdp.obs(state, sim.a_prev, sim.xi, sim.f)
    if noise:
        obs = sim.mdp.add_noise(obs)
    goal = sim.mdp.goal(sim.f, GOAL_FRAMES)
    return torch.cat([obs, goal], dim=-1)


def critic_parts(sim) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Raw critic blocks ``(inv, goal, xi, priv, obj_or_none)`` before ``InputNorms``."""
    p, quat, v, w, q, qd = sim.state()
    inv = invariant_features(p, quat, v, w, q, qd)
    goal = sim.mdp.goal(sim.f, GOAL_FRAMES)
    xi_n = normalize_xi(sim.xi)
    obj = sim.object_state() if sim.task.has_object else None
    priv = sim.mdp.privileged(p, quat, v, w, q, qd, sim.f, object_state=obj)
    obj_feat = object_features(p, quat, *obj) if obj is not None else None
    return inv, goal, xi_n, priv, obj_feat


def critic_input_raw(sim) -> torch.Tensor:
    """Unnormalized critic input (same layout as :func:`critic_input`).

    Stored in the replay / prior archive so a training run can renormalize with
    its own ``InputNorms``.
    """
    inv, goal, xi_n, priv, obj_feat = critic_parts(sim)
    parts = [inv]
    if obj_feat is not None:
        parts.append(obj_feat)
    parts.extend((goal, xi_n, priv))
    return torch.cat(parts, dim=-1)


def critic_input(sim, norms) -> torch.Tensor:
    """Normalized critic input, including the object block for HOI tasks."""
    inv, goal, xi_n, priv, obj_feat = critic_parts(sim)
    return norms.critic(inv, goal, xi_n, priv, obj_feat)


def critic_feat_dim(task) -> int:
    """Width of :func:`critic_input`, matching SuperMimic's critic layout."""
    return (
        task.inv_dim
        + task.object_feat_dim
        + task.goal_dim
        + task.xi_dim
        + task.priv_dim
    )


def update_norms(sim, norms) -> None:
    """Advance the running normalizers from the current state."""
    p, quat, v, w, q, qd = sim.state()
    inv = invariant_features(p, quat, v, w, q, qd)
    state = wm_state(sim)
    obs = sim.mdp.obs(state, sim.a_prev, sim.xi, sim.f)
    goal = sim.mdp.goal(sim.f, GOAL_FRAMES)
    xi_n = normalize_xi(sim.xi)
    obj = sim.object_state() if sim.task.has_object else None
    priv = sim.mdp.privileged(p, quat, v, w, q, qd, sim.f, object_state=obj)
    norms.state.update(inv)
    norms.goal.update(goal)
    norms.xi.update(xi_n)
    norms.priv.update(priv)
    norms.obs.update(obs)
    if obj is not None and norms.obj is not None:
        norms.obj.update(object_features(p, quat, *obj))
