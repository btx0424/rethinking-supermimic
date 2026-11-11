"""Shared real-state policy loading for eval and visual playback."""
from dataclasses import fields

import torch

from supermimic.mdp import normalize_xi
from .mdp import configure_sim, goal
from .models import actor_features, build_models
from .normalization import build_vecnorms, bind_vecnorms
from .state import build_maxcoord_full_state
from .trainer import MaxCoordConfig


def load_policy(saved, sim, device):
    if saved.get("format") not in ("maxcoord_v1", "maxcoord_v2", "maxcoord_v3", "maxcoord_v4"):
        raise ValueError("Expected MaxCoord checkpoint")
    names = {f.name for f in fields(MaxCoordConfig)}
    raw = {k: v for k, v in saved["cfg"].items() if k in names}
    if saved["format"] == "maxcoord_v1":
        raw.update(wm_xi=False, simnorm_dim=0)
    raw.setdefault("critic_xi", False)  # v1/v2 had actor-only critic inputs
    raw.setdefault("input_normalization", "legacy")
    cfg = MaxCoordConfig(**raw)
    configure_sim(sim)
    wm, actor = build_models(cfg, sim.task.n_body, sim.task.num_joints)
    wm.to(device).load_state_dict(saved["wm"])
    actor.to(device).load_state_dict(saved["pi"])
    if cfg.input_normalization == "vecnorm":
        vecnorms = build_vecnorms(cfg, wm.representation.rep_state_dim, sim.task.num_joints).to(device)
        vecnorms.load_state_dict(saved["vecnorms"])
        bind_vecnorms(wm, actor, vecnorms)
    wm.eval()
    actor.eval()

    @torch.no_grad()
    def act():
        state = wm.initialize(build_maxcoord_full_state(*sim.tracked_bodies()), normalize_xi(sim.xi))
        g = goal(sim.motion, sim.f, state.aux[..., 3:7], cfg.goal_frames)
        return actor(actor_features(state, cfg.actor_kind), g)

    return act
