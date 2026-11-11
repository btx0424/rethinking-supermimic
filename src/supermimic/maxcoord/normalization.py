"""Algo-owned VecNorm, matching mimic-lite's vector EMA moments.

Reference: active_adaptation/learning/modules/vecnorm.py and mimic-lite PPO's
vecnorms (decay=0.9999). Unlike auto-updating forward, update is explicit here:
only real collection may change stats; all network calls are read-only.
"""
import torch
from torch import nn


class VecNorm(nn.Module):
    def __init__(self, width, decay=0.9999):
        super().__init__()
        if not 0 < decay < 1:
            raise ValueError("VecNorm decay must be in (0, 1)")
        self.decay = decay
        self.eps = 1e-5
        self.register_buffer("sum", torch.zeros(width))
        self.register_buffer("ssq", torch.zeros(width))
        self.register_buffer("count", torch.tensor(0.0))

    @torch.no_grad()
    def update(self, values):
        values = values.reshape(-1, self.sum.numel()).float()
        self.count.mul_(self.decay).add_(values.shape[0])
        self.sum.mul_(self.decay).add_(values.sum(0))
        self.ssq.mul_(self.decay).add_(values.square().sum(0))

    def forward(self, values):
        denom = self.count.clamp_min(1.0)
        mean = self.sum / denom
        variance = (self.ssq / denom - mean.square()).clamp_min(self.eps)
        return (values - mean) / variance.sqrt()


def build_vecnorms(cfg, feature_dim, action_dim):
    context_dim = action_dim + 4
    state_dim = feature_dim if cfg.actor_kind == "explicit" else cfg.latent_dim
    goal_dim = cfg.goal_frames * (9 + action_dim)
    dims = {"actor": state_dim + goal_dim,
            "critic": state_dim + goal_dim + (context_dim if cfg.critic_xi else 0)}
    wm_context = context_dim if cfg.wm_xi else 0
    if cfg.wm_kind == "explicit":
        dims["wm_delta"] = feature_dim + action_dim + wm_context
    else:
        dims.update(wm_encoder=feature_dim,
                    wm_dynamics=cfg.latent_dim + action_dim + wm_context,
                    wm_decoder=cfg.latent_dim + action_dim + wm_context)
    return nn.ModuleDict({key: VecNorm(width, cfg.vecnorm_decay) for key, width in dims.items()})


def bind_vecnorms(wm, actor, vecnorms):
    """Only a callable enters model containers; VecNorm modules belong to algo.

    Normalization runs immediately before each neural module. Physics features,
    integrated states and SimNorm latents remain in their original representations.
    """
    def transform(name, inputs):
        return vecnorms[name](inputs)
    wm.input_transform = transform
    actor.input_transform = transform
