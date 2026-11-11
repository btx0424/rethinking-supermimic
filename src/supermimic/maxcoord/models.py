"""World models and actors are independent modules sharing a rollout state."""
from __future__ import annotations

import torch
from torch import nn

from supermimic.supermimic.network import MLP, RunningNorm
from supermimic.supermimic.latent_wm import SimNorm
from .representation import MaxCoordRepresentation
from .interface import WorldModel, RolloutState


class ExplicitWM(WorldModel):
    def __init__(self, representation: MaxCoordRepresentation, action_dim: int,
                 hidden=(512, 512, 512), dt=0.02, xi_dim=0, legacy_norm=False):
        super().__init__()
        self.representation = representation
        self.dt = dt
        self.xi_dim = xi_dim
        self.legacy_norm = legacy_norm
        if legacy_norm:
            self.feature_norm = RunningNorm(representation.rep_state_dim)
            self.action_norm = RunningNorm(action_dim)
        self.delta = MLP(representation.rep_state_dim + action_dim + xi_dim, list(hidden),
                         representation.prediction_dim, norm="rms")

    def forward(self, state: RolloutState, action: torch.Tensor) -> RolloutState:
        inputs = self.transition_network_inputs(state, action)["wm_delta"]
        delta = self.delta(self.network_input("wm_delta", inputs))
        aux, features = self.representation.apply_prediction(state.aux, state.f, delta, self.dt)
        return RolloutState(aux, features, None, state.xi_n)

    def transition_network_inputs(self, state, action):
        f = self.feature_norm(state.f) if self.legacy_norm else state.f
        a = self.action_norm(action) if self.legacy_norm else action
        return {"wm_delta": torch.cat((f, a, state.xi_n if self.xi_dim else state.f[..., :0]), -1)}


class LatentWM(WorldModel):
    def __init__(self, representation: MaxCoordRepresentation, action_dim: int,
                 hidden=(512, 512, 512), latent_dim=256, dt=0.02, xi_dim=0, simnorm_dim=8, legacy_norm=False):
        super().__init__()
        self.representation = representation
        self.dt = dt
        self.xi_dim = xi_dim
        self.latent_dim = latent_dim
        if simnorm_dim < 0 or (simnorm_dim and latent_dim % simnorm_dim):
            raise ValueError("latent_dim must be divisible by simnorm_dim")
        self.simnorm = SimNorm(simnorm_dim)
        self.legacy_norm = legacy_norm
        if legacy_norm:
            self.feature_norm = RunningNorm(representation.rep_state_dim)
            self.action_norm = RunningNorm(action_dim)
        self.encoder = MLP(representation.rep_state_dim, list(hidden), latent_dim, norm="rms")
        self.dynamics = MLP(latent_dim + action_dim + xi_dim, list(hidden), latent_dim, norm="rms")
        self.decoder = MLP(latent_dim + action_dim + xi_dim, list(hidden), representation.prediction_dim,
                           norm="rms")

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        inputs = self.feature_norm(features) if self.legacy_norm else features
        return self.simnorm(self.encoder(self.network_input("wm_encoder", inputs)))

    def encode_features(self, features):
        return self.encode(features)

    def auxiliary_loss(self, trajectory, targets):
        from .loss import latent_diagnostics
        consistency = (trajectory.z - targets.z).square().mean()
        return consistency, latent_diagnostics(self, trajectory.z, targets.z)

    def forward(self, state: RolloutState, action: torch.Tensor) -> RolloutState:
        if state.z is None:
            raise ValueError("LatentWM requires an initialized latent state")
        inputs = self.transition_network_inputs(state, action)
        # Both heads see z_t and a_t. Never decode z_{t+1}, and never re-encode
        # predicted f during imagination: dynamics must carry its own latent.
        delta = self.decoder(self.network_input("wm_decoder", inputs["wm_decoder"]))
        next_z = self.simnorm(self.dynamics(self.network_input("wm_dynamics", inputs["wm_dynamics"])))
        aux, features = self.representation.apply_prediction(state.aux, state.f, delta, self.dt)
        return RolloutState(aux, features, next_z, state.xi_n)

    def observation_network_inputs(self, full):
        return {"wm_encoder": self.representation.represent(full)}

    def transition_network_inputs(self, state, action):
        a = self.action_norm(action) if self.legacy_norm else action
        inputs = torch.cat((state.z, a, state.xi_n if self.xi_dim else state.f[..., :0]), -1)
        return {"wm_dynamics": inputs, "wm_decoder": inputs}


def actor_features(state: RolloutState, kind: str) -> torch.Tensor:
    if kind == "explicit":
        return state.f
    if kind == "latent" and state.z is not None:
        return state.z
    raise ValueError(f"Cannot use {kind!r} actor with this world-model state")


class Actor(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim, hidden=(512, 512, 512), legacy_norm=False):
        super().__init__()
        self.input_dim = state_dim + goal_dim
        self.legacy_norm = legacy_norm
        self.input_transform = None
        if legacy_norm:
            self.state_norm = RunningNorm(state_dim)
            self.goal_norm = RunningNorm(goal_dim)
        self.net = MLP(state_dim + goal_dim, list(hidden), action_dim)
        nn.init.zeros_(self.net.head.weight)
        nn.init.zeros_(self.net.head.bias)

    def inputs(self, state, goal):
        if self.legacy_norm:
            state, goal = self.state_norm(state), self.goal_norm(goal)
        return torch.cat((state, goal), -1)

    def forward(self, state, goal):
        inputs = self.inputs(state, goal)
        if self.input_transform is not None:
            inputs = self.input_transform("actor", inputs)
        return self.net(inputs)


def build_models(cfg, num_bodies, action_dim, decimation=4, dt=0.02):
    if cfg.wm_kind not in ("explicit", "latent"):
        raise ValueError("wm_kind must be explicit or latent")
    if cfg.actor_kind not in ("explicit", "latent"):
        raise ValueError("actor_kind must be explicit or latent")
    if cfg.wm_kind == "explicit" and cfg.actor_kind == "latent":
        raise ValueError("A latent actor requires a latent world model")
    rep = MaxCoordRepresentation(num_bodies, decimation)
    args = (rep, action_dim, cfg.wm_hidden)
    legacy = getattr(cfg, "input_normalization", "vecnorm") == "legacy"
    wm = (ExplicitWM(*args, dt=dt, legacy_norm=legacy, xi_dim=action_dim + 4 if cfg.wm_xi else 0) if cfg.wm_kind == "explicit" else
          LatentWM(*args, latent_dim=cfg.latent_dim, dt=dt,
                   xi_dim=action_dim + 4 if cfg.wm_xi else 0, simnorm_dim=cfg.simnorm_dim, legacy_norm=legacy))
    state_dim = rep.rep_state_dim if cfg.actor_kind == "explicit" else cfg.latent_dim
    actor = Actor(state_dim, cfg.goal_frames * (9 + action_dim), action_dim, cfg.actor_hidden, legacy_norm=legacy)
    return wm, actor
