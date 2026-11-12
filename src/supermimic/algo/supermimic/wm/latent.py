import torch
from torch import nn
from supermimic.nn.mlp import MLP
from .interface import WorldModel, WMState
from .loss import latent_diagnostics


class SimNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.reshape(*x.shape[:-1], -1, self.dim).softmax(-1).reshape(x.shape)


class LatentWorldModel(WorldModel):
    def __init__(
        self,
        representation,
        action_dim,
        hidden=(512, 512, 512),
        latent_dim=256,
        simnorm_dim=8,
        dt=0.02,
    ):
        super().__init__()
        if simnorm_dim < 1 or latent_dim % simnorm_dim:
            raise ValueError("simnorm_dim must divide latent_dim")
        self.representation = representation
        self.dt = dt
        self.xi_dim = action_dim + 4
        self.latent_dim = latent_dim
        self.simnorm = SimNorm(simnorm_dim)
        self.encoder = MLP(representation.rep_state_dim, hidden, latent_dim, norm="rms")
        width = latent_dim + action_dim + self.xi_dim
        self.dynamics = MLP(width, hidden, latent_dim, norm="rms")
        self.decoder = MLP(width, hidden, representation.prediction_dim, norm="rms")

    def encode_features(self, f):
        return self.simnorm(self.encoder(self.network_input("wm_encoder", f)))

    def observation_network_inputs(self, full):
        return {"wm_encoder": self.representation.represent(full)}

    def transition_network_inputs(self, state, action):
        inputs = torch.cat((state.z, action, state.xi_n), -1)
        return {"wm_dynamics": inputs, "wm_decoder": inputs}

    def forward(self, state, action):
        if state.z is None:
            raise ValueError("Latent state must be initialized")
        inputs = self.transition_network_inputs(state, action)
        delta = self.decoder(self.network_input("wm_decoder", inputs["wm_decoder"]))
        z = self.simnorm(self.dynamics(self.network_input("wm_dynamics", inputs["wm_dynamics"])))
        aux, f = self.representation.apply_prediction(state.aux, state.f, delta, self.dt)
        return WMState(aux, f, z, state.xi_n)

    def auxiliary_loss(self, trajectory, targets):
        return (trajectory.z - targets.z).square().mean(), latent_diagnostics(
            self, trajectory.z, targets.z
        )
