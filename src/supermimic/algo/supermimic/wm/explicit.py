import torch
from supermimic.nn.mlp import MLP
from .interface import WorldModel, WMState


class ExplicitWorldModel(WorldModel):
    def __init__(self, representation, action_dim, hidden=(512, 512, 512), dt=0.02):
        super().__init__()
        self.representation = representation
        self.dt = dt
        self.xi_dim = action_dim + 4
        self.delta = MLP(
            representation.rep_state_dim + action_dim + self.xi_dim,
            hidden,
            representation.prediction_dim,
            norm="rms",
        )

    def transition_network_inputs(self, state, action):
        return {"wm_delta": torch.cat((state.f, action, state.xi_n), -1)}

    def forward(self, state, action):
        inputs = self.transition_network_inputs(state, action)["wm_delta"]
        delta = self.delta(self.network_input("wm_delta", inputs))
        aux, f = self.representation.apply_prediction(state.aux, state.f, delta, self.dt)
        return WMState(aux, f, None, state.xi_n)
