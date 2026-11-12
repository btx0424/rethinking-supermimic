from torch import nn
from supermimic.nn.mlp import MLP


class Actor(nn.Module):
    """Explicit features + full goal. Algo normalizes before this network."""

    def __init__(self, input_dim, action_dim, hidden=(512, 512, 512)):
        super().__init__()
        self.net = MLP(input_dim, hidden, action_dim)
        nn.init.zeros_(self.net.head.weight)
        nn.init.zeros_(self.net.head.bias)

    def forward(self, inputs):
        return self.net(inputs)
