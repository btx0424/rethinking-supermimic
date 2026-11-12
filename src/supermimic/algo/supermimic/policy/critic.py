import torch
from torch import nn
from supermimic.nn.mlp import MLP


class Critic(nn.Module):
    """Independent value heads; policy bootstrap takes their mean."""

    def __init__(self, input_dim, heads=2):
        super().__init__()
        if heads < 1:
            raise ValueError("Need at least one value head")
        self.heads = nn.ModuleList(
            MLP(input_dim, (512, 256, 256), 1, norm="layer") for _ in range(heads)
        )

    def forward(self, inputs):
        return torch.stack([head(inputs).squeeze(-1) for head in self.heads], -1)
