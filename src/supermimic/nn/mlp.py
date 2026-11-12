"""Small feed-forward networks shared by the two world models and policy."""

from torch import nn


class Block(nn.Module):
    def __init__(self, inp, out, norm):
        super().__init__()
        self.fc = nn.Linear(inp, out)
        self.norm = {
            "none": nn.Identity,
            "rms": lambda: nn.RMSNorm(out),
            "layer": lambda: nn.LayerNorm(out),
        }[norm]()
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.fc(x)))


class MLP(nn.Module):
    def __init__(self, inp, hidden, out, norm="none"):
        super().__init__()
        blocks = []
        for width in hidden:
            blocks.append(Block(inp, width, norm))
            inp = width
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(inp, out)
        if blocks:
            for p in blocks[0].fc.parameters():
                p._non_muon = True
        for p in self.head.parameters():
            p._non_muon = True

    def forward(self, x):
        return self.head(self.blocks(x))
