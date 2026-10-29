"""Small SAC helpers (no tensordict / torchrl)."""

from __future__ import annotations

import contextlib
import math
from contextlib import contextmanager

import torch
import torch.nn as nn


def gaussian_target_entropy(act_dim: int, sigma: float) -> float:
    """Target **differential** entropy for i.i.d. ``Normal(0, sigma^2)`` per action dim."""
    if sigma <= 0:
        raise ValueError("target_entropy_sigma must be positive.")
    return 0.5 * float(act_dim) * math.log(2.0 * math.pi * math.e * sigma * sigma)


def diagonal_gaussian_entropy(scale: torch.Tensor) -> torch.Tensor:
    """Sum of marginal entropies for ``IndependentNormal(0, scale)`` (matches ``gaussian_target_entropy``)."""
    c = 0.5 * math.log(2.0 * math.pi * math.e)
    return (scale.log() + c).sum(dim=-1)


@torch.no_grad()
def soft_copy_(src: nn.Module, tgt: nn.Module, tau: float) -> None:
    for p_src, p_tgt in zip(src.parameters(), tgt.parameters()):
        p_tgt.data.lerp_(p_src.data, tau)


class IndependentNormal:
    """Diagonal Gaussian; log_prob sums over action dims."""

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor):
        self.loc = loc
        self.scale = scale.clamp_min(1e-6)

    def sample(self) -> torch.Tensor:
        return self.loc + self.scale * torch.randn_like(self.loc)

    def rsample(self, sample_shape: tuple[int, ...] = ()) -> torch.Tensor:
        if not sample_shape:
            return self.sample()
        loc = self.loc.unsqueeze(0).expand(*sample_shape, *self.loc.shape)
        scale = self.scale.unsqueeze(0).expand_as(loc)
        return loc + scale * torch.randn_like(loc)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        var = self.scale.square()
        log_scale = self.scale.log()
        return (
            -((actions - self.loc) ** 2) / (2.0 * var)
            - log_scale
            - 0.5 * math.log(2.0 * math.pi)
        ).sum(dim=-1)


class ObsVecNorm(nn.Module):
    """Online mean/var normalization for flat observations."""

    def __init__(self, dim: int, decay: float = 0.995, eps: float = 1e-8):
        super().__init__()
        self.decay = decay
        self.eps = eps
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.detach().float().reshape(-1, self.mean.numel())
        if x.shape[0] == 0:
            return
        batch_mean = x.mean(0)
        batch_var = x.var(0, unbiased=False)
        if self.count.item() == 0:
            self.mean.copy_(batch_mean)
            self.var.copy_(batch_var.clamp_min(self.eps))
            self.count += float(x.shape[0])
            return
        d = self.decay
        self.mean.lerp_(batch_mean, 1.0 - d)
        self.var.lerp_(batch_var, 1.0 - d)
        self.count += float(x.shape[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.count.item() == 0:
            return x
        return (x - self.mean) / (self.var.sqrt() + self.eps)


@contextmanager
def hold_out_net(network: nn.Module):
    params = [p for p in network.parameters() if p.requires_grad]
    if not params:
        yield
        return
    req = [p.requires_grad for p in params]
    try:
        for p in params:
            p.requires_grad_(False)
        yield
    finally:
        for p, r in zip(params, req):
            p.requires_grad_(r)


def autocast(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device.type, dtype=torch.float16)
