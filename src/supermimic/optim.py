"""Muon for hidden 2D weights, AdamW for everything else.

A parameter stays on AdamW when it is not a matrix or when ``p._non_muon`` is set.
MLPs mark their first Linear and their output head that way.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn


class OptimizerGroup(torch.optim.Optimizer):
    """One step/zero_grad/state_dict over several optimizers."""

    def __init__(self, optimizers: list[torch.optim.Optimizer]):
        if not optimizers:
            raise ValueError("OptimizerGroup requires at least one optimizer.")
        params = [p for opt in optimizers for group in opt.param_groups for p in group["params"]]
        if not params:
            raise ValueError("OptimizerGroup underlying optimizers have no parameters.")
        super().__init__(params=params, defaults={})
        self.optimizers = optimizers
        self.param_groups = []
        for opt in self.optimizers:
            self.param_groups.extend(opt.param_groups)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = self.optimizers[0].step(closure)
            for opt in self.optimizers[1:]:
                opt.step()
            return loss
        for opt in self.optimizers:
            result = opt.step()
            if loss is None:
                loss = result
        return loss

    def zero_grad(self, set_to_none: bool | None = None):
        for opt in self.optimizers:
            if set_to_none is None:
                opt.zero_grad()
            else:
                opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {"optimizers": [opt.state_dict() for opt in self.optimizers], "class": self.__class__.__name__}

    def load_state_dict(self, state_dict):
        opt_states = state_dict.get("optimizers")
        if opt_states is None:
            return
        if len(opt_states) != len(self.optimizers):
            warnings.warn(
                f"OptimizerGroup state has {len(opt_states)} optimizers, "
                f"but this instance has {len(self.optimizers)}. Loading the matching prefix.",
                stacklevel=2,
            )
        for opt, opt_state in zip(self.optimizers, opt_states):
            opt.load_state_dict(opt_state)


class MuonAdamW(OptimizerGroup):
    """Hidden matrices use Muon. Biases, 1D parameters, and ``_non_muon`` weights use AdamW."""

    def __init__(
        self,
        modules: list[nn.Module],
        lr: float,
        weight_decay: float = 0.01,
        adam_betas: tuple[float, float] = (0.9, 0.999),
        ignore_frozen: bool = True,
    ):
        seen: set[int] = set()
        muon_params: list[nn.Parameter] = []
        adamw_params: list[nn.Parameter] = []
        for module in modules:
            for _, param in module.named_parameters():
                if id(param) in seen:
                    continue
                if ignore_frozen and not param.requires_grad:
                    continue
                seen.add(id(param))
                if param.dim() == 2 and not getattr(param, "_non_muon", False):
                    muon_params.append(param)
                else:
                    adamw_params.append(param)

        optimizers: list[torch.optim.Optimizer] = []
        if muon_params:
            optimizers.append(torch.optim.Muon(muon_params, lr=lr, adjust_lr_fn="match_rms_adamw"))
        if adamw_params:
            optimizers.append(
                torch.optim.AdamW(adamw_params, lr=lr, weight_decay=weight_decay, betas=adam_betas, fused=True)
            )
        if not optimizers:
            raise ValueError("MuonAdamW found no trainable parameters.")
        super().__init__(optimizers)


def build_optimizer(
    modules: list[nn.Module] | nn.Module,
    lr: float,
    kind: str,
    adam_betas: tuple[float, float] = (0.9, 0.999),
) -> torch.optim.Optimizer:
    """``adam`` or ``muon``. ``muon`` still uses AdamW on non-matrix and ``_non_muon`` weights."""
    if isinstance(modules, nn.Module):
        modules = [modules]
    if kind == "adam":
        params = [p for module in modules for p in module.parameters() if p.requires_grad]
        return torch.optim.Adam(params, lr=lr, betas=adam_betas)
    if kind == "muon":
        return MuonAdamW(modules, lr=lr, adam_betas=adam_betas)
    raise ValueError(f"Unknown optim {kind!r}. Expected 'adam' or 'muon'.")
