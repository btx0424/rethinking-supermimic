"""Architecture-independent physical world-model contract.

NamedTuple containers remain PyTorch pytrees for compile/CUDA Graph. World models
own physical transitions and supervision; learners own optimizers, and tracking
MDPs own reference time, goals, rewards and termination.
"""
from typing import NamedTuple

import torch
from torch import nn


class RolloutState(NamedTuple):
    aux: torch.Tensor
    f: torch.Tensor
    z: torch.Tensor | None = None
    xi_n: torch.Tensor | None = None

    @property
    def features(self):
        return self.f

    @property
    def latent(self):
        return self.z


class WMBatch(NamedTuple):
    initial_full: torch.Tensor
    actions: torch.Tensor  # [B, H, A]
    next_full: torch.Tensor  # [B, H, S], terminal successors before reset
    xi_n: torch.Tensor | None = None  # [B, C], constant within the window


class WMTargets(NamedTuple):
    f: torch.Tensor
    aux: torch.Tensor
    z: torch.Tensor | None


class WMTrajectory(NamedTuple):
    initial: RolloutState
    aux: torch.Tensor  # [B, H, ...], successors only
    f: torch.Tensor
    z: torch.Tensor | None
    xi_n: torch.Tensor | None

    @property
    def final(self):
        return RolloutState(self.aux[:, -1], self.f[:, -1],
                            None if self.z is None else self.z[:, -1], self.xi_n)


class LossOutput(NamedTuple):
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]


class WorldModel(nn.Module):
    """`train(mode)` keeps nn.Module semantics; compute_loss never steps an optimizer."""

    input_transform = None

    def network_input(self, name, inputs):
        return inputs if self.input_transform is None else self.input_transform(name, inputs)

    def observation_network_inputs(self, full):
        return {}

    def transition_network_inputs(self, state, action):
        raise NotImplementedError

    def context(self, full, xi_n):
        if self.xi_dim and (xi_n is None or xi_n.shape != (*full.shape[:-1], self.xi_dim)):
            raise ValueError("World model requires normalized per-episode xi")
        # Preserve context for the critic even when WM conditioning is disabled.
        return full[..., :0] if xi_n is None else xi_n

    def encode_features(self, features):
        return None

    def initialize(self, full, xi_n=None):
        features = self.representation.represent(full)
        return RolloutState(self.representation.extract_aux_state(full), features,
                            self.encode_features(features), self.context(full, xi_n))

    def forward(self, state, action):
        raise NotImplementedError

    def step(self, state, action):
        """Compatibility alias; new callers use wm(state, action)."""
        return self(state, action)

    def rollout(self, state, actions, *, step=None):
        if actions.ndim != 3 or actions.shape[1] < 1:
            raise ValueError("actions must have nonempty [B, H, A] shape")
        initial, states = state, []
        transition = self if step is None else step
        for action in actions.unbind(1):
            state = transition(state, action)
            states.append(state)
        return WMTrajectory(initial, torch.stack([s.aux for s in states], 1),
                            torch.stack([s.f for s in states], 1),
                            None if state.z is None else torch.stack([s.z for s in states], 1),
                            initial.xi_n)

    @torch.no_grad()
    def update_normalization(self, full, actions=None):
        if not self.legacy_norm:
            raise RuntimeError("VecNorm belongs to the algo and updates only from collect")
        if isinstance(full, WMBatch):
            batch = full
            full = torch.cat((batch.initial_full[:, None], batch.next_full[:, :-1]), 1)
            actions = batch.actions
        if actions is None:
            raise ValueError("Normalization requires a WMBatch or full states and actions")
        self.feature_norm.update(self.representation.represent(full))
        self.action_norm.update(actions)

    @torch.no_grad()
    def prepare_targets(self, next_full):
        f = self.representation.represent(next_full)
        return WMTargets(f, self.representation.extract_aux_state(next_full), self.encode_features(f))

    def auxiliary_loss(self, trajectory, targets):
        return trajectory.f.new_zeros(()), trajectory.f.new_zeros(3)

    def compute_loss(self, batch, consistency_weight=1.0, *, targets=None, step=None):
        if batch.actions.shape[:2] != batch.next_full.shape[:2]:
            raise ValueError("Need one true successor per action")
        targets = self.prepare_targets(batch.next_full) if targets is None else targets
        return self.loss_from_targets(batch.initial_full, batch.actions, targets,
                                      batch.xi_n, consistency_weight, step=step)

    def loss_from_targets(self, initial_full, actions, targets, xi_n=None,
                          consistency_weight=1.0, *, step=None):
        """Same loss with precomputed detached targets, for compiled learning kernels."""
        from .loss import state_loss, LATENT_METRICS

        if consistency_weight < 0:
            raise ValueError("consistency_weight must be nonnegative")
        trajectory = self.rollout(self.initialize(initial_full, xi_n), actions, step=step)
        loss, groups = state_loss(self.representation, trajectory.f, trajectory.aux, targets.f, targets.aux)
        consistency, diagnostics = self.auxiliary_loss(trajectory, targets)
        metrics = {"wm_state": loss, "wm_consistency": consistency,
                   "wm_consistency_weighted": consistency_weight * consistency,
                   **{f"wm_mse_{k}": v for k, v in groups.items()},
                   **dict(zip(LATENT_METRICS, diagnostics.unbind()))}
        return LossOutput(loss + consistency_weight * consistency, metrics)

# Descriptive public alias; old RolloutState imports remain compatible.
WMState = RolloutState
