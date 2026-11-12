"""Architecture-independent physical world-model contract.

NamedTuple containers remain PyTorch pytrees for compile/CUDA Graph. World models
own physical transitions and supervision; learners own optimizers, and tracking
MDPs own reference time, goals, rewards and termination.
"""

from typing import NamedTuple

import torch
from torch import nn


class WMState(NamedTuple):
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
    initial: WMState
    aux: torch.Tensor  # [B, H, ...], successors only
    f: torch.Tensor
    z: torch.Tensor | None
    xi_n: torch.Tensor | None

    @property
    def final(self):
        return WMState(
            self.aux[:, -1], self.f[:, -1], None if self.z is None else self.z[:, -1], self.xi_n
        )


class LossOutput(NamedTuple):
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]


class WorldModel(nn.Module):
    """Physical dynamics and supervision; algo owns input statistics and optimizers."""

    input_transform = None

    def network_input(self, name, inputs):
        return inputs if self.input_transform is None else self.input_transform(name, inputs)

    def observation_network_inputs(self, full):
        return {}

    def transition_network_inputs(self, state, action):
        raise NotImplementedError

    def context(self, full, xi_n):
        if xi_n is None or xi_n.shape != (*full.shape[:-1], self.xi_dim):
            raise ValueError("World model requires normalized per-episode xi")
        return xi_n

    def encode_features(self, features):
        return None

    def initialize(self, full, xi_n=None):
        features = self.representation.represent(full)
        return WMState(
            self.representation.extract_aux_state(full),
            features,
            self.encode_features(features),
            self.context(full, xi_n),
        )

    def forward(self, state, action):
        raise NotImplementedError

    def rollout(self, state, actions):
        if actions.ndim != 3 or actions.shape[1] < 1:
            raise ValueError("actions must have nonempty [B, H, A] shape")
        initial, states = state, []
        for action in actions.unbind(1):
            state = self(state, action)
            states.append(state)
        return WMTrajectory(
            initial,
            torch.stack([s.aux for s in states], 1),
            torch.stack([s.f for s in states], 1),
            None if state.z is None else torch.stack([s.z for s in states], 1),
            initial.xi_n,
        )

    @torch.no_grad()
    def prepare_targets(self, next_full):
        f = self.representation.represent(next_full)
        return WMTargets(
            f, self.representation.extract_aux_state(next_full), self.encode_features(f)
        )

    def auxiliary_loss(self, trajectory, targets):
        return trajectory.f.new_zeros(()), trajectory.f.new_zeros(3)

    def compute_loss(self, batch, consistency_weight=1.0):
        """Supervise integrated physical states; latent models add their own auxiliary loss."""
        from .loss import state_loss, LATENT_METRICS

        if batch.actions.shape[:2] != batch.next_full.shape[:2]:
            raise ValueError("Need one true successor per action")
        if consistency_weight < 0:
            raise ValueError("consistency_weight must be nonnegative")
        targets = self.prepare_targets(batch.next_full)
        trajectory = self.rollout(self.initialize(batch.initial_full, batch.xi_n), batch.actions)
        loss, groups = state_loss(
            self.representation, trajectory.f, trajectory.aux, targets.f, targets.aux
        )
        consistency, diagnostics = self.auxiliary_loss(trajectory, targets)
        metrics = {
            "wm_state": loss,
            "wm_consistency": consistency,
            "wm_consistency_weighted": consistency_weight * consistency,
            **{f"wm_mse_{k}": v for k, v in groups.items()},
            **dict(zip(LATENT_METRICS, diagnostics.unbind())),
        }
        return LossOutput(loss + consistency_weight * consistency, metrics)
