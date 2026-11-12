"""Common policy/environment contract for real and imagined tracking."""

from abc import ABC, abstractmethod
from typing import NamedTuple
import torch
from supermimic.env.mdp.state import PhysicalState
from supermimic.env.mdp.command import CommandState


class TrackingState(NamedTuple):
    physical: PhysicalState
    command: CommandState
    previous_action: torch.Tensor
    xi: torch.Tensor  # raw physical context, preserved in both rollout paths
    episode_step: torch.Tensor
    full: torch.Tensor | None = None  # raw world-body state; real replay only
    latent: torch.Tensor | None = None  # WM internal continuation, never passed to terms


class PolicyObservation(NamedTuple):
    features: torch.Tensor
    goal: torch.Tensor
    xi_n: torch.Tensor
    full: torch.Tensor | None


class Transition(NamedTuple):
    state: TrackingState  # true successor, BEFORE any real reset
    continuation: TrackingState  # next policy call; may contain reset environments
    reward: torch.Tensor
    terminated: torch.Tensor  # hard bool in sim; soft probability in WM
    truncated: torch.Tensor
    terms: dict[str, torch.Tensor]
    info: dict[str, torch.Tensor]


class RolloutEnv(ABC):
    def observe(self, state):
        return self.task.observe(state)

    @abstractmethod
    def step(self, state, action) -> Transition: ...
