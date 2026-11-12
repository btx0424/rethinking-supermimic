"""The same policy step/rollout works with both envs.

The policy callback may add real exploration/collect-only normalization. The
imagined callback stays differentiable and read-only. Reset semantics belong
to the env's continuation field, not to this loop.
"""

from typing import NamedTuple
from torch import Tensor
from .interface import PolicyObservation, TrackingState, Transition


class PolicyStep(NamedTuple):
    state: TrackingState
    observation: PolicyObservation
    action: Tensor
    transition: Transition


def policy_step(env, state, policy):
    observation = env.observe(state)
    action = policy(observation)
    return PolicyStep(state, observation, action, env.step(state, action))


def rollout(env, state, policy, steps):
    for _ in range(steps):
        record = policy_step(env, state, policy)
        yield record
        state = record.transition.continuation
