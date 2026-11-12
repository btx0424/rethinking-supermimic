"""Differentiable imagined rollout; soft termination weights returns and never resets state."""

import torch
from .interface import WMState
from supermimic.env.mdp.state import PhysicalState
from supermimic.robots.g1 import normalize_context
from supermimic.env.rollout.interface import RolloutEnv, TrackingState, Transition


class WorldModelEnv(RolloutEnv):
    def __init__(self, wm, task):
        self.wm = wm
        self.task = task

    def reset(self, full, command, previous_action, xi, episode_step):
        model = self.wm.initialize(full, normalize_context(xi))
        return TrackingState(
            PhysicalState(model.aux, model.f, model.xi_n),
            command,
            previous_action,
            xi,
            episode_step,
            latent=model.z,
        )

    def step(self, state, action):
        model = self.wm(
            WMState(state.physical.aux, state.physical.f, state.latent, state.physical.xi_n), action
        )
        physical = PhysicalState(model.aux, model.f, model.xi_n)
        command = self.task.command.advance(state.command)
        frame = self.task.command.frame(command)
        next_state = TrackingState(
            physical, command, action, state.xi, state.episode_step + 1, latent=model.z
        )
        reward, terms = self.task.reward(physical, frame, action, state.previous_action)
        terminated = self.task.soft_termination(physical, frame)
        return Transition(
            next_state,
            next_state,
            reward,
            terminated,
            torch.zeros_like(frame, dtype=torch.bool),
            terms,
            {},
        )
