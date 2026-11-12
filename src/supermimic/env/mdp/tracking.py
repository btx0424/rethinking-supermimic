from supermimic.env.rollout.interface import PolicyObservation
from .observation import goal
from .termination import TrackingTermination
from .command import MotionCommand
from .config import TaskConfig
from .reward import reward_terms
from . import CONTROL_DT


class TrackingTask(TrackingTermination):
    def __init__(self, motion, representation, cfg=None):
        self.motion = motion
        self.command = MotionCommand(motion)
        self.representation = representation
        self.cfg = cfg or TaskConfig()
        self.goal_frames = self.cfg.goal_frames
        self.term_names = tuple(self.cfg.reward_weights)
        self.max_step_reward = sum(max(v, 0) for v in self.cfg.reward_weights.values()) * CONTROL_DT
        self.num_joints = len(motion.joint_names)
        self.num_bodies = len(motion.body_names)

    def observe(self, state):
        physical = state.physical
        return PolicyObservation(
            physical.f,
            goal(
                self.motion,
                self.command.frame(state.command),
                physical.aux[..., :3],
                physical.aux[..., 3:7],
                self.goal_frames,
            ),
            physical.xi_n,
            state.full,
        )

    def reward(self, state, frame, action, previous):
        terms = reward_terms(
            self.representation, state, self.motion, frame, action, previous, self.cfg
        )
        return sum(terms.values()), terms
