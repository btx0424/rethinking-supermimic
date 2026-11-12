"""G1 LAFAN task choices, including the explicit tracking-body order."""

from dataclasses import dataclass, field
from pathlib import Path
from supermimic.robots.g1 import ROOT_BODY


@dataclass
class TaskConfig:
    motion: str = str(Path(__file__).resolve().parents[4] / "data/lafan1_g1.npz")
    tracking_bodies: tuple[str, ...] = (
        "pelvis",
        "left_hip_roll_link",
        "left_knee_link",
        "left_ankle_roll_link",
        "right_hip_roll_link",
        "right_knee_link",
        "right_ankle_roll_link",
        "torso_link",
        "left_shoulder_roll_link",
        "left_elbow_link",
        "left_wrist_yaw_link",
        "right_shoulder_roll_link",
        "right_elbow_link",
        "right_wrist_yaw_link",
    )
    anchor_body: str = "pelvis"
    torso_body: str = "torso_link"
    goal_frames: int = 10
    actor_observations: tuple[str, ...] = ("features", "goal")
    critic_observations: tuple[str, ...] = ("features", "goal", "xi_n")
    reward_weights: dict[str, float] = field(
        default_factory=lambda: {
            "body_pos": 1.0,
            "body_ori": 1.0,
            "body_lin_vel": 1.0,
            "body_ang_vel": 1.0,
            "root_pos": 0.5,
            "root_ori": 0.5,
            "action_rate": -0.1,
        }
    )
    reward_sigmas: dict[str, float] = field(
        default_factory=lambda: {
            "body_pos": 0.3,
            "body_ori": 0.4,
            "body_lin_vel": 1.0,
            "body_ang_vel": 3.14,
            "root_pos": 0.3,
            "root_ori": 0.4,
        }
    )
    height_threshold: float = 0.25
    tilt_threshold: float = 0.8
    height_softness: float = 0.02
    tilt_softness: float = 0.04
    max_episode_steps: int = 500

    def validate(self):
        import math
        from .observation import OBSERVATION_TERMS
        from .reward import TERM_NAMES

        for names in (self.actor_observations, self.critic_observations):
            if not names or len(set(names)) != len(names) or set(names) - set(OBSERVATION_TERMS):
                raise ValueError("Invalid observation terms")
        if not self.reward_weights or set(self.reward_weights) - set(TERM_NAMES):
            raise ValueError("Invalid reward terms")
        if any(not math.isfinite(v) for v in self.reward_weights.values()):
            raise ValueError("Reward weights must be finite")
        if not any(v > 0 for v in self.reward_weights.values()):
            raise ValueError("At least one positive reward weight is required")
        for name in self.reward_weights:
            if name != "action_rate" and not (
                math.isfinite(self.reward_sigmas[name]) and self.reward_sigmas[name] > 0
            ):
                raise ValueError("Reward sigma must be positive and finite")
        if (
            min(
                self.height_threshold,
                self.tilt_threshold,
                self.height_softness,
                self.tilt_softness,
                self.max_episode_steps,
            )
            <= 0
        ):
            raise ValueError("Termination settings must be positive")
        if len(self.tracking_bodies) < 2 or len(set(self.tracking_bodies)) != len(
            self.tracking_bodies
        ):
            raise ValueError("tracking_bodies must contain at least two unique names")
        if self.anchor_body != ROOT_BODY or self.tracking_bodies[0] != ROOT_BODY:
            raise ValueError(
                "G1 MaxCoord uses the pelvis root anchor; put pelvis first in tracking_bodies"
            )
        if self.goal_frames < 1:
            raise ValueError("goal_frames must be positive")
