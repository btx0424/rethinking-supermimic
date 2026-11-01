"""Task layout shared by the simulator, the MDP, and the networks.

A task fixes the robot asset and the tensor widths. LAFAN1 and Weave use the
same ``TrackingSim`` and ``MDP``; only this spec changes.
"""

from __future__ import annotations

from dataclasses import dataclass

from supermimic.kinematics import TRACKED_BODIES


@dataclass(frozen=True)
class TaskSpec:
    """Robot, observation layout, and whether the reference includes an object."""

    name: str
    num_joints: int
    tracked_bodies: tuple[str, ...]
    contact_bodies: tuple[str, ...]
    has_object: bool
    foot_geom_expr: str
    object_name: str | None = None
    goal_frames: int = 10
    hist: int = 10

    def __post_init__(self) -> None:
        if self.num_joints <= 0:
            raise ValueError(f"{self.name}: num_joints must be positive.")
        if not self.tracked_bodies:
            raise ValueError(f"{self.name}: tracked_bodies is empty.")
        if not self.contact_bodies:
            raise ValueError(f"{self.name}: contact_bodies is empty.")
        if self.has_object and not self.object_name:
            raise ValueError(f"{self.name}: has_object requires object_name.")
        if not self.has_object and self.object_name is not None:
            raise ValueError(f"{self.name}: object_name set but has_object is False.")

    @property
    def n_body(self) -> int:
        return len(self.tracked_bodies)

    @property
    def n_contact(self) -> int:
        return len(self.contact_bodies)

    @property
    def xi_dim(self) -> int:
        """Friction, per-joint encoder bias, torso COM shift."""
        return 1 + self.num_joints + 3

    @property
    def inv_dim(self) -> int:
        return 1 + 6 + 3 + 3 + self.num_joints + self.num_joints

    @property
    def pelvis_feat_dim(self) -> int:
        return 1 + 3 + 3 + 3 + self.num_joints + self.num_joints

    @property
    def state_dim(self) -> int:
        """Root pose and twist plus the hinge joints. The object is not in this vector."""
        return 3 + 4 + 3 + 3 + self.num_joints + self.num_joints

    @property
    def object_state_dim(self) -> int:
        """Flat object pose and twist when ``has_object``, else 0."""
        return 13 if self.has_object else 0

    @property
    def object_feat_dim(self) -> int:
        """Yaw-frame object pose and object-local twist for the WM / critic."""
        return 3 + 6 + 3 + 3 if self.has_object else 0

    @property
    def priv_dim(self) -> int:
        width = self.num_joints + self.n_body * 3 + 3 + 6
        if self.has_object:
            width += 3 + 6
        return width

    @property
    def body_pos_dim(self) -> int:
        return self.n_body * 3

    @property
    def goal_per_frame(self) -> int:
        """Yaw-frame root, joints, and, for object tasks, the object pose."""
        width = 3 + 6 + 3 + 3 + self.num_joints + self.num_joints
        if self.has_object:
            width += 3 + 6 + 3 + 3
        return width

    @property
    def goal_dim(self) -> int:
        return self.goal_per_frame * self.goal_frames

    @property
    def wm_out_dim(self) -> int:
        width = 3 + 3 + 3 + 3 + self.num_joints + self.num_joints + self.n_contact
        if self.has_object:
            width += 3 + 3 + 3 + 3
        return width

    def obs_term(self, name: str) -> tuple[int, float]:
        """``(dimension, noise half-width)`` for one observation term."""
        terms = {
            "joint_pos": (self.num_joints, 0.01),
            "joint_vel": (self.num_joints, 0.5),
            "base_ang_vel": (3, 0.2),
            "base_lin_vel": (3, 0.2),
            "projected_gravity": (3, 0.05),
            "last_action": (self.num_joints, 0.0),
            "anchor_ori": (6, 0.05),
            "anchor_pos": (3, 0.01),
            "body_pos": (self.n_body * 3, 0.01),
            "body_ori": (self.n_body * 6, 0.05),
            "object_pos": (3, 0.01),
            "object_ori": (6, 0.05),
        }
        try:
            return terms[name]
        except KeyError as exc:
            raise KeyError(f"Unknown observation term {name!r}.") from exc

    def nominal_error_groups(self) -> tuple[tuple[int, float], ...]:
        groups = (
            (3, 0.25),
            (3, 0.5),
            (3, 1.0),
            (3, 3.14),
            (self.num_joints, 0.5),
            (self.num_joints, 5.0),
        )
        if self.has_object:
            groups = groups + ((3, 0.25), (3, 0.5), (3, 1.0), (3, 3.14))
        return groups

    def object_cfg(self):
        """mjlab ``EntityCfg`` for the free-joint object. Only for object tasks."""
        raise NotImplementedError(f"{self.name} has no object asset.")

    def compile_model(self):
        """MuJoCo model and hinge-joint names in qpos order."""
        raise NotImplementedError

    def robot_cfg(self):
        """mjlab ``EntityCfg`` for the simulated robot."""
        raise NotImplementedError

    def action_scale(self, joint_names: tuple[str, ...]) -> list[float]:
        """Position-target scale for each hinge, in ``joint_names`` order."""
        raise NotImplementedError

    def load_motion(self, device: str):
        """Reference library whose joint order matches :meth:`compile_model`."""
        raise NotImplementedError


def _g1_scale(joint_names: tuple[str, ...], *, allow_fingers: bool) -> list[float]:
    from mjlab.asset_zoo.robots import G1_ACTION_SCALE
    from mjlab.utils.string import resolve_expr

    scale = list(resolve_expr(G1_ACTION_SCALE, joint_names))
    missing = [name for name, value in zip(joint_names, scale) if value is None]
    if not missing:
        return [float(value) for value in scale]
    if not allow_fingers:
        raise KeyError(f"No action scale for joints {missing}")
    fingers = [name for name in missing if name[:2] in ("L_", "R_")]
    if fingers != missing:
        raise KeyError(f"No action scale for joints {missing}")
    # Inspire finger hinges are not in the 29-DoF G1 map.
    return [0.5 if value is None else float(value) for value in scale]


class Lafan1Task(TaskSpec):
    def __init__(self) -> None:
        super().__init__(
            name="lafan1",
            num_joints=29,
            tracked_bodies=TRACKED_BODIES,
            contact_bodies=TRACKED_BODIES,
            has_object=False,
            foot_geom_expr=r"^(left|right)_foot[1-7]_collision$",
        )

    def compile_model(self):
        from supermimic.kinematics import compile_g1

        return compile_g1()

    def robot_cfg(self):
        from mjlab.asset_zoo.robots import get_g1_robot_cfg

        return get_g1_robot_cfg()

    def action_scale(self, joint_names: tuple[str, ...]) -> list[float]:
        return _g1_scale(joint_names, allow_fingers=False)

    def load_motion(self, device: str):
        from supermimic.motion import MotionLib

        return MotionLib.load(device=device)


class WeaveTask(TaskSpec):
    """Inspire-hand G1 plus a free-joint object tracking a Weave clip.

    Finger hinges are independent position actuators, matching the recorded
    joint angles. The scene object mesh matches ``object_name``; the motion
    cache is built from that object's rollout file.
    """

    def __init__(self, object_name: str = "smalltable") -> None:
        super().__init__(
            name="weave",
            num_joints=53,
            tracked_bodies=TRACKED_BODIES,
            contact_bodies=TRACKED_BODIES,
            has_object=True,
            foot_geom_expr=r"^(left|right)_foot_collision_[1-7]$",
            object_name=object_name,
        )

    def compile_model(self):
        from supermimic.robots.g1_inspire import compile_g1_inspire

        model, names = compile_g1_inspire()
        if len(names) != self.num_joints:
            raise RuntimeError(f"Inspire-hand model has {len(names)} hinges, expected {self.num_joints}.")
        return model, names

    def robot_cfg(self):
        from supermimic.robots.g1_inspire import robot_cfg

        return robot_cfg()

    def object_cfg(self):
        from supermimic.robots.object import object_cfg

        assert self.object_name is not None
        return object_cfg(self.object_name)

    def action_scale(self, joint_names: tuple[str, ...]) -> list[float]:
        from supermimic.robots.g1_inspire import action_scale

        return action_scale(joint_names)

    def load_motion(self, device: str):
        from supermimic.motions.weave import WeaveMotion

        assert self.object_name is not None
        return WeaveMotion.load(device=device, objects=(self.object_name,))


_TASKS: dict[str, type[TaskSpec]] = {
    "lafan1": Lafan1Task,
    "weave": WeaveTask,
}


def get_task(name: str) -> TaskSpec:
    try:
        return _TASKS[name]()
    except KeyError as exc:
        known = ", ".join(sorted(_TASKS))
        raise KeyError(f"Unknown task {name!r}. Available: {known}.") from exc
