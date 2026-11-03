"""Task layout shared by the simulator, the MDP, and the networks.

A task fixes the robot asset and the tensor widths. LAFAN1 and Weave use the
same ``TrackingSim`` and ``MDP``; only this spec changes.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from supermimic.utils.kinematics import HAND_BODIES, TRACKED_BODIES


@dataclass(frozen=True)
class SimCfg:
    """Per-task MuJoCo / mjlab solver limits used by ``TrackingSim``.

    Hand–object scenes need larger ``njmax`` / ``nconmax`` than plain G1
    locomotion; overflow messages come from mujoco_warp when these are tight.
    """

    nconmax: int = 128
    njmax: int = 500
    timestep: float = 0.005
    iterations: int = 10
    ls_iterations: int = 20

    def __post_init__(self) -> None:
        if self.nconmax <= 0:
            raise ValueError(f"nconmax must be positive, got {self.nconmax}.")
        if self.njmax <= 0:
            raise ValueError(f"njmax must be positive, got {self.njmax}.")
        if self.timestep <= 0.0:
            raise ValueError(f"timestep must be positive, got {self.timestep}.")
        if self.iterations <= 0:
            raise ValueError(f"iterations must be positive, got {self.iterations}.")
        if self.ls_iterations <= 0:
            raise ValueError(f"ls_iterations must be positive, got {self.ls_iterations}.")


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
    # The object asset to load motions for, kept even when ``has_object`` is off.
    # ``None`` means "same as ``object_name``".
    cache_object_name: str | None = None
    hand_bodies: tuple[str, ...] = ()
    goal_frames: int = 10
    hist: int = 10
    sim: SimCfg = field(default_factory=SimCfg)
    # Object tasks: which clip split the motion cache was built from.
    split: str = "train"
    # Every other key of the task config node. Tasks may read their own flags from
    # here; the shared fields above are the ones the MDP and the networks need.
    options: dict = field(default_factory=dict)

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
        if self.hand_bodies and not self.has_object:
            raise ValueError(f"{self.name}: hand_bodies require has_object.")

    @property
    def motions_object_name(self) -> str | None:
        """Object whose clip library to load, whether or not it is in the scene."""
        return self.cache_object_name or self.object_name

    # ---------------------------------------------------------------- variants

    def with_options(self, has_object: bool | None = None) -> "TaskSpec":
        """Copy of this task with per-run options applied.

        ``has_object=False`` drops the scene object and every object term, so the
        same motion data can be tracked as robot-only ("sanity check" mode).

        Subclasses take their own ``__init__`` arguments, so ``dataclasses.replace``
        cannot rebuild them; the copy is made field-by-field instead.
        """
        if has_object is None or bool(has_object) == self.has_object:
            return self
        if has_object:
            # Robot-only task asked to track an object: it has no object asset.
            raise ValueError(f"{self.name} has no object to track.")
        if not self.has_object:
            return self
        clone = copy.copy(self)
        object.__setattr__(clone, "has_object", False)
        # Keep the clip library: the cache is per object even when the object
        # itself is left out of the scene.
        object.__setattr__(clone, "cache_object_name", self.motions_object_name)
        object.__setattr__(clone, "object_name", None)
        object.__setattr__(clone, "hand_bodies", ())
        # Object rewards/observations are the widest terms; fall back to the
        # robot-only solver budget.
        object.__setattr__(clone, "sim", SimCfg())
        return clone

    @property
    def n_body(self) -> int:
        return len(self.tracked_bodies)

    @property
    def run_name(self) -> str:
        """Run label used for the log directory and the logger's experiment name.

        ``<task>`` for robot-only tasks, ``<task>_<object>`` for object tasks, with
        ``_<split>`` appended when the split is not ``train``. An object task run
        with ``has_object=False`` is labelled ``<task>_noobj`` so it never shares a
        directory with the object run.
        """
        if self.object_name is None:
            return f"{self.name}_noobj" if self.name != "lafan1" else self.name
        label = f"{self.name}_{self.object_name}"
        if self.split != "train":
            label += f"_{self.split}"
        return label

    @property
    def n_hand(self) -> int:
        return len(self.hand_bodies)

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

    def object_mesh_path(self):
        """Triangle mesh of the scene object, for playback overlays."""
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
        from supermimic.utils.kinematics import compile_g1

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

    ``split`` is recorded for parity with :class:`WeaveXHandTask` but does not
    select a cache: this release ships one rollout file per object, and the
    cached library keeps its ``motion_lengths`` across both.
    """

    def __init__(
        self,
        object_name: str = "smalltable",
        split: str = "train",
        options: dict | None = None,
    ) -> None:
        super().__init__(
            name="weave",
            num_joints=53,
            tracked_bodies=TRACKED_BODIES,
            contact_bodies=TRACKED_BODIES,
            has_object=True,
            foot_geom_expr=r"^(left|right)_foot_collision_[1-7]$",
            object_name=object_name,
            hand_bodies=HAND_BODIES,
            # Finger + object contacts: overflow asked for njmax≈554.
            sim=SimCfg(nconmax=256, njmax=768),
            split=split,
            options=dict(options or {}),
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

    def object_mesh_path(self):
        from supermimic.robots.object import object_mesh_path

        assert self.object_name is not None
        return object_mesh_path(self.object_name)

    def action_scale(self, joint_names: tuple[str, ...]) -> list[float]:
        from supermimic.robots.g1_inspire import action_scale

        return action_scale(joint_names)

    def load_motion(self, device: str):
        from supermimic.motions.weave import WeaveMotion

        name = self.motions_object_name
        assert name is not None
        lib = WeaveMotion.load(device=device, objects=(name,))
        return lib if self.has_object else lib.robot_only()


class WeaveXHandTask(TaskSpec):
    """XHand G1 plus a primitive-collided free object tracking a G1-Xhand-OMOMO clip.

    Same 53-DoF layout as :class:`WeaveTask` but a different robot asset (XHand
    instead of Inspire hands) and a different motion release, whose object
    colliders are fitted primitives (``assets/primitives``) rather than the
    visual mesh. Hand reward bodies are the XHand fingertips.

    ``simple_hands`` swaps the hand *mesh* colliders for the box/cylinder
    primitives of ``g1_hoi_learning``'s simplified URDF (see
    ``supermimic.robots.xhand_primitives``); the feet are capsules either way.
    """

    def __init__(
        self,
        object_name: str = "smalltable",
        split: str = "train",
        simple_hands: bool = False,
        options: dict | None = None,
    ) -> None:
        from supermimic.utils.kinematics import XHAND_BODIES

        super().__init__(
            name="weave_xhand",
            num_joints=53,
            tracked_bodies=TRACKED_BODIES,
            contact_bodies=TRACKED_BODIES,
            has_object=True,
            foot_geom_expr=r"^(left|right)_foot_collision_[1-9]$",
            object_name=object_name,
            hand_bodies=XHAND_BODIES,
            # 7 foot capsules per foot (from the Inspire model) plus the hand geoms
            # push the active constraints past the old 768; 1200 leaves headroom for
            # object contact and the solver's warm-start rows.
            sim=SimCfg(nconmax=256, njmax=1200),
            split=split,
            options=dict(options or {}),
        )
        self.simple_hands = bool(simple_hands)
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}.")

    def compile_model(self):
        from supermimic.robots.g1_xhand import compile_g1_xhand

        model, names = compile_g1_xhand(simple=self.simple_hands)
        if len(names) != self.num_joints:
            raise RuntimeError(f"XHand model has {len(names)} hinges, expected {self.num_joints}.")
        return model, names

    def robot_cfg(self):
        from supermimic.robots.g1_xhand import robot_cfg

        return robot_cfg(simple=self.simple_hands)

    def object_cfg(self):
        from supermimic.robots.object_xhand import object_cfg

        assert self.object_name is not None
        return object_cfg(self.object_name)

    def object_mesh_path(self):
        from supermimic.motions.weave_xhand import object_mesh_path

        assert self.object_name is not None
        return object_mesh_path(self.object_name)

    def action_scale(self, joint_names: tuple[str, ...]) -> list[float]:
        from supermimic.robots.g1_xhand import action_scale

        return action_scale(joint_names)

    def load_motion(self, device: str):
        from supermimic.motions.weave_xhand import WeaveXHandMotion

        name = self.motions_object_name
        assert name is not None
        lib = WeaveXHandMotion.load(device=device, objects=(name,), split=self.split)
        return lib if self.has_object else lib.robot_only()


_TASKS: dict[str, type[TaskSpec]] = {
    "lafan1": Lafan1Task,
    "weave": WeaveTask,
    "weave_xhand": WeaveXHandTask,
}

# Keys of the ``task`` config node that map to explicit constructor arguments.
_TASK_FIELDS = ("name", "object_name", "split", "simple_hands", "has_object")


def _task_options(task_cfg) -> dict:
    """Every key of the task config node, minus the ones handled explicitly."""
    if task_cfg is None:
        return {}
    if hasattr(task_cfg, "items"):
        items = task_cfg.items()
    else:
        items = dict(task_cfg).items()
    return {key: value for key, value in items if key not in _TASK_FIELDS}


def make_task_from_config(task_cfg) -> TaskSpec:
    """Build the task from a ``task`` config node (one per task config group).

    The node carries the shared keys (``name``, ``object_name``, ``split``,
    ``simple_hands``, ``has_object``) plus any task-specific flags, which are
    forwarded verbatim as :attr:`TaskSpec.options` so a task can define its own
    without touching the trainer config.
    """
    if task_cfg is None:
        raise ValueError("A task config node is required (defaults: `task: lafan1`).")

    def opt(key: str, cast):
        value = task_cfg.get(key) if hasattr(task_cfg, "get") else None
        return None if value is None else cast(value)

    return make_task(
        str(task_cfg.name),
        object_name=opt("object_name", str),
        split=opt("split", str),
        simple_hands=opt("simple_hands", bool),
        has_object=opt("has_object", bool),
        options=_task_options(task_cfg),
    )


def get_task(name: str) -> TaskSpec:
    try:
        factory = _TASKS[name]
    except KeyError as exc:
        known = ", ".join(sorted(_TASKS))
        raise KeyError(f"Unknown task {name!r}. Available: {known}.") from exc
    return factory()


def make_task(
    name: str,
    *,
    object_name: str | None = None,
    split: str | None = None,
    simple_hands: bool | None = None,
    has_object: bool | None = None,
    options: dict | None = None,
) -> TaskSpec:
    """Task with per-run overrides.

    ``object_name`` and ``split`` apply to the object tasks. ``split`` reaches
    ``weave_xhand`` only: its cache path is split-tagged, while the older
    ``weave`` cache holds both splits under one file. ``simple_hands`` likewise
    reaches ``weave_xhand`` only.

    ``has_object=False`` switches an object task to robot-only tracking: the scene
    object and every object observation/reward/termination term are dropped, while
    the motion cache (which does carry the object) is reused.

    ``options`` is the rest of the task config node; tasks may read their own flags
    from :attr:`TaskSpec.options`.
    """
    if name not in _TASKS:
        known = ", ".join(sorted(_TASKS))
        raise KeyError(f"Unknown task {name!r}. Available: {known}.")
    kwargs: dict = {"options": dict(options or {})}
    if name in ("weave", "weave_xhand") and object_name is not None:
        kwargs["object_name"] = object_name
    if name == "weave_xhand" and split is not None:
        kwargs["split"] = split
    if name == "weave_xhand" and simple_hands is not None:
        kwargs["simple_hands"] = simple_hands
    if name not in ("weave", "weave_xhand"):
        # Robot-only tasks have no options to forward.
        kwargs.pop("options")
    return _TASKS[name](**kwargs).with_options(has_object=has_object)
