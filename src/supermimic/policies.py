"""Load a trained policy from a checkpoint, for ``play`` and ``collect``.

``supermimic`` and ``nablamimic`` share the same actor (``pi``: an encoder-decoder
over a history of observations plus the goal), so one loader covers both. ``sac``
stored a different actor (``actor``: a flat ``[obs, goal]`` input behind its own
running observation normalizer).

The distinction that matters to callers is ``kind``:

``history``
    The policy consumed a ``HIST``-frame observation history, built from the
    simulator's observations. Used by SuperMimic / NablaMimic.
``flat``
    The policy consumed a single concatenated ``[obs, goal]`` vector, optionally
    normalized. Used by SAC.
"""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf

from supermimic.mdp import GOAL_FRAMES, HIST, PAPER_OBS_TERMS, WorldModelState
from supermimic.supermimic.network import InputNorms, build_policy
from supermimic.tasks import TaskSpec, make_task_from_config
from supermimic.utils.kinematics import make_kinematics

HISTORY = "history"
FLAT = "flat"


def load_checkpoint(ckpt: Path | str, device: str = "cuda") -> tuple[dict, dict, dict]:
    """Raw checkpoint plus its trainer config and task node."""
    saved = torch.load(Path(ckpt), map_location=device, weights_only=False)
    cfg = dict(saved.get("cfg", {}))
    node = saved.get("task_cfg")
    if node is None:
        raise ValueError(
            f"{ckpt} has no task_cfg; re-save it with a current training run "
            "(older checkpoints only store the flattened config keys)."
        )
    return saved, cfg, dict(node)


def task_from_checkpoint(ckpt: Path | str, device: str = "cpu") -> tuple[TaskSpec, dict, dict]:
    """Task spec, trainer config and raw checkpoint, without building a simulator."""
    saved, cfg, node = load_checkpoint(ckpt, device)
    return make_task_from_config(OmegaConf.create(node)), cfg, saved


def load_critic_norms(saved: dict, task: TaskSpec, obs_dim: int, device: str):
    """``InputNorms`` for building critic inputs, from any checkpoint.

    SuperMimic and NablaMimic store the critic normalizers under ``norms``; SAC
    stores them in the same place (its trainer saves ``self.norms``) even though
    SAC's actor has its own separate observation normalizer.
    """
    from supermimic.supermimic.network import InputNorms

    if "norms" not in saved:
        raise ValueError(
            "Checkpoint has no 'norms'; cannot build normalized critic inputs for "
            "the archive."
        )
    norms = InputNorms(obs_dim, task).to(device)
    norms.load_state_dict(saved["norms"])
    return norms


def obs_terms_from_cfg(cfg: dict) -> tuple[str, ...]:
    terms = cfg.get("obs_terms")
    return tuple(terms) if terms else PAPER_OBS_TERMS


def build_sim(
    task: TaskSpec,
    cfg: dict,
    device: str,
    *,
    num_envs: int,
    survival_bonus: float = 0.0,
):
    """A :class:`TrackingSim` reproducing the checkpoint's training setup.

    ``eval_mode`` starts every environment on a distinct clip and disables reset
    noise, which is what both the evaluation protocol and rollout collection want.
    """
    from supermimic.sim import TrackingSim

    model, joint_names = task.compile_model()
    kin = make_kinematics(
        model,
        joint_names,
        str(cfg.get("fk", "warp")),
        device,
        hand_bodies=task.hand_bodies,
    )
    kin.set_anchor(str(cfg.get("anchor", "torso")))
    return TrackingSim(
        num_envs,
        task.load_motion(device),
        kin,
        device,
        gamma=float(cfg.get("gamma", 0.99)),
        eval_mode=True,
        reset_noise=False,
        anchor=str(cfg.get("anchor", "torso")),
        obs_terms=obs_terms_from_cfg(cfg),
        reward_style=str(cfg.get("reward_style", "gaussian")),
        survival_bonus=survival_bonus,
        task=task,
    )


def wm_state(sim) -> WorldModelState:
    """Current robot (and object) state for the analytic MDP."""
    p, quat, v, w, q, qd = sim.state()
    bodies = sim.tracked_bodies()
    obj = sim.object_state() if sim.task.has_object else None
    return WorldModelState(
        p, quat, v, w, q, qd, bodies[0], bodies[1],
        *(obj if obj is not None else (None, None, None, None)),
    )


class LoadedPolicy:
    """A policy bound to a simulator, ready to act on its current state.

    ``act()`` returns the deterministic action for the simulator's *current* state
    and advances any internal observation history. Call ``reset(clip)`` before
    replaying a new clip.
    """

    def __init__(
        self,
        algo: str,
        kind: str,
        sim,
        actor,
        *,
        vecnorm=None,
        norms=None,
        live: bool = False,
    ):
        self.algo = algo
        self.kind = kind
        self.sim = sim
        self.actor = actor
        self.vecnorm = vecnorm
        self.norms = norms
        self.live = live
        self.obs_hist = (
            torch.zeros(sim.num_envs, HIST, sim.mdp.obs_dim, device=sim.device)
            if kind == HISTORY
            else None
        )

    def reset(self, clip: int) -> None:
        start = self.sim.motion.clip_start[clip].to(self.sim.device).view(1)
        env_ids = torch.zeros(1, dtype=torch.long, device=self.sim.device)
        self.sim.reset(env_ids, start)
        if self.obs_hist is not None:
            self.obs_hist.zero_()

    @torch.no_grad()
    def act(self) -> torch.Tensor:
        sim = self.sim
        state = wm_state(sim)
        obs = sim.mdp.obs(state, sim.a_prev, sim.xi, sim.f)
        goal = sim.motion.goal(sim.f, GOAL_FRAMES)
        if self.kind == HISTORY:
            assert self.obs_hist is not None
            fresh = sim.ep_step == 0
            self.obs_hist = torch.where(
                fresh[:, None, None],
                obs[:, None, :].expand(-1, HIST, -1),
                torch.cat((self.obs_hist[:, 1:], obs[:, None, :]), dim=1),
            )
            return self.actor(self.obs_hist, goal, self.norms, deterministic=True)
        policy_obs = torch.cat([obs, goal], dim=-1)
        obs_n = self.vecnorm(policy_obs) if self.vecnorm is not None else policy_obs
        loc, _scale = self.actor(obs_n)
        return loc


def policy_state_dict(saved: dict, *, live: bool = False) -> dict:
    """Select the policy weights to use, preferring the EMA average.

    Checkpoints write the Polyak-averaged actor under ``pi`` / ``actor`` and the
    online one under ``pi_live`` / ``actor_live``, so the default (``live=False``)
    is the EMA. ``live=True`` picks the online weights when they were saved.
    Mirrors ``supermimic.eval._policy_state_dict``.
    """
    if live and "pi_live" in saved:
        return saved["pi_live"]
    if "pi" in saved:
        return saved["pi"]
    if live and "actor_live" in saved:
        return saved["actor_live"]
    if "actor" in saved:
        return saved["actor"]
    keys = ", ".join(sorted(saved.keys()))
    raise KeyError(f"No policy weights in checkpoint. Keys: {keys}")


def has_ema(saved: dict) -> bool:
    """Whether the checkpoint carries a Polyak-averaged actor."""
    return "pi" in saved and "pi_live" in saved or "actor" in saved and "actor_live" in saved


def load_policy(
    saved: dict, cfg: dict, task: TaskSpec, sim, device: str, *, live: bool = False
) -> LoadedPolicy:
    """Bind the checkpoint's actor to ``sim``.

    Accepts SuperMimic, NablaMimic (both store ``pi``) and SAC (``actor``).
    Defaults to the EMA weights (``live=False``), matching ``supermimic.eval``.
    """
    weights = policy_state_dict(saved, live=live)
    if "pi" in saved or "pi_live" in saved:
        actor = build_policy(
            sim.mdp.obs_dim, task.goal_dim, HIST, cfg, task.num_joints
        ).to(device)
        actor.load_state_dict(weights)
        actor.eval()
        norms = InputNorms(sim.mdp.obs_dim, task).to(device)
        norms.load_state_dict(saved["norms"])
        # NablaMimic shares SuperMimic's actor; report which trainer produced it.
        algo = "nablamimic" if "act_cot_clip" in cfg else "supermimic"
        return LoadedPolicy(algo, HISTORY, sim, actor, norms=norms, live=live)

    if "actor" in saved:
        from supermimic.sac.network import NormalActor
        from supermimic.utils.nn import ObsVecNorm

        policy_obs_dim = sim.mdp.obs_dim + task.goal_dim
        actor = NormalActor(
            policy_obs_dim,
            task.num_joints,
            action_init=str(cfg.get("actor_init", "zeros")),
        ).to(device)
        actor.load_state_dict(weights)
        actor.eval()
        vecnorm = None
        if bool(cfg.get("vecnorm", True)) and "vecnorm" in saved:
            vecnorm = ObsVecNorm(policy_obs_dim).to(device)
            vecnorm.load_state_dict(saved["vecnorm"])
            vecnorm.eval()
        return LoadedPolicy("sac", FLAT, sim, actor, vecnorm=vecnorm, live=live)

    keys = ", ".join(sorted(saved.keys()))
    raise ValueError(f"Checkpoint is not SuperMimic/NablaMimic (pi) or SAC (actor). Keys: {keys}")


def build_policy_from_checkpoint(
    ckpt: Path | str,
    device: str = "cuda",
    *,
    num_envs: int = 1,
    survival_bonus: float = 0.0,
) -> LoadedPolicy:
    """Convenience: load the task from the checkpoint and bind its policy."""
    task, cfg, saved = task_from_checkpoint(ckpt, device)
    sim = build_sim(task, cfg, device, num_envs=num_envs, survival_bonus=survival_bonus)
    return load_policy(saved, cfg, task, sim, device)
