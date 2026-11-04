"""Play each LAFAN1 clip from its first frame and report normalized return."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from supermimic.utils.kinematics import make_kinematics
from supermimic.mdp import (
    GOAL_FRAMES,
    HIST,
    PAPER_OBS_TERMS,
    TERM_HEIGHT,
    TERM_TILT,
    WorldModelState,
)
from supermimic.motion import CONTROL_DT, MotionLib
from supermimic.sac import features as sac_features
from supermimic.sim import TrackingSim
from supermimic.tasks import TaskSpec, make_task_from_config
from supermimic.supermimic.network import InputNorms, build_policy


def _flatten_saved(saved: dict) -> dict:
    """Checkpoint payload with its ``cfg`` dict flattened in, plus ``task_cfg``.

    Older code passed only the ``cfg`` dict around; the task node lives beside it,
    so the two are merged into one mapping here.
    """
    flat = dict(saved.get("cfg", {}))
    flat["task_cfg"] = saved.get("task_cfg")
    return flat


def _task_from_cfg(saved_cfg: dict) -> TaskSpec:
    """Task from a checkpoint payload (or its flattened ``cfg`` dict).

    The checkpoint stores the whole ``task`` config node under ``task_cfg``, so any
    task-specific option round-trips. Older checkpoints only have the flattened
    keys, which are rebuilt into an equivalent node. ``weave_xhand`` is evaluated on
    its held-out test split; ``weave`` keeps the split it trained on, because its
    cache is not split-tagged.
    """
    node = saved_cfg.get("task_cfg")
    if node is None:
        node = {
            "name": saved_cfg.get("task", "lafan1"),
            "object_name": saved_cfg.get("object_name"),
            "split": saved_cfg.get("split", "train"),
            "simple_hands": saved_cfg.get("simple_hands", True),
            "has_object": saved_cfg.get("has_object"),
        }
    node = dict(node)
    if node.get("name") == "weave_xhand":
        node["split"] = "test"
    return make_task_from_config(OmegaConf.create(node))


def _build_sim(saved_cfg: dict, num_envs: int, device: str) -> TrackingSim:
    task = _task_from_cfg(saved_cfg)
    motion = task.load_motion(device)
    model, names = task.compile_model()
    kin = make_kinematics(
        model, names, str(saved_cfg.get("fk", "warp")), device, hand_bodies=task.hand_bodies
    )
    kin.set_anchor(str(saved_cfg.get("anchor", "torso")))
    return TrackingSim(
        num_envs,
        motion,
        kin,
        device,
        eval_mode=True,
        reset_noise=False,
        anchor=str(saved_cfg.get("anchor", "torso")),
        obs_terms=tuple(saved_cfg.get("obs_terms", PAPER_OBS_TERMS)),
        reward_style=str(saved_cfg.get("reward_style", "gaussian")),
        task=task,
    )


def _policy_state_dict(saved: dict, *, live: bool) -> dict:
    """Checkpoints store EMA weights under ``pi`` when ``pi_live`` exists."""
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


def _run_rollout(
    sim: TrackingSim,
    motion: MotionLib,
    n: int,
    device: str,
    step_fn,
    *,
    progress: bool = True,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Roll every clip and report ``returns, terminated, truncated, fail_*``.

    ``terminated`` marks clips the policy actually failed; ``truncated`` marks
    clips that simply ran out (time limit or end of the reference), which is not
    a failure. A clip can be both, in which case it counts as terminated.
    """
    alive = torch.ones(n, dtype=torch.bool, device=device)
    returns = torch.zeros(n, device=device)
    terminated = torch.zeros(n, dtype=torch.bool, device=device)
    truncated = torch.zeros(n, dtype=torch.bool, device=device)
    fail_step = torch.full((n,), -1, dtype=torch.long, device=device)
    fail_height = torch.zeros(n, device=device)
    fail_tilt = torch.zeros(n, device=device)
    pbar = tqdm(total=n, desc="Eval clips", unit="clip", disable=not progress)
    step = 0
    while bool(alive.any()):
        out = step_fn(sim, motion, alive)
        # ``terminated`` is a genuine failure (the clip is marked failed and the
        # error that caused it is recorded). ``truncated`` is a clean cut by the
        # time limit or the end of the clip and must not count as a failure.
        first = out["terminated"] & alive
        if bool(first.any()):
            p1, quat1, _, _, _, _ = sim.state()
            height, tilt = sim.mdp._errors(p1, quat1, sim.f)
            fail_step = torch.where(first, sim.ep_step, fail_step)
            fail_height = torch.where(first, height, fail_height)
            fail_tilt = torch.where(first, tilt, fail_tilt)
        returns = returns + out["r"] * alive.float()
        terminated = terminated | first
        truncated = truncated | (out["truncated"] & alive)
        alive = alive & ~out["done"]
        step += 1
        pbar.n = n - int(alive.sum().item())
    pbar.set_postfix(steps=step, alive=int(alive.sum().item()), refresh=False)
    pbar.refresh()
    pbar.close()
    return returns, terminated, truncated, fail_step, fail_height, fail_tilt


def evaluate_supermimic(
    saved: dict,
    saved_cfg: dict,
    device: str,
    *,
    live: bool,
    progress: bool,
) -> None:
    task = _task_from_cfg(saved_cfg)
    motion = task.load_motion(device)
    n = motion.num_clips
    sim = _build_sim(saved_cfg, n, device)
    motion = sim.motion
    policy = build_policy(
        sim.mdp.obs_dim, sim.mdp.task.goal_dim, HIST, saved_cfg, sim.mdp.task.num_joints
    ).to(device)
    policy.load_state_dict(_policy_state_dict(saved, live=live))
    policy.eval()
    norms = InputNorms(sim.mdp.obs_dim, sim.mdp.task).to(device)
    norms.load_state_dict(saved["norms"])
    obs_hist = torch.zeros(n, HIST, sim.mdp.obs_dim, device=device)

    def step_fn(sim: TrackingSim, motion: MotionLib, alive: torch.Tensor) -> dict:
        p, quat, v, w, q, qd = sim.state()
        bodies = sim.tracked_bodies()
        if sim.task.has_object:
            state = WorldModelState(
                p, quat, v, w, q, qd, bodies[0], bodies[1], *sim.object_state()
            )
        else:
            state = WorldModelState(p, quat, v, w, q, qd, bodies[0], bodies[1])
        obs = sim.mdp.obs(state, sim.a_prev, sim.xi, sim.f)
        fresh = sim.ep_step == 0
        obs_hist[:] = torch.where(
            fresh[:, None, None],
            obs[:, None, :].expand(-1, HIST, -1),
            torch.cat((obs_hist[:, 1:], obs[:, None, :]), dim=1),
        )
        goal = motion.goal(sim.f, GOAL_FRAMES)
        with torch.no_grad():
            action = policy(obs_hist, goal, norms, deterministic=True)
        return sim.step(action, auto_reset=False)

    returns, terminated, truncated, fail_step, fail_height, fail_tilt = _run_rollout(
        sim, motion, n, device, step_fn, progress=progress
    )
    _print_results(
        motion,
        sim,
        returns,
        terminated,
        truncated,
        fail_step,
        fail_height,
        fail_tilt,
        live=live,
        algo="supermimic",
    )


def evaluate_sac(saved: dict, saved_cfg: dict, device: str, *, live: bool, progress: bool) -> None:
    from supermimic.sac.network import NormalActor
    from supermimic.utils.nn import ObsVecNorm

    task = _task_from_cfg(saved_cfg)
    motion = task.load_motion(device)
    n = motion.num_clips
    sim = _build_sim(saved_cfg, n, device)
    motion = sim.motion
    policy_obs_dim = sim.mdp.obs_dim + sim.mdp.task.goal_dim
    actor = NormalActor(
        policy_obs_dim,
        sim.mdp.task.num_joints,
        action_init=str(saved_cfg.get("actor_init", "zeros")),
    ).to(device)
    actor.load_state_dict(_policy_state_dict(saved, live=live))
    actor.eval()
    use_vecnorm = bool(saved_cfg.get("vecnorm", True))
    vecnorm = ObsVecNorm(policy_obs_dim).to(device) if use_vecnorm else None
    if vecnorm is not None and "vecnorm" in saved:
        vecnorm.load_state_dict(saved["vecnorm"])
    if vecnorm is not None:
        vecnorm.eval()
    norms = InputNorms(sim.mdp.obs_dim, sim.mdp.task).to(device)
    if "norms" in saved:
        norms.load_state_dict(saved["norms"])

    def step_fn(sim: TrackingSim, motion: MotionLib, alive: torch.Tensor) -> dict:
        # Shared builder so HOI tasks include the object block.
        policy_obs = sac_features.policy_obs(sim, noise=False)
        with torch.no_grad():
            obs_n = vecnorm(policy_obs) if vecnorm is not None else policy_obs
            loc, _scale = actor(obs_n)
            action = loc
        return sim.step(action, auto_reset=False)

    returns, terminated, truncated, fail_step, fail_height, fail_tilt = _run_rollout(
        sim, motion, n, device, step_fn, progress=progress
    )
    _print_results(
        motion,
        sim,
        returns,
        terminated,
        truncated,
        fail_step,
        fail_height,
        fail_tilt,
        live=live,
        algo="sac",
    )


def _print_results(
    motion: MotionLib,
    sim: TrackingSim,
    returns: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    fail_step: torch.Tensor,
    fail_height: torch.Tensor,
    fail_tilt: torch.Tensor,
    *,
    live: bool,
    algo: str,
) -> None:
    n = motion.num_clips
    lengths = (motion.clip_end - motion.clip_start - 1).clamp(min=1)
    score = returns / (sim.mdp.max_step_reward * lengths.float())
    which = "online" if live else "eval"
    print(
        f"[{algo} {which}] normalized return {float(score.mean()):.4f}  "
        f"terminated {int(terminated.sum())}/{n}  truncated {int(truncated.sum())}/{n}"
    )
    for name, value, term, trunc, step, height, tilt in zip(
        motion.clip_names,
        score.tolist(),
        terminated.tolist(),
        truncated.tolist(),
        fail_step.tolist(),
        fail_height.tolist(),
        fail_tilt.tolist(),
    ):
        if trunc and not term:
            print(f"  {name:28} {value:.3f}  ok (truncated)")
            continue
        if not term:
            print(f"  {name:28} {value:.3f}  ok")
            continue
        parts = []
        if height > TERM_HEIGHT:
            parts.append(f"height {height:.3f} m > {TERM_HEIGHT:.2f}")
        if tilt > TERM_TILT:
            parts.append(f"tilt {tilt:.3f} > {TERM_TILT:.2f}")
        cause = " and ".join(parts) if parts else "hard termination"
        print(f"  {name:28} {value:.3f}  terminated  {cause} at frame {step} ({step * CONTROL_DT:.2f} s)")


def evaluate(ckpt: Path, device: str, *, live: bool = False, progress: bool = True) -> None:
    torch.manual_seed(0)
    saved = torch.load(ckpt, map_location=device, weights_only=False)
    # One flat mapping: the trainer ``cfg`` plus the ``task_cfg`` node beside it.
    saved_cfg = _flatten_saved(saved)
    if "pi" in saved or "pi_live" in saved:
        evaluate_supermimic(saved, saved_cfg, device, live=live, progress=progress)
    elif "actor" in saved or "actor_live" in saved:
        evaluate_sac(saved, saved_cfg, device, live=live, progress=progress)
    else:
        keys = ", ".join(sorted(saved.keys()))
        raise ValueError(f"Checkpoint is not SuperMimic (pi) or SAC (actor). Keys: {keys}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Evaluate the online policy (pi_live / actor_live) instead of the EMA checkpoint weights.",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable the rollout progress bar.")
    args = parser.parse_args()
    evaluate(args.ckpt, args.device, live=args.live, progress=not args.quiet)


if __name__ == "__main__":
    main()
