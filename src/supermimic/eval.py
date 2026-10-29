"""Play each LAFAN1 clip from its first frame and report normalized return."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from supermimic.kinematics import compile_g1, make_kinematics
from supermimic.mdp import (
    GOAL_DIM,
    GOAL_FRAMES,
    HIST,
    NUM_J,
    PAPER_OBS_TERMS,
    TERM_HEIGHT,
    TERM_TILT,
    WorldModelState,
)
from supermimic.motion import CONTROL_DT, MotionLib
from supermimic.sim import TrackingSim
from supermimic.supermimic.network import InputNorms, build_policy


def _build_sim(saved_cfg: dict, num_envs: int, device: str) -> TrackingSim:
    motion = MotionLib.load(device=device)
    model, names = compile_g1()
    kin = make_kinematics(model, names, str(saved_cfg.get("fk", "warp")), device)
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    alive = torch.ones(n, dtype=torch.bool, device=device)
    returns = torch.zeros(n, device=device)
    failed = torch.zeros(n, dtype=torch.bool, device=device)
    fail_step = torch.full((n,), -1, dtype=torch.long, device=device)
    fail_height = torch.zeros(n, device=device)
    fail_tilt = torch.zeros(n, device=device)
    pbar = tqdm(total=n, desc="Eval clips", unit="clip", disable=not progress)
    step = 0
    while bool(alive.any()):
        out = step_fn(sim, motion, alive)
        first = out["d"] & alive
        if bool(first.any()):
            p1, quat1, _, _, _, _ = sim.state()
            height, tilt = sim.mdp._errors(p1, quat1, sim.f)
            fail_step = torch.where(first, sim.ep_step, fail_step)
            fail_height = torch.where(first, height, fail_height)
            fail_tilt = torch.where(first, tilt, fail_tilt)
        returns = returns + out["r"] * alive.float()
        failed = failed | first
        alive = alive & ~out["done"]
        step += 1
        pbar.n = n - int(alive.sum().item())
        pbar.set_postfix(steps=step, alive=int(alive.sum().item()), refresh=False)
        pbar.refresh()
    pbar.close()
    return returns, failed, fail_step, fail_height, fail_tilt


def evaluate_supermimic(
    saved: dict,
    saved_cfg: dict,
    device: str,
    *,
    live: bool,
    progress: bool,
) -> None:
    motion = MotionLib.load(device=device)
    n = motion.num_clips
    sim = _build_sim(saved_cfg, n, device)
    policy = build_policy(sim.mdp.obs_dim, GOAL_DIM, HIST, saved_cfg).to(device)
    policy.load_state_dict(_policy_state_dict(saved, live=live))
    policy.eval()
    norms = InputNorms(sim.mdp.obs_dim).to(device)
    norms.load_state_dict(saved["norms"])
    obs_hist = torch.zeros(n, HIST, sim.mdp.obs_dim, device=device)

    def step_fn(sim: TrackingSim, motion: MotionLib, alive: torch.Tensor) -> dict:
        p, quat, v, w, q, qd = sim.state()
        bodies = sim.tracked_bodies()
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

    returns, failed, fail_step, fail_height, fail_tilt = _run_rollout(
        sim, motion, n, device, step_fn, progress=progress
    )
    _print_results(motion, sim, returns, failed, fail_step, fail_height, fail_tilt, live=live, algo="supermimic")


def evaluate_sac(saved: dict, saved_cfg: dict, device: str, *, live: bool, progress: bool) -> None:
    from supermimic.sac.network import NormalActor
    from supermimic.sac.util import ObsVecNorm

    motion = MotionLib.load(device=device)
    n = motion.num_clips
    sim = _build_sim(saved_cfg, n, device)
    policy_obs_dim = sim.mdp.obs_dim + GOAL_DIM
    actor = NormalActor(
        policy_obs_dim,
        NUM_J,
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
    norms = InputNorms(sim.mdp.obs_dim).to(device)
    if "norms" in saved:
        norms.load_state_dict(saved["norms"])

    def step_fn(sim: TrackingSim, motion: MotionLib, alive: torch.Tensor) -> dict:
        p, quat, v, w, q, qd = sim.state()
        bodies = sim.tracked_bodies()
        state = WorldModelState(p, quat, v, w, q, qd, bodies[0], bodies[1])
        obs = sim.mdp.obs(state, sim.a_prev, sim.xi, sim.f)
        goal = motion.goal(sim.f, GOAL_FRAMES)
        policy_obs = torch.cat([obs, goal], dim=-1)
        with torch.no_grad():
            obs_n = vecnorm(policy_obs) if vecnorm is not None else policy_obs
            loc, _scale = actor(obs_n)
            action = loc
        return sim.step(action, auto_reset=False)

    returns, failed, fail_step, fail_height, fail_tilt = _run_rollout(
        sim, motion, n, device, step_fn, progress=progress
    )
    _print_results(motion, sim, returns, failed, fail_step, fail_height, fail_tilt, live=live, algo="sac")


def _print_results(
    motion: MotionLib,
    sim: TrackingSim,
    returns: torch.Tensor,
    failed: torch.Tensor,
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
    print(f"[{algo} {which}] normalized return {float(score.mean()):.4f}  failed {int(failed.sum())}/{n}")
    for name, value, fail, step, height, tilt in zip(
        motion.clip_names,
        score.tolist(),
        failed.tolist(),
        fail_step.tolist(),
        fail_height.tolist(),
        fail_tilt.tolist(),
    ):
        if not fail:
            print(f"  {name:28} {value:.3f}  ok")
            continue
        parts = []
        if height > TERM_HEIGHT:
            parts.append(f"height {height:.3f} m > {TERM_HEIGHT:.2f}")
        if tilt > TERM_TILT:
            parts.append(f"tilt {tilt:.3f} > {TERM_TILT:.2f}")
        cause = " and ".join(parts) if parts else "hard termination"
        print(f"  {name:28} {value:.3f}  fail  {cause} at frame {step} ({step * CONTROL_DT:.2f} s)")


def evaluate(ckpt: Path, device: str, *, live: bool = False, progress: bool = True) -> None:
    torch.manual_seed(0)
    saved = torch.load(ckpt, map_location=device, weights_only=False)
    saved_cfg = saved.get("cfg", {})
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
