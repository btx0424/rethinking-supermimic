"""Play each LAFAN1 clip from its first frame and report normalized return."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .kinematics import compile_g1, make_kinematics
from .mdp import GOAL_DIM, GOAL_FRAMES, HIST, PAPER_OBS_TERMS, TERM_HEIGHT, TERM_TILT
from .motion import CONTROL_DT, MotionLib
from .mdp import WorldModelState
from .network import InputNorms, build_policy
from .sim import TrackingSim


def evaluate(ckpt: Path, device: str) -> None:
    torch.manual_seed(0)
    motion = MotionLib.load(device=device)
    model, names = compile_g1()
    saved = torch.load(ckpt, map_location=device, weights_only=False)
    saved_cfg = saved.get("cfg", {})
    kin = make_kinematics(model, names, str(saved_cfg.get("fk", "warp")), device)
    n = motion.num_clips
    kin.set_anchor(str(saved_cfg.get("anchor", "torso")))
    sim = TrackingSim(
        n,
        motion,
        kin,
        device,
        eval_mode=True,
        reset_noise=False,
        anchor=str(saved_cfg.get("anchor", "torso")),
        obs_terms=tuple(saved_cfg.get("obs_terms", PAPER_OBS_TERMS)),
        reward_style=str(saved_cfg.get("reward_style", "gaussian")),
    )
    policy = build_policy(sim.mdp.obs_dim, GOAL_DIM, HIST, saved_cfg).to(device)
    policy.load_state_dict(saved["pi"])
    norms = InputNorms(sim.mdp.obs_dim).to(device)
    norms.load_state_dict(saved["norms"])
    policy.eval()
    obs_hist = torch.zeros(n, HIST, sim.mdp.obs_dim, device=device)
    alive = torch.ones(n, dtype=torch.bool, device=device)
    returns = torch.zeros(n, device=device)
    failed = torch.zeros(n, dtype=torch.bool, device=device)
    fail_step = torch.full((n,), -1, dtype=torch.long, device=device)
    fail_height = torch.zeros(n, device=device)
    fail_tilt = torch.zeros(n, device=device)
    lengths = (motion.clip_end - motion.clip_start - 1).clamp(min=1)
    while bool(alive.any()):
        p, quat, v, w, q, qd = sim.state()
        bodies = sim.tracked_bodies()
        state = WorldModelState(p, quat, v, w, q, qd, bodies[0], bodies[1])
        obs = sim.mdp.obs(state, sim.a_prev, sim.xi, sim.f)
        fresh = sim.ep_step == 0
        obs_hist = torch.where(
            fresh[:, None, None],
            obs[:, None, :].expand(-1, HIST, -1),
            torch.cat((obs_hist[:, 1:], obs[:, None, :]), dim=1),
        )
        goal = motion.goal(sim.f, GOAL_FRAMES)
        with torch.no_grad():
            action = policy(obs_hist, goal, norms, deterministic=True)
        out = sim.step(action, auto_reset=False)
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
    score = returns / (sim.mdp.max_step_reward * lengths.float())
    print(f"normalized return {float(score.mean()):.4f}  failed {int(failed.sum())}/{n}")
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    evaluate(args.ckpt, args.device)


if __name__ == "__main__":
    main()
