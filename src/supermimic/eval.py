"""Play each LAFAN1 clip from its first frame and report normalized return."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .kinematics import compile_g1, make_kinematics
from .mdp import GOAL_DIM, GOAL_FRAMES, HIST, PAPER_OBS_TERMS
from .motion import MotionLib
from .network import InputNorms, Policy, policy_widths
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
    policy = Policy(
        sim.mdp.obs_dim,
        GOAL_DIM,
        HIST,
        z_dim=int(saved_cfg.get("policy_z_dim", 32)),
        stochastic=bool(saved_cfg.get("stochastic_actor", False)),
        encoder_hidden=policy_widths(saved_cfg, "policy_encoder"),
        decoder_hidden=policy_widths(saved_cfg, "policy_decoder"),
        norm=str(saved_cfg.get("policy_norm", "none")),
    ).to(device)
    policy.load_state_dict(saved["pi"])
    norms = InputNorms(sim.mdp.obs_dim).to(device)
    norms.load_state_dict(saved["norms"])
    policy.eval()
    obs_hist = torch.zeros(n, HIST, sim.mdp.obs_dim, device=device)
    alive = torch.ones(n, dtype=torch.bool, device=device)
    returns = torch.zeros(n, device=device)
    failed = torch.zeros(n, dtype=torch.bool, device=device)
    lengths = (motion.clip_end - motion.clip_start - 1).clamp(min=1)
    while bool(alive.any()):
        p, quat, v, w, q, qd = sim.state()
        obs = sim.mdp.obs(p, quat, v, w, q, qd, sim.a_prev, sim.xi, sim.f)
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
        returns = returns + out["r"] * alive.float()
        failed = failed | (out["d"] & alive)
        alive = alive & ~out["done"]
    score = returns / (sim.mdp.max_step_reward * lengths.float())
    print(f"normalized return {float(score.mean()):.4f}  failed {int(failed.sum())}/{n}")
    for name, value, fail in zip(motion.clip_names, score.tolist(), failed.tolist()):
        print(f"  {name:28} {value:.3f}  {'fail' if fail else 'ok'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    evaluate(args.ckpt, args.device)


if __name__ == "__main__":
    main()
