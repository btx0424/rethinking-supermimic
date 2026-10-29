"""Horizon ablation of the actor gradient through one world model.

Same actor, critic, world model, and imagination starts. Only the horizon
changes. For each H the script records the gradient SNR, the mean per-start
cosine, the mean and individual gradient norms, and the simulator return after
one step along the mean gradient.

Usage:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/check_horizon_snr.py \
    --ckpt logs/lafan1/<run>/ckpt_002000.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from supermimic.algo import Trainer
from supermimic.mdp import GOAL_FRAMES, HIST, WorldModelState
from check_grad_variance import COLLECT, NUM_ENVS, SAMPLES, _cfg, _dot_cosine, imagine_grads, snr_stats, write_report

HORIZONS = (1, 2, 4, 8, 16, 32)


def _grads(trainer: Trainer, start: dict[str, torch.Tensor], horizon: int) -> dict[str, torch.Tensor]:
    trainer.cfg.horizon = horizon
    return imagine_grads(trainer, start)


def _sim_return(trainer: Trainer, steps: int, frames: torch.Tensor) -> float:
    """Undiscounted return over ``steps``, normalized by the max per-step reward."""
    sim = trainer.sim
    n = sim.num_envs
    env_ids = torch.arange(n, device=trainer.device)
    sim.reset(env_ids, frames)
    obs_hist = torch.zeros(n, HIST, sim.mdp.obs_dim, device=trainer.device)
    total = torch.zeros(n, device=trainer.device)
    alive = torch.ones(n, device=trainer.device)
    for _ in range(steps):
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
        goal = trainer.motion.goal(sim.f, GOAL_FRAMES)
        with torch.no_grad():
            action = trainer.policy(obs_hist, goal, trainer.norms, deterministic=True)
        out = sim.step(action, auto_reset=False)
        total = total + out["r"] * alive
        alive = alive * (~out["done"]).float()
        if not bool(alive.any()):
            break
    return float(total.mean() / trainer.r_max)


def _apply(params: list[torch.nn.Parameter], mean: torch.Tensor, lr: float) -> None:
    offset = 0
    for param in params:
        count = param.numel()
        delta = mean[offset : offset + count].to(device=param.device, dtype=param.dtype).view_as(param)
        param.data.add_(delta, alpha=-lr)
        offset += count


def _plot(rows: list[dict], baseline: float, path: Path) -> None:
    h = [row["horizon"] for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    axes[0].plot(h, [row["param_snr"] for row in rows], marker="o", lw=2, label="total")
    axes[0].plot(h, [row["reward_snr"] for row in rows], marker="o", lw=2, label="reward")
    axes[0].plot(h, [row["critic_snr"] for row in rows], marker="o", lw=2, label="critic")
    axes[0].set_ylabel("SNR")
    axes[0].set_title("parameter SNR")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(h, [row["action_snr"] for row in rows], marker="o", lw=2, label="total")
    axes[1].plot(h, [row["action_reward_snr"] for row in rows], marker="o", lw=2, label="reward")
    axes[1].plot(h, [row["action_critic_snr"] for row in rows], marker="o", lw=2, label="critic")
    axes[1].set_ylabel("SNR")
    axes[1].set_title("action SNR")
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].plot(h, [row["sim"] for row in rows], marker="o", lw=2, color="#d62728", label="after update")
    axes[2].axhline(baseline, color="k", lw=1, ls="--", label="before update")
    axes[2].set_ylabel("normalized return")
    axes[2].set_title("simulator return")
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set_xlabel("horizon")
        ax.set_xticks(h)
        ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=160)
    print(f"wrote {path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    parser.add_argument("--num-envs", type=int, default=NUM_ENVS)
    parser.add_argument("--collect", type=int, default=COLLECT)
    parser.add_argument("--samples", type=int, default=SAMPLES)
    parser.add_argument("--eval-steps", type=int, default=150)
    parser.add_argument("--lr", type=float, default=None, help="Step size on the mean gradient. Defaults to pi_lr.")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    saved = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    cfg = _cfg(saved.get("cfg", {}), args.num_envs, args.device)
    cfg.horizon = max(args.horizons)
    cfg.pi_batch = args.samples
    cfg.buffer_steps = max(args.collect + 4, max(args.horizons) + 4)
    out = args.out or (args.ckpt.parent / "horizon_snr.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(cfg, out.parent / "horizon_snr")
    trainer.load_checkpoint(args.ckpt)
    trainer.policy.train()
    lr = float(saved.get("cfg", {}).get("pi_lr", 5e-4) if args.lr is None else args.lr)
    with torch.no_grad():
        trainer.collect(args.collect)
    start = {key: value.detach() for key, value in trainer.sample_policy_starts().items()}
    frames, _bins = trainer.sim._sample_frames(trainer.sim.num_envs)
    params = [p for p in trainer.policy.parameters() if p.requires_grad]
    baseline_weights = {id(p): p.detach().clone() for p in params}
    trainer.policy.eval()
    baseline = _sim_return(trainer, args.eval_steps, frames)
    print(f"baseline normalized return {baseline:.4f}  lr {lr:g}  starts {args.samples}")

    rows = []
    for horizon in args.horizons:
        print(f"horizon {horizon}", flush=True)
        trainer.policy.train()
        packed = _grads(trainer, start, horizon)
        param = snr_stats(packed["param"])
        action = snr_stats(packed["action"])
        reward = snr_stats(packed["param_reward"])
        critic = snr_stats(packed["param_critic"])
        action_reward = snr_stats(packed["action_reward"])
        action_critic = snr_stats(packed["action_critic"])
        for p in params:
            p.data.copy_(baseline_weights[id(p)])
        _apply(params, param["mean"], lr)
        trainer.policy.eval()
        sim = _sim_return(trainer, args.eval_steps, frames)
        for p in params:
            p.data.copy_(baseline_weights[id(p)])
        row = {
            "horizon": horizon,
            "param_snr": param["snr"],
            "action_snr": action["snr"],
            "param_cos": param["cosine"],
            "action_cos": action["cosine"],
            "mean_norm": param["mean_norm"],
            "individual_norm": param["individual_norm"],
            "action_mean_norm": action["mean_norm"],
            "action_individual_norm": action["individual_norm"],
            "reward_snr": reward["snr"],
            "critic_snr": critic["snr"],
            "reward_cos": reward["cosine"],
            "critic_cos": critic["cosine"],
            "reward_norm": reward["mean_norm"],
            "critic_norm": critic["mean_norm"],
            "action_reward_snr": action_reward["snr"],
            "action_critic_snr": action_critic["snr"],
            "reward_vs_critic": _dot_cosine(reward["mean"], critic["mean"]),
            "sim": sim,
        }
        rows.append(row)
        print(
            f"  param SNR {row['param_snr']:.3f} cos {row['param_cos']:.3f} "
            f"||μ|| {row['mean_norm']:.4g} ||g|| {row['individual_norm']:.4g}",
            flush=True,
        )
        print(
            f"  action SNR {row['action_snr']:.3f} cos {row['action_cos']:.3f} "
            f"||μ|| {row['action_mean_norm']:.4g} ||g|| {row['action_individual_norm']:.4g}  sim {row['sim']:.4f}",
            flush=True,
        )
    print(f"\n{'H':>4} {'pSNR':>8} {'rSNR':>8} {'cSNR':>8} {'r·c':>8} {'aSNR':>8} {'ar':>8} {'ac':>8} {'sim':>8}")
    lines = [
        f"# Horizon SNR {args.ckpt.parent.name} / {args.ckpt.name}",
        "",
        f"N={args.samples}, collect={args.collect}, num_envs={args.num_envs}, eval_steps={args.eval_steps}, lr={lr:g}.",
        f"Baseline normalized return {baseline:.4f}.",
        "Reward is the imagined return. Critic is the bootstrap. The policy loss is their sum.",
        "",
        "| H | param SNR | reward SNR | critic SNR | reward‖μ‖ | critic‖μ‖ | reward·critic | action SNR | action reward SNR | action critic SNR | sim |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        print(
            f"{row['horizon']:4d} {row['param_snr']:8.3f} {row['reward_snr']:8.3f} {row['critic_snr']:8.3f} "
            f"{row['reward_vs_critic']:+8.3f} {row['action_snr']:8.3f} {row['action_reward_snr']:8.3f} "
            f"{row['action_critic_snr']:8.3f} {row['sim']:8.4f}"
        )
        lines.append(
            f"| {row['horizon']} | {row['param_snr']:.3f} | {row['reward_snr']:.3f} | {row['critic_snr']:.3f} | "
            f"{row['reward_norm']:.4g} | {row['critic_norm']:.4g} | {row['reward_vs_critic']:+.3f} | "
            f"{row['action_snr']:.3f} | {row['action_reward_snr']:.3f} | {row['action_critic_snr']:.3f} | "
            f"{row['sim']:.4f} |"
        )
    _plot(rows, baseline, out)
    write_report(out.with_suffix(".md"), "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
