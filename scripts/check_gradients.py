"""Policy-gradient diagnostics along one imagined horizon.

Loads a checkpoint, collects a short buffer, and plots:

- weighted vs undiscounted per-step gradient norm, and alignment with the full update
- encoder vs decoder share of that step
- ``||∂G/∂a_t||`` against the parameter gradient
- per-reward-term gradient over time
- bootstrap share of the update

Usage:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/check_gradients.py \
    --ckpt logs/lafan1/<run>/last.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from supermimic.algo import TrainConfig, Trainer
from supermimic.mdp import PAPER_OBS_TERMS
from supermimic.network import PAPER_POLICY_HIDDEN


def _cfg(saved: dict, num_envs: int, device: str) -> TrainConfig:
    def get(key: str, default):
        return saved.get(key, default)

    hidden = tuple(int(x) for x in PAPER_POLICY_HIDDEN)
    return TrainConfig(
        num_envs=num_envs,
        horizon=int(get("horizon", 32)),
        gamma=float(get("gamma", 0.99)),
        lam=float(get("lam", 0.95)),
        action_noise=float(get("action_noise", 0.1)),
        wm_norm=str(get("wm_norm", "none")),
        wm_residual=get("wm_residual", False),
        stochastic_actor=bool(get("stochastic_actor", False)),
        policy_z_dim=int(get("policy_z_dim", 32)),
        policy_encoder=tuple(int(x) for x in get("policy_encoder", hidden)),
        policy_decoder=tuple(int(x) for x in get("policy_decoder", hidden)),
        policy_norm=str(get("policy_norm", "none")),
        anchor=str(get("anchor", "torso")),
        obs_terms=tuple(get("obs_terms", PAPER_OBS_TERMS)),
        reward_style=str(get("reward_style", "gaussian")),
        device=device,
        normalize_reward=bool(get("normalize_reward", False)),
        compile=False,
        amp=str(get("amp", "bf16")),
        fk=str(get("fk", "warp")),
        seed=0,
    )


def _plot(diag: dict, path: Path) -> None:
    steps = list(range(len(diag["marginal"])))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(steps, diag["marginal"], color="#1f77b4", lw=2, label=r"$\gamma^t S_t r_t$")
    ax.plot(steps, diag["raw"], color="#ff7f0e", lw=2, label="reward only")
    ax.set_ylabel(r"$||\nabla_\theta||$")
    ax.set_xlabel("imagination step")
    ax.set_title("Does the tail still move the actor?")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(steps, diag["cosine"], color="#2ca02c", lw=1.5, ls="--", label="cosine with full update")
    ax2.set_ylabel("cosine")
    ax2.set_ylim(-1.05, 1.05)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, fontsize=8)

    ax = axes[0, 1]
    ax.plot(steps, diag["encoder"], lw=2, label="encoder")
    ax.plot(steps, diag["decoder"], lw=2, label="decoder")
    ax.plot(steps, diag["action"], lw=2, label=r"$||\partial G/\partial a_t||$")
    ax.set_xlabel("imagination step")
    ax.set_ylabel("grad norm")
    ax.set_title("Actor depth vs dynamics chain")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    for name in ("body_pos", "body_ori", "body_lin_vel", "body_ang_vel", "root_pos", "root_ori"):
        ax.plot(steps, diag["terms"][name], lw=1.6, label=name)
    ax.set_xlabel("imagination step")
    ax.set_ylabel(r"$||\nabla_\theta||$ of weighted term")
    ax.set_title("Tracking terms over the horizon")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=7, ncol=2)

    ax = axes[1, 1]
    ax.plot(steps, diag["terms"]["action_rate"], lw=2, label="action_rate")
    ax.plot(steps, diag["terms"]["joint_limit"], lw=2, label="joint_limit")
    ax.plot(steps, diag["survival"], color="k", lw=1.2, ls=":", label="survival")
    full = diag["full"] + 1e-12
    boot_share = diag["bootstrap"] / full
    ax.set_title(
        f"penalties, survival\n"
        f"bootstrap/full={boot_share:.2f}  cosine(boot, reward)={diag['bootstrap_cosine']:.2f}"
    )
    ax.set_xlabel("imagination step")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)

    fig.savefig(path, dpi=160)
    print(f"wrote {path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("logs/grad_diagnostics.png"))
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--collect", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    saved = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    cfg = _cfg(saved.get("cfg", {}), args.num_envs, args.device)
    out_dir = args.out.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(cfg, out_dir / "grad_check")
    trainer.load_checkpoint(args.ckpt)
    trainer.collect(args.collect)
    trainer.cfg.pi_batch = args.batch
    diag = trainer.diagnose_imagination_step_grads()
    printable = {key: diag[key] for key in ("bootstrap", "bootstrap_cosine", "full", "reward")}
    print(printable)
    print("marginal", [round(x, 4) for x in diag["marginal"]])
    print("raw", [round(x, 4) for x in diag["raw"]])
    print("cosine", [round(x, 3) for x in diag["cosine"]])
    _plot(diag, args.out)


if __name__ == "__main__":
    main()
