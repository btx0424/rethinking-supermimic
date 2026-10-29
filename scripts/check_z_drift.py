"""Open-loop root-height drift of the world model.

The integrator adds a height residual and never overwrites z. This rolls the
world model on recorded actions and compares predicted root z with the
simulator. A positive error is floating above the true trajectory. A negative
error is sinking through it.

Usage:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/check_z_drift.py \
    --ckpt logs/lafan1/<run>/last.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from supermimic.algo import TrainConfig, Trainer
from supermimic.mdp import PAPER_OBS_TERMS, normalize_xi, unflatten_state
from supermimic.network import PAPER_POLICY_HIDDEN, WorldModelState


def _cfg(saved: dict, num_envs: int, device: str, horizon: int) -> TrainConfig:
    hidden = tuple(int(x) for x in PAPER_POLICY_HIDDEN)

    def get(key: str, default):
        return saved.get(key, default)

    return TrainConfig(
        num_envs=num_envs,
        buffer_steps=max(horizon + 8, 64),
        horizon=horizon,
        wm_norm=str(get("wm_norm", "none")),
        wm_residual=get("wm_residual", False),
        wm_frame=str(get("wm_frame", "heading")),
        stochastic_actor=bool(get("stochastic_actor", False)),
        critic_ensemble=int(get("critic_ensemble", 1)),
        policy_z_dim=int(get("policy_z_dim", 32)),
        policy_encoder=tuple(int(x) for x in get("policy_encoder", hidden)),
        policy_decoder=tuple(int(x) for x in get("policy_decoder", hidden)),
        policy_trunk=tuple(int(x) for x in get("policy_trunk", (512, 512, 512))),
        policy_norm=str(get("policy_norm", "none")),
        policy_kind=str(get("policy_kind", "split")),
        anchor=str(get("anchor", "torso")),
        obs_terms=tuple(get("obs_terms", PAPER_OBS_TERMS)),
        reward_style=str(get("reward_style", "gaussian")),
        device=device,
        compile=False,
        amp=str(get("amp", "bf16")),
        fk=str(get("fk", "warp")),
        swanlab_mode="disabled",
        seed=0,
    )


@torch.no_grad()
def _rollout(trainer: Trainer, batch: int, horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return predicted and true root z, shape (horizon + 1, batch). Step 0 is the start."""
    window = trainer.buffer.sample_windows(batch, horizon)
    state = WorldModelState(
        *unflatten_state(window["s"][:, 0]),
        window["body_pos"][:, 0],
        window["body_quat"][:, 0],
    )
    xi_n = normalize_xi(window["xi"][:, 0])
    contact = window["c"][:, 0]
    pred = [state.p[:, 2]]
    true = [window["s"][:, 0, 2]]
    for step in range(horizon):
        state, logit = trainer.wm.step(
            state, window["a"][:, step], xi_n, trainer.norms, contact, trainer.sim.mdp
        )
        contact = logit.sigmoid()
        pred.append(state.p[:, 2])
        true.append(window["s1"][:, step, 2])
    return torch.stack(pred), torch.stack(true)


def _plot(pred: torch.Tensor, true: torch.Tensor, path: Path) -> None:
    err = (pred - true).cpu()
    steps = list(range(err.shape[0]))
    mean = err.mean(1)
    std = err.std(1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    ax = axes[0]
    ax.plot(steps, mean, color="#1f77b4", lw=2, label="mean pred − true")
    ax.fill_between(steps, mean - std, mean + std, color="#1f77b4", alpha=0.2, label="±1 std")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("open-loop step")
    ax.set_ylabel("root z error (m)")
    ax.set_title("positive = floating, negative = sinking")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.plot(steps, true.mean(1).cpu(), color="#2ca02c", lw=2, label="simulator")
    ax.plot(steps, pred.mean(1).cpu(), color="#d62728", lw=2, label="world model")
    ax.axhline(0.0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("open-loop step")
    ax.set_ylabel("root z (m)")
    ax.set_title("mean height")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)

    fig.savefig(path, dpi=160)
    print(f"wrote {path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None, help="Defaults to <ckpt dir>/z_drift.png")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--collect", type=int, default=48, help="Simulator steps to fill the buffer.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.collect < args.horizon:
        raise SystemExit(f"--collect ({args.collect}) must be at least --horizon ({args.horizon}).")

    saved = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    cfg = _cfg(saved.get("cfg", {}), args.num_envs, args.device, args.horizon)
    out = args.out or (args.ckpt.parent / "z_drift.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(cfg, out.parent / "z_drift_check")
    trainer.load_checkpoint(args.ckpt)
    trainer.collect(args.collect)
    pred, true = _rollout(trainer, args.batch, args.horizon)
    err = pred - true
    last = err[-1]
    below = pred[-1] < 0.0
    print(
        f"horizon {args.horizon}  batch {args.batch}\n"
        f"final z error  mean {last.mean():+.4f} m  std {last.std():.4f}  "
        f"p10 {last.quantile(0.1):+.4f}  p90 {last.quantile(0.9):+.4f}\n"
        f"pred z<0 at last step: {100 * below.float().mean():.1f}%  "
        f"(simulator {100 * (true[-1] < 0).float().mean():.1f}%)"
    )
    _plot(pred, true, out)


if __name__ == "__main__":
    main()
