"""Saturation of the policy's tanh latent code.

The encoder output is z = tanh(h) in R^{z_dim}. A unit is saturated when
|z| exceeds ``--level`` (default 0.99), i.e. |h| is past artanh(level).

Hooks the encoder during a short simulator rollout of the loaded policy and
writes ``tanh.png`` next to the checkpoint. With no paths, the newest runs
are used.

Usage:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/check_tanh.py --latest 5
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
    hidden = tuple(int(x) for x in PAPER_POLICY_HIDDEN)

    def get(key: str, default):
        return saved.get(key, default)

    return TrainConfig(
        num_envs=num_envs,
        horizon=int(get("horizon", 32)),
        gamma=float(get("gamma", 0.99)),
        lam=float(get("lam", 0.95)),
        action_noise=float(get("action_noise", 0.1)),
        wm_norm=str(get("wm_norm", "none")),
        wm_residual=get("wm_residual", False),
        wm_frame=str(get("wm_frame", "heading")),
        stochastic_actor=bool(get("stochastic_actor", False)),
        critic_ensemble=int(get("critic_ensemble", 1)),
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


def _latest_runs(n: int) -> list[Path]:
    root = Path("logs/lafan1")
    runs = [path for path in sorted(root.iterdir()) if path.is_dir() and (path / "last.pt").exists()]
    return [run / "last.pt" for run in runs[-n:]]


def _plot(z: torch.Tensor, path: Path, level: float, title: str) -> None:
    abs_z = z.abs()
    mean_abs = abs_z.mean(0).cpu()
    saturated = (abs_z > level).float().mean(0).cpu()
    overall = float((abs_z > level).float().mean())
    units = list(range(abs_z.shape[-1]))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    ax = axes[0]
    colors = ["#d62728" if value > 0.5 else "#1f77b4" for value in saturated]
    ax.bar(units, mean_abs, color=colors)
    ax.axhline(level, color="k", lw=0.8, ls="--", label=f"|z|={level}")
    ax.set_xlabel("latent unit")
    ax.set_ylabel("mean |tanh|")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False)

    ax = axes[1]
    ax.hist(abs_z.flatten().cpu().numpy(), bins=40, range=(0, 1), color="#1f77b4")
    ax.axvline(level, color="k", lw=0.8, ls="--")
    ax.set_xlabel("|tanh|")
    ax.set_ylabel("count")
    ax.set_title(f"{100 * overall:.1f}% of activations above {level}")
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(path, dpi=160)
    print(f"wrote {path.resolve()}")


def _one(ckpt: Path, args: argparse.Namespace) -> None:
    saved = torch.load(ckpt, map_location=args.device, weights_only=False)
    cfg = _cfg(saved.get("cfg", {}), args.num_envs, args.device)
    trainer = Trainer(cfg, ckpt.parent / "tanh_check")
    trainer.load_checkpoint(ckpt)
    trainer.policy.eval()
    captured: list[torch.Tensor] = []

    def hook(_module, _inputs, output: torch.Tensor) -> None:
        captured.append(output.detach())

    if trainer.policy.encoder is None:
        raise SystemExit(f"{ckpt} has no tanh latent.")
    handle = trainer.policy.encoder.register_forward_hook(hook)
    try:
        with torch.no_grad():
            trainer.collect(args.steps)
    finally:
        handle.remove()
    pre = torch.cat([t.reshape(-1, t.shape[-1]) for t in captured], dim=0)
    z = pre.tanh()
    level = args.level
    abs_z = z.abs()
    frac = float((abs_z > level).float().mean())
    per_unit = (abs_z > level).float().mean(0)
    n_sat_units = int((per_unit > 0.5).sum())
    iteration = saved.get("iteration")
    print(
        f"\n== {ckpt.parent.name} iter {iteration}  "
        f"z={z.shape[-1]} samples={z.shape[0]}  "
        f"|z|>{level}: {100 * frac:.2f}% of activations, "
        f"{n_sat_units}/{z.shape[-1]} units saturated on most samples  "
        f"mean|z|={float(abs_z.mean()):.3f}"
    )
    _plot(
        z,
        ckpt.parent / "tanh.png",
        level,
        f"{ckpt.parent.name}  it={iteration}  mean|z|={float(abs_z.mean()):.2f}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, nargs="*", default=None)
    parser.add_argument("--latest", type=int, default=5)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--level", type=float, default=0.99)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    for ckpt in args.ckpt or _latest_runs(args.latest):
        _one(ckpt, args)


if __name__ == "__main__":
    main()
