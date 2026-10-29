"""Dormant-unit fractions for the policy, world model, and critic.

Sokar et al. (ICML 2023): a unit is dormant when its mean absolute activation,
divided by the layer mean, is at most ``tau``. SiLU outputs are the activations.

The simulator rollout is the trained policy for ``--steps`` control steps.
The world model and the critic are measured on one imagined horizon from those
states, which is when they actually run.

Usage:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/check_dormancy.py \
    --ckpt logs/lafan1/<run>/last.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from supermimic.algo import TrainConfig, Trainer, _flatten_time, _unflatten_time
from supermimic.mdp import PAPER_OBS_TERMS, invariant_features, normalize_xi
from supermimic.network import PAPER_POLICY_HIDDEN

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "dormancy_tracker",
    "/mnt/workspace/btx0424/active-adaptation/active_adaptation/learning/utils/dormancy.py",
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
DormancyTracker = _mod.DormancyTracker


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


def _ordered(scores: dict[str, float]) -> list[tuple[str, float]]:
    def key(item: tuple[str, float]) -> tuple[int, str]:
        name = item[0]
        # blocks.0, blocks.1, ... in module order.
        return (name.count("."), name)

    return sorted(scores.items(), key=key)


def _plot(groups: dict[str, dict[str, float]], path: Path, tau: float) -> None:
    fig, axes = plt.subplots(len(groups), 1, figsize=(10, 2.4 * len(groups)), constrained_layout=True)
    if len(groups) == 1:
        axes = [axes]
    for ax, (title, scores) in zip(axes, groups.items()):
        rows = _ordered(scores)
        labels = [name.replace(".blocks.", ".").replace(".act", "") for name, _ in rows]
        values = [100.0 * value for _, value in rows]
        ax.bar(range(len(values)), values, color="#1f77b4")
        ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("dormant %")
        ax.set_ylim(0, 100)
        ax.axhline(tau * 100, color="k", lw=0.6, alpha=0.4)
        ax.set_title(f"{title}  (τ={tau})")
        ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(path, dpi=160)
    print(f"wrote {path.resolve()}")


def _latest_runs(n: int) -> list[Path]:
    root = Path("logs/lafan1")
    runs = [path for path in sorted(root.iterdir()) if path.is_dir() and (path / "last.pt").exists()]
    return [run / "last.pt" for run in runs[-n:]]


def _one(ckpt: Path, args: argparse.Namespace) -> None:
    saved = torch.load(ckpt, map_location=args.device, weights_only=False)
    saved_cfg = saved.get("cfg", {})
    cfg = _cfg(saved_cfg, args.num_envs, args.device)
    out = ckpt.parent / "dormancy.png"
    trainer = Trainer(cfg, ckpt.parent / "dormancy_check")
    trainer.load_checkpoint(ckpt)
    trainer.policy.eval()
    trainer.wm.eval()
    trainer.critic.eval()

    policy_tracker = DormancyTracker(trainer.policy)
    wm_tracker = DormancyTracker(trainer.wm)
    critic_tracker = DormancyTracker(trainer.critic)
    try:
        with torch.no_grad(), policy_tracker.track():
            trainer.collect(args.steps)
        trainer.cfg.pi_batch = args.batch
        with torch.no_grad():
            start = trainer.sample_policy_starts()
            with wm_tracker.track(), policy_tracker.track():
                graph = trainer._imagine_graph(start)
            batch = graph["critic_batch"]
            horizon = trainer.cfg.horizon
            flat = _unflatten_time(batch["states"], horizon + 1)
            priv = trainer.sim.mdp.privileged(*flat, batch["frames"])
            features = trainer.norms.critic(
                invariant_features(*flat), batch["goals"], normalize_xi(batch["xi"]), priv
            )
            with critic_tracker.track():
                trainer.critic(features)
        iteration = saved.get("iteration", "?")
        groups = {
            f"policy  it={iteration}  enc={list(cfg.policy_encoder)} dec={list(cfg.policy_decoder)} norm={cfg.policy_norm}": policy_tracker.compute_dormancy(args.tau),
            f"world model  norm={cfg.wm_norm} residual={cfg.wm_residual}": wm_tracker.compute_dormancy(args.tau),
            "critic": critic_tracker.compute_dormancy(args.tau),
        }
    finally:
        policy_tracker.close()
        wm_tracker.close()
        critic_tracker.close()

    print(f"\n== {ckpt.parent.name} iter {saved.get('iteration')}")
    for title, scores in groups.items():
        print(title)
        for name, value in _ordered(scores):
            print(f"  {name:24s} {100 * value:6.2f}%")
    _plot(groups, out, args.tau)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, nargs="*", default=None)
    parser.add_argument("--latest", type=int, default=5, help="Newest runs when --ckpt is omitted.")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    ckpts = args.ckpt or _latest_runs(args.latest)
    for ckpt in ckpts:
        _one(ckpt, args)


if __name__ == "__main__":
    main()
