"""Do world models from different checkpoints agree on the actor update?

The actor, the critic, and the imagination starts are taken once, from
``--actor`` (default: the last ``--ckpt``). Each checkpoint then contributes
only its world model and the normalizers that model reads. The policy gradient
is backpropagation of the same starts through that world model.

Pairwise cosine of the mean gradients is the agreement of the update. The
per-start cosine is the agreement on each trajectory before averaging.

Usage:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/check_wm_grad_consistency.py \
    --ckpt logs/lafan1/<run>/ckpt_001000.pt logs/lafan1/<run>/ckpt_002000.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from supermimic.algo import Trainer
from supermimic.network import WorldModel
from check_grad_variance import COLLECT, NUM_ENVS, SAMPLES, _cfg, imagine_grads, write_report


def _load_wm(trainer: Trainer, path: Path) -> None:
    saved = torch.load(path, map_location=trainer.device, weights_only=False)
    cfg = saved.get("cfg", {})
    wm = WorldModel(
        norm=str(cfg.get("wm_norm", "none")),
        residual=cfg.get("wm_residual", False),
        frame=str(cfg.get("wm_frame", "heading")),
    ).to(trainer.device)
    wm.load_state_dict(saved["wm"])
    trainer.wm = wm
    norms = saved.get("norms")
    if norms is None:
        return
    own = trainer.norms.state_dict()
    for key in own:
        if key.startswith(("state.", "action.", "xi.", "body.")) and key in norms:
            own[key] = norms[key]
    trainer.norms.load_state_dict(own)


def _grads(trainer: Trainer, start: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return imagine_grads(trainer, start)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, nargs="+", required=True)
    parser.add_argument("--actor", type=Path, default=None, help="Policy and critic. Defaults to the last --ckpt.")
    parser.add_argument("--num-envs", type=int, default=NUM_ENVS)
    parser.add_argument("--collect", type=int, default=COLLECT)
    parser.add_argument("--samples", type=int, default=SAMPLES)
    parser.add_argument("--out", type=Path, default=Path("logs/wm_grad_consistency.png"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    actor_path = args.actor or args.ckpt[-1]
    saved = torch.load(actor_path, map_location=args.device, weights_only=False)
    cfg = _cfg(saved.get("cfg", {}), args.num_envs, args.device)
    cfg.pi_batch = args.samples
    cfg.buffer_steps = max(args.collect + 4, int(cfg.horizon) + 4)
    out_dir = args.out.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(cfg, out_dir / "wm_grad_consistency")
    trainer.load_checkpoint(actor_path)
    trainer.policy.train()
    with torch.no_grad():
        trainer.collect(args.collect)
    start = {key: value.detach() for key, value in trainer.sample_policy_starts().items()}

    labels = []
    stored: dict[str, list[torch.Tensor]] = {
        name: []
        for name in ("param", "action", "param_reward", "param_critic", "action_reward", "action_critic")
    }
    for ckpt in args.ckpt:
        print(f"world model {ckpt.parent.name}/{ckpt.name}", flush=True)
        _load_wm(trainer, ckpt)
        packed = _grads(trainer, start)
        labels.append(f"{ckpt.parent.name[-8:]}\n{ckpt.stem.replace('ckpt_', '')}")
        for name, rows in stored.items():
            rows.append(packed[name])

    def _matrix(starts: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        count = len(starts)
        mean_cos = torch.eye(count)
        start_cos = torch.eye(count)
        for i in range(count):
            for j in range(i + 1, count):
                mean_cos[i, j] = mean_cos[j, i] = _cosine(starts[i].mean(0), starts[j].mean(0))
                pair = [_cosine(starts[i][k], starts[j][k]) for k in range(args.samples)]
                start_cos[i, j] = start_cos[j, i] = sum(pair) / len(pair)
        return mean_cos, start_cos

    param_mean_cos, param_start_cos = _matrix(stored["param"])
    action_mean_cos, action_start_cos = _matrix(stored["action"])
    reward_mean_cos, reward_start_cos = _matrix(stored["param_reward"])
    critic_mean_cos, critic_start_cos = _matrix(stored["param_critic"])
    n = len(args.ckpt)
    names = [f"{ckpt.parent.name}/{ckpt.name}" for ckpt in args.ckpt]

    def _print_matrix(title: str, matrix: torch.Tensor) -> list[str]:
        print(title)
        lines = [f"### {title}", "", "| | " + " | ".join(names) + " |", "|---|" + "---|" * n]
        for i, name in enumerate(names):
            row = " ".join(f"{matrix[i, j]:+.3f}" for j in range(n))
            print(f"  {name}  {row}")
            cells = " | ".join(f"{matrix[i, j]:+.3f}" for j in range(n))
            lines.append(f"| {name} | {cells} |")
        lines.append("")
        return lines

    print(f"\nactor {actor_path.parent.name}/{actor_path.name}  starts {args.samples} collect {args.collect}")
    report = [
        "# World-model gradient consistency",
        "",
        f"Actor `{actor_path}`. N={args.samples}, collect={args.collect}, num_envs={args.num_envs}.",
        "Reward is the imagined return. Critic is the bootstrap. The policy loss is their sum.",
        "",
    ]
    report.extend(_print_matrix("parameter space, cosine of mean gradients", param_mean_cos))
    report.extend(_print_matrix("parameter space, mean per-start cosine", param_start_cos))
    report.extend(_print_matrix("parameter reward, cosine of mean gradients", reward_mean_cos))
    report.extend(_print_matrix("parameter reward, mean per-start cosine", reward_start_cos))
    report.extend(_print_matrix("parameter critic, cosine of mean gradients", critic_mean_cos))
    report.extend(_print_matrix("parameter critic, mean per-start cosine", critic_start_cos))
    report.extend(_print_matrix("action space, cosine of mean gradients", action_mean_cos))
    report.extend(_print_matrix("action space, mean per-start cosine", action_start_cos))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for ax, matrix, title in (
        (axes[0, 0], param_mean_cos, "parameter, mean update"),
        (axes[0, 1], param_start_cos, "parameter, per-start"),
        (axes[1, 0], action_mean_cos, "action, mean update"),
        (axes[1, 1], action_start_cos, "action, per-start"),
    ):
        image = ax.imshow(matrix.numpy(), vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(n), labels, fontsize=7)
        ax.set_yticks(range(n), labels, fontsize=7)
        ax.set_title(title)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(image, ax=ax, fraction=0.046)
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out.resolve()}")
    write_report(args.out.with_suffix(".md"), "\n".join(report) + "\n")


if __name__ == "__main__":
    main()
