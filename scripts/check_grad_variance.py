"""Variance of the actor gradient that flows through the world model.

Definition. Draw N imagination starts. Let L_i be the policy loss of trajectory
i alone, and let g_i = ∇_θ L_i be its gradient with respect to the actor
parameters. The world model and the critic are held fixed, so g_i reaches θ
only by backpropagation through the imagined dynamics. With

    μ = (1/N) Σ_i g_i
    v = (1/N) Σ_i ||g_i - μ||² = (1/N) Σ_i ||g_i||² - ||μ||²

the signal-to-noise ratio is

    SNR = ||μ||² / v.

SNR ≫ 1 means the trajectories agree on one update. SNR ≪ 1 means the mean
step is small compared with the scatter of the individual world-model
gradients. The plot shows cos(g_i, μ) for each start.

Sign conflict. A coordinate is large in g when |g_i| exceeds the root-mean-square
of g. For a pair of trajectories, overlap is the Jaccard index of their large
sets, and conflict is the fraction of the intersection whose signs differ.
Disjoint supports give a small overlap. Destructive interference gives a large
overlap whose conflict is near one half.

Early checkpoints of the newest runs are the default.

Usage:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/check_grad_variance.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from supermimic.algo import TrainConfig, Trainer, hold_out_net
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
        policy_trunk=tuple(int(x) for x in get("policy_trunk", (512, 512, 512))),
        policy_kind=str(get("policy_kind", "split")),
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


def _early_runs(n: int) -> list[Path]:
    root = Path("logs/lafan1")
    runs = [path for path in sorted(root.iterdir()) if path.is_dir() and any(path.glob("ckpt_*.pt"))]
    chosen = []
    for run in runs[-n:]:
        chosen.append(sorted(run.glob("ckpt_*.pt"))[0])
    return chosen


# Shared by the gradient-consistency scripts so the numbers can be compared.
NUM_ENVS = 128
COLLECT = 40
SAMPLES = 16


def _flat_grad(params: list[torch.nn.Parameter]) -> torch.Tensor:
    grads = [p.grad.detach().reshape(-1).float().cpu() for p in params if p.grad is not None]
    if not grads:
        return torch.zeros(1)
    return torch.cat(grads)


def _param_groups(policy) -> dict[str, list[torch.nn.Parameter]]:
    """Actor blocks whose gradients can conflict separately."""
    groups = {}
    if policy.encoder is not None:
        groups["encoder"] = [p for p in policy.encoder.parameters() if p.requires_grad]
        groups["decoder"] = [p for p in policy.decoder.parameters() if p.requires_grad]
    elif policy.film is not None:
        groups["film"] = [p for p in policy.film.parameters() if p.requires_grad]
    else:
        groups["trunk"] = [p for p in policy.trunk.parameters() if p.requires_grad]
    return groups


def snr_stats(grads: torch.Tensor) -> dict[str, float | torch.Tensor]:
    """SNR, mean cosine, and norms of per-start gradients. ``grads`` is (N, D)."""
    mean = grads.mean(0)
    mean_norm = float(mean.norm())
    individual = float(grads.norm(dim=1).mean())
    variance = max(float(grads.square().sum(1).mean()) - mean_norm**2, 0.0)
    snr = mean_norm**2 / max(variance, 1e-12)
    cosine = (grads @ mean) / (grads.norm(dim=1) * mean.norm().clamp_min(1e-12))
    return {
        "snr": snr,
        "cosine": float(cosine.mean()),
        "mean_norm": mean_norm,
        "individual_norm": individual,
        "variance": variance,
        "mean": mean,
        "cosine_vec": cosine,
    }


def _flat_autograd(grads: tuple[torch.Tensor | None, ...], refs: list[torch.Tensor]) -> torch.Tensor:
    parts = []
    for grad, ref in zip(grads, refs):
        if grad is None:
            parts.append(torch.zeros(ref.numel()))
        else:
            parts.append(grad.detach().reshape(-1).float().cpu())
    return torch.cat(parts)


def imagine_grads(trainer: Trainer, start: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Per-start gradients of the policy loss, split by source and by space.

    ``param`` / ``action`` are ∇_θ L and ∇_a L of the full loss. ``reward`` is
    the imagined return only. ``critic`` is the bootstrap value only. L is
    their sum.
    """
    params = [p for p in trainer.policy.parameters() if p.requires_grad]
    buckets = {name: [] for name in ("param", "action", "param_reward", "param_critic", "action_reward", "action_critic")}
    n = next(iter(start.values())).shape[0]
    for i in range(n):
        row = {key: value[i : i + 1] for key, value in start.items()}
        with hold_out_net(trainer.wm), hold_out_net(trainer.critic):
            graph = trainer._imagine_graph(row)
            actions = graph["actions"]
            pieces = {
                "total": graph["policy_loss"],
                "reward": -graph["step_return"].sum(0).mean(),
                "critic": -graph["bootstrap"].mean(),
            }
            for name, loss in pieces.items():
                action_grad = torch.autograd.grad(loss, actions, retain_graph=True, allow_unused=True)
                param_grad = torch.autograd.grad(loss, params, retain_graph=name != "critic", allow_unused=True)
                suffix = "" if name == "total" else f"_{name}"
                buckets[f"action{suffix}"].append(_flat_autograd(action_grad, actions))
                buckets[f"param{suffix}"].append(_flat_autograd(param_grad, params))
    return {name: torch.stack(rows) for name, rows in buckets.items()}


def _dot_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-12))


def write_report(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f"wrote {path.resolve()}")


def _pair_conflict(stacked: torch.Tensor) -> tuple[float, float, float]:
    """Mean overlap, sign-conflict inside the overlap, and both-large count.

    A coordinate is large when it exceeds that trajectory's RMS. Overlap is
    ``|A ∩ B| / |A ∪ B|``. Conflict is the opposite-sign share of ``A ∩ B``.
    """
    rms = stacked.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-12)
    large = stacked.abs() > rms
    n = stacked.shape[0]
    overlaps, conflicts, both_counts = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            both = large[i] & large[j]
            union = large[i] | large[j]
            n_both = int(both.sum())
            n_union = int(union.sum())
            overlaps.append(n_both / max(n_union, 1))
            both_counts.append(n_both)
            if n_both == 0:
                conflicts.append(0.0)
            else:
                oppose = both & (stacked[i] * stacked[j] < 0)
                conflicts.append(float(oppose.sum()) / n_both)
    return (
        sum(overlaps) / len(overlaps),
        sum(conflicts) / len(conflicts),
        sum(both_counts) / len(both_counts),
    )


def _one(ckpt: Path, args: argparse.Namespace) -> None:
    saved = torch.load(ckpt, map_location=args.device, weights_only=False)
    cfg = _cfg(saved.get("cfg", {}), args.num_envs, args.device)
    cfg.buffer_steps = max(args.collect + 4, int(cfg.horizon) + 4)
    trainer = Trainer(cfg, ckpt.parent / "grad_variance_check")
    trainer.load_checkpoint(ckpt)
    trainer.policy.train()
    with torch.no_grad():
        trainer.collect(args.collect)
    trainer.cfg.pi_batch = args.samples
    start = trainer.sample_policy_starts()
    packed = imagine_grads(trainer, start)
    groups = _param_groups(trainer.policy)
    iteration = saved.get("iteration")
    lines = [
        f"## {ckpt.parent.name} / {ckpt.name}",
        "",
        f"iteration {iteration}. N={args.samples}, collect={args.collect}, num_envs={args.num_envs}, horizon={trainer.cfg.horizon}.",
        "",
        "| source | space | SNR | mean cos | ||μ|| | mean ||g_i|| |",
        "|---|---|---|---|---|---|",
    ]
    print(f"\n== {ckpt.parent.name} {ckpt.name} iter {iteration}  N={args.samples} collect={args.collect}")
    spaces = {}
    for source, param_key, action_key in (
        ("total", "param", "action"),
        ("reward", "param_reward", "action_reward"),
        ("critic", "param_critic", "action_critic"),
    ):
        for space, key in (("parameter", param_key), ("action", action_key)):
            stats = snr_stats(packed[key])
            spaces[f"{source}_{space}"] = stats
            overlap, conflict, both = _pair_conflict(packed[key])
            print(
                f"   {source:<8} {space:<10} SNR={stats['snr']:.3f}  cos={stats['cosine']:.3f}  "
                f"||μ||={stats['mean_norm']:.4g}  mean ||g||={stats['individual_norm']:.4g}  "
                f"overlap={overlap:.3f}  conflict={conflict:.3f}"
            )
            lines.append(
                f"| {source} | {space} | {stats['snr']:.3f} | {stats['cosine']:.3f} | "
                f"{stats['mean_norm']:.4g} | {stats['individual_norm']:.4g} |"
            )
            lines.append("")
            lines.append(
                f"{source} {space} overlap {overlap:.3f}, sign conflict {conflict:.3f}, both-large {both:.0f}."
            )
            lines.append("")
    reward_vs_critic = _dot_cosine(spaces["reward_parameter"]["mean"], spaces["critic_parameter"]["mean"])
    print(f"   reward vs critic parameter cosine {reward_vs_critic:+.3f}")
    lines.append(f"Cosine of the mean parameter gradients, reward against critic: {reward_vs_critic:+.3f}.")
    lines.append("")
    for name, group in groups.items():
        print(f"   block {name:<8} params={sum(p.numel() for p in group)}")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, space in zip(axes, ("parameter", "action")):
        cosine = spaces[f"total_{space}"]["cosine_vec"]
        ax.bar(range(args.samples), cosine.numpy(), color="#1f77b4")
        ax.axhline(0.0, color="k", lw=0.8)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("imagination start")
        ax.set_ylabel(r"$\cos(g_i, \mu)$")
        ax.set_title(f"{space} total  SNR={spaces[f'total_{space}']['snr']:.2f}")
        ax.grid(True, axis="y", alpha=0.3)
    out = ckpt.parent / f"grad_variance_{ckpt.stem}.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out.resolve()}")
    report = ckpt.parent / "grad_variance.md"
    previous = report.read_text() if report.exists() else "# Gradient variance\n\n"
    marker = f"## {ckpt.parent.name} / {ckpt.name}"
    if marker in previous:
        previous = previous.split(marker)[0].rstrip() + "\n\n"
    write_report(report, previous + "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, nargs="*", default=None)
    parser.add_argument("--latest", type=int, default=5)
    parser.add_argument("--num-envs", type=int, default=NUM_ENVS)
    parser.add_argument("--collect", type=int, default=COLLECT)
    parser.add_argument("--samples", type=int, default=SAMPLES)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    for ckpt in args.ckpt or _early_runs(args.latest):
        _one(ckpt, args)


if __name__ == "__main__":
    main()
