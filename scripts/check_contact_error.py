"""8-step world-model error on ground contact versus walking.

Both sets are windows from the replay buffer filled by the loaded policy, so
this is the distribution the world model is trained on. A window is ground
when a tracked body other than the feet is below 0.20 m. It is walking when
the pelvis is above 0.60 m, every non-foot body is above 0.30 m, and a foot
is below 0.12 m. The error is the training ``wm_err8``: root-mean-square of
the nominal-scaled local state error at the eighth step.

Usage:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/check_contact_error.py \
    --ckpt logs/lafan1/<run>/last.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from supermimic.algo import TrainConfig, Trainer, world_model_window
from supermimic.kinematics import TRACKED_BODIES
from supermimic.mdp import PAPER_OBS_TERMS, unflatten_state
from supermimic.network import PAPER_POLICY_HIDDEN

FOOT = {TRACKED_BODIES.index("left_ankle_roll_link"), TRACKED_BODIES.index("right_ankle_roll_link")}
NON_FOOT = [i for i in range(len(TRACKED_BODIES)) if i not in FOOT]
PELVIS = TRACKED_BODIES.index("pelvis")
GROUPS = (("pos", 0, 3), ("rot", 3, 6), ("lin", 6, 9), ("ang", 9, 12), ("q", 12, 41), ("qd", 41, 70))


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


def _family(name: str) -> str:
    for prefix in ("fallAndGetUp", "dance", "fight", "jump", "sprint", "run", "walk"):
        if name.startswith(prefix):
            return prefix
    return "other"


def _summarize(err: torch.Tensor, mask: torch.Tensor) -> tuple[int, float]:
    n = int(mask.sum())
    if n == 0:
        return 0, float("nan")
    return n, float(err[mask].mean())


def _one(ckpt: Path, args: argparse.Namespace) -> None:
    saved = torch.load(ckpt, map_location=args.device, weights_only=False)
    cfg = _cfg(saved.get("cfg", {}), args.num_envs, args.device)
    trainer = Trainer(cfg, ckpt.parent / "contact_error_check")
    trainer.load_checkpoint(ckpt)
    trainer.policy.eval()
    trainer.wm.eval()
    with torch.no_grad():
        trainer.collect(args.collect)
        batch = trainer.buffer.sample_windows(args.windows, trainer.cfg.wm_window)
        errors, _, _, _ = world_model_window(
            trainer.wm,
            trainer.norms,
            batch["s"],
            batch["s1"],
            batch["a"],
            batch["xi"],
            batch["c"],
            batch["body_pos"],
            batch["body_quat"],
            trainer.sim.mdp,
        )
        scaled = errors / trainer.mse_loss.nominal
        last = scaled[:, -1]
        err8 = last.square().mean(-1).sqrt()
        p, quat, v, w, q, qd = unflatten_state(batch["s"][:, 0])
        body_z = trainer.sim.mdp.bodies(p, quat, v, w, q, qd)[0][..., 2]
    non_foot = body_z[:, NON_FOOT]
    feet = body_z[:, list(FOOT)]
    ground = non_foot.min(-1).values < 0.20
    walk = (body_z[:, PELVIS] > 0.60) & (non_foot.min(-1).values > 0.30) & (feet.min(-1).values < 0.12)
    iteration = saved.get("iteration")
    print(f"\n== {ckpt.parent.name} {ckpt.name} iter {iteration}  windows={err8.shape[0]}")
    for label, mask in (("ground", ground), ("walk", walk), ("other", ~(ground | walk))):
        n, mean = _summarize(err8, mask)
        print(f"  {label:<8} n={n:5d}  err8={mean:.4f}")
        if n == 0:
            continue
        parts = []
        for name, lo, hi in GROUPS:
            part = last[mask, lo:hi].square().mean().sqrt()
            parts.append(f"{name}={float(part):.3f}")
        print(f"           {'  '.join(parts)}")

    names = trainer.motion.clip_names
    clip = trainer.motion.clip_id[batch["f"][:, 0]].tolist()
    families = ["fallAndGetUp", "dance", "fight", "jump", "sprint", "run", "walk"]
    print("  by clip family")
    family_means = []
    for family in families:
        mask = torch.tensor([_family(names[i]) == family for i in clip], device=err8.device)
        n, mean = _summarize(err8, mask)
        family_means.append(mean)
        print(f"  {family:<14} n={n:5d}  err8={mean:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    class_labels = ["ground", "walk", "other"]
    class_masks = [ground, walk, ~(ground | walk)]
    class_means = [_summarize(err8, mask)[1] for mask in class_masks]
    axes[0].bar(class_labels, class_means, color=["#d62728", "#2ca02c", "#7f7f7f"])
    axes[0].set_ylabel("8-step error")
    axes[0].set_title("Body near the ground vs walking")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(families, family_means, color="#1f77b4")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].set_title("Buffered windows by clip")
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.suptitle(f"{ckpt.parent.name}  it={iteration}")
    out = ckpt.parent / f"contact_error_{ckpt.stem}.png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out.resolve()}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, nargs="+", required=True)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--collect", type=int, default=32)
    parser.add_argument("--windows", type=int, default=2048)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    for ckpt in args.ckpt:
        _one(ckpt, args)


if __name__ == "__main__":
    main()
