"""Do the critic heads agree on the gradient they give the actor?

The policy loss adds γ^H S_H times the mean of the heads at the imagined
terminal state. This script separates that term by head.

- Actor gradient: g_i = ∇_θ (γ^H S_H V_i(s_H)), world model and critic weights
  held fixed, one shared imagination. Agreement is cos(g_0, g_1).
- State gradient: at each imagined state, cos(∇_x V_0, ∇_x V_1) for the critic
  input x. The value loss never trains this Jacobian.
- Scalars: correlation of V_0 and V_1 on the same states.

Usage:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/check_critic_heads.py \
    --ckpt logs/lafan1/<run>/ckpt_003000.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from supermimic.algo import Trainer, hold_out_net
from supermimic.mdp import invariant_features, normalize_xi
from check_grad_variance import _cfg


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = a.norm() * b.norm()
    if float(denom) < 1e-12:
        return 0.0
    return float((a @ b) / denom)


def _row_grads(values: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Per-row ∇_x V. ``values`` is (N,), ``inputs`` is (N, D) and requires grad."""
    grads = torch.empty_like(inputs)
    for i in range(inputs.shape[0]):
        grad = torch.autograd.grad(values[i], inputs, retain_graph=True)[0]
        grads[i] = grad[i]
    return grads


def _jacobian_cosine(trainer: Trainer, batch: dict, horizon: int, limit: int) -> tuple[float, float, float, float]:
    """Mean cosine of ∇V at terminal states and at a sample of every imagined state."""
    n = batch["reward"].shape[1]
    flat = batch["states"]
    priv = trainer.sim.mdp.privileged(*flat, batch["frames"])
    features = trainer.norms.critic(
        invariant_features(*flat), batch["goals"], normalize_xi(batch["xi"]), priv
    )
    features = features.detach().view(horizon + 1, n, -1)
    terminal = features[-1]
    earlier = features[:-1].reshape(-1, features.shape[-1])
    if earlier.shape[0] > limit:
        pick = torch.randperm(earlier.shape[0], device=earlier.device)[:limit]
        earlier = earlier[pick]

    def cosines(x: torch.Tensor) -> torch.Tensor:
        x = x.detach().requires_grad_(True)
        values = trainer.critic(x)
        g0 = _row_grads(values[:, 0], x)
        g1 = _row_grads(values[:, 1], x)
        denom = g0.norm(dim=-1) * g1.norm(dim=-1)
        return (g0 * g1).sum(-1) / denom.clamp_min(1e-12)

    with hold_out_net(trainer.critic):
        term = cosines(terminal)
        mid = cosines(earlier)
    return float(term.mean()), float(term.median()), float(mid.mean()), float(mid.median())


def _one(ckpt: Path, args: argparse.Namespace) -> None:
    saved = torch.load(ckpt, map_location=args.device, weights_only=False)
    cfg = _cfg(saved.get("cfg", {}), args.num_envs, args.device)
    if cfg.critic_ensemble < 2:
        raise SystemExit(f"{ckpt} has critic_ensemble={cfg.critic_ensemble}")
    trainer = Trainer(cfg, ckpt.parent / "critic_heads_check")
    trainer.load_checkpoint(ckpt)
    trainer.policy.train()
    with torch.no_grad():
        trainer.collect(args.collect)
    trainer.cfg.pi_batch = args.batch
    start = trainer.sample_policy_starts()
    trainer.critic._head_trace = []
    params = [p for p in trainer.policy.parameters() if p.requires_grad]
    with hold_out_net(trainer.wm), hold_out_net(trainer.critic):
        loss, critic_batch, *_ = trainer.imagine(start)
        heads, alive = trainer.critic._head_trace[-1]
        gamma_h = cfg.gamma ** cfg.horizon
        flats = []
        for i in range(heads.shape[-1]):
            term = -(gamma_h * alive * heads[:, i]).mean()
            got = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
            flats.append(torch.cat([g.detach().reshape(-1).float().cpu() for g in got if g is not None]))
        values = heads.detach()
        del loss, heads, alive
    trainer.critic._head_trace.clear()

    g0, g1 = flats
    v0, v1 = values[:, 0], values[:, 1]
    v_cos = _cosine(v0.float().cpu(), v1.float().cpu())
    term_cos, term_med, mid_cos, mid_med = _jacobian_cosine(trainer, critic_batch, cfg.horizon, args.jacobian_samples)
    iteration = saved.get("iteration")
    print(
        f"\n== {ckpt.parent.name} {ckpt.name} iter {iteration}  batch {args.batch}\n"
        f"scalars   corr(V0,V1)={v_cos:.3f}  "
        f"mean |V0-V1|={float((v0 - v1).abs().mean()):.4g}  "
        f"mean |V|={float(values.abs().mean()):.4g}\n"
        f"∇_x V     terminal mean cos={term_cos:.3f}  median={term_med:.3f}  "
        f"earlier mean cos={mid_cos:.3f}  median={mid_med:.3f}\n"
        f"actor ∇   ||g0||={float(g0.norm()):.4g}  ||g1||={float(g1.norm()):.4g}  "
        f"cos(g0,g1)={_cosine(g0, g1):.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, nargs="+")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--collect", type=int, default=4)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--jacobian-samples", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    for ckpt in args.ckpt:
        _one(ckpt, args)


if __name__ == "__main__":
    main()
