"""Benchmark torch FK throughput and check that torch.compile keeps gradients.

Training hot path: reward FK on H×B flattened states (32×4096), and optional
per-step obs FK at B=4096 when obs.anchor_ori=true.

Usage:
  PYTHONPATH=src python benchmark/torch_fk.py
  PYTHONPATH=src python benchmark/torch_fk.py --no-compile
  PYTHONPATH=src python benchmark/torch_fk.py --batches 4096 131072 --warmup 5 --iters 20
"""

from __future__ import annotations

import argparse
import time

import torch

from supermimic.kinematics import TorchKinematics, compile_g1


def _make_state(batch: int, n_j: int, device: torch.device, requires_grad: bool, seed: int = 0):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    p = torch.empty(batch, 3).uniform_(-0.2, 0.2, generator=gen)
    p[:, 2] = 0.8
    yaw = torch.empty(batch).uniform_(-1.0, 1.0, generator=gen)
    quat = torch.stack(
        (yaw.mul(0.5).cos(), torch.zeros_like(yaw), torch.zeros_like(yaw), yaw.mul(0.5).sin()),
        dim=-1,
    )
    v = torch.empty(batch, 3).uniform_(-0.5, 0.5, generator=gen)
    w = torch.empty(batch, 3).uniform_(-0.5, 0.5, generator=gen)
    q = torch.empty(batch, n_j).uniform_(-0.4, 0.4, generator=gen)
    qd = torch.empty(batch, n_j).uniform_(-1.0, 1.0, generator=gen)
    tensors = [t.to(device) for t in (p, quat, v, w, q, qd)]
    if requires_grad:
        tensors = [t.detach().requires_grad_(True) for t in tensors]
    return tensors


def _loss(outs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return sum(o.float().square().mean() for o in outs)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time(fn, iters: int, device: torch.device) -> float:
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(device)
    return (time.perf_counter() - t0) / iters


def _grads(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    return [t.grad.detach().clone() for t in tensors]


def _max_rel(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = b.abs().clamp_min(1e-8)
    return float((a - b).abs().div(denom).max().item())


def _print_row(name: str, batch: int, fwd_ms: float, bwd_ms: float) -> None:
    print(
        f"{name:14s} {batch:8d} {fwd_ms:10.3f} {bwd_ms:12.3f} "
        f"{batch / (fwd_ms * 1e-3) / 1e6:10.3f} {batch / (bwd_ms * 1e-3) / 1e6:10.3f}",
        flush=True,
    )


def bench(
    name: str,
    forward_fn,
    batch: int,
    n_j: int,
    device: torch.device,
    warmup: int,
    iters: int,
) -> None:
    state = _make_state(batch, n_j, device, requires_grad=False)

    def fwd():
        outs = forward_fn(*state)
        _ = outs[0]

    for _ in range(warmup):
        fwd()
    t_fwd = _time(fwd, iters, device)

    state_g = _make_state(batch, n_j, device, requires_grad=True)

    def fwd_bwd():
        for t in state_g:
            t.grad = None
        _loss(forward_fn(*state_g)).backward()

    for _ in range(warmup):
        fwd_bwd()
    t_bwd = _time(fwd_bwd, iters, device)
    _print_row(name, batch, 1e3 * t_fwd, 1e3 * t_bwd)


def check_grads(eager_fn, compiled_fn, batch: int, n_j: int, device: torch.device) -> None:
    state = _make_state(batch, n_j, device, requires_grad=True)
    state_c = [t.detach().clone().requires_grad_(True) for t in state]

    loss_e = _loss(eager_fn(*state))
    loss_e.backward()
    g_e = _grads(state)

    loss_c = _loss(compiled_fn(*state_c))
    loss_c.backward()
    g_c = _grads(state_c)

    loss_rel = abs(float(loss_e - loss_c)) / max(abs(float(loss_e)), 1e-8)
    names = ("p", "quat", "v", "w", "q", "qd")
    print(f"\ngrad check batch={batch}  loss_rel={loss_rel:.3e}", flush=True)
    ok = loss_rel < 1e-4
    for name, ge, gc in zip(names, g_e, g_c):
        abs_err = float((ge - gc).abs().max().item())
        rel_err = _max_rel(ge, gc)
        print(f"  {name:4s}  max_abs={abs_err:.3e}  max_rel={rel_err:.3e}", flush=True)
        ok = ok and abs_err < 1e-4
    if not ok:
        raise SystemExit("Gradient / loss mismatch between eager and compiled FK.")
    print("  OK", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=[4096, 32 * 4096],
        help="Batch sizes (default: 4096 and H×B=131072)",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument(
        "--compile-mode",
        default="default",
        choices=("default", "reduce-overhead", "max-autotune-no-cudagraphs"),
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    model, joint_names = compile_g1()
    kin = TorchKinematics(model, joint_names)
    n_j = len(joint_names)

    def eager(p, quat, v, w, q, qd):
        return kin.forward(p, quat, v, w, q, qd)

    print(
        f"device={device} joints={n_j} chain_bodies={len(kin._bodies)} "
        f"tracked={len(kin.tracked_ids)}",
        flush=True,
    )
    print(f"warmup={args.warmup} iters={args.iters}", flush=True)
    print(
        f"{'mode':14s} {'batch':>8s} {'fwd_ms':>10s} {'fwd+bwd_ms':>12s} "
        f"{'fwd_Mps':>10s} {'f+b_Mps':>10s}",
        flush=True,
    )

    for batch in args.batches:
        bench("eager", eager, batch, n_j, device, args.warmup, args.iters)

    if args.no_compile or device.type != "cuda":
        return

    # Compile once on a small batch so inductor does not choke on the first timed call.
    print(f"\ncompiling forward (mode={args.compile_mode})…", flush=True)
    compiled = torch.compile(eager, mode=args.compile_mode, fullgraph=False)
    t0 = time.perf_counter()
    _ = compiled(*_make_state(64, n_j, device, requires_grad=False))
    _sync(device)
    print(f"  first fwd compile: {time.perf_counter() - t0:.1f}s", flush=True)
    t0 = time.perf_counter()
    state = _make_state(64, n_j, device, requires_grad=True)
    _loss(compiled(*state)).backward()
    _sync(device)
    print(f"  first bwd compile: {time.perf_counter() - t0:.1f}s", flush=True)

    check_grads(eager, compiled, 256, n_j, device)

    print(
        f"\n{'mode':14s} {'batch':>8s} {'fwd_ms':>10s} {'fwd+bwd_ms':>12s} "
        f"{'fwd_Mps':>10s} {'f+b_Mps':>10s}",
        flush=True,
    )
    for batch in args.batches:
        bench(f"compile/{args.compile_mode}", compiled, batch, n_j, device, max(args.warmup, 5), args.iters)


if __name__ == "__main__":
    main()
