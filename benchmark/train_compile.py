"""Benchmark world-model compile units against a training-like WM update.

Mirrors ``Trainer.update_world_model`` (minus the discriminator):
  - batch 2048, K=8, 10 batches per iteration
  - bf16 MLP AMP, MuonAdamW, contact BCE, MSELoss
  - fresh window tensors every batch (like ``buffer.sample_windows``)
  - Torch FK prewarm; donated_buffer=False when compiling

Compares:
  eager | window/default | window/reduce-overhead | step/default | step/reduce-overhead

``window`` compiles the whole K-loop. ``step`` compiles one transition and keeps
the K-loop eager (same pattern as ``imagine_step``).

Usage:
  CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src .venv/bin/python benchmark/train_compile.py
  CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src .venv/bin/python benchmark/train_compile.py --force

Results are cached under ``benchmark/.train_compile_cache.json`` so finished
cases are not re-run. ``--force`` rebuilds the cache. Compile of one batch must
finish within ``--timeout`` (default 60s); slower cases are recorded and skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import signal
import statistics
import time
import traceback
from pathlib import Path

# Prefer the default caching allocator so reduce-overhead CUDA graphs can run.
os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

import torch

WM_MODES = ("default", "reduce-overhead")
CACHE_PATH = Path(__file__).resolve().parent / ".train_compile_cache.json"


class CompileTimeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise CompileTimeout()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _is_timeout(exc: BaseException | None) -> bool:
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, CompileTimeout):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def _timed(fn, timeout: float, device: torch.device) -> float | None:
    """Run ``fn`` and return seconds, or None if it exceeds ``timeout``."""
    signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(max(int(timeout), 1))
    start = time.perf_counter()
    try:
        fn()
        _sync(device)
    except CompileTimeout:
        return None
    except Exception as exc:
        if _is_timeout(exc):
            return None
        raise
    finally:
        signal.alarm(0)
    return time.perf_counter() - start


def _steady_samples(fn, warmup: int, iters: int, device: torch.device) -> list[float]:
    """Per-call wall times after ``warmup`` (seconds)."""
    for _ in range(warmup):
        fn()
    _sync(device)
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        _sync(device)
        samples.append(time.perf_counter() - start)
    return samples


def _setup_wm(batch: int, k: int, device: torch.device, amp: str, optim: str, wm_lr: float):
    from supermimic import network
    from supermimic.algo import MSELoss, contact_bce
    from supermimic.kinematics import TorchKinematics, compile_g1
    from supermimic.mdp import (
        N_BODY,
        N_CONTACT,
        STATE_DIM,
        XI_DIM,
        MDP,
        unflatten_state,
    )
    from supermimic.motion import MotionLib
    from supermimic.network import InputNorms, WorldModel
    from supermimic.optim import build_optimizer

    torch.manual_seed(0)
    if amp not in ("none", "bf16"):
        raise ValueError(amp)
    network.AMP_DTYPE = torch.bfloat16 if amp == "bf16" else None
    torch.backends.cuda.matmul.allow_tf32 = amp == "none"
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high" if amp == "none" else "highest")

    motion = MotionLib.load(device=str(device))
    model, names = compile_g1()
    kin = TorchKinematics(model, names)
    kin.set_anchor("pelvis")
    zeros = torch.zeros(29, device=device)
    mdp = MDP(
        kin,
        motion,
        zeros,
        zeros - 1,
        zeros + 1,
        obs_terms=(
            "joint_pos",
            "joint_vel",
            "base_ang_vel",
            "projected_gravity",
            "last_action",
            "anchor_ori",
            "anchor_pos",
            "body_pos",
            "body_ori",
        ),
    )
    # Materialize FK device buffers before any compile / CUDAGraph capture.
    with torch.no_grad():
        kin.forward(*unflatten_state(torch.zeros(1, STATE_DIM, device=device)))

    wm = WorldModel("rms", False).to(device)
    norms = InputNorms(mdp.obs_dim).to(device)
    with torch.no_grad():
        wm.net.head.weight.normal_(0, 0.01)
    mse_loss = MSELoss(device).to(device)
    opt = build_optimizer(wm, wm_lr, optim)

    def rand_quat(shape: tuple[int, ...]) -> torch.Tensor:
        q = torch.randn(*shape, 4, device=device)
        return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    def make_batch() -> dict[str, torch.Tensor]:
        """Fresh tensors each call — same shape as ``ReplayBuffer.sample_windows``."""
        s = torch.randn(batch, k, STATE_DIM, device=device)
        s[..., 3:7] = rand_quat((batch, k))
        s1 = torch.randn(batch, k, STATE_DIM, device=device)
        s1[..., 3:7] = rand_quat((batch, k))
        body_quat = rand_quat((batch, k, N_BODY))
        return {
            "s": s,
            "s1": s1,
            "a": torch.randn(batch, k, 29, device=device),
            "xi": torch.randn(batch, k, XI_DIM, device=device),
            "c": torch.zeros(batch, k, N_CONTACT, device=device),
            "c1": torch.zeros(batch, k, N_CONTACT, device=device),
            "body_pos": torch.randn(batch, k, N_BODY, 3, device=device),
            "body_quat": body_quat,
        }

    return wm, norms, mdp, mse_loss, opt, make_batch, contact_bce


def _bind_rollout(unit: str, mode: str | None, compile_on: bool):
    """Return ``(rollout_fn, use_cudagraph_mark)``.

    ``rollout_fn`` has the same signature as ``world_model_window``.
    """
    from supermimic.algo import world_model_step, world_model_window, world_model_window_steps

    use_graphs = bool(compile_on and mode == "reduce-overhead")
    if not compile_on:
        return world_model_window, False
    if unit == "window":
        return torch.compile(world_model_window, mode=mode, dynamic=False), use_graphs
    if unit == "step":
        compiled_step = torch.compile(world_model_step, mode=mode, dynamic=False)

        def rollout(wm, norms, s, s1, actions, xi, contact, body_pos, body_quat, mdp, detach_pred=True):
            return world_model_window_steps(
                wm, norms, s, s1, actions, xi, contact, body_pos, body_quat, mdp, detach_pred, compiled_step
            )

        return rollout, use_graphs
    raise ValueError(f"unknown unit {unit!r}")


def _wm_update(
    *,
    wm,
    norms,
    mdp,
    mse_loss,
    opt,
    make_batch,
    contact_bce,
    rollout_fn,
    use_graphs: bool,
    n_batches: int,
    contact_coef: float,
    grad_clip: float,
):
    """One training-like WM update: ``n_batches`` fresh windows, loss, backward, Muon step."""
    from supermimic.algo import _clip_grads

    for _ in range(n_batches):
        batch = make_batch()
        if use_graphs:
            torch.compiler.cudagraph_mark_step_begin()
        errors, _, _, contact_logits = rollout_fn(
            wm,
            norms,
            batch["s"],
            batch["s1"],
            batch["a"],
            batch["xi"],
            batch["c"],
            batch["body_pos"],
            batch["body_quat"],
            mdp,
            True,
        )
        mse, _err8 = mse_loss(errors)
        contact_loss = contact_bce(contact_logits, batch["c1"].transpose(0, 1))
        loss = mse + contact_coef * contact_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        _clip_grads(wm.parameters(), grad_clip)
        opt.step()


def _worker(
    conn,
    unit: str,
    mode: str | None,
    batch: int,
    k: int,
    n_batches: int,
    timeout: float,
    device_str: str,
    warmup: int,
    iters: int,
    amp: str,
    optim: str,
    wm_lr: float,
    contact_coef: float,
    grad_clip: float,
) -> None:
    try:
        device = torch.device(device_str)
        compile_on = mode is not None
        if compile_on:
            import torch._functorch.config as functorch_config

            functorch_config.donated_buffer = False

        wm, norms, mdp, mse_loss, opt, make_batch, contact_bce = _setup_wm(
            batch, k, device, amp, optim, wm_lr
        )
        rollout_fn, use_graphs = _bind_rollout(unit, mode, compile_on)
        conn.send({"event": "ready"})
        if conn.recv() != "go":
            return

        result: dict = {"unit": unit if compile_on else "eager", "mode": mode or "eager", "status": "ok"}

        def one_batch():
            _wm_update(
                wm=wm,
                norms=norms,
                mdp=mdp,
                mse_loss=mse_loss,
                opt=opt,
                make_batch=make_batch,
                contact_bce=contact_bce,
                rollout_fn=rollout_fn,
                use_graphs=use_graphs,
                n_batches=1,
                contact_coef=contact_coef,
                grad_clip=grad_clip,
            )

        def one_update():
            _wm_update(
                wm=wm,
                norms=norms,
                mdp=mdp,
                mse_loss=mse_loss,
                opt=opt,
                make_batch=make_batch,
                contact_bce=contact_bce,
                rollout_fn=rollout_fn,
                use_graphs=use_graphs,
                n_batches=n_batches,
                contact_coef=contact_coef,
                grad_clip=grad_clip,
            )

        # Compile budget covers one batch only (fwd+bwd+opt). Do not inflate timeout.
        compile_s = _timed(one_batch, timeout, device)
        result["compile_s"] = compile_s
        if compile_s is None:
            result["status"] = "compile timeout"
            conn.send(result)
            return

        try:
            samples = _steady_samples(one_update, warmup, iters, device)
        except Exception as exc:
            result["status"] = f"steady error: {type(exc).__name__}: {exc}"
            conn.send(result)
            return

        result["steady_ms"] = statistics.mean(samples) * 1e3
        result["steady_std_ms"] = statistics.pstdev(samples) * 1e3 if len(samples) > 1 else 0.0
        result["batch_ms"] = result["steady_ms"] / n_batches
        conn.send(result)
    except Exception:
        conn.send(
            {
                "unit": unit if mode is not None else "eager",
                "mode": mode or "eager",
                "status": "error",
                "error": traceback.format_exc(),
            }
        )


def _run_case(ctx, unit: str, mode: str | None, args) -> dict:
    parent, child = ctx.Pipe()
    tag = "eager" if mode is None else unit
    proc = ctx.Process(
        target=_worker,
        args=(
            child,
            unit,
            mode,
            args.batch,
            args.k,
            args.wm_batches,
            args.timeout,
            args.device,
            args.warmup,
            args.iters,
            args.amp,
            args.optim,
            args.wm_lr,
            args.contact_coef,
            args.grad_clip,
        ),
    )
    label = f"{tag}/{mode or 'eager'}"
    print(f"start {label}", flush=True)
    proc.start()
    child.close()
    if not parent.poll(args.setup_timeout):
        proc.terminate()
        proc.join(5)
        print(f"  {label} setup timeout", flush=True)
        return {"unit": tag, "mode": mode or "eager", "status": "setup timeout"}
    message = parent.recv()
    if message.get("event") != "ready":
        proc.terminate()
        proc.join(5)
        message.setdefault("unit", tag)
        message.setdefault("mode", mode or "eager")
        message.setdefault("status", "setup failed")
        _print_row(message)
        return message
    parent.send("go")
    # Compile budget + steady updates (each update is wm_batches batches).
    limit = args.timeout + (args.warmup + args.iters) * max(30.0, 2.0 * args.wm_batches) + 30
    if not parent.poll(limit):
        proc.terminate()
        proc.join(5)
        print(f"  {label} killed after {limit:.0f}s", flush=True)
        return {"unit": tag, "mode": mode or "eager", "status": "killed"}
    result = parent.recv()
    proc.join(10)
    _print_row(result)
    return result


def _fmt(value, spec: str) -> str:
    if value is None:
        return "-"
    return format(value, spec)


def _print_row(row: dict, *, cached: bool = False) -> None:
    tag = "cached" if cached else row.get("status")
    print(
        f"  {row.get('unit', '?'):6} {str(row.get('mode', '-')):16} "
        f"compile {_fmt(row.get('compile_s'), '6.1f')}s  "
        f"update {_fmt(row.get('steady_ms'), '7.1f')}±{_fmt(row.get('steady_std_ms'), '5.1f')} ms  "
        f"batch {_fmt(row.get('batch_ms'), '6.1f')} ms  {tag}",
        flush=True,
    )
    if row.get("error"):
        print(row["error"], flush=True)


def _case_key(unit: str, mode: str | None, args) -> str:
    payload = {
        "unit": unit if mode is not None else "eager",
        "mode": mode or "eager",
        "batch": args.batch,
        "k": args.k,
        "wm_batches": args.wm_batches,
        "amp": args.amp,
        "optim": args.optim,
        "wm_lr": args.wm_lr,
        "contact_coef": args.contact_coef,
        "grad_clip": args.grad_clip,
        "warmup": args.warmup,
        "iters": args.iters,
        "timeout": args.timeout,
        # Bump when the measured path changes so stale rows are ignored.
        "bench": "wm_update_v2",
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    label = f"{payload['unit']}/{payload['mode']}"
    return f"{label}:{digest}"


def _load_cache(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _save_cache(path: Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=2048, help="wm_batch (train default 2048)")
    parser.add_argument("--k", type=int, default=8, help="wm_window")
    parser.add_argument("--wm-batches", type=int, default=10, dest="wm_batches")
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Seconds allowed for compiling one batch (fwd+bwd+opt). Default 60.",
    )
    parser.add_argument("--setup-timeout", type=float, default=180.0, dest="setup_timeout")
    parser.add_argument("--warmup", type=int, default=2, help="Steady-state warmups after compile.")
    parser.add_argument("--iters", type=int, default=6, help="Timed steady-state updates.")
    parser.add_argument("--amp", default="bf16", choices=("bf16", "none"))
    parser.add_argument("--optim", default="muon", choices=("muon", "adam"))
    parser.add_argument("--wm-lr", type=float, default=2e-4, dest="wm_lr")
    parser.add_argument("--contact-coef", type=float, default=0.01, dest="contact_coef")
    parser.add_argument("--grad-clip", type=float, default=1.0, dest="grad_clip")
    parser.add_argument(
        "--units",
        default="eager,window,step",
        help="Comma list: eager, window, step.",
    )
    parser.add_argument(
        "--modes",
        default=",".join(WM_MODES),
        help="Inductor modes for compiled units (ignored for eager).",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=CACHE_PATH,
        help=f"JSON result cache (default {CACHE_PATH.name}).",
    )
    parser.add_argument("--force", action="store_true", help="Ignore cache and re-benchmark.")
    parsed = parser.parse_args()
    modes = tuple(m.strip() for m in parsed.modes.split(",") if m.strip())
    units = tuple(u.strip() for u in parsed.units.split(",") if u.strip())
    for unit in units:
        if unit not in ("eager", "window", "step"):
            raise SystemExit(f"unknown unit {unit!r}")
    ctx = mp.get_context("spawn")
    cases: list[tuple[str, str | None]] = []
    for unit in units:
        if unit == "eager":
            cases.append(("window", None))  # path is window; labeled eager via mode=None
        else:
            for mode in modes:
                cases.append((unit, mode))

    cache = {} if parsed.force else _load_cache(parsed.cache)
    print(
        f"WM update bench: batch={parsed.batch} K={parsed.k} batches/iter={parsed.wm_batches} "
        f"amp={parsed.amp} optim={parsed.optim} contact_coef={parsed.contact_coef} "
        f"timeout={parsed.timeout:.0f}s cache={parsed.cache}",
        flush=True,
    )
    rows = []
    for unit, mode in cases:
        key = _case_key(unit, mode, parsed)
        if key in cache and not parsed.force:
            row = dict(cache[key])
            _print_row(row, cached=True)
            rows.append(row)
            continue
        row = _run_case(ctx, unit, mode, parsed)
        cache[key] = {k: v for k, v in row.items() if k != "error" or row.get("status") == "error"}
        if row.get("error"):
            cache[key]["error"] = row["error"]
        _save_cache(parsed.cache, cache)
        rows.append(row)

    print(
        "\nunit   mode             compile_s  update_ms±std   batch_ms  status",
        flush=True,
    )
    for row in rows:
        print(
            f"{row.get('unit', '?'):6} {str(row.get('mode', '-')):16} "
            f"{_fmt(row.get('compile_s'), '9.1f')} "
            f"{_fmt(row.get('steady_ms'), '8.1f')}±{_fmt(row.get('steady_std_ms'), '5.1f')} "
            f"{_fmt(row.get('batch_ms'), '8.1f')}  {row.get('status')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
