"""Compile-time and runtime of the policy loss.

Compares an eager imagination horizon with torch.compile on one imagination
step and on the whole horizon. Each compile phase (forward, then backward)
is limited to ``--timeout`` seconds so a long Inductor compile is recorded
and killed instead of running for minutes.

Usage:
  CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src python benchmark/train_compile.py
  CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src python benchmark/train_compile.py --timeout 40
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import signal
import time
import traceback

import torch

MODES = ("default", "reduce-overhead", "max-autotune-no-cudagraphs")


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
    """Run ``fn`` and return seconds, or None if it exceeds ``timeout``.

    Inductor wraps an alarm raised inside the compiler as ``BackendCompilerFailed``.
    That still counts as a timeout.
    """
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


def _setup(batch: int, device: torch.device):
    from supermimic.kinematics import TorchKinematics, compile_g1
    from supermimic.mdp import (
        GOAL_DIM,
        HIST,
        INV_DIM,
        PRIV_DIM,
        STATE_DIM,
        XI_DIM,
        MDP,
    )
    from supermimic.motion import MotionLib
    from supermimic.network import Critic, InputNorms, Policy, WorldModel

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
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
        ),
    )
    policy = Policy(mdp.obs_dim, GOAL_DIM, HIST).to(device)
    wm = WorldModel().to(device)
    critic = Critic(INV_DIM + GOAL_DIM + XI_DIM + PRIV_DIM).to(device)
    norms = InputNorms(mdp.obs_dim).to(device)
    with torch.no_grad():
        wm.net.head.weight.normal_(0, 0.01)
    wm.requires_grad_(False)
    critic.requires_grad_(False)

    def rand_state(n: int) -> torch.Tensor:
        state = torch.randn(n, STATE_DIM, device=device)
        quat = state[..., 3:7]
        state[..., 3:7] = quat / quat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return state

    args = (
        policy,
        wm,
        critic,
        norms,
        mdp,
        rand_state(batch * HIST).view(batch, HIST, STATE_DIM),
        torch.randn(batch, HIST, 29, device=device),
        torch.randn(batch, HIST, XI_DIM, device=device),
        torch.zeros(batch, HIST, dtype=torch.long, device=device),
        rand_state(batch),
        torch.randn(batch, 29, device=device),
        torch.randn(batch, XI_DIM, device=device),
        torch.zeros(batch, dtype=torch.long, device=device),
        torch.ones((), device=device),
    )
    return policy, wm, norms, mdp, args


def _imagine_step(
    policy,
    wm,
    norms,
    mdp,
    obs_hist,
    goal,
    p,
    quat,
    v,
    w,
    q,
    qd,
    a_prev,
    xi_n,
    xi,
    frame,
    deterministic: bool,
):
    """One imagination step: actor, world model, next observation and goal."""
    from supermimic.mdp import GOAL_FRAMES

    action = policy(obs_hist, goal, norms, deterministic=deterministic)
    p, quat, v, w, q, qd = wm.step(p, quat, v, w, q, qd, action, xi_n, norms)
    frame = frame + 1
    goal = mdp.motion.goal(frame, GOAL_FRAMES)
    obs = mdp.add_noise(mdp.obs(p, quat, v, w, q, qd, action, xi, frame))
    obs_hist = torch.cat((obs_hist[:, 1:], obs.unsqueeze(1)), dim=1)
    return obs_hist, goal, p, quat, v, w, q, qd, frame, action


def _loss_from_step(step, bundle, horizon: int):
    """Policy loss over ``horizon`` calls of ``step``, with the eager reward tail."""
    from supermimic.algo import _flatten_time, _stack
    from supermimic.mdp import GOAL_FRAMES, HIST, invariant_features, normalize_xi, unflatten_state

    policy, wm, critic, norms, mdp, hist_s, hist_a, hist_xi, hist_f, s, a_prev, xi, frame, scale = bundle
    batch = s.shape[0]
    device = s.device
    p, quat, v, w, q, qd = unflatten_state(hist_s)
    n = p.shape[0] * p.shape[1]
    obs = mdp.obs(
        p.reshape(n, 3),
        quat.reshape(n, 4),
        v.reshape(n, 3),
        w.reshape(n, 3),
        q.reshape(n, -1),
        qd.reshape(n, -1),
        hist_a.reshape(n, -1),
        hist_xi.reshape(n, -1),
        hist_f.reshape(n),
    )
    obs_hist = mdp.add_noise(obs).view(batch, HIST, -1)
    p, quat, v, w, q, qd = unflatten_state(s)
    xi_n = normalize_xi(xi)
    goal = mdp.motion.goal(frame, GOAL_FRAMES)
    states = [(p, quat, v, w, q, qd)]
    frames = [frame]
    goals = [goal]
    actions = []
    prevs = []
    for _ in range(horizon):
        prevs.append(a_prev)
        obs_hist, goal, p, quat, v, w, q, qd, frame, action = step(
            policy, wm, norms, mdp, obs_hist, goal, p, quat, v, w, q, qd, a_prev, xi_n, xi, frame, True
        )
        actions.append(action)
        a_prev = action
        states.append((p, quat, v, w, q, qd))
        frames.append(frame)
        goals.append(goal)
    flat = _flatten_time(states[1:])
    reward = mdp.reward(*flat, _stack(frames[1:]), _stack(actions), _stack(prevs))
    done = mdp.soft_termination(flat[0], flat[1], _stack(frames[1:]))
    reward = (reward / scale).view(horizon, batch)
    done = done.view(horizon, batch)
    alive = torch.cumprod(1.0 - done, dim=0)
    survival = torch.cat((torch.ones(1, batch, device=device), alive[:-1]), dim=0)
    discount = 0.99 ** torch.arange(horizon, device=device)
    returns = (discount[:, None] * survival * reward).sum(0)
    bootstrap = (0.99**horizon) * alive[-1] * critic(
        norms.critic(
            invariant_features(*states[-1]),
            goals[-1],
            xi_n,
            mdp.privileged(*states[-1], frames[-1]),
        )
    )
    return -(returns + bootstrap).mean()


def _steady(fn, iters: int, device: torch.device) -> float:
    fn()
    _sync(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(device)
    return (time.perf_counter() - start) / iters


def _worker(conn, kind: str, mode: str | None, batch: int, horizon: int, timeout: float, device_str: str, iters: int) -> None:
    try:
        device = torch.device(device_str)
        policy, wm, norms, mdp, bundle = _setup(batch, device)
        conn.send({"event": "ready"})
        if conn.recv() != "go":
            return
        result: dict = {"kind": kind, "mode": mode or "-", "status": "ok"}
        if kind == "eager":
            from supermimic.algo import imagine_horizon

            def run():
                policy.zero_grad(set_to_none=True)
                imagine_horizon(*bundle, horizon, 0.99, True)[0].backward()

            result["steady_ms"] = _steady(run, iters, device) * 1e3
        elif kind == "step":
            compiled = torch.compile(_imagine_step, mode=mode, dynamic=False)
            obs = torch.randn(batch, 10, mdp.obs_dim, device=device)
            goal = torch.randn(batch, 730, device=device)
            p, quat, v, w, q, qd = (torch.randn(batch, n, device=device) for n in (3, 4, 3, 3, 29, 29))
            quat = quat / quat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            step_args = (
                policy,
                wm,
                norms,
                mdp,
                obs,
                goal,
                p,
                quat,
                v,
                w,
                q,
                qd,
                torch.randn(batch, 29, device=device),
                torch.randn(batch, 33, device=device),
                torch.randn(batch, 33, device=device),
                torch.zeros(batch, dtype=torch.long, device=device),
                True,
            )

            def fwd():
                compiled(*step_args)

            def bwd():
                policy.zero_grad(set_to_none=True)
                out = compiled(*step_args)
                out[-1].square().mean().backward()

            fwd_s = _timed(fwd, timeout, device)
            result["fwd_s"] = fwd_s
            if fwd_s is None:
                result["status"] = "fwd timeout"
            else:
                bwd_s = _timed(bwd, timeout, device)
                result["bwd_s"] = bwd_s
                if bwd_s is None:
                    result["status"] = "bwd timeout"
                else:
                    def run():
                        policy.zero_grad(set_to_none=True)
                        _loss_from_step(compiled, bundle, horizon).backward()

                    result["steady_ms"] = _steady(run, iters, device) * 1e3
        elif kind == "horizon":
            from supermimic.algo import imagine_horizon

            compiled = torch.compile(imagine_horizon, mode=mode, dynamic=False)

            def fwd():
                compiled(*bundle, horizon, 0.99, True)

            def bwd():
                policy.zero_grad(set_to_none=True)
                compiled(*bundle, horizon, 0.99, True)[0].backward()

            fwd_s = _timed(fwd, timeout, device)
            result["fwd_s"] = fwd_s
            if fwd_s is None:
                result["status"] = "fwd timeout"
            else:
                bwd_s = _timed(bwd, timeout, device)
                result["bwd_s"] = bwd_s
                if bwd_s is None:
                    result["status"] = "bwd timeout"
                else:
                    def run():
                        policy.zero_grad(set_to_none=True)
                        compiled(*bundle, horizon, 0.99, True)[0].backward()

                    result["steady_ms"] = _steady(run, iters, device) * 1e3
        else:
            result["status"] = f"unknown kind {kind}"
        conn.send(result)
    except Exception:
        conn.send({"kind": kind, "mode": mode or "-", "status": "error", "error": traceback.format_exc()})


def _run_case(ctx, kind, mode, args) -> dict:
    parent, child = ctx.Pipe()
    proc = ctx.Process(
        target=_worker,
        args=(child, kind, mode, args.batch, args.horizon, args.timeout, args.device, args.iters),
    )
    label = f"{kind}/{mode or 'eager'}"
    print(f"start {label}", flush=True)
    proc.start()
    child.close()
    ready = parent.poll(args.setup_timeout)
    if not ready:
        proc.terminate()
        proc.join(5)
        print(f"  {label} setup timeout", flush=True)
        return {"kind": kind, "mode": mode or "-", "status": "setup timeout"}
    message = parent.recv()
    if message.get("event") != "ready":
        proc.terminate()
        proc.join(5)
        message.setdefault("kind", kind)
        message.setdefault("mode", mode or "-")
        message.setdefault("status", "setup failed")
        _print_row(message)
        return message
    parent.send("go")
    # Forward and backward each get their own alarm. Extra slack covers the steady-state loop.
    limit = 2 * args.timeout + 60
    if not parent.poll(limit):
        proc.terminate()
        proc.join(5)
        print(f"  {label} killed after {limit:.0f}s", flush=True)
        return {"kind": kind, "mode": mode or "-", "status": "killed"}
    result = parent.recv()
    proc.join(10)
    _print_row(result)
    return result


def _fmt(value, spec: str) -> str:
    if value is None:
        return "-"
    return format(value, spec)


def _print_row(row: dict) -> None:
    print(
        f"  {row.get('kind', '?'):8} {str(row.get('mode', '-')):28} "
        f"fwd {_fmt(row.get('fwd_s'), '6.1f')}s  bwd {_fmt(row.get('bwd_s'), '6.1f')}s  "
        f"steady {_fmt(row.get('steady_ms'), '7.1f')} ms  {row.get('status')}",
        flush=True,
    )
    if row.get("error"):
        print(row["error"], flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=40.0, help="Seconds allowed for one compile phase.")
    parser.add_argument("--setup-timeout", type=float, default=180.0, dest="setup_timeout")
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--modes", default=",".join(MODES))
    parsed = parser.parse_args()
    modes = tuple(mode.strip() for mode in parsed.modes.split(",") if mode.strip())
    ctx = mp.get_context("spawn")
    cases = [("eager", None)]
    for mode in modes:
        cases.append(("step", mode))
        cases.append(("horizon", mode))
    rows = [_run_case(ctx, kind, mode, parsed) for kind, mode in cases]
    print("\nregion   mode                          fwd_s   bwd_s   steady_ms  status")
    for row in rows:
        print(
            f"{row.get('kind', '?'):8} {str(row.get('mode', '-')):28} "
            f"{_fmt(row.get('fwd_s'), '7.1f')} {_fmt(row.get('bwd_s'), '7.1f')} "
            f"{_fmt(row.get('steady_ms'), '10.1f')}  {row.get('status')}"
        )


if __name__ == "__main__":
    main()
