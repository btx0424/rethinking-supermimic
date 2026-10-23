"""Compile and run world_model_window / imagine_step with the torch FK backend.

Usage:
  CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src python scripts/check_torch_fk_compile.py
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from supermimic.algo import TrainConfig, Trainer
from supermimic.mdp import HIST, WorldModelState, normalize_xi, unflatten_state


def _sync() -> None:
    torch.cuda.synchronize()


def main() -> None:
    device = "cuda"
    collect_steps = 16
    cfg = TrainConfig(
        num_envs=64,
        buffer_steps=64,
        wm_batch=32,
        wm_window=8,
        pi_batch=32,
        horizon=8,
        compile=True,
        fk="torch",
        amp="bf16",
        swanlab_mode="disabled",
        anchor="pelvis",
        obs_terms=(
            "joint_pos",
            "joint_vel",
            "base_ang_vel",
            "projected_gravity",
            "last_action",
            "anchor_ori",
            "body_pos",
            "body_ori",
        ),
    )
    out = Path("/tmp/torch_fk_compile_check")
    print(f"Trainer(fk={cfg.fk}, compile={cfg.compile})…", flush=True)
    t0 = time.perf_counter()
    trainer = Trainer(cfg, out)
    print(f"  setup {time.perf_counter() - t0:.1f}s  kin={type(trainer.kin).__name__}", flush=True)
    assert type(trainer.kin).__name__ == "TorchKinematics"

    print(f"collect {collect_steps}…", flush=True)
    trainer.collect(collect_steps)
    assert trainer.buffer.filled >= cfg.wm_window

    batch = trainer.buffer.sample_windows(cfg.wm_batch, cfg.wm_window)
    print("compile+run world_model_window fwd/bwd…", flush=True)
    t0 = time.perf_counter()
    errors, _starts, _preds, logits = trainer._wm_rollout(
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
        False,
    )
    loss = errors.square().mean() + logits.square().mean()
    loss.backward()
    _sync()
    print(f"  wm_window ok  {time.perf_counter() - t0:.1f}s  loss={float(loss):.4f}", flush=True)

    start = {k: v[: cfg.pi_batch] for k, v in trainer.sample_policy_starts().items()}
    print("compile+run imagine_horizon (compiled step) fwd/bwd…", flush=True)
    t0 = time.perf_counter()
    trainer.opt_pi.zero_grad(set_to_none=True)
    pieces = trainer._imagine_graph(start)
    pieces["policy_loss"].backward()
    _sync()
    print(
        f"  imagine ok  {time.perf_counter() - t0:.1f}s  pi_loss={float(pieces['policy_loss']):.4f}",
        flush=True,
    )

    state = WorldModelState(*unflatten_state(start["s"]), start["body_pos"], start["body_quat"])
    obs_hist = torch.randn(cfg.pi_batch, HIST, trainer.sim.mdp.obs_dim, device=device)
    goal = trainer.motion.goal(start["f"], 10)
    print("compiled imagine_step alone…", flush=True)
    t0 = time.perf_counter()
    out_step = trainer._imagine_step(
        trainer.policy,
        trainer.wm,
        trainer.norms,
        trainer.sim.mdp,
        obs_hist,
        goal,
        state,
        start["a_prev"],
        normalize_xi(start["xi"]),
        start["xi"],
        start["f"],
        True,
        start["c"],
    )
    out_step[-1].square().mean().backward()
    _sync()
    print(f"  step ok  {time.perf_counter() - t0:.1f}s", flush=True)
    print("all passed", flush=True)


if __name__ == "__main__":
    main()
