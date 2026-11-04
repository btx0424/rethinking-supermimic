"""Smoke-test the episode-end contract of :meth:`TrackingSim.step`.

A transition reports ``terminated`` (the MDP's own failure), ``truncated`` (the
time limit or the end of the reference clip) and ``done``, with the invariant::

    done == terminated | truncated     # done is only their union

``truncated`` is itself split into ``timeout`` and ``clip_end``
(``truncated == timeout | clip_end``; these two may not overlap, since the
episode length can exceed the clip length). Termination and truncation are
reported independently and *may* co-occur: a fall on the very last frame of a
clip is both. Nothing may assume they are disjoint; only the union invariant
above is guaranteed.

Run with the usual environment::

    WEAVE_ROOT=... python scripts/check_done_contract.py task=weave_xhand
"""

from __future__ import annotations

import argparse

import torch
from hydra import compose, initialize

from supermimic.tasks import make_task_from_config


def _check(out: dict, step: int) -> None:
    terminated = out["terminated"]
    truncated = out["truncated"]
    done = out["done"]
    assert terminated.dtype == torch.bool, terminated.dtype
    assert truncated.dtype == torch.bool, truncated.dtype
    assert done.dtype == torch.bool, done.dtype
    if not torch.equal(done, terminated | truncated):
        raise AssertionError(f"step {step}: done != terminated | truncated")
    if not torch.equal(truncated, out["timeout"] | out["clip_end"]):
        raise AssertionError(f"step {step}: truncated != timeout | clip_end")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Shorten the episode limit so the timeout truncation path is exercised.",
    )
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    with initialize(version_base=None, config_path="../src/supermimic/conf"):
        cfg = compose(config_name="train", overrides=list(args.overrides))
    task_cfg = cfg.task
    task = make_task_from_config(task_cfg)

    from supermimic.sim import TrackingSim
    from supermimic.utils.kinematics import make_kinematics

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, joint_names = task.compile_model()
    kin = make_kinematics(model, joint_names, "warp", device, hand_bodies=task.hand_bodies)
    kin.set_anchor("torso")
    # ``eval_mode=False`` so the automatic reset can sample fresh clips; the
    # contract under test is on ``step``'s return value, not on the reset policy.
    sim = TrackingSim(args.envs, task.load_motion(device), kin, device, task=task)
    if args.max_steps is not None:
        # Exercise the timeout branch: a limit below any clip length, so the
        # episode ends by truncation before the reference clip runs out.
        sim.max_steps = int(args.max_steps)
    n = args.envs
    counts = {"terminated": 0, "truncated": 0, "timeout": 0, "clip_end": 0, "done": 0}
    action = torch.zeros(n, task.num_joints, device=device)
    for step in range(args.steps):
        out = sim.step(action, auto_reset=True)
        _check(out, step)
        for key in counts:
            counts[key] += int(out[key].sum())
    print(f"ok: {args.steps} steps x {n} envs")
    print("  " + "  ".join(f"{k}={v}" for k, v in counts.items()))

    # The clip-end branch: start every env on the last frame of its clip, so the
    # very next step must report ``clip_end`` truncation (and nothing else).
    motion = sim.motion
    env_ids = torch.arange(n, device=device)
    clip = torch.arange(n, device=device) % motion.num_clips
    last = (motion.clip_end[clip] - 1).long()
    sim.reset(env_ids, frames=last)
    out = sim.step(torch.zeros(n, task.num_joints, device=device), auto_reset=False)
    _check(out, args.steps)
    if not bool(out["clip_end"].all()):
        raise AssertionError("expected every env to report clip_end at the last frame")
    print(
        f"clip_end ok: {int(out['clip_end'].sum())}/{n} clip_end, "
        f"{int(out['truncated'].sum())}/{n} truncated, "
        f"{int(out['terminated'].sum())}/{n} terminated"
    )


if __name__ == "__main__":
    main()
