"""Collect a rollout archive from any trained policy, for RLPD prior mixing.

Loads a ``supermimic``, ``nablamimic`` or ``sac`` checkpoint, replays every clip
from its first frame with the deterministic policy (the same protocol as
``supermimic.eval``), and records exactly the fields
:class:`~supermimic.sac.replay_buffer.ReplayBuffer` stores. The archive is written
by :func:`supermimic.sac.prior.save_rollout` and fed back to a training run with
``algo=sac prior_data=<dir>``.

The archive is always in SAC's layout (``obs`` is ``[observation, goal]``, critic
is the normalized critic input) whichever trainer produced the policy, because it
is consumed by SAC's replay buffer. Observations are stored **raw** (before the
policy's own normalization) and rewards unnormalized, so the consuming run applies
its own running statistics.

    PYTHONPATH=src python -m supermimic.collect \
        --ckpt logs/lafan1/2026-11-04_07-39-08/ckpt_000004.pt

The output directory defaults to ``data/prior/<ckpt path under logs>``, so a
checkpoint at ``logs/<task>/<timestamp>/ckpt_N.pt`` is archived at
``data/prior/<task>/<timestamp>/ckpt_N/rollout.pt``. Pass ``--out`` to override.

Add ``--action-noise`` to record exploratory actions instead of the deterministic
mean; plain RLPD mixes demonstrations, so the default is deterministic.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from supermimic.policies import (
    build_sim,
    has_ema,
    load_critic_norms,
    load_policy,
    task_from_checkpoint,
)
from supermimic.sac import features as sac_features
from supermimic.sac.prior import save_rollout
from supermimic.sac.replay_buffer import ReplayBuffer

_REPO_ROOT = Path(__file__).resolve().parents[2]
# Prior archives live beside the motion caches; ``data/`` is git-ignored.
PRIOR_DIR = _REPO_ROOT / "data" / "prior"


def default_out_dir(ckpt: Path | str) -> Path:
    """Archive directory derived from the checkpoint's own location.

    A checkpoint under ``logs/`` keeps its path, so the archive names the run and
    the checkpoint that produced it: ``logs/<task>/<stamp>/ckpt_N.pt`` becomes
    ``data/prior/<task>/<stamp>/ckpt_N``. Anything else falls back to
    ``data/prior/<parent name>/<ckpt stem>``.
    """
    ckpt = Path(ckpt).expanduser()
    try:
        resolved = ckpt.resolve()
    except OSError:
        resolved = ckpt
    try:
        rel = resolved.relative_to((_REPO_ROOT / "logs").resolve())
    except ValueError:
        rel = Path(ckpt.parent.name) / ckpt.stem
    else:
        rel = rel.with_suffix("")
    return PRIOR_DIR / rel


def collect(
    ckpt: Path,
    out_dir: Path | None = None,
    *,
    device: str = "cuda",
    num_envs: int | None = None,
    action_noise: float = 0.0,
    seed: int = 0,
    live: bool = False,
) -> Path:
    """Replay the clips with ``ckpt``'s policy and write a prior archive.

    Uses the Polyak-averaged (EMA) policy when the checkpoint has one, matching
    ``supermimic.eval``; ``live=True`` selects the online weights instead.
    """
    ckpt = Path(ckpt)
    if out_dir is None:
        out_dir = default_out_dir(ckpt)
    out_dir = Path(out_dir)
    print(f"prior archive -> {out_dir}", flush=True)
    torch.manual_seed(seed)

    task, cfg, saved = task_from_checkpoint(ckpt, device)
    n = int(num_envs) if num_envs is not None else int(task.load_motion(device).num_clips)
    # ``survival_bonus`` matches SAC's training reward, so the archive's rewards
    # are on the same scale as the run that consumes them. The *score* is computed
    # from the tracking reward alone (bonus removed) to match ``supermimic.eval``.
    sim = build_sim(task, cfg, device, num_envs=n, survival_bonus=1.0)
    policy = load_policy(saved, cfg, task, sim, device, live=live)
    # The archive always uses SAC's critic layout, so it needs SAC's normalizers
    # regardless of which trainer produced the policy.
    critic_norms = load_critic_norms(saved, task, sim.mdp.obs_dim, device)
    weights = "online" if live else ("EMA" if has_ema(saved) else "only")
    print(f"policy: {policy.algo} ({policy.kind}, {weights}) · {n} envs", flush=True)

    buffer = ReplayBuffer(
        int(cfg.get("buffer_steps", 1000)),
        n,
        sim.mdp.obs_dim + task.goal_dim,
        sac_features.critic_feat_dim(task),
        task.num_joints,
        device,
    )

    # ``eval_mode`` puts every env on its own clip start with no reset noise.
    # Score each clip with the evaluation protocol (no noise, deterministic
    # policy) so the archive's quality is visible before it is used as prior data.
    steps = 0
    horizon = buffer.max_size
    live_mask = torch.ones(n, dtype=torch.bool, device=device)
    # Tracking reward only (the survival bonus is added by the simulator for the
    # stored transition); this is what makes the printed score comparable to
    # ``supermimic.eval``.
    track_returns = torch.zeros(n, device=device)
    terminated_any = torch.zeros(n, dtype=torch.bool, device=device)
    truncated_any = torch.zeros(n, dtype=torch.bool, device=device)
    bonus = sim.survival_bonus
    t0 = time.time()
    with torch.no_grad():
        while steps < horizon and bool(live_mask.any()):
            policy_obs = sac_features.policy_obs(sim, noise=False)
            critic = sac_features.critic_input(sim, critic_norms)
            fresh = (sim.ep_step == 0).float().unsqueeze(-1)
            action = policy.act()
            recorded = action
            if action_noise > 0:
                recorded = action + action_noise * torch.randn_like(action)
            out = sim.step(recorded, auto_reset=False)
            buffer.push(
                {
                    "obs": policy_obs,
                    "critic": critic,
                    "action": recorded,
                    "reward": out["r"].unsqueeze(-1),
                    "terminated": out["terminated"].unsqueeze(-1).float(),
                    "truncated": out["truncated"].unsqueeze(-1).float(),
                    "done": out["done"].unsqueeze(-1).float(),
                    "is_init": fresh,
                }
            )
            # The bonus is paid while the episode is alive (``~terminated``), so
            # removing it recovers the pure tracking reward the MDP's max refers to.
            tracking = out["r"] - bonus * (~out["terminated"]).float()
            track_returns = track_returns + tracking * live_mask.float()
            terminated_any = terminated_any | (out["terminated"].bool() & live_mask)
            truncated_any = truncated_any | (out["truncated"].bool() & live_mask)
            live_mask = live_mask & ~out["done"]
            steps += 1
    if steps < 1:
        raise RuntimeError("Rollout produced no transitions.")

    score = _report_scores(
        sim,
        track_returns,
        terminated_any,
        truncated_any,
        steps,
        live=live,
        algo=policy.algo,
    )

    path = save_rollout(
        out_dir,
        buffer,
        task_cfg=dict(saved["task_cfg"]),
        train_cfg={"actor_init": cfg.get("actor_init", "zeros")},
        extras={
            "source_ckpt": str(ckpt),
            "algo": policy.algo,
            "policy_kind": policy.kind,
            "policy_weights": weights,
            "normalized_return": score,
            "terminated_clips": int(terminated_any.sum()),
            "truncated_clips": int(truncated_any.sum()),
            "steps": steps,
            "episodes": n,
            "action_noise": float(action_noise),
            "deterministic": action_noise <= 0,
        },
    )
    print(
        f"wrote {path}: {steps} steps x {n} envs = {buffer.num_samples} transitions "
        f"in {time.time() - t0:.1f}s",
        flush=True,
    )
    return path


def _report_scores(
    sim,
    returns: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    steps: int,
    *,
    live: bool,
    algo: str,
) -> float:
    """Print the evaluation-protocol score of the collected policy.

    ``normalized_return`` is the clip return over the maximum attainable reward —
    the same quantity ``supermimic.eval`` reports — so the archive can be judged
    before it is used as prior data. Each env rolls out one clip, so the clip
    lengths are gathered by the env's clip id rather than taking all 779.

    ``terminated`` and ``truncated`` are reported separately: a clip that was
    cut by the time limit or the end of the reference clip did not fail, so it
    must not be counted as one. ``returns`` are discounted only to the extent
    the simulator accumulates them undiscounted; ``sim.ep_ret`` already does the
    same accounting, so the ratio is the evaluation metric either way.
    """
    motion = sim.motion
    clip_of_env = motion.clip_id[sim.f].long()
    lengths = (motion.clip_end[clip_of_env] - motion.clip_start[clip_of_env] - 1).clamp(min=1).float()
    score = returns / (sim.mdp.max_step_reward * lengths)
    mean = float(score.mean())
    which = "online" if live else "policy"
    print(
        f"[{algo} {which}] normalized return {mean:.4f}  "
        f"terminated {int(terminated.sum())}/{len(score)}  "
        f"truncated {int(truncated.sum())}/{len(score)}  steps {steps}",
        flush=True,
    )
    # Compact per-clip listing, worst first: the clips that terminated or scored
    # low are what make a prior archive not worth mixing in.
    names = list(motion.clip_names)
    for idx in torch.argsort(score).tolist():
        if bool(terminated[idx]):
            flag = "TERM"
        elif bool(truncated[idx]):
            flag = "TRNC"
        else:
            flag = "    "
        print(f"  {flag} {names[int(clip_of_env[idx])]:32s} {float(score[idx]):.3f}", flush=True)
    return mean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt",
        type=Path,
        required=True,
        help="supermimic / nablamimic / sac checkpoint.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Archive directory (default: data/prior/<checkpoint path under logs>).",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Environments to roll out (default: one per clip).",
    )
    parser.add_argument(
        "--action-noise",
        type=float,
        default=0.0,
        help="Gaussian std added to the deterministic action (0 records the mean).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use the online policy weights instead of the EMA average.",
    )
    args = parser.parse_args()
    collect(
        args.ckpt,
        args.out,
        device=args.device,
        num_envs=args.num_envs,
        action_noise=args.action_noise,
        seed=args.seed,
        live=args.live,
    )


if __name__ == "__main__":
    main()
