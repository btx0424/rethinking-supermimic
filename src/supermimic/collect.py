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
import math
import time
from pathlib import Path

import torch
from tqdm import tqdm

from supermimic.policies import (
    build_sim,
    has_ema,
    load_policy,
    task_from_checkpoint,
)
from supermimic.sac import features as sac_features
from supermimic.sac.prior import save_rollout
from supermimic.sac.replay_buffer import ReplayBuffer

_REPO_ROOT = Path(__file__).resolve().parents[2]
# Prior archives live beside the motion caches; ``data/`` is git-ignored.
PRIOR_DIR = _REPO_ROOT / "data" / "prior"
# Matches ``TrainConfig.soft_bound`` in ``sac.sac`` (mean-action L6 penalty scale).
_SAC_SOFT_BOUND = 3.0 * math.pi
# Trim this fraction from each tail when reporting a "practical" range.
_ACTION_TAIL = 0.005


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
    # Actual steps taken while the episode was live (not the full clip length).
    ep_lengths = torch.zeros(n, device=device)
    # Live-step actions only (post-done steps are still pushed to the ring but
    # should not set the practical action range).
    live_actions: list[torch.Tensor] = []
    bonus = sim.survival_bonus
    t0 = time.time()
    pbar = tqdm(total=n, desc="Collect clips", unit="clip")
    with torch.no_grad():
        while steps < horizon and bool(live_mask.any()):
            policy_obs = sac_features.policy_obs(sim, noise=False)
            # Raw critic features: the consuming SAC run renormalizes with its own
            # ``InputNorms`` (same path as online transitions).
            critic = sac_features.critic_input_raw(sim)
            fresh = (sim.ep_step == 0).float().unsqueeze(-1)
            action = policy.act()
            recorded = action
            if action_noise > 0:
                recorded = action + action_noise * torch.randn_like(action)
            if bool(live_mask.any()):
                live_actions.append(recorded[live_mask].detach())
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
            finished = live_mask & out["done"]
            ep_lengths = torch.where(finished, out["ep_len"].float(), ep_lengths)
            live_mask = live_mask & ~out["done"]
            steps += 1
            pbar.n = n - int(live_mask.sum().item())
            pbar.set_postfix(steps=steps, alive=int(live_mask.sum().item()), refresh=False)
            pbar.refresh()
    # Horizon cut: still-live envs contribute their steps so far.
    ep_lengths = torch.where(live_mask, sim.ep_step.float(), ep_lengths).clamp(min=1.0)
    pbar.close()
    if steps < 1:
        raise RuntimeError("Rollout produced no transitions.")

    score = _report_scores(
        sim,
        track_returns,
        terminated_any,
        truncated_any,
        ep_lengths,
        steps,
        live=live,
        algo=policy.algo,
    )
    action_stats = _report_action_ranges(
        torch.cat(live_actions, dim=0) if live_actions else recorded.new_zeros(0, task.num_joints),
        joint_names=tuple(sim.robot.joint_names),
        soft_bound=_SAC_SOFT_BOUND,
        tail=_ACTION_TAIL,
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
            "action_ranges": action_stats,
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
    ep_lengths: torch.Tensor,
    steps: int,
    *,
    live: bool,
    algo: str,
) -> float:
    """Print the evaluation-protocol score of the collected policy.

    ``normalized_return`` is the episode return over the maximum attainable
    reward for the steps actually taken (termination, truncation, or horizon
    cut), not the full reference clip length. Early falls are not diluted by
    the remaining clip frames the policy never saw.

    ``terminated`` and ``truncated`` are reported separately: a clip that was
    cut by the time limit or the end of the reference clip did not fail, so it
    must not be counted as one.
    """
    motion = sim.motion
    clip_of_env = motion.clip_id[sim.f].long()
    score = returns / (sim.mdp.max_step_reward * ep_lengths)
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
        print(
            f"  {flag} {names[int(clip_of_env[idx])]:32s} {float(score[idx]):.3f}  "
            f"len {int(ep_lengths[idx])}",
            flush=True,
        )
    return mean


def _report_action_ranges(
    actions: torch.Tensor,
    *,
    joint_names: tuple[str, ...],
    soft_bound: float,
    tail: float = _ACTION_TAIL,
) -> dict:
    """Robust per-joint and global action ranges from live rollout actions.

    ``tail`` is trimmed from each end (default 0.5%), so a few spikes do not set
    the reported range. Compares the practical |a| envelope to SAC's
    ``soft_bound`` (the L6 mean penalty scale), which is what motivated this
    report.
    """
    if actions.numel() == 0:
        print("[action ranges] no live actions recorded", flush=True)
        return {}
    if actions.dim() != 2:
        raise ValueError(f"actions must be [N, act_dim], got {tuple(actions.shape)}")
    n, act_dim = actions.shape
    low_q = float(tail)
    high_q = 1.0 - float(tail)
    # ``quantile`` needs a float tensor; keep on the action device.
    a = actions.float()
    per_low = torch.quantile(a, low_q, dim=0)
    per_high = torch.quantile(a, high_q, dim=0)
    per_abs = torch.maximum(per_low.abs(), per_high.abs())
    flat = a.reshape(-1)
    abs_flat = flat.abs()
    global_low = float(torch.quantile(flat, low_q))
    global_high = float(torch.quantile(flat, high_q))
    global_abs = float(torch.quantile(abs_flat, high_q))
    global_abs_max = float(abs_flat.max())
    frac_above = float((abs_flat > soft_bound).float().mean())
    names = list(joint_names) if len(joint_names) == act_dim else [f"a{i}" for i in range(act_dim)]

    print(
        f"[action ranges] n={n}  trim={100 * tail:.2f}% each tail  "
        f"soft_bound={soft_bound:.4g} (=3π)",
        flush=True,
    )
    print(
        f"  global p{100 * low_q:.1f}..p{100 * high_q:.1f} "
        f"[{global_low:.4g}, {global_high:.4g}]  "
        f"|a| p{100 * high_q:.1f}={global_abs:.4g}  "
        f"|a|_max={global_abs_max:.4g}  "
        f"frac(|a|>soft_bound)={frac_above:.3g}",
        flush=True,
    )
    # Joints with the largest practical |a|, so a soft_bound that is too tight
    # shows up as a short list of offenders rather than a 29-line dump.
    order = torch.argsort(per_abs, descending=True).tolist()
    print("  per-joint practical |a| (largest first):", flush=True)
    for idx in order:
        lo = float(per_low[idx])
        hi = float(per_high[idx])
        print(
            f"    {names[idx]:32s}  [{lo: .4g}, {hi: .4g}]  "
            f"|a|≤{float(per_abs[idx]):.4g}",
            flush=True,
        )

    return {
        "tail": float(tail),
        "soft_bound": float(soft_bound),
        "n_live_actions": int(n),
        "global_low": global_low,
        "global_high": global_high,
        "global_abs_p_high": global_abs,
        "global_abs_max": global_abs_max,
        "frac_abs_above_soft_bound": frac_above,
        "joint_names": names,
        "per_joint_low": per_low.cpu().tolist(),
        "per_joint_high": per_high.cpu().tolist(),
        "per_joint_abs": per_abs.cpu().tolist(),
    }


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
