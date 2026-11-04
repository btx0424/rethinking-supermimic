"""Collect a rollout archive with a trained SAC policy, for RLPD prior mixing.

Replays every clip from its first frame with the deterministic policy (the same
protocol as ``supermimic.eval``), recording exactly the fields
:class:`~supermimic.sac.replay_buffer.ReplayBuffer` stores. The archive is written
by :func:`supermimic.sac.prior.save_rollout` and can be fed back to a training run
with ``algo=sac prior_data=<dir>``.

Observations are stored **raw** (before ``ObsVecNorm``) and rewards
unnormalized, so the training run applies its own normalization to them.

    PYTHONPATH=src python -m supermimic.sac.collect \
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

from supermimic.sac import features as sac_features
from supermimic.sac.network import NormalActor
from supermimic.sac.prior import save_rollout
from supermimic.sac.replay_buffer import ReplayBuffer
from supermimic.sim import TrackingSim
from supermimic.supermimic.network import InputNorms
from supermimic.utils.kinematics import make_kinematics
from supermimic.utils.nn import ObsVecNorm

_REPO_ROOT = Path(__file__).resolve().parents[3]
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


def _load_agent(ckpt: Path, device: str):
    """Actor, observation normalizer and critic normalizers from a checkpoint."""
    saved = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = dict(saved.get("cfg", {}))
    task_cfg = saved.get("task_cfg")
    if task_cfg is None:
        raise ValueError(
            f"{ckpt} has no task_cfg; re-save it with a current training run."
        )
    return saved, cfg, task_cfg


def collect(
    ckpt: Path,
    out_dir: Path | None = None,
    *,
    device: str = "cuda",
    num_envs: int | None = None,
    action_noise: float = 0.0,
    seed: int = 0,
) -> Path:
    from supermimic.tasks import make_task_from_config
    from omegaconf import OmegaConf

    ckpt = Path(ckpt)
    if out_dir is None:
        out_dir = default_out_dir(ckpt)
    out_dir = Path(out_dir)
    print(f"prior archive -> {out_dir}", flush=True)
    torch.manual_seed(seed)
    saved, cfg, task_cfg = _load_agent(ckpt, device)
    task = make_task_from_config(OmegaConf.create(dict(task_cfg)))
    motion = task.load_motion(device)
    n = int(num_envs) if num_envs is not None else int(motion.num_clips)
    model, names = task.compile_model()
    kin = make_kinematics(
        model, names, str(cfg.get("fk", "warp")), device, hand_bodies=task.hand_bodies
    )
    sim = TrackingSim(
        n,
        motion,
        kin,
        device,
        gamma=float(cfg.get("gamma", 0.99)),
        anchor=str(cfg.get("anchor", "pelvis")),
        obs_terms=tuple(cfg["obs_terms"]) if cfg.get("obs_terms") else None,
        reward_style=str(cfg.get("reward_style", "gaussian")),
        survival_bonus=1.0,
        eval_mode=True,
        task=task,
    )
    policy_obs_dim = sim.mdp.obs_dim + task.goal_dim
    critic_dim = sac_features.critic_feat_dim(task)

    actor = NormalActor(
        policy_obs_dim,
        task.num_joints,
        action_init=str(cfg.get("actor_init", "zeros")),
    ).to(device)
    actor.load_state_dict(saved["actor"])
    actor.eval()
    vecnorm = None
    if bool(cfg.get("vecnorm", True)):
        vecnorm = ObsVecNorm(policy_obs_dim).to(device)
        vecnorm.load_state_dict(saved["vecnorm"])
        vecnorm.eval()
    norms = InputNorms(sim.mdp.obs_dim, task).to(device)
    norms.load_state_dict(saved["norms"])

    horizon = int(cfg.get("buffer_steps", 1000))
    buffer = ReplayBuffer(
        horizon, n, policy_obs_dim, critic_dim, task.num_joints, device
    )

    # ``eval_mode`` starts every env on a distinct clip and disables reset noise,
    # matching the evaluation protocol. Record until every env terminates.
    alive = torch.ones(n, dtype=torch.bool, device=device)
    steps = 0
    t0 = time.time()
    with torch.no_grad():
        while bool(alive.any()) and steps < horizon:
            # No sensor noise: the archive should look like clean expert data.
            policy_obs = sac_features.policy_obs(sim, noise=False)
            critic = sac_features.critic_input(sim, norms)
            fresh = (sim.ep_step == 0).float().unsqueeze(-1)
            obs_in = vecnorm(policy_obs) if vecnorm is not None else policy_obs
            loc, _scale = actor(obs_in)
            action = loc
            if action_noise > 0:
                action = action + action_noise * torch.randn_like(action)
            out = sim.step(action, auto_reset=False)
            buffer.push(
                {
                    "obs": policy_obs,
                    "critic": critic,
                    "action": action,
                    "reward": out["r"].unsqueeze(-1),
                    "terminated": out["d"].unsqueeze(-1).float(),
                    "done": out["done"].unsqueeze(-1).float(),
                    "is_init": fresh,
                }
            )
            alive = alive & ~out["done"]
            steps += 1
    if steps < 1:
        raise RuntimeError("Rollout produced no transitions.")

    path = save_rollout(
        out_dir,
        buffer,
        task_cfg=dict(task_cfg),
        train_cfg={"actor_init": cfg.get("actor_init", "zeros")},
        extras={
            "source_ckpt": str(ckpt),
            "steps": steps,
            "episodes": n,
            "action_noise": float(action_noise),
            "deterministic": action_noise <= 0,
        },
    )
    total = buffer.num_samples
    print(
        f"wrote {path}: {steps} steps x {n} envs = {total} transitions "
        f"in {time.time() - t0:.1f}s",
        flush=True,
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True, help="SAC checkpoint.")
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
    args = parser.parse_args()
    collect(
        args.ckpt,
        args.out,
        device=args.device,
        num_envs=args.num_envs,
        action_noise=args.action_noise,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
