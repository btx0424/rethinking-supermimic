"""Rollout archives for RLPD-style prior mixing.

A prior archive is a replay ring collected by a trained policy and stored
chronologically, so a later SAC run can draw a fraction of every update batch
from it. The on-disk layout is exactly the fields of
:class:`~supermimic.sac.replay_buffer.ReplayBuffer`, which keeps loading trivial
and means prior rows go through the same `sample` / n-step / bootstrap path as
online rows.

Everything is stored **raw**: the policy observation before `ObsVecNorm`, the
critic input before `InputNorms`, and the unnormalized reward. Normalization is
applied at train time with the *training run's* running statistics, which is what
makes prior rows comparable to online rows.
"""

from __future__ import annotations

from pathlib import Path

import torch

from supermimic.sac.replay_buffer import ReplayBuffer
from supermimic.utils.episode import check_transition_flags

ROLLOUT_FORMAT_VERSION = 1
ARCHIVE_NAME = "rollout.pt"
# Tensor fields of ``ReplayBuffer``, in the order they are written.
FIELDS = (
    "obs",
    "critic",
    "action",
    "reward",
    "terminated",
    "truncated",
    "done",
    "is_init",
)


def replay_mix_sizes(
    batch_size: int, prior_ratio: float, *, use_prior: bool
) -> tuple[int, int]:
    """Split ``batch_size`` into ``(online, prior)`` counts.

    ``prior_ratio`` is the share drawn from the archive. The two counts always sum
    to ``batch_size``, so the update sees the batch size the config asks for.
    """
    if not use_prior or prior_ratio <= 0.0:
        return batch_size, 0
    if prior_ratio >= 1.0:
        return 0, batch_size
    prior = int(round(batch_size * float(prior_ratio)))
    prior = max(1, min(batch_size - 1, prior))
    return batch_size - prior, prior


def save_rollout(
    root: Path | str,
    buffer: ReplayBuffer,
    *,
    task_cfg: dict | None = None,
    train_cfg: dict | None = None,
    extras: dict | None = None,
) -> Path:
    """Write ``buffer`` as a chronological archive under ``root``.

    The ring is unrolled so row 0 is the oldest transition, which lets the loader
    rebuild it with ``ptr = 0`` and no wrap-around bookkeeping.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    steps = int(buffer._size)
    if steps < 1:
        raise RuntimeError("Refusing to save an empty rollout.")
    full = buffer._size >= buffer.max_size
    head = int(buffer._ptr) if full else 0

    fields = {}
    for name in FIELDS:
        ring = getattr(buffer, name)
        if full and head:
            ring = torch.roll(ring, shifts=-head, dims=0)
        fields[name] = ring[:steps].detach().cpu()

    payload = {
        "format_version": ROLLOUT_FORMAT_VERSION,
        "fields": fields,
        "steps": steps,
        "num_envs": int(buffer.num_envs),
        "obs_dim": int(buffer.obs_dim),
        "critic_dim": int(buffer.critic_dim),
        "act_dim": int(buffer.act_dim),
        "task_cfg": task_cfg,
        "train_cfg": train_cfg,
        **(extras or {}),
    }
    path = root / ARCHIVE_NAME
    torch.save(payload, path)
    return path


def load_rollout(root: Path | str, *, map_location="cpu") -> dict:
    """Read an archive written by :func:`save_rollout`."""
    root = Path(root)
    path = root if root.suffix == ".pt" else root / ARCHIVE_NAME
    if not path.is_file():
        raise FileNotFoundError(f"No rollout archive at {path}.")
    payload = torch.load(path, map_location=map_location, weights_only=False)
    version = payload.get("format_version")
    if version != ROLLOUT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported rollout format_version={version!r}; "
            f"expected {ROLLOUT_FORMAT_VERSION}."
        )
    fields = payload.get("fields")
    if not isinstance(fields, dict):
        raise ValueError(f"{path} has no 'fields' dict.")
    if "truncated" not in fields and "done" in fields and "terminated" in fields:
        # Archives written before the terminated/truncated split stored only
        # ``done``; recover the missing flag from the invariant:
        # truncated = done & ~terminated.
        fields["truncated"] = fields["done"].bool() & ~fields["terminated"].bool()
    missing = [name for name in FIELDS if name not in fields]
    if missing:
        raise ValueError(f"{path} is missing fields {missing}.")
    check_transition_flags(fields["terminated"], fields["truncated"], fields["done"], what=str(path))
    return payload


def build_prior(
    root: Path | str,
    device: torch.device | str,
    *,
    obs_dim: int | None = None,
    critic_dim: int | None = None,
    act_dim: int | None = None,
) -> tuple[ReplayBuffer, dict]:
    """Load an archive into a read-only :class:`ReplayBuffer`.

    The optional ``*_dim`` arguments assert the archive matches the run it is
    being mixed into; a mismatch would otherwise surface as a shape error deep
    inside the critic.
    """
    payload = load_rollout(root)
    for name, expected in (
        ("obs_dim", obs_dim),
        ("critic_dim", critic_dim),
        ("act_dim", act_dim),
    ):
        if expected is not None and int(payload[name]) != int(expected):
            raise ValueError(
                f"Prior archive {name}={payload[name]} does not match this run "
                f"({expected}). Collect the prior with the same task and "
                "observation configuration."
            )
    steps = int(payload["steps"])
    buffer = ReplayBuffer(
        max_size=steps,
        num_envs=int(payload["num_envs"]),
        obs_dim=int(payload["obs_dim"]),
        critic_dim=int(payload["critic_dim"]),
        act_dim=int(payload["act_dim"]),
        device=device,
    )
    for name in FIELDS:
        getattr(buffer, name).copy_(payload["fields"][name].to(buffer.device))
    # Chronological archive: the ring starts empty and is exactly full.
    buffer._ptr = 0
    buffer._size = steps
    return buffer, payload


def split_prior_fraction(buffer: ReplayBuffer, fraction: float, *, seed: int = 0) -> None:
    """Keep a random subset of the archive's env columns, in place.

    The archive is ``[steps, num_envs, ...]``; taking a column subset keeps whole
    trajectories, so n-step windows stay valid. Useful to shrink a very large
    archive or to build a smaller ablation from the same file.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"prior_fraction must be in (0, 1], got {fraction}.")
    keep = max(1, int(round(buffer.num_envs * fraction)))
    if keep >= buffer.num_envs:
        return
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    cols = torch.randperm(buffer.num_envs, generator=gen)[:keep].to(buffer.device)
    for name in FIELDS:
        ring = getattr(buffer, name)
        setattr(buffer, name, ring[:, cols].contiguous())
    buffer.num_envs = keep


def mix_batches(
    online: dict[str, torch.Tensor] | None,
    prior: dict[str, torch.Tensor] | None,
    dim: int,
) -> dict[str, torch.Tensor] | None:
    """Concatenate an online and a prior batch along the batch dimension."""
    if prior is None:
        return online
    if online is None:
        return prior
    return {key: torch.cat([online[key], prior[key]], dim=dim) for key in online}
