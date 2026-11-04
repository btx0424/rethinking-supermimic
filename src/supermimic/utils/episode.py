"""The episode-end convention shared by every rollout in the repo.

A step produces two independent signals and their union::

    terminated  the MDP's own absorbing failure (a fall). Never bootstrapped.
    truncated   the episode stopped for a reason outside the MDP: a time limit
                or the end of the reference clip. Bootstrapped as usual.
    done        exactly ``terminated | truncated``, i.e. "no successor".

``done`` exists only to mark the last valid step of a trajectory, for storage and
window bookkeeping. Anything that means *the policy failed* must read
``terminated``; anything that decides *whether the return beyond this step is
real* must read ``terminated`` too (a truncated step keeps its bootstrap).

The causes are reported independently and may co-occur: a fall on the very last
frame of a clip is both terminated and truncated. Only the union invariant is
guaranteed, never disjointness. Where they conflict, ``terminated`` wins, because
the bootstrap discount is zeroed by termination alone.
"""

from __future__ import annotations

import torch


def done_from(terminated: torch.Tensor, truncated: torch.Tensor) -> torch.Tensor:
    """``done = terminated | truncated``, the only sanctioned way to build it."""
    terminated = terminated.bool()
    return terminated | truncated.bool()


def check_transition_flags(terminated, truncated, done, *, what: str = "transition") -> None:
    """Assert the convention on one batch of flags.

    Flags may be bool or any numeric dtype holding exactly ``0``/``1`` (the
    replay ring stores them as floats so they can be concatenated with the rest
    of a batch), but they may not be probabilities.
    """
    flags = {"terminated": terminated, "truncated": truncated, "done": done}
    shapes = set()
    for name, flag in flags.items():
        if not isinstance(flag, torch.Tensor):
            raise TypeError(f"{what}: {name} must be a tensor, got {type(flag).__name__}.")
        if flag.dtype.is_floating_point:
            # Storage dtype: must still be exactly boolean-valued.
            if not torch.equal(flag, flag.bool()):
                raise ValueError(
                    f"{what}: {name} has non-boolean values; flags are indicators, "
                    "not termination probabilities."
                )
        shapes.add(tuple(flag.shape))
    if len(shapes) != 1:
        raise ValueError(f"{what}: flag shapes differ ({sorted(shapes)}).")
    term, trunc, dn = terminated.bool(), truncated.bool(), done.bool()
    if not torch.equal(dn, term | trunc):
        raise ValueError(f"{what}: done != terminated | truncated.")
