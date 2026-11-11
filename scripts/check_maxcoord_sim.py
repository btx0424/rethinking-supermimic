"""Measure exact-twist-delta integration against real G1 simulator transitions."""
import argparse
import json

import torch
from omegaconf import OmegaConf

from supermimic.eval import _build_sim
from supermimic.maxcoord.mdp import configure_sim, full_from_flat
from supermimic.maxcoord.representation import MaxCoordRepresentation
from supermimic.maxcoord.state import build_maxcoord_full_state
from supermimic.maxcoord.loss import state_loss
from supermimic.mdp import flatten_state
from supermimic.utils.math import quat_apply_inverse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", default=32, type=int)
    args = parser.parse_args()
    torch.manual_seed(0)
    sim = _build_sim({"task_cfg": {"name": "lafan1"}, "anchor": "pelvis", "fk": "torch"}, 16, args.device)
    configure_sim(sim)
    sim.reset(torch.arange(16, device=args.device), sim.motion.clip_start[:16])
    rep = MaxCoordRepresentation(14)
    groups = {}
    fk_max = 0.0
    terminations = 0
    with torch.no_grad():
        for _ in range(args.steps):
            bodies = tuple(x.clone() for x in sim.tracked_bodies())
            initial = build_maxcoord_full_state(*bodies)
            fk = full_from_flat(sim.mdp, flatten_state(*sim.state()))
            # Quaternion sign is arbitrary, so compare invariant representation.
            fk_error = (rep.represent(fk) - rep.represent(initial)).abs().max()
            fk_max = max(fk_max, float(fk_error))
            out = sim.step(0.1 * torch.randn(16, 29, device=args.device), auto_reset=False)
            successor = tuple(x.clone() for x in sim.tracked_bodies())
            true = build_maxcoord_full_state(*successor)
            root_q = bodies[1][:, :1].expand_as(bodies[1])
            delta = torch.cat([quat_apply_inverse(root_q, successor[i] - bodies[i]).flatten(-2)
                               for i in (2, 3)], -1)
            aux, f = rep.apply_prediction(rep.extract_aux_state(initial), rep.represent(initial), delta, 0.02)
            _, errors = state_loss(rep, f, aux, rep.represent(true), rep.extract_aux_state(true))
            for key, value in errors.items():
                groups.setdefault(key, []).append(value)
            terminations += int(out["terminated"].sum())
            if out["done"].any():
                sim.reset(out["done"].nonzero().flatten())
    assert fk_max < 1e-3, f"Replay FK disagrees with simulator: {fk_max}"
    print(json.dumps({"steps": args.steps, "envs": 16, "substep_coeff": 0.625,
                      "fk_feature_max_abs": fk_max, "terminated_transitions": terminations,
                      "integration_rmse": {k: float(torch.stack(v).mean().sqrt()) for k, v in groups.items()}}, indent=2))


if __name__ == "__main__":
    main()
