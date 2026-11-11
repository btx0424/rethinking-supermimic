"""Numerical/gradient checks. Optional --reference-root compares upstream in isolation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import torch


def fixture(dtype=torch.float32):
    torch.manual_seed(17)
    p = torch.randn(3, 14, 3, dtype=dtype)
    q = torch.nn.functional.normalize(torch.randn(3, 14, 4, dtype=dtype), dim=-1)
    v, w = torch.randn_like(p), torch.randn_like(p)
    full = torch.cat([x.flatten(-2) for x in (p, q, v, w)], -1)
    delta = torch.randn(3, 32, 84, dtype=dtype) * 0.1
    return full, delta


def reference_output(path):
    from supermimic.representations import MaxCoordRepresentation
    full, delta = fixture()
    rep = MaxCoordRepresentation(14, torch.zeros(29))
    rep.substep_coeff = 0.625
    f, aux = rep.represent(full), rep.extract_aux_state(full)
    states = [(aux, f)]
    for d in delta.unbind(1):
        aux, f = rep.apply_prediction(aux, f, d, 0.02)
        states.append((aux, f))
    torch.save(states, path)


def check(reference_root=None):
    from supermimic.maxcoord.representation import MaxCoordRepresentation
    from supermimic.maxcoord.state import decompose_maxcoord_state
    from supermimic.maxcoord.models import build_models, actor_features
    from supermimic.maxcoord.loss import rollout_loss, state_loss, NOMINAL_ERRORS
    from supermimic.maxcoord.mdp import goal
    from supermimic.utils.math import quat_to_mat

    torch.set_num_threads(2)
    full, delta = fixture()
    rep = MaxCoordRepresentation(14)
    aux, f = rep.extract_aux_state(full), rep.represent(full)
    rebuilt = rep.reconstruct_full_state(aux, f)
    expected, actual = decompose_maxcoord_state(full, 14), decompose_maxcoord_state(rebuilt, 14)
    roundtrip = 0.0
    for i, (a, b) in enumerate(zip(expected, actual)):
        if i == 1:
            a, b = quat_to_mat(a), quat_to_mat(b)
        torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-6)
        roundtrip = max(roundtrip, float((a - b).abs().max()))
    report = {"roundtrip_max_abs": roundtrip}
    if reference_root:
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "reference.pt"
            env = {**os.environ, "PYTHONPATH": str(Path(reference_root).resolve() / "src")}
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--reference-worker", str(output)],
                           env=env, cwd=td, check=True)
            ref = torch.load(output, weights_only=True)
        state = (aux, f)
        maximum = 0.0
        for i in range(33):
            for ours, theirs in zip(state, ref[i]):
                torch.testing.assert_close(ours, theirs, rtol=2e-5, atol=1e-5)
                maximum = max(maximum, float((ours - theirs).abs().max()))
            if i < 32:
                state = rep.apply_prediction(*state, delta[:, i], 0.02)
        report["hybrid_32_step_max_abs"] = maximum

    # Identity/zero angular velocity must have finite backward, not sqrt(0) NaNs.
    identity = full.clone()
    identity[:, 42:98] = torch.tensor([1., 0., 0., 0.]).repeat(14)
    identity[:, 98:] = 0
    ia, iff = rep.extract_aux_state(identity), rep.represent(identity)
    zero = torch.zeros(3, 84, requires_grad=True)
    na, nf = rep.apply_prediction(ia, iff, zero, 0.02)
    (na.square().sum() + nf.square().sum()).backward()
    assert torch.isfinite(zero.grad).all() and zero.grad.abs().sum() > 0

    # Confirm Bryson weighting is over channels, not over six equally weighted groups.
    changed = aux.clone()
    changed[:, 0] += 0.01
    weighted, groups = state_loss(rep, f, changed, f, aux)
    widths = {k: v.shape[-1] for k, v in rep.loss_blocks(f, aux).items()}
    expected_loss = sum(groups[k] * widths[k] / NOMINAL_ERRORS[k]**2 for k in widths) / sum(widths.values())
    torch.testing.assert_close(weighted, expected_loss)
    torch.testing.assert_close(weighted, weighted.new_tensor(1 / sum(widths.values())))

    for wm_kind, actor_kind in (("explicit", "explicit"), ("latent", "latent"), ("latent", "explicit")):
        cfg = SimpleNamespace(wm_kind=wm_kind, actor_kind=actor_kind, latent_dim=16,
                              wm_hidden=(32, 32), actor_hidden=(32, 32), goal_frames=10, wm_xi=True, simnorm_dim=8)
        wm, actor = build_models(cfg, 14, 29)
        xi_n = torch.randn(3, 33)
        actions = torch.randn(3, 8, 29, requires_grad=True)
        targets = full[:, None].expand(-1, 8, -1)
        target_modes = []
        hook = None
        if wm_kind == "latent":
            hook = wm.encoder.register_forward_hook(lambda *_: target_modes.append(torch.is_grad_enabled()))
        loss, metrics = rollout_loss(wm, full, actions, targets, xi_n=xi_n)
        loss.backward()
        if hook:
            hook.remove()
            assert target_modes == [False, True], target_modes
            # Every latent component is supervised, including the final dynamics step.
            for module in (wm.encoder, wm.dynamics, wm.decoder):
                grads = [p.grad for p in module.parameters() if p.grad is not None]
                assert grads and all(torch.isfinite(g).all() for g in grads)
                assert sum(g.abs().sum() for g in grads) > 0
            seen = []
            h = wm.decoder.register_forward_pre_hook(lambda m, args: seen.append(args[0].detach().clone()))
            st = wm.initialize(full, xi_n)
            wm.step(st, actions[:, 0])
            h.remove()
            torch.testing.assert_close(seen[0][..., :16], st.z)
        first = wm.step(wm.initialize(full, xi_n), actions[:, 0])
        altered = wm.step(wm.initialize(full, xi_n + 1), actions[:, 0])
        assert not torch.allclose(first.f, altered.f), "WM ignored xi"
        if wm_kind == "latent":
            assert not torch.allclose(first.z, altered.z), "Dynamics ignored xi"
            for code in (wm.initialize(full, xi_n).z, first.z):
                torch.testing.assert_close(code.reshape(3, -1, 8).sum(-1), torch.ones(3, 2))
        assert torch.isfinite(actions.grad).all() and actions.grad.abs().sum() > 0
        for horizon in (8, 32):
            wm.zero_grad(set_to_none=True)
            actor.zero_grad(set_to_none=True)
            wm.requires_grad_(False)
            st = wm.initialize(identity, xi_n)
            for _ in range(horizon):
                action = actor(actor_features(st, actor_kind), torch.zeros(3, 380))
                st = wm.step(st, action)
            objective = st.aux.square().mean() + st.f.square().mean()
            objective.backward()
            grads = [p.grad for p in actor.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            assert sum(g.abs().sum() for g in grads) > 0
            assert all(p.grad is None for p in wm.parameters())
            wm.requires_grad_(True)
        # Architecture and normalizers round-trip in checkpoints.
        wm2, actor2 = build_models(cfg, 14, 29)
        wm2.load_state_dict(wm.state_dict())
        actor2.load_state_dict(actor.state_dict())
        for x, y in zip(wm.initialize(full, xi_n), wm2.initialize(full, xi_n)):
            if x is not None:
                torch.testing.assert_close(x, y)
        report[f"{wm_kind}/{actor_kind}"] = {"loss8": float(loss.detach()), "gradient8_32": "pass"}

    # The goal sees reference displacement, robot-relative orientation, and no robot XY.
    motion = SimpleNamespace(root_p=torch.randn(20, 3), root_quat=torch.tensor([1., 0., 0., 0.]).repeat(20, 1),
                             q=torch.randn(20, 29),
                             future_frames=lambda frames, k: (frames[:, None] + torch.arange(1, k + 1)).clamp(max=19))
    frames = torch.tensor([0, 19])
    g = goal(motion, frames, motion.root_quat[frames])
    assert g.shape == (2, 380)
    torch.testing.assert_close(g[0, :3], motion.root_p[1] - motion.root_p[0])
    torch.testing.assert_close(g[1].reshape(10, 38)[:, :3], torch.zeros(10, 3))
    shifted = SimpleNamespace(**vars(motion))
    shifted.root_p = motion.root_p + torch.tensor([3., -2., 1.])
    torch.testing.assert_close(g, goal(shifted, frames, motion.root_quat[frames]))
    # A 90-degree reference pitch maps world +X displacement to local +Z.
    pitched = SimpleNamespace(**vars(motion))
    pitched.root_p = torch.zeros(20, 3); pitched.root_p[1:, 0] = 1
    pitched.root_quat = torch.tensor([2**-0.5, 0., 2**-0.5, 0.]).repeat(20, 1)
    pitch_goal = goal(pitched, torch.tensor([0]), pitched.root_quat[:1])
    torch.testing.assert_close(pitch_goal[0, :3], torch.tensor([0., 0., 1.]), atol=1e-6, rtol=1e-6)
    report["goal_and_loss_contract"] = "pass"
    report["xi_simnorm_full_orientation"] = "pass"
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--reference-worker", type=Path)
    args = parser.parse_args()
    if args.reference_worker:
        reference_output(args.reference_worker)
    else:
        check(args.reference_root)
