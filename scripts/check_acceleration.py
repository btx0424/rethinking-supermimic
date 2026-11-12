"""Compare two eager and graph learner updates, refreshing replay inputs each time."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import argparse, copy, json
import torch
from supermimic.algo.supermimic.config import TrainConfig, WandbConfig
from supermimic.algo.supermimic.trainer import Trainer
from supermimic.algo.supermimic.updates import Updates
from supermimic.algo.supermimic.capture import _tensor_state, _optimizers
from supermimic.nn.optim import build_optimizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wm", choices=["explicit", "latent"], required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--full-size", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(8)
    cfg = TrainConfig(
        wm_kind=args.wm,
        run_dir=str(args.output),
        num_envs=16,
        buffer_steps=64,
        wm_hidden=(32, 32),
        actor_hidden=(32, 32),
        latent_dim=16,
        wm_batch=8,
        wm_batches=2,
        pi_batch=8,
        wm_window=3,
        horizon=4,
        critic_epochs=2,
        critic_minibatches=2,
        pi_warmup=0,
        wandb=WandbConfig(mode="disabled"),
    )
    if args.full_size:
        defaults = TrainConfig()
        cfg.num_envs = defaults.num_envs
        cfg.buffer_steps = 1000
        cfg.wm_hidden = cfg.actor_hidden = (512, 512, 512)
        cfg.latent_dim = 256
        cfg.wm_batch = defaults.wm_batch
        cfg.pi_batch = defaults.pi_batch
        cfg.wm_batches = 10
        cfg.wm_window = 8
        cfg.horizon = 32
        cfg.critic_epochs = 5
        cfg.critic_minibatches = 4
    algo = Trainer(cfg)
    algo.collect(32)
    algo.buffer.prepare(cfg.wm_window, cfg.horizon, algo.motion)
    windows = [
        [algo.buffer.sample_windows(cfg.wm_batch, cfg.wm_window) for _ in range(cfg.wm_batches)]
        for _ in range(2)
    ]
    starts = [algo.buffer.sample_starts(cfg.pi_batch) for _ in range(2)]
    selection = 0
    cursor = 0

    def sample_windows(*_):
        nonlocal cursor
        result = windows[selection][cursor % cfg.wm_batches]
        cursor += 1
        return result

    algo.buffer.sample_windows = sample_windows
    algo.buffer.sample_starts = lambda *_: starts[selection]
    fast = algo.updates
    for _ in range(3):
        fast.fit_world_model()
        fast.fit_policy(200)
    torch.cuda.synchronize()
    models = {"wm": algo.wm, "actor": algo.actor, "critic": algo.critic}
    refs = _tensor_state([*models.values(), algo.vecnorms], [algo.opt_wm, algo.opt_pi, algo.opt_v])
    initial = [x.detach().clone() for x in refs]
    initial_params = {
        k: torch.cat([p.detach().flatten() for p in m.parameters()]).clone()
        for k, m in models.items()
    }
    fast_opts = (algo.opt_wm, algo.opt_pi, algo.opt_v)
    reference_opts = (
        build_optimizer(algo.wm, cfg.wm_lr, cfg.optim),
        build_optimizer(algo.actor, cfg.pi_lr, cfg.optim, adam_betas=(0.9, 0.95)),
        build_optimizer(algo.critic, cfg.v_lr, "adam"),
    )
    for source, target in zip(fast_opts, reference_opts, strict=True):
        target.load_state_dict(copy.deepcopy(source.state_dict()))
        for opt in _optimizers(target):
            if isinstance(opt, (torch.optim.Adam, torch.optim.AdamW)):
                for group in opt.param_groups:
                    group["capturable"] = False
                    if not group.get("fused", False):
                        for parameter in group["params"]:
                            state = opt.state.get(parameter, {})
                            if isinstance(state.get("step"), torch.Tensor):
                                state["step"] = state["step"].cpu()
    algo.opt_wm, algo.opt_pi, algo.opt_v = reference_opts
    cfg.cuda_graphs = False
    eager = Updates(algo)
    cfg.cuda_graphs = True
    expected = []
    for j in range(2):
        selection = j
        torch.manual_seed(1234 + j)
        metrics = {**eager.fit_world_model(), **eager.fit_policy(200)}
        torch.cuda.synchronize()
        expected.append(
            {
                "metrics": {k: v.clone() for k, v in metrics.items()},
                "params": {
                    k: torch.cat([p.detach().flatten() for p in m.parameters()]).clone()
                    for k, m in models.items()
                },
            }
        )
    with torch.no_grad():
        for tensor, value in zip(refs, initial, strict=True):
            tensor.copy_(value)
    algo.opt_wm, algo.opt_pi, algo.opt_v = fast_opts
    results = []
    failures = []
    for j in range(2):
        selection = j
        torch.manual_seed(1234 + j)
        metrics = {**fast.fit_world_model(), **fast.fit_policy(200)}
        torch.cuda.synchronize()
        row = {"update": j + 1, "parameters": {}, "metrics": {}}
        for k, v in metrics.items():
            target = expected[j]["metrics"][k]
            row["metrics"][k] = {"eager": float(target), "graph": float(v)}
            try:
                torch.testing.assert_close(v, target, rtol=3e-3, atol=3e-5)
            except AssertionError:
                failures.append(f"{j + 1}: metric {k}")
        for k, m in models.items():
            value = torch.cat([p.detach().flatten() for p in m.parameters()])
            target = expected[j]["params"][k]
            reference_update = target - initial_params[k]
            update = value - initial_params[k]
            relative = (value - target).norm() / reference_update.norm().clamp_min(1e-12)
            cosine = torch.nn.functional.cosine_similarity(reference_update, update, dim=0)
            row["parameters"][k] = {
                "relative_update_error": float(relative),
                "cosine": float(cosine),
                "max_abs_error": float((value - target).abs().max()),
            }
            if not torch.isfinite(value).all() or relative >= 0.02 or cosine <= 0.999:
                failures.append(f"{j + 1}: parameter {k}")
        results.append(row)
    assert fast.wm_update.graph is not None and fast.policy_update.graph is not None
    (args.output / "equivalence.json").write_text(
        json.dumps({"passed": not failures, "failures": failures, "results": results}, indent=2)
    )
    print(
        json.dumps(
            {
                "passed": not failures,
                "failures": failures,
                "parameters": [r["parameters"] for r in results],
            }
        )
    )
    assert not failures, failures


if __name__ == "__main__":
    main()
