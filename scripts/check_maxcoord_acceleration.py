"""Compare two eager updates with two compiled/CUDA-graph updates on fixed batches."""
import argparse
import copy
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from supermimic.maxcoord.accelerated import AcceleratedUpdates, _tensor_state
from supermimic.maxcoord.trainer import MaxCoordConfig, Trainer
from supermimic.supermimic.network import RunningNorm
from supermimic.utils.optim import build_optimizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--graphs", action="store_true")
    parser.add_argument("--full-size", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--wm", choices=("explicit", "latent"), default="latent")
    parser.add_argument("--actor", choices=("explicit", "latent"), default="explicit")
    args = parser.parse_args()
    torch.set_num_threads(8)
    cfg = MaxCoordConfig(num_envs=16, buffer_steps=64, wm_kind="latent", actor_kind="explicit",
        wm_hidden=(32, 32), actor_hidden=(32, 32), latent_dim=16, wm_batches=2, wm_batch=8,
        pi_batch=8, wm_window=3, horizon=4, critic_epochs=2, critic_minibatches=2,
        pi_warmup=0, swanlab_mode="disabled", optim="muon", normalize_reward=True)
    if args.full_size:
        cfg.num_envs=1024; cfg.buffer_steps=1000
        cfg.wm_hidden=cfg.actor_hidden=(512, 512, 512); cfg.latent_dim=256
        cfg.wm_batches=10; cfg.wm_batch=cfg.pi_batch=512
        cfg.wm_window=8; cfg.horizon=32; cfg.critic_epochs=5; cfg.critic_minibatches=4
    cfg.wm_kind, cfg.actor_kind = args.wm, args.actor
    cfg.resume = args.checkpoint
    trainer = Trainer(cfg, OmegaConf.create({"name": "lafan1"}), args.output)
    trainer.collect(32)
    trainer.buffer.prepare(cfg.wm_window, cfg.horizon, trainer.mdp.episode_last(trainer.buffer.f))
    window_sets = [[trainer.buffer.sample_windows(cfg.wm_batch, cfg.wm_window)
                    for _ in range(cfg.wm_batches)] for _ in range(2)]
    start_sets = [trainer.buffer.sample_starts(cfg.pi_batch, 1) for _ in range(2)]
    selection = 0
    cursor = 0
    def sample_windows(*_):
        nonlocal cursor
        windows = window_sets[selection]
        result = windows[cursor % len(windows)]; cursor += 1
        return result
    trainer.buffer.sample_windows = sample_windows
    trainer.buffer.sample_starts = lambda *_: start_sets[selection]
    cfg.compile_updates=args.compile; cfg.cuda_graphs=args.graphs
    fast = AcceleratedUpdates(trainer)
    # Initialize optimizer state and compile both grad modes before comparing.
    for _ in range(3):
        fast.fit_wm(); fast.fit_policy(200)
    torch.cuda.synchronize()
    refs = _tensor_state([trainer.wm, trainer.actor, trainer.critic],
                         [trainer.opt_wm, trainer.opt_pi, trainer.opt_v])
    initial = [x.detach().clone() for x in refs]
    models = {"wm": trainer.wm, "actor": trainer.actor, "critic": trainer.critic}
    initial_params = {name: torch.cat([p.detach().flatten() for p in model.parameters()]).clone()
                      for name, model in models.items()}
    # Use the original optimizer/norm modes for the eager reference, not the
    # capturable adaptations installed by AcceleratedUpdates above.
    fast_opts = (trainer.opt_wm, trainer.opt_pi, trainer.opt_v)
    reference_opts = (build_optimizer(trainer.wm, cfg.wm_lr, cfg.optim),
                      build_optimizer(trainer.actor, cfg.pi_lr, cfg.optim, adam_betas=(0.9, 0.95)),
                      build_optimizer(trainer.critic, cfg.v_lr, "adam"))
    for source, target in zip(fast_opts, reference_opts, strict=True):
        target.load_state_dict(copy.deepcopy(source.state_dict()))
        for optimizer in getattr(target, "optimizers", [target]):
            if isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
                for group in optimizer.param_groups:
                    group["capturable"] = False
                    if not group.get("fused", False):
                        for parameter in group["params"]:
                            state = optimizer.state.get(parameter, {})
                            if isinstance(state.get("step"), torch.Tensor):
                                state["step"] = state["step"].cpu()
    trainer.opt_wm, trainer.opt_pi, trainer.opt_v = reference_opts
    for model in (trainer.wm, trainer.actor):
        for module in model.modules():
            if isinstance(module, RunningNorm): module.no_sync = False
    expected = []
    for index in range(2):
        selection = index
        torch.manual_seed(1234 + index)
        stats = {**trainer.fit_world_model(), **trainer.train_actor_critic(200)}
        torch.cuda.synchronize()
        expected.append({"metrics": {k:v.detach().clone() for k,v in stats.items()},
                         "params": {name: torch.cat([p.detach().flatten() for p in model.parameters()]).clone()
                                    for name,model in models.items()},
                         "grads": {name: torch.cat([p.grad.detach().flatten() for p in model.parameters() if p.grad is not None]).clone()
                                   for name,model in models.items()}})
    with torch.no_grad():
        for ref,value in zip(refs,initial,strict=True):ref.copy_(value)
    trainer.opt_wm, trainer.opt_pi, trainer.opt_v = fast_opts
    for model in (trainer.wm, trainer.actor):
        for module in model.modules():
            if isinstance(module, RunningNorm): module.no_sync = True
    results=[]
    failures=[]
    for index in range(2):
        selection = index
        torch.manual_seed(1234 + index)
        stats={**fast.fit_wm(),**fast.fit_policy(200)}
        torch.cuda.synchronize()
        row={"update":index+1,"metrics":{},"parameters":{}}
        for key,value in stats.items():
            target=expected[index]["metrics"][key]
            row["metrics"][key]={"eager":float(target),"candidate":float(value),"abs_error":float((value-target).abs())}
            try:
                torch.testing.assert_close(value,target,rtol=3e-3,atol=3e-5,msg=key)
            except AssertionError:
                failures.append(f"update {index+1}: metric {key}")
        for name,model in models.items():
            value=torch.cat([p.detach().flatten() for p in model.parameters()])
            target=expected[index]["params"][name]
            update_ref=target-initial_params[name];update_new=value-initial_params[name]
            relative=(value-target).norm()/update_ref.norm().clamp_min(1e-12)
            cosine=torch.nn.functional.cosine_similarity(update_ref,update_new,dim=0)
            row["parameters"][name]={"max_abs_error":float((value-target).abs().max()),
                                     "relative_update_error":float(relative),"update_cosine":float(cosine)}
            grad=torch.cat([p.grad.detach().flatten() for p in model.parameters() if p.grad is not None])
            grad_ref=expected[index]["grads"][name]
            row["parameters"][name]["relative_gradient_error"]=float((grad-grad_ref).norm()/grad_ref.norm().clamp_min(1e-12))
            assert torch.isfinite(value).all()
            if relative >= 0.02 or cosine <= 0.999:
                failures.append(f"update {index+1}: parameter update {name}")
        results.append(row)
        (args.output/"equivalence.json").write_text(json.dumps(results,indent=2))
    if args.graphs:
        assert fast.wm_update.graph is not None and fast.policy_update.graph is not None
    print(json.dumps({"passed":not failures,"compiled":args.compile,"cuda_graphs":args.graphs,"results":results,"failures":failures},indent=2))
    assert not failures, failures


if __name__ == "__main__":
    main()
