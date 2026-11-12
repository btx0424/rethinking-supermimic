"""Six real training iterations, both rollout contracts, raw replay and checkpoint checks."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import argparse
import csv
import json

import torch

from supermimic.algo.supermimic.checkpoint import FORMAT, load_policy
from supermimic.algo.supermimic.config import TrainConfig, WandbConfig
from supermimic.algo.supermimic.trainer import Trainer
from supermimic.env.rollout.policy import policy_step


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wm", choices=["explicit", "latent"], required=True)
    p.add_argument("--output", required=True)
    p.add_argument("overrides", nargs="*", help="TrainConfig key=value overrides")
    args = p.parse_args()
    torch.set_num_threads(8)
    cfg = TrainConfig(
        wm_kind=args.wm,
        run_dir=args.output,
        num_envs=16,
        buffer_steps=96,
        wm_hidden=(32, 32),
        actor_hidden=(32, 32),
        latent_dim=16,
        wm_batches=1,
        wm_batch=16,
        pi_batch=8,
        iterations=6,
        critic_epochs=1,
        critic_minibatches=2,
        pi_warmup=0,
        log_every=3,
        ckpt_every=6,
        wandb=WandbConfig(mode="disabled"),
    )
    from omegaconf import OmegaConf

    cfg = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(cfg), OmegaConf.from_dotlist(args.overrides))
    )
    algo = Trainer(cfg)
    calls = []
    add = algo.buffer.add
    collect = algo.collect

    def checked_add(record):
        i = algo.buffer.ptr
        raw = record.state.full.clone()
        successor = record.transition.state.full.clone()
        add(record)
        torch.testing.assert_close(algo.buffer.motion_id[i], record.state.command.motion_id)
        torch.testing.assert_close(algo.buffer.motion_t[i], record.state.command.motion_t)
        torch.testing.assert_close(algo.buffer.episode_step[i], record.state.episode_step)
        torch.testing.assert_close(algo.buffer.xi[i], record.state.xi)
        torch.testing.assert_close(algo.buffer.s[i], raw, rtol=0, atol=0)
        torch.testing.assert_close(algo.buffer.s1[i], successor, rtol=0, atol=0)

    algo.buffer.add = checked_add

    def checked_collect(steps):
        before = algo.real.lifetime.clone()
        counts = {k: v.count.clone() for k, v in algo.vecnorms.items()}
        result = collect(steps)
        calls.append(steps)
        torch.testing.assert_close(algo.real.lifetime - before, torch.full_like(before, steps))
        for k, norm in algo.vecnorms.items():
            expected = counts[k]
            for _ in range(steps):
                expected.mul_(norm.decay).add_(cfg.num_envs)
            torch.testing.assert_close(norm.count, expected, rtol=0, atol=0)
        return result

    algo.collect = checked_collect

    def frozen(fn):
        def checked(*a, **kw):
            before = {k: v.clone() for k, v in algo.vecnorms.state_dict().items()}
            result = fn(*a, **kw)
            for k, v in algo.vecnorms.state_dict().items():
                torch.testing.assert_close(v, before[k], rtol=0, atol=0)
            return result

        return checked

    algo.updates.fit_world_model = frozen(algo.updates.fit_world_model)
    algo.updates.fit_policy = frozen(algo.updates.fit_policy)
    algo.fit()
    assert calls == [max(2 * cfg.wm_window, 32)] + [cfg.collect_steps] * cfg.iterations
    assert algo.updates.wm_update.graph is not None and algo.updates.policy_update.graph is not None
    rows = list(csv.DictReader((algo.log_dir / "train_log.csv").open()))
    for row in rows:
        for prefix in ["SimReward", "WMReward"]:
            assert (
                abs(
                    sum(float(row[f"{prefix}/{k}"]) for k in algo.task.term_names)
                    - float(row[f"{prefix}/total"])
                )
                < 1e-5
            )
    perf = list(csv.DictReader((algo.log_dir / "performance.csv").open()))
    assert [int(r["iteration"]) for r in perf] == list(range(1, cfg.iterations + 1))
    assert all(float(r["collect_time"]) > 0 and float(r["train_time"]) > 0 for r in perf)
    policy, data = load_policy(algo.log_dir / "last.pt", cfg.device)
    assert data["format"] == FORMAT
    with torch.no_grad():
        obs = algo.real.observe(algo.real.read_state())
        torch.testing.assert_close(policy(obs), algo.policy(obs), rtol=0, atol=0)
        before = algo.real.read_state()
        after = algo.real.reset(
            torch.tensor([0], device=cfg.device), torch.tensor([0], device=cfg.device)
        )
        torch.testing.assert_close(before.full[1:], after.full[1:], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(before.episode_step[1:], after.episode_step[1:])
        torch.testing.assert_close(before.command.motion_id[1:], after.command.motion_id[1:])
        torch.testing.assert_close(before.command.motion_t[1:], after.command.motion_t[1:])
        assert after.episode_step[0] == 0
        algo.real.max_steps = 1
        start = algo.real.read_state()
        record = policy_step(algo.real, start, algo.policy)
        assert record.transition.truncated.all()
        torch.testing.assert_close(record.transition.state.episode_step, start.episode_step + 1)
        assert (record.transition.continuation.episode_step == 0).all()
        assert not torch.equal(record.transition.state.full, record.transition.continuation.full)
        torch.testing.assert_close(
            algo.task.command.frame(record.transition.state.command),
            algo.task.command.frame(algo.task.command.advance(start.command)),
        )
    result = {
        "passed": True,
        "wm_kind": args.wm,
        "collect_calls": calls,
        "body_names": algo.motion.body_names,
        "reward_terms": algo.task.term_names,
        "raw_full_body_replay": True,
        "collect_only_vecnorm": True,
        "timing_every_iteration": True,
        "partial_reset": True,
        "checkpoint_policy_exact": True,
    }
    (algo.log_dir / "checks.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
