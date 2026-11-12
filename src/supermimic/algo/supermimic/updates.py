"""One learner implementation for eager execution and whole-update CUDA Graphs."""

import torch

from supermimic.algo.supermimic.wm.interface import WMBatch
from supermimic.env.mdp import CONTROL_DT
from supermimic.env.mdp.command import CommandState
from supermimic.env.rollout.policy import rollout
from supermimic.nn.utils import hold_out_net
from supermimic.robots.g1 import normalize_context

from .capture import CapturedUpdate, _optimizers, _tensor_state, optimize_capturable


class Updates:
    def __init__(self, algo):
        self.a = algo
        self.cfg = algo.cfg
        for group in (algo.opt_wm, algo.opt_pi, algo.opt_v):
            for opt in _optimizers(group):
                if isinstance(opt, (torch.optim.Adam, torch.optim.AdamW)) and self.cfg.cuda_graphs:
                    for params in opt.param_groups:
                        params["capturable"] = True
                        for p in params["params"]:
                            state = opt.state.get(p, {})
                            if isinstance(state.get("step"), torch.Tensor):
                                state["step"] = state["step"].to(p.device)
        self.wm_update = CapturedUpdate(
            self._wm_update,
            lambda: _tensor_state([algo.wm, algo.vecnorms], [algo.opt_wm]),
            self.cfg.cuda_graphs,
        )
        self.policy_update = CapturedUpdate(
            self._policy_update,
            lambda: _tensor_state(
                [algo.wm, algo.actor, algo.critic, algo.vecnorms], [algo.opt_pi, algo.opt_v]
            ),
            self.cfg.cuda_graphs,
        )

    def fit_world_model(self):
        cfg = self.cfg
        batches = [
            self.a.buffer.sample_windows(cfg.wm_batch, cfg.wm_window) for _ in range(cfg.wm_batches)
        ]
        return self.wm_update(
            *(torch.stack([b[k] for b in batches]) for k in ("s", "s1", "a", "xi"))
        )

    def _wm_update(self, s, s1, actions, xi):
        algo = self.a
        cfg = self.cfg
        rows = []
        for j in range(cfg.wm_batches):
            result = algo.wm.compute_loss(
                WMBatch(s[j, :, 0], actions[j], s1[j], normalize_context(xi[j, :, 0])),
                cfg.consistency_weight,
            )
            grad = optimize_capturable(result.loss, algo.wm, algo.opt_wm, cfg.grad_clip_norm)
            rows.append(
                {
                    **{
                        f"WMLearning/{k.removeprefix('wm_')}": v.detach()
                        for k, v in result.metrics.items()
                    },
                    "WMLearning/loss": result.loss.detach(),
                    "WMLearning/grad_norm": grad,
                }
            )
        return {k: torch.stack([r[k] for r in rows]).mean() for k in rows[0]}

    def fit_policy(self, iteration):
        algo = self.a
        cfg = self.cfg
        b = algo.buffer.sample_starts(cfg.pi_batch)
        n = cfg.horizon * cfg.pi_batch
        permutations = torch.stack(
            [torch.randperm(n, device=cfg.device) for _ in range(cfg.critic_epochs)]
        )
        scale = (
            algo.reward_norm.scale() if cfg.normalize_reward else torch.ones((), device=cfg.device)
        )
        ratio = min(1.0, iteration / max(cfg.pi_warmup, 1)) if cfg.pi_warmup else 1.0
        for group in algo.opt_pi.param_groups:
            group["lr"] = cfg.pi_lr * ratio
        return self.policy_update(
            b["s"],
            b["motion_id"],
            b["motion_t"],
            b["episode_step"],
            b["a_prev"],
            b["xi"],
            scale,
            permutations,
            allow_capture=iteration >= cfg.pi_warmup,
        )

    def _policy_update(
        self, full, motion_id, motion_t, episode_step, previous, xi, scale, permutations
    ):
        algo = self.a
        cfg = self.cfg
        with torch.no_grad():
            state = algo.imagined.reset(
                full, CommandState(motion_id, motion_t), previous, xi, episode_step
            )
        inputs = []
        rewards = []
        terms = []
        components = []
        returns = torch.zeros(cfg.pi_batch, device=cfg.device)
        alive = torch.ones_like(returns)
        with hold_out_net(algo.wm), hold_out_net(algo.critic):
            for t, record in enumerate(rollout(algo.imagined, state, algo.policy, cfg.horizon)):
                inputs.append(algo.critic_input(record.observation).detach())
                result = record.transition
                reward = result.reward / scale
                term = result.terminated
                returns = returns + cfg.gamma**t * alive * reward
                alive = alive * (1 - term)
                rewards.append(reward.detach())
                terms.append(term.detach())
                components.append(
                    torch.stack([result.terms[k].detach().mean() for k in algo.task.term_names])
                )
                state = result.continuation
            last = algo.imagined.observe(state)
            final_inputs = algo.critic_input(last)
            inputs.append(final_inputs.detach())
            value = algo.critic(final_inputs).mean(-1)
            pi_loss = -(returns + cfg.gamma**cfg.horizon * alive * value).mean()
            grad_pi = optimize_capturable(
                pi_loss, algo.actor, algo.opt_pi, cfg.grad_clip_norm, cfg.actor_grad_clip_value
            )
        x = torch.stack(inputs)
        reward = torch.stack(rewards)
        term = torch.stack(terms)
        with torch.no_grad():
            values = algo.critic(x.flatten(0, 1)).reshape(
                cfg.horizon + 1, cfg.pi_batch, cfg.critic_heads
            )
            target = values[-1]
            targets = []
            for t in reversed(range(cfg.horizon)):
                target = reward[t, :, None] + cfg.gamma * (1 - term[t, :, None]) * (
                    (1 - cfg.lam) * values[t + 1] + cfg.lam * target
                )
                targets.append(target)
            targets = torch.stack(targets[::-1]).flatten(0, 1)
        x = x[:-1].flatten(0, 1)
        losses = []
        norms = []
        for order in permutations:
            for idx in order.tensor_split(min(cfg.critic_minibatches, len(x))):
                loss = (algo.critic(x[idx]) - targets[idx]).square().mean()
                norms.append(optimize_capturable(loss, algo.critic, algo.opt_v, cfg.grad_clip_norm))
                losses.append(loss.detach())
        component_mean = torch.stack(components).mean(0) / CONTROL_DT
        return {
            "ActorCriticLearning/actor_loss": pi_loss.detach(),
            "ActorCriticLearning/critic_loss": torch.stack(losses).mean(),
            "ActorCriticLearning/actor_grad_norm": grad_pi,
            "ActorCriticLearning/critic_grad_norm": torch.stack(norms).mean(),
            "WMReward/total": reward.mean() * scale / CONTROL_DT,
            "WMReward/normalized_total": reward.mean() / CONTROL_DT,
            "WMReward/survival": alive.detach().mean(),
            **{
                f"WMReward/{k}": v
                for k, v in zip(algo.task.term_names, component_mean.unbind(), strict=True)
            },
        }
