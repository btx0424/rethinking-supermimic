"""SuperMimic with the critic fit immediately before the actor step.

The base trainer differentiates the current policy against the value from the
previous iteration, and that value was fit under the previous world model. This
variant, after the world-model update:

1. imagines with the current policy, without an actor step;
2. fits the critic on that rollout;
3. imagines again and takes the actor step, so the terminal value and its
   state Jacobian are the value just fit to this policy and this world model.

The base :class:`~supermimic.supermimic.supermimic.Trainer` is unchanged.
"""

from __future__ import annotations

import torch
from omegaconf import DictConfig

from supermimic.supermimic.supermimic import (
    Trainer,
    hold_out_net,
    lr_at,
    pi_lr_at,
    standalone_config,
    timed,
)

VFirstTrainConfig = standalone_config("VFirstTrainConfig", "supermimic_vfirst")


class VFirstTrainer(Trainer):
    """SuperMimic trainer that evaluates V, then improves the policy."""

    def _step(self, iteration: int) -> None:
        cfg = self.cfg
        schedules = (
            (self.opt_wm, lr_at(iteration, cfg.wm_lr, cfg.lr_decay_iters)),
            (self.opt_pi, pi_lr_at(iteration, cfg.pi_lr, cfg.lr_decay_iters, cfg.pi_warmup)),
            (self.opt_v, lr_at(iteration, cfg.v_lr, cfg.lr_decay_iters)),
            (self.opt_disc, lr_at(iteration, cfg.wm_lr, cfg.lr_decay_iters)),
        )
        lr = schedules[0][1]
        pi_lr = schedules[1][1]
        for opt, group_lr in schedules:
            for group in opt.param_groups:
                group["lr"] = group_lr
        self.step_acc += 1.0 / cfg.utd
        n_steps = int(self.step_acc)
        self.step_acc -= n_steps

        with timed(self.device):
            self.collect(n_steps)
        self.timing["collect"].append(timed.last)

        if self.buffer.filled < cfg.wm_window:
            for key in ("wm", "imagine", "policy", "critic"):
                self.timing[key].append(0.0)
            return

        log_now = (iteration + 1) % cfg.log_every == 0
        if cfg.critic_source not in ("imagined", "sim"):
            raise ValueError(
                f"critic_source={cfg.critic_source!r}. Expected 'imagined' or 'sim'."
            )
        # Policy starts must leave enough reference frames for the actor rollout.
        # An imagined critic of a different length needs the longer of the two.
        start_horizon = self.horizon_pi
        if cfg.critic_source == "imagined":
            start_horizon = max(start_horizon, self.horizon_v)
        clip_last = self.sim.mdp.episode_last(self.buffer.f)
        self.buffer.prepare(
            cfg.wm_window, start_horizon=start_horizon, clip_last=clip_last
        )

        with timed(self.device):
            wm_loss, err8, gn_wm, disc_stats = self.update_world_model(log_now=log_now)
        self.timing["wm"].append(timed.last)

        # Value pass before the actor step. Imagined targets use horizon_v, which
        # may differ from the actor's horizon_pi; sim targets are the latest real
        # transitions instead of a world-model rollout.
        diagnose_reward = log_now
        policy_wm = self.policy_wm()
        with hold_out_net(policy_wm), hold_out_net(self.critic):
            start = self.sample_policy_starts()
            if cfg.critic_source == "sim":
                critic_batch = self.sim_critic_batch()
            else:
                with timed(self.device):
                    with torch.no_grad():
                        _, critic_batch, _, _, _, _ = self.imagine(start, horizon=self.horizon_v)
                self.timing["imagine"].append(timed.last)

        with timed(self.device):
            if critic_batch is None:
                v_loss = start["s"].new_zeros(())
                gn_v = 0.0
                value_mean = v_loss
            else:
                v_loss, gn_v, value_mean = self.update_critic(critic_batch)
        self.timing["critic"].append(timed.last)

        # Actor pass at horizon_pi. A new forward is required: the graph built
        # above ran the previous value, and a critic step does not rewrite it.
        with hold_out_net(policy_wm), hold_out_net(self.critic):
            pi_loss, _, imag_score, survival, imag_terms, gn_pi = self.improve_policy(
                start, diagnose_reward=diagnose_reward, iteration=iteration
            )

        self.agg["wm"].append(wm_loss.detach())
        self._record_wm_stats(disc_stats)
        self.agg["err"].append(err8.detach())
        self.agg["pi"].append(pi_loss.detach())
        self.agg["v"].append(v_loss.detach() if isinstance(v_loss, torch.Tensor) else v_loss)
        self.agg["value_mean"].append(value_mean.detach() if isinstance(value_mean, torch.Tensor) else value_mean)
        self.agg["score"].append(imag_score.detach() if isinstance(imag_score, torch.Tensor) else imag_score)
        self.agg["surv"].append(survival.detach() if isinstance(survival, torch.Tensor) else survival)
        self.agg["gn_wm"].append(gn_wm)
        self.agg["gn_pi"].append(gn_pi)
        self.agg["gn_v"].append(gn_v)
        for name, value in imag_terms.items():
            self.agg["imag_terms"][name].append(value)

        if (iteration + 1) % cfg.log_every == 0:
            self._log(iteration + 1, lr, pi_lr)
        if (iteration + 1) % cfg.ckpt_every == 0 or iteration + 1 == cfg.iterations:
            self.save_checkpoint(iteration + 1)


def run(cfg: VFirstTrainConfig, task_cfg: DictConfig, log_dir) -> None:
    VFirstTrainer(cfg, task_cfg, log_dir).fit()
