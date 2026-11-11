"""Optional compiled learning kernels and CUDA-graph learner updates.

Simulation, replay sampling and random minibatch permutations remain outside
capture. Network widths, losses, normalization order and update counts are the
same as the eager trainer. The original trainer remains the default.
"""
from __future__ import annotations

import torch

from supermimic.supermimic.network import RunningNorm, critic_forward
from supermimic.utils.nn import hold_out_net
from supermimic.motion import CONTROL_DT
from supermimic.mdp import normalize_xi
from .loss import LATENT_METRICS
from .mdp import full_from_flat, goal, reward_state
from .models import actor_features


def _optimizers(optimizer):
    return getattr(optimizer, "optimizers", [optimizer])


def _tensor_state(modules, optimizers):
    values = []
    for module in modules:
        values.extend(module.parameters())
        values.extend(module.buffers())
    for group in optimizers:
        for optimizer in _optimizers(group):
            for state in optimizer.state.values():
                values.extend(x for x in state.values() if isinstance(x, torch.Tensor))
    return values


class CapturedUpdate:
    """Warm up real updates, capture once, then refresh inputs and replay.

    Separate graph pools keep WM and policy outputs independent. Capture itself
    records operations; the first replay is the one update for that call.
    """
    def __init__(self, fn, state, enabled, warmup=3):
        self.fn, self.state, self.enabled = fn, state, enabled
        self.warmup = warmup
        self.inputs = self.graph = self.outputs = None
        self.identity = None
        self.stream = torch.cuda.Stream() if enabled else None

    def __call__(self, *inputs, allow_capture=True):
        if not self.enabled or not allow_capture:
            return self.fn(*inputs)
        if self.inputs is None:
            self.inputs = tuple(x.clone() for x in inputs)
        else:
            for destination, source in zip(self.inputs, inputs, strict=True):
                if destination.shape != source.shape or destination.dtype != source.dtype:
                    raise ValueError("Captured update input shape/dtype changed")
                destination.copy_(source)
        if self.warmup:
            self.warmup -= 1
            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                result = self.fn(*self.inputs)
            torch.cuda.current_stream().wait_stream(self.stream)
            return result
        if self.graph is None:
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=self.stream):
                self.outputs = self.fn(*self.inputs)
            self.identity = tuple(t.data_ptr() for t in self.state())
        elif self.identity != tuple(t.data_ptr() for t in self.state()):
            raise RuntimeError("Captured parameter, normalizer or optimizer storage moved")
        self.graph.replay()
        return self.outputs


def optimize_capturable(loss, module, optimizer, max_norm):
    # Device assertions preserve fail-fast checks without Tensor -> bool syncs.
    torch._assert_async(torch.isfinite(loss), "Non-finite learner loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm, error_if_nonfinite=False)
    torch._assert_async(torch.isfinite(norm), "Non-finite learner gradient norm")
    optimizer.step()
    return norm.detach()


class AcceleratedUpdates:
    def __init__(self, trainer):
        self.t = trainer
        self.cfg = trainer.cfg
        if self.cfg.fk != "torch":
            raise ValueError("Compiled MaxCoord updates require fk=torch")
        if self.cfg.cuda_graphs and not str(self.cfg.device).startswith("cuda"):
            raise ValueError("cuda_graphs requires a CUDA device")
        for model in (trainer.wm, trainer.actor):
            for module in model.modules():
                if isinstance(module, RunningNorm):
                    module.no_sync = True
        for combined in (trainer.opt_wm, trainer.opt_pi, trainer.opt_v):
            for optimizer in _optimizers(combined):
                if isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
                    for group in optimizer.param_groups:
                        group["capturable"] = True
                        # Adam (critic) has CPU step tensors in old eager checkpoints.
                        for parameter in group["params"]:
                            state = optimizer.state.get(parameter, {})
                            if isinstance(state.get("step"), torch.Tensor):
                                state["step"] = state["step"].to(parameter.device)
        compile_fn = (lambda fn: torch.compile(fn, dynamic=False, fullgraph=True)) if self.cfg.compile_updates else (lambda fn: fn)
        self.prepare_wm = compile_fn(self._prepare_wm)
        self.wm_loss = compile_fn(self._wm_loss)
        self.prepare_actor = compile_fn(self._prepare_actor)
        self.actor_step = compile_fn(self._actor_step)
        self.bootstrap = compile_fn(self._bootstrap)
        self.critic_targets = compile_fn(self._critic_targets)
        self.critic_loss = compile_fn(self._critic_loss)
        norm_modules = [] if trainer.vecnorms is None else [trainer.vecnorms]
        self.wm_update = CapturedUpdate(self._wm_update,
            lambda: _tensor_state([trainer.wm] + norm_modules, [trainer.opt_wm]), self.cfg.cuda_graphs)
        self.policy_update = CapturedUpdate(self._policy_update,
            lambda: _tensor_state([trainer.wm, trainer.actor, trainer.critic] + norm_modules, [trainer.opt_pi, trainer.opt_v]),
            self.cfg.cuda_graphs)

    @torch.no_grad()
    def _prepare_wm(self, s, s1, actions, xi):
        wm = self.t.wm
        full = full_from_flat(self.t.mdp, s)
        successors = full_from_flat(self.t.mdp, s1)
        if self.t.vecnorms is None:
            wm.update_normalization(full, actions)
        return full[:, 0], wm.prepare_targets(successors), normalize_xi(xi[:, 0])

    def _wm_loss(self, initial, actions, targets, xi_n):
        result = self.t.wm.loss_from_targets(initial, actions, targets, xi_n,
                                             self.cfg.consistency_weight)
        return result.loss, torch.stack(list(result.metrics.values()))

    def _wm_update(self, s, s1, actions, xi):
        t, cfg = self.t, self.cfg
        rows = []
        for j in range(cfg.wm_batches):
            initial, targets, xi_n = self.prepare_wm(s[j], s1[j], actions[j], xi[j])
            loss, metrics = self.wm_loss(initial, actions[j], targets, xi_n)
            norm = optimize_capturable(loss, t.wm, t.opt_wm, cfg.grad_clip_norm)
            rows.append(torch.cat((metrics.detach(), loss.detach().reshape(1), norm.reshape(1))))
        return torch.stack(rows).mean(0)

    def fit_wm(self):
        t, cfg = self.t, self.cfg
        batches = [t.buffer.sample_windows(cfg.wm_batch, cfg.wm_window) for _ in range(cfg.wm_batches)]
        values = self.wm_update(*(torch.stack([b[k] for b in batches]) for k in ("s", "s1", "a", "xi")))
        names = ["wm_state", "wm_consistency", "wm_consistency_weighted",
                 "wm_mse_root_pos", "wm_mse_root_ori", "wm_mse_body_pos", "wm_mse_body_rot",
                 "wm_mse_body_vel", "wm_mse_body_avel", *LATENT_METRICS, "wm_loss", "grad_wm"]
        return dict(zip(names, values.unbind(), strict=True))

    @torch.no_grad()
    def _prepare_actor(self, s, frame, xi):
        t, cfg = self.t, self.cfg
        state = t.wm.initialize(full_from_flat(t.mdp, s), normalize_xi(xi))
        g = goal(t.sim.motion, frame, state.aux[..., 3:7], cfg.goal_frames)
        if t.vecnorms is None:
            t.actor.state_norm.update(actor_features(state, cfg.actor_kind))
            t.actor.goal_norm.update(g)
        return state

    def _actor_step(self, state, frame, previous, scale):
        t, cfg = self.t, self.cfg
        features = actor_features(state, cfg.actor_kind)
        g = goal(t.sim.motion, frame, state.aux[..., 3:7], cfg.goal_frames)
        critic_input = t.critic_inputs(state, g)
        action = t.actor(features, g)
        state = t.wm(state, action)
        frame = frame + 1
        rs = reward_state(t.wm.representation, state, t.task.num_joints)
        # All body fields are already present, so MDP.reward's completion is a no-op.
        parts = [term(rs, action, previous, frame) for term in t.mdp._reward_terms]
        reward = parts[0]
        for part in parts[1:]:
            reward = reward + part
        term = t.mdp.soft_termination(rs.p, rs.quat, frame)
        return state, frame, action, reward / scale, term, critic_input, torch.stack([p.detach().mean() for p in parts])

    def _bootstrap(self, state, frame, returns, alive):
        t, cfg = self.t, self.cfg
        g = goal(t.sim.motion, frame, state.aux[..., 3:7], cfg.goal_frames)
        inputs = t.critic_inputs(state, g)
        value = t.critic.expected_values(critic_forward(t.critic, inputs)).mean(-1)
        loss = -(returns + cfg.gamma**cfg.horizon * alive * value).mean()
        return loss, inputs

    @torch.no_grad()
    def _critic_targets(self, features, reward, term):
        cfg = self.cfg
        values = self.t.critic.expected_values(critic_forward(self.t.critic, features.flatten(0, 1)))
        values = values.reshape(cfg.horizon + 1, cfg.pi_batch, cfg.critic_ensemble)
        target, targets = values[-1], []
        for j in reversed(range(cfg.horizon)):
            target = reward[j, :, None] + cfg.gamma * (1 - term[j, :, None]) * (
                (1 - cfg.lam) * values[j + 1] + cfg.lam * target)
            targets.append(target)
        return torch.stack(targets[::-1]).flatten(0, 1)

    def _critic_loss(self, x, targets):
        return self.t.critic.compute_loss(critic_forward(self.t.critic, x), targets)

    def _policy_update(self, s, frame, previous, xi, scale, permutations):
        t, cfg = self.t, self.cfg
        state = self.prepare_actor(s, frame, xi)
        inputs, rewards, terminated, reward_parts = [], [], [], []
        returns = torch.zeros(cfg.pi_batch, device=cfg.device)
        alive = torch.ones_like(returns)
        with hold_out_net(t.wm), hold_out_net(t.critic):
            for j in range(cfg.horizon):
                state, frame, action, reward, term, critic_input, parts = self.actor_step(state, frame, previous, scale)
                inputs.append(critic_input.detach())
                reward_parts.append(parts)
                returns = returns + cfg.gamma**j * alive * reward
                alive = alive * (1 - term)
                rewards.append(reward.detach())
                terminated.append(term.detach())
                previous = action
            pi_loss, last_inputs = self.bootstrap(state, frame, returns, alive)
            inputs.append(last_inputs.detach())
            grad_pi = optimize_capturable(pi_loss, t.actor, t.opt_pi, cfg.grad_clip_norm)
        features = torch.stack(inputs)
        reward, term = torch.stack(rewards), torch.stack(terminated)
        targets = self.critic_targets(features, reward, term)
        x = features[:-1].flatten(0, 1)
        losses, critic_norms = [], []
        for order in permutations:
            for idx in order.tensor_split(min(cfg.critic_minibatches, x.shape[0])):
                loss = self.critic_loss(x[idx], targets[idx])
                critic_norms.append(optimize_capturable(loss, t.critic, t.opt_v, cfg.grad_clip_norm))
                losses.append(loss.detach())
        return torch.stack((pi_loss.detach(), torch.stack(losses).mean(), grad_pi,
                            reward.mean() / CONTROL_DT, alive.detach().mean(), reward.mean() * scale / CONTROL_DT, torch.stack(critic_norms).mean(),
                            *(torch.stack(reward_parts).mean(0) / CONTROL_DT).unbind()))

    def fit_policy(self, iteration):
        t, cfg = self.t, self.cfg
        batch = t.buffer.sample_starts(cfg.pi_batch, 1)
        # Neither actor nor optimizer consumes randomness before the eager critic
        # shuffle, so drawing these permutations here preserves the RNG order.
        count = cfg.horizon * cfg.pi_batch
        permutations = torch.stack([torch.randperm(count, device=cfg.device) for _ in range(cfg.critic_epochs)])
        scale = t.reward_norm.scale() if cfg.normalize_reward else torch.ones((), device=cfg.device)
        lr_scale = min(1.0, iteration / max(cfg.pi_warmup, 1)) if cfg.pi_warmup else 1.0
        for group in t.opt_pi.param_groups:
            group["lr"] = cfg.pi_lr * lr_scale
        # Muon's learning rate is a Python float; capture only once warmup ends.
        values = self.policy_update(batch["s"], batch["f"], batch["a_prev"], batch["xi"], scale, permutations,
                                    allow_capture=iteration >= cfg.pi_warmup)
        names = ("pi_loss", "v_loss", "grad_pi", "imag_reward", "imag_survival", "imag_raw_reward", "grad_v",
                 *(f"imag_term_{name}" for name in t.mdp.term_names))
        return dict(zip(names, values.unbind(), strict=True))
