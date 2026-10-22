"""SuperMimic training loop: collect, fit the world model, BPTT the policy, fit the critic."""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import swanlab
import torch
import torch.nn as nn

from .buffer import ReplayBuffer
from .mdp import (
    GOAL_DIM,
    GOAL_FRAMES,
    HIST,
    INV_DIM,
    MAX_STEP_REWARD,
    NUM_J,
    XI_DIM,
    invariant_features,
    local_state_error,
    nominal_error,
    normalize_xi,
    unflatten_state,
)
from .network import Critic, InputNorms, MLP, Policy, WorldModel
from .reward_norm import RewardNormalizer


@dataclass
class TrainConfig:
    num_envs: int = 4096
    iterations: int = 320_000
    buffer_steps: int = 1000
    utd: float = 0.4
    horizon: int = 32
    wm_window: int = 8
    wm_batches: int = 10
    wm_batch: int = 2048
    pi_batch: int = 4096
    lr: float = 2e-4
    gamma: float = 0.99
    lam: float = 0.95
    action_noise: float = 0.1
    critic_epochs: int = 5
    critic_minibatches: int = 4
    lr_decay_iters: int = 250_000
    wm_norm: str = "none"
    wm_residual: bool = False
    stochastic_actor: bool = False
    anchor: str = "torso"
    obs_terms: tuple[str, ...] = ("joint_pos", "joint_vel", "base_ang_vel", "projected_gravity", "last_action")
    swanlab_project: str = "rethinking-supermimic"
    swanlab_mode: str = "offline"
    swanlab_experiment: str = "lafan1"
    device: str = "cuda"
    # None / <=0 disables. Grad norms are always logged.
    grad_clip_norm: float | None = None
    grad_clip_value: float | None = None  # policy only; None disables
    # Scale policy/critic rewards by 1/max(√Var(G), G_r_max/G_max). Stats from sim collect.
    # G_max = MAX_STEP_REWARD / (1-γ) is fixed by the MDP.
    normalize_reward: bool = False
    log_every: int = 50
    ckpt_every: int = 2000
    seed: int = 0


@contextmanager
def timed(device: str):
    """Wall time of a block, after the GPU has finished the work inside it."""
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        yield
    finally:
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        timed.last = time.perf_counter() - start


timed.last = 0.0


def lr_at(iteration: int, base: float, decay_iters: int) -> float:
    if iteration >= decay_iters:
        return base * 0.03
    cosine = 0.5 * (1.0 + math.cos(math.pi * iteration / decay_iters))
    return base * (0.03 + 0.97 * cosine)


def _set_grad(module: nn.Module, enabled: bool) -> None:
    for param in module.parameters():
        param.requires_grad_(enabled)


def _stack(xs: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(xs).reshape(-1, *xs[0].shape[1:])


def _flatten_time(states: list[tuple]) -> tuple:
    return tuple(torch.stack([s[i] for s in states]).reshape(-1, states[0][i].shape[-1]) for i in range(6))


def _unflatten_time(flat: tuple, steps: int) -> tuple:
    return tuple(t.view(steps, -1, t.shape[-1]).reshape(-1, t.shape[-1]) for t in flat)


def _mean(xs: list[float]) -> float:
    return sum(xs) / max(len(xs), 1)


def _grad_norm(params) -> float:
    grads = [p.grad.detach() for p in params if p.grad is not None]
    if not grads:
        return 0.0
    return float(nn.utils.get_total_norm(grads))


def _clip_grads(params, max_norm: float | None, max_value: float | None = None) -> float:
    """Return total grad norm (pre-clip). Clip only when limits are set and positive."""
    params = list(params)
    norm = _grad_norm(params)
    if max_norm is not None and max_norm > 0:
        nn.utils.clip_grad_norm_(params, max_norm)
    if max_value is not None and max_value > 0:
        nn.utils.clip_grad_value_(params, max_value)
    return norm


class MSELoss(nn.Module):
    """Normalized MSE on frame-local state errors (current SuperMimic WM objective)."""

    def __init__(self, device: torch.device | str):
        super().__init__()
        self.register_buffer("nominal", nominal_error(torch.device(device)))

    def forward(self, errors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``errors`` is ``(B, K, D)`` from ``local_state_error`` over the WM window."""
        scaled = errors / self.nominal
        loss = scaled.square().mean()
        err_last = scaled[:, -1].square().mean(-1).sqrt().mean()
        return loss, err_last


class AdversarialLoss(nn.Module):
    """LSGAN world-model loss. Owns the discriminator; not used for WM training yet.

    Discriminates real vs predicted next-state invariant features, conditioned on
    ``(inv(s), a, ξ)``. Labels: real → +1, fake → −1; the generator fools D to +1
    (Mao et al. LSGAN / AMP convention).
    """

    def __init__(self, hidden: list[int] | None = None):
        super().__init__()
        hidden = hidden or [512, 512]
        self.disc = MLP(INV_DIM + NUM_J + XI_DIM + INV_DIM, hidden, 1)

    def score(
        self,
        inv_s: torch.Tensor,
        action: torch.Tensor,
        xi_n: torch.Tensor,
        inv_s1: torch.Tensor,
    ) -> torch.Tensor:
        return self.disc(torch.cat((inv_s, action, xi_n, inv_s1), dim=-1)).squeeze(-1)

    def discriminator_loss(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        return 0.5 * ((real - 1.0).square().mean() + (fake + 1.0).square().mean())

    def generator_loss(self, fake: torch.Tensor) -> torch.Tensor:
        return 0.5 * (fake - 1.0).square().mean()

    def forward(
        self,
        inv_s: torch.Tensor,
        action: torch.Tensor,
        xi_n: torch.Tensor,
        inv_pred: torch.Tensor,
        inv_true: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(generator_loss, discriminator_loss)`` for one transition batch."""
        real = self.score(inv_s, action, xi_n, inv_true.detach())
        fake_d = self.score(inv_s, action, xi_n, inv_pred.detach())
        fake_g = self.score(inv_s, action, xi_n, inv_pred)
        return self.generator_loss(fake_g), self.discriminator_loss(real, fake_d)


class Trainer:
    """Owns the simulator, buffer, networks, and the Algorithm A.1 loop."""

    def __init__(self, cfg: TrainConfig, log_dir: Path | str):
        from .kinematics import TorchKinematics, compile_g1
        from .motion import MotionLib
        from .sim import TrackingSim

        self.cfg = cfg
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.device = cfg.device

        torch.manual_seed(cfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        self.motion = MotionLib.load(device=self.device)
        model, names = compile_g1()
        self.kin = TorchKinematics(model, names)
        self.sim = TrackingSim(
            cfg.num_envs,
            self.motion,
            self.kin,
            self.device,
            gamma=cfg.gamma,
            anchor=cfg.anchor,
            obs_terms=cfg.obs_terms,
        )
        obs_dim = self.sim.mdp.obs_dim
        self.wm = WorldModel(cfg.wm_norm, cfg.wm_residual).to(self.device)
        self.policy = Policy(obs_dim, GOAL_DIM, HIST, stochastic=cfg.stochastic_actor).to(self.device)
        self.critic = Critic(INV_DIM + GOAL_DIM + XI_DIM).to(self.device)
        self.norms = InputNorms(obs_dim).to(self.device)
        self.opt_wm = torch.optim.Adam(self.wm.parameters(), lr=cfg.lr)
        self.opt_pi = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr, betas=(0.9, 0.95))
        self.opt_v = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(cfg.buffer_steps, cfg.num_envs, self.device)
        self.mse_loss = MSELoss(self.device).to(self.device)
        # Built for later experiments; WM training still uses ``mse_loss`` only.
        self.adv_loss = AdversarialLoss().to(self.device)
        self.reward_norm = (
            RewardNormalizer(
                cfg.num_envs,
                cfg.gamma,
                g_max=MAX_STEP_REWARD / (1.0 - cfg.gamma),
                device=self.device,
            )
            if cfg.normalize_reward
            else None
        )
        self.obs_hist = torch.zeros(cfg.num_envs, HIST, obs_dim, device=self.device)

        self.stats = {"r": [], "ep": [], "term": [], "done_n": []}
        self.agg = {
            "wm": [],
            "err": [],
            "pi": [],
            "v": [],
            "score": [],
            "surv": [],
            "gn_wm": [],
            "gn_pi": [],
            "gn_v": [],
        }
        self.timing = {"collect": [], "wm": [], "imagine": [], "policy": [], "critic": []}
        self.step_acc = 0.0
        self.started = 0.0
        self.logged = 0.0
        self._log_f = None

    def fit(self) -> None:
        cfg = self.cfg
        logged_cfg = {
            key: list(value) if isinstance(value, tuple) else value for key, value in cfg.__dict__.items()
        }
        swanlab.init(
            project=cfg.swanlab_project,
            name=cfg.swanlab_experiment or None,
            config={**logged_cfg, "obs_dim": self.sim.mdp.obs_dim, "anchor_body": self.kin.anchor_name},
            mode=cfg.swanlab_mode,
        )
        self._log_f = (self.log_dir / "train_log.csv").open("w")
        self._log_f.write(
            "iteration,wall_h,wm_loss,wm_err8,pi_loss,v_loss,imag_score,imag_survival,"
            "sim_score,ep_score,term_frac,it_per_s,lr,entropy,"
            "grad_wm,grad_pi,grad_v,"
            "t_collect,t_wm,t_imagine,t_policy,t_critic\n"
        )
        self.started = time.time()
        self.logged = self.started
        try:
            for iteration in range(cfg.iterations):
                self._step(iteration)
        finally:
            swanlab.finish()
            if self._log_f is not None:
                self._log_f.close()

    def _step(self, iteration: int) -> None:
        cfg = self.cfg
        lr = lr_at(iteration, cfg.lr, cfg.lr_decay_iters)
        for opt in (self.opt_wm, self.opt_pi, self.opt_v):
            for group in opt.param_groups:
                group["lr"] = lr
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

        with timed(self.device):
            wm_loss, err8, gn_wm = self.update_world_model()
        self.timing["wm"].append(timed.last)

        with timed(self.device):
            start = self.sample_policy_starts()
            pi_loss, critic_batch, imag_score, survival = self.imagine(start)
        self.timing["imagine"].append(timed.last)

        with timed(self.device):
            gn_pi = self.update_policy(pi_loss)
        self.timing["policy"].append(timed.last)

        with timed(self.device):
            v_loss, gn_v = self.update_critic(critic_batch)
        self.timing["critic"].append(timed.last)

        self.agg["wm"].append(float(wm_loss.detach()))
        self.agg["err"].append(float(err8.detach()))
        self.agg["pi"].append(float(pi_loss.detach()))
        self.agg["v"].append(float(v_loss))
        self.agg["score"].append(float(imag_score))
        self.agg["surv"].append(float(survival))
        self.agg["gn_wm"].append(gn_wm)
        self.agg["gn_pi"].append(gn_pi)
        self.agg["gn_v"].append(gn_v)

        if (iteration + 1) % cfg.log_every == 0:
            self._log(iteration + 1, lr)
        if (iteration + 1) % cfg.ckpt_every == 0 or iteration + 1 == cfg.iterations:
            self.save_checkpoint(iteration + 1)

    def collect(self, n_steps: int) -> None:
        cfg = self.cfg
        for _ in range(n_steps):
            p, quat, v, w, q, qd = self.sim.state()
            obs = self.sim.mdp.add_noise(
                self.sim.mdp.obs(p, quat, v, w, q, qd, self.sim.a_prev, self.sim.xi, self.sim.f)
            )
            fresh = self.sim.ep_step == 0
            self.obs_hist = torch.where(
                fresh[:, None, None],
                obs[:, None, :].expand(-1, HIST, -1),
                torch.cat((self.obs_hist[:, 1:], obs[:, None, :]), dim=1),
            )
            goal = self.motion.goal(self.sim.f, GOAL_FRAMES)
            xi_n = normalize_xi(self.sim.xi)
            self.norms.state.update(invariant_features(p, quat, v, w, q, qd))
            self.norms.obs.update(obs)
            self.norms.goal.update(goal)
            self.norms.xi.update(xi_n)
            with torch.no_grad():
                action = self.policy(
                    self.obs_hist, goal, self.norms, deterministic=not cfg.stochastic_actor
                )
            executed = action + cfg.action_noise * torch.randn_like(action)
            self.norms.action.update(executed)
            out = self.sim.step(executed)
            self.buffer.add(out)
            if self.reward_norm is not None:
                self.reward_norm.update(out["r"], out["done"])
            self.stats["r"].append(float(out["r"].mean() / MAX_STEP_REWARD))
            done = out["done"]
            if bool(done.any()):
                length = out["ep_len"][done].float().clamp(min=1)
                self.stats["ep"].append(
                    float((out["ep_ret"][done] / (MAX_STEP_REWARD * length)).mean())
                )
                self.stats["term"].append(float(out["d"][done].float().sum()))
                self.stats["done_n"].append(float(done.sum()))

    def update_world_model(self) -> tuple[torch.Tensor, torch.Tensor, float]:
        cfg = self.cfg
        loss = err8 = None
        norms: list[float] = []
        for _ in range(cfg.wm_batches):
            batch = self.buffer.sample_windows(cfg.wm_batch, cfg.wm_window)
            loss, err8 = self.world_model_loss(batch)
            self.opt_wm.zero_grad(set_to_none=True)
            loss.backward()
            norms.append(_clip_grads(self.wm.parameters(), cfg.grad_clip_norm))
            self.opt_wm.step()
        assert loss is not None and err8 is not None
        return loss, err8, _mean(norms)

    def world_model_rollout(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        """Autoregressive WM window. Returns stacked local errors and per-step transition feats."""
        p, quat, v, w, q, qd = unflatten_state(batch["s"][:, 0])
        xi_n = normalize_xi(batch["xi"][:, 0])
        errors = []
        steps = []
        for j in range(batch["a"].shape[1]):
            inv_s = invariant_features(p, quat, v, w, q, qd)
            action = batch["a"][:, j]
            p, quat, v, w, q, qd = self.wm.step(
                p, quat, v, w, q, qd, action, xi_n, self.norms
            )
            true = unflatten_state(batch["s1"][:, j])
            errors.append(
                local_state_error(
                    true[0], true[1], true[2], true[3], true[4], true[5], p, quat, v, w, q, qd
                )
            )
            steps.append(
                {
                    "inv_s": inv_s,
                    "action": action,
                    "xi_n": xi_n,
                    "inv_pred": invariant_features(p, quat, v, w, q, qd),
                    "inv_true": invariant_features(*true),
                }
            )
        return torch.stack(errors, dim=1), steps

    def world_model_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        errors, _steps = self.world_model_rollout(batch)
        return self.mse_loss(errors)

    def sample_policy_starts(self) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        motion = self.sim.mdp.motion
        pieces = []
        guard = 0
        while sum(p["s"].shape[0] for p in pieces) < cfg.pi_batch and guard < 40:
            guard += 1
            start = self.buffer.sample_starts(cfg.pi_batch * 2, HIST)
            last = motion.clip_end[motion.clip_id[start["f"]]] - 1
            keep = (start["f"] + cfg.horizon <= last).nonzero(as_tuple=False).flatten()
            keep = keep[: cfg.pi_batch - sum(p["s"].shape[0] for p in pieces)]
            pieces.append({key: value[keep] for key, value in start.items()})
        out = {key: torch.cat([p[key] for p in pieces]) for key in pieces[0]}
        if out["s"].shape[0] < cfg.pi_batch:
            raise RuntimeError("Could not sample enough imagination starts inside the clips.")
        return out

    def imagine(self, start: dict[str, torch.Tensor]):
        """One differentiable rollout. Returns the policy loss and detached critic data."""
        cfg = self.cfg
        mdp = self.sim.mdp
        device = start["s"].device
        batch = start["s"].shape[0]
        p, quat, v, w, q, qd = unflatten_state(start["hist_s"])
        hist_n = p.shape[0] * p.shape[1]
        obs = mdp.obs(
            p.reshape(hist_n, 3),
            quat.reshape(hist_n, 4),
            v.reshape(hist_n, 3),
            w.reshape(hist_n, 3),
            q.reshape(hist_n, -1),
            qd.reshape(hist_n, -1),
            start["hist_a_prev"].reshape(hist_n, -1),
            start["hist_xi"].reshape(hist_n, -1),
            start["hist_f"].reshape(hist_n),
        )
        obs_hist = mdp.add_noise(obs).view(batch, HIST, -1)
        p, quat, v, w, q, qd = unflatten_state(start["s"])
        a_prev = start["a_prev"]
        xi = start["xi"]
        xi_n = normalize_xi(xi)
        frame = start["f"]
        goal = mdp.motion.goal(frame, GOAL_FRAMES)
        _set_grad(self.wm, False)
        _set_grad(self.critic, False)
        states = [(p, quat, v, w, q, qd)]
        frames = [frame]
        goals = [goal]
        actions = []
        prev_actions = []
        for _ in range(cfg.horizon):
            action = self.policy(obs_hist, goal, self.norms, deterministic=not cfg.stochastic_actor)
            prev_actions.append(a_prev)
            actions.append(action)
            p, quat, v, w, q, qd = self.wm.step(p, quat, v, w, q, qd, action, xi_n, self.norms)
            frame = frame + 1
            goal = mdp.motion.goal(frame, GOAL_FRAMES)
            obs = mdp.add_noise(mdp.obs(p, quat, v, w, q, qd, action, xi, frame))
            obs_hist = torch.cat((obs_hist[:, 1:], obs.unsqueeze(1)), dim=1)
            states.append((p, quat, v, w, q, qd))
            frames.append(frame)
            goals.append(goal)
            a_prev = action
        flat = _flatten_time(states[1:])
        reward = mdp.reward(*flat, _stack(frames[1:]), _stack(actions), _stack(prev_actions))
        done = mdp.soft_termination(flat[0], flat[1], _stack(frames[1:]))
        reward = reward.view(cfg.horizon, batch)
        done = done.view(cfg.horizon, batch)
        raw_score = reward.detach().mean() / MAX_STEP_REWARD
        if self.reward_norm is not None:
            reward = self.reward_norm.normalize(reward)
        alive = torch.cumprod(1.0 - done, dim=0)
        survival = torch.cat((torch.ones(1, batch, device=device), alive[:-1]), dim=0)
        discount = cfg.gamma ** torch.arange(cfg.horizon, device=device, dtype=torch.float32)
        returns = (discount[:, None] * survival * reward).sum(0)
        bootstrap = cfg.gamma**cfg.horizon * alive[-1] * self.critic(
            self.norms.critic(invariant_features(*states[-1]), goals[-1], xi_n)
        )
        policy_loss = -(returns + bootstrap).mean()
        critic_batch = {
            "states": tuple(t.detach() for t in _flatten_time(states)),
            "frames": _stack(frames).detach(),
            "goals": _stack(goals).detach(),
            "xi": xi.detach().unsqueeze(0).expand(cfg.horizon + 1, -1, -1).reshape(-1, xi.shape[-1]),
            "reward": reward.detach(),
            "done": done.detach(),
        }
        return (
            policy_loss,
            critic_batch,
            raw_score,
            alive[-1].detach().mean(),
        )

    def update_policy(self, pi_loss: torch.Tensor) -> float:
        cfg = self.cfg
        self.opt_pi.zero_grad(set_to_none=True)
        pi_loss.backward()
        grad_norm = _clip_grads(self.policy.parameters(), cfg.grad_clip_norm, cfg.grad_clip_value)
        if math.isfinite(grad_norm):
            self.opt_pi.step()
        _set_grad(self.wm, True)
        _set_grad(self.critic, True)
        return grad_norm

    def update_critic(self, batch: dict) -> tuple[torch.Tensor, float]:
        cfg = self.cfg
        horizon, n = batch["reward"].shape
        flat = _unflatten_time(batch["states"], horizon + 1)
        features = self.norms.critic(
            invariant_features(*flat), batch["goals"], normalize_xi(batch["xi"])
        )
        with torch.no_grad():
            values = self.critic(features).view(horizon + 1, n)
            target = values[-1]
            targets = []
            for t in range(horizon - 1, -1, -1):
                nxt = (1.0 - cfg.lam) * values[t + 1] + cfg.lam * target
                target = batch["reward"][t] + cfg.gamma * (1.0 - batch["done"][t]) * nxt
                targets.append(target)
            targets = torch.stack(targets[::-1])
        pred_in = features.view(horizon + 1, n, -1)[:-1].reshape(-1, features.shape[-1])
        target_flat = targets.reshape(-1)
        count = pred_in.shape[0]
        minibatch = count // cfg.critic_minibatches
        last = pred_in.new_zeros(())
        norms: list[float] = []
        for _ in range(cfg.critic_epochs):
            order = torch.randperm(count, device=pred_in.device)
            for i in range(cfg.critic_minibatches):
                idx = order[i * minibatch : (i + 1) * minibatch]
                last = (self.critic(pred_in[idx]) - target_flat[idx]).square().mean()
                self.opt_v.zero_grad(set_to_none=True)
                last.backward()
                norms.append(_clip_grads(self.critic.parameters(), cfg.grad_clip_norm))
                self.opt_v.step()
        return last.detach(), _mean(norms)

    def _log(self, iteration: int, lr: float) -> None:
        cfg = self.cfg
        now = time.time()
        term = sum(self.stats["term"]) / max(sum(self.stats["done_n"]), 1.0)
        seconds = {key: _mean(self.timing[key]) for key in self.timing}
        gn_wm = _mean(self.agg["gn_wm"])
        gn_pi = _mean(self.agg["gn_pi"])
        gn_v = _mean(self.agg["gn_v"])
        entropy = (
            float(self.policy.entropy().detach())
            if cfg.stochastic_actor and self.policy.log_std is not None
            else float("nan")
        )
        row = [
            iteration,
            (now - self.started) / 3600,
            _mean(self.agg["wm"]),
            _mean(self.agg["err"]),
            _mean(self.agg["pi"]),
            _mean(self.agg["v"]),
            _mean(self.agg["score"]),
            _mean(self.agg["surv"]),
            _mean(self.stats["r"]),
            _mean(self.stats["ep"]),
            term,
            cfg.log_every / (now - self.logged),
            lr,
            entropy,
            gn_wm,
            gn_pi,
            gn_v,
            seconds["collect"],
            seconds["wm"],
            seconds["imagine"],
            seconds["policy"],
            seconds["critic"],
        ]
        assert self._log_f is not None
        self._log_f.write(",".join(f"{x:.6g}" for x in row) + "\n")
        self._log_f.flush()
        metrics = {
            "train/wm_loss": row[2],
            "train/wm_err8": row[3],
            "train/pi_loss": row[4],
            "train/v_loss": row[5],
            "train/imag_score": row[6],
            "train/imag_survival": row[7],
            "train/sim_score": row[8],
            "train/ep_score": row[9],
            "train/term_frac": term,
            "train/it_per_s": row[11],
            "train/lr": lr,
            "grad/wm": gn_wm,
            "grad/pi": gn_pi,
            "grad/v": gn_v,
            "time/collect_s": seconds["collect"],
            "time/wm_s": seconds["wm"],
            "time/imagine_s": seconds["imagine"],
            "time/policy_s": seconds["policy"],
            "time/critic_s": seconds["critic"],
        }
        if math.isfinite(entropy):
            metrics["train/entropy"] = entropy
        if self.reward_norm is not None:
            metrics["train/reward_scale"] = float(self.reward_norm.scale())
        swanlab.log(metrics, step=iteration)
        ent_s = f" H {entropy:.3f} |" if math.isfinite(entropy) else ""
        print(
            f"it {iteration} | {row[1]:.3f} h | wm {row[2]:.4f} e8 {row[3]:.4f} | "
            f"pi {row[4]:.3f} v {row[5]:.4f} imag {row[6]:.3f} surv {row[7]:.3f} | "
            f"sim {row[8]:.3f} ep {row[9]:.3f} term {term:.2f} |{ent_s} {row[11]:.2f} it/s | "
            f"gn wm {gn_wm:.3g} pi {gn_pi:.3g} v {gn_v:.3g} | "
            f"collect {seconds['collect']:.3f}s wm {seconds['wm']:.3f}s "
            f"imagine {seconds['imagine']:.3f}s policy {seconds['policy']:.3f}s critic {seconds['critic']:.3f}s",
            flush=True,
        )
        self.logged = now
        self.stats = {"r": [], "ep": [], "term": [], "done_n": []}
        self.agg = {key: [] for key in self.agg}
        self.timing = {key: [] for key in self.timing}

    def save_checkpoint(self, iteration: int) -> None:
        ckpt = {
            "wm": self.wm.state_dict(),
            "pi": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "norms": self.norms.state_dict(),
            "opt_wm": self.opt_wm.state_dict(),
            "opt_pi": self.opt_pi.state_dict(),
            "opt_v": self.opt_v.state_dict(),
            "iteration": iteration,
            "cfg": self.cfg.__dict__,
        }
        if self.reward_norm is not None:
            ckpt["reward_norm"] = self.reward_norm.state_dict()
        path = self.log_dir / f"ckpt_{iteration:06d}.pt"
        torch.save(ckpt, path)
        torch.save(ckpt, self.log_dir / "last.pt")


def run(cfg: TrainConfig, log_dir) -> None:
    Trainer(cfg, log_dir).fit()
