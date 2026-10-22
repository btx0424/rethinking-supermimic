"""SuperMimic training loop: collect, fit the world model, BPTT the policy, fit the critic."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .buffer import ReplayBuffer
from .mdp import (
    GOAL_DIM,
    GOAL_FRAMES,
    HIST,
    INV_DIM,
    MAX_STEP_REWARD,
    OBS_DIM,
    XI_DIM,
    local_state_error,
    nominal_error,
    normalize_xi,
    unflatten_state,
    invariant_features,
)
from .network import Critic, InputNorms, Policy, WorldModel


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
    device: str = "cuda"
    log_every: int = 50
    ckpt_every: int = 2000
    seed: int = 0


def lr_at(iteration: int, base: float, decay_iters: int) -> float:
    if iteration >= decay_iters:
        return base * 0.03
    cosine = 0.5 * (1.0 + math.cos(math.pi * iteration / decay_iters))
    return base * (0.03 + 0.97 * cosine)


def build_models(cfg: TrainConfig, device: str) -> tuple[WorldModel, Policy, Critic, InputNorms]:
    wm = WorldModel(cfg.wm_norm, cfg.wm_residual).to(device)
    policy = Policy(OBS_DIM, GOAL_DIM, HIST, stochastic=cfg.stochastic_actor).to(device)
    critic = Critic(INV_DIM + GOAL_DIM + XI_DIM).to(device)
    norms = InputNorms().to(device)
    return wm, policy, critic, norms


def _set_grad(module: nn.Module, enabled: bool) -> None:
    for param in module.parameters():
        param.requires_grad_(enabled)


def world_model_loss(wm: WorldModel, batch: dict[str, torch.Tensor], nominal: torch.Tensor, norms: InputNorms):
    p, quat, v, w, q, qd = unflatten_state(batch["s"][:, 0])
    xi_n = normalize_xi(batch["xi"][:, 0])
    errors = []
    for j in range(batch["a"].shape[1]):
        p, quat, v, w, q, qd = wm.step(p, quat, v, w, q, qd, batch["a"][:, j], xi_n, norms)
        true = unflatten_state(batch["s1"][:, j])
        errors.append(local_state_error(true[0], true[1], true[2], true[3], true[4], true[5], p, quat, v, w, q, qd))
    scaled = torch.stack(errors, dim=1) / nominal
    loss = scaled.square().mean()
    err8 = scaled[:, -1].square().mean(-1).sqrt().mean()
    return loss, err8


def imagine(wm, policy, critic, mdp, start, cfg: TrainConfig, norms: InputNorms):
    """One differentiable rollout. Returns the policy loss and detached critic data."""
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
    )
    obs_hist = mdp.add_noise(obs).view(batch, HIST, -1)
    p, quat, v, w, q, qd = unflatten_state(start["s"])
    a_prev = start["a_prev"]
    xi = start["xi"]
    xi_n = normalize_xi(xi)
    frame = start["f"]
    goal = mdp.motion.goal(frame, GOAL_FRAMES)
    _set_grad(wm, False)
    _set_grad(critic, False)
    states = [(p, quat, v, w, q, qd)]
    frames = [frame]
    goals = [goal]
    actions = []
    prev_actions = []
    for _ in range(cfg.horizon):
        action = policy(obs_hist, goal, norms, deterministic=not cfg.stochastic_actor)
        prev_actions.append(a_prev)
        actions.append(action)
        p, quat, v, w, q, qd = wm.step(p, quat, v, w, q, qd, action, xi_n, norms)
        frame = frame + 1
        goal = mdp.motion.goal(frame, GOAL_FRAMES)
        obs = mdp.add_noise(mdp.obs(p, quat, v, w, q, qd, action, xi))
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
    alive = torch.cumprod(1.0 - done, dim=0)
    survival = torch.cat((torch.ones(1, batch, device=device), alive[:-1]), dim=0)
    discount = cfg.gamma ** torch.arange(cfg.horizon, device=device, dtype=torch.float32)
    returns = (discount[:, None] * survival * reward).sum(0)
    bootstrap = cfg.gamma**cfg.horizon * alive[-1] * critic(
        norms.critic(invariant_features(*states[-1]), goals[-1], xi_n)
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
    return policy_loss, critic_batch, reward.detach().mean() / MAX_STEP_REWARD, alive[-1].detach().mean()


def fit_critic(critic, opt, batch, cfg: TrainConfig, norms: InputNorms) -> torch.Tensor:
    horizon, n = batch["reward"].shape
    flat = _unflatten_time(batch["states"], horizon + 1)
    features = norms.critic(invariant_features(*flat), batch["goals"], normalize_xi(batch["xi"]))
    with torch.no_grad():
        values = critic(features).view(horizon + 1, n)
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
    for _ in range(cfg.critic_epochs):
        order = torch.randperm(count, device=pred_in.device)
        for i in range(cfg.critic_minibatches):
            idx = order[i * minibatch : (i + 1) * minibatch]
            last = (critic(pred_in[idx]) - target_flat[idx]).square().mean()
            opt.zero_grad(set_to_none=True)
            last.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            opt.step()
    return last.detach()


def _stack(xs: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(xs).reshape(-1, *xs[0].shape[1:])


def _flatten_time(states: list[tuple]) -> tuple:
    return tuple(torch.stack([s[i] for s in states]).reshape(-1, states[0][i].shape[-1]) for i in range(6))


def _unflatten_time(flat: tuple, steps: int) -> tuple:
    return tuple(t.view(steps, -1, t.shape[-1]).reshape(-1, t.shape[-1]) for t in flat)


def sample_policy_starts(buffer: ReplayBuffer, mdp, cfg: TrainConfig) -> dict[str, torch.Tensor]:
    motion = mdp.motion
    pieces = []
    guard = 0
    while sum(p["s"].shape[0] for p in pieces) < cfg.pi_batch and guard < 40:
        guard += 1
        start = buffer.sample_starts(cfg.pi_batch * 2, HIST)
        last = motion.clip_end[motion.clip_id[start["f"]]] - 1
        keep = (start["f"] + cfg.horizon <= last).nonzero(as_tuple=False).flatten()
        keep = keep[: cfg.pi_batch - sum(p["s"].shape[0] for p in pieces)]
        pieces.append({key: value[keep] for key, value in start.items()})
    out = {key: torch.cat([p[key] for p in pieces]) for key in pieces[0]}
    if out["s"].shape[0] < cfg.pi_batch:
        raise RuntimeError("Could not sample enough imagination starts inside the clips.")
    return out


def run(cfg: TrainConfig, log_dir) -> None:
    import time
    from pathlib import Path

    from .kinematics import TorchKinematics, compile_g1
    from .motion import MotionLib
    from .sim import TrackingSim

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = cfg.device
    motion = MotionLib.load(device=device)
    model, names = compile_g1()
    kin = TorchKinematics(model, names)
    sim = TrackingSim(cfg.num_envs, motion, kin, device, gamma=cfg.gamma)
    wm, policy, critic, norms = build_models(cfg, device)
    opt_wm = torch.optim.Adam(wm.parameters(), lr=cfg.lr)
    opt_pi = torch.optim.Adam(policy.parameters(), lr=cfg.lr, betas=(0.9, 0.95))
    opt_v = torch.optim.Adam(critic.parameters(), lr=cfg.lr)
    buffer = ReplayBuffer(cfg.buffer_steps, cfg.num_envs, device)
    nominal = nominal_error(torch.device(device))
    obs_hist = torch.zeros(cfg.num_envs, HIST, OBS_DIM, device=device)
    log_path = log_dir / "train_log.csv"
    log_f = log_path.open("w")
    log_f.write(
        "iteration,wall_h,wm_loss,wm_err8,pi_loss,v_loss,imag_score,imag_survival,"
        "sim_score,ep_score,term_frac,it_per_s,lr\n"
    )

    def collect(n_steps: int, stats: dict) -> None:
        nonlocal obs_hist
        for _ in range(n_steps):
            p, quat, v, w, q, qd = sim.state()
            obs = sim.mdp.add_noise(sim.mdp.obs(p, quat, v, w, q, qd, sim.a_prev, sim.xi))
            fresh = sim.ep_step == 0
            obs_hist = torch.where(
                fresh[:, None, None],
                obs[:, None, :].expand(-1, HIST, -1),
                torch.cat((obs_hist[:, 1:], obs[:, None, :]), dim=1),
            )
            goal = motion.goal(sim.f, GOAL_FRAMES)
            xi_n = normalize_xi(sim.xi)
            norms.state.update(invariant_features(p, quat, v, w, q, qd))
            norms.obs.update(obs)
            norms.goal.update(goal)
            norms.xi.update(xi_n)
            with torch.no_grad():
                action = policy(obs_hist, goal, norms, deterministic=not cfg.stochastic_actor)
            executed = action + cfg.action_noise * torch.randn_like(action)
            norms.action.update(executed)
            out = sim.step(executed)
            buffer.add(out)
            stats["r"].append(float(out["r"].mean() / MAX_STEP_REWARD))
            done = out["done"]
            if bool(done.any()):
                length = out["ep_len"][done].float().clamp(min=1)
                stats["ep"].append(float((out["ep_ret"][done] / (MAX_STEP_REWARD * length)).mean()))
                stats["term"].append(float(out["d"][done].float().sum()))
                stats["done_n"].append(float(done.sum()))

    stats = {"r": [], "ep": [], "term": [], "done_n": []}
    step_acc = 0.0
    started = time.time()
    logged = time.time()
    agg = {"wm": [], "err": [], "pi": [], "v": [], "score": [], "surv": []}
    for iteration in range(cfg.iterations):
        lr = lr_at(iteration, cfg.lr, cfg.lr_decay_iters)
        for opt in (opt_wm, opt_pi, opt_v):
            for group in opt.param_groups:
                group["lr"] = lr
        step_acc += 1.0 / cfg.utd
        n_steps = int(step_acc)
        step_acc -= n_steps
        collect(n_steps, stats)
        if buffer.filled < cfg.wm_window:
            continue
        wm_loss = err8 = None
        for _ in range(cfg.wm_batches):
            batch = buffer.sample_windows(cfg.wm_batch, cfg.wm_window)
            wm_loss, err8 = world_model_loss(wm, batch, nominal, norms)
            opt_wm.zero_grad(set_to_none=True)
            wm_loss.backward()
            nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
            opt_wm.step()
        start = sample_policy_starts(buffer, sim.mdp, cfg)
        pi_loss, critic_batch, imag_score, survival = imagine(wm, policy, critic, sim.mdp, start, cfg, norms)
        opt_pi.zero_grad(set_to_none=True)
        pi_loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        nn.utils.clip_grad_value_(policy.parameters(), 0.1)
        if torch.isfinite(grad_norm):
            opt_pi.step()
        _set_grad(wm, True)
        _set_grad(critic, True)
        v_loss = fit_critic(critic, opt_v, critic_batch, cfg, norms)
        agg["wm"].append(float(wm_loss.detach()))
        agg["err"].append(float(err8.detach()))
        agg["pi"].append(float(pi_loss.detach()))
        agg["v"].append(float(v_loss))
        agg["score"].append(float(imag_score))
        agg["surv"].append(float(survival))
        if (iteration + 1) % cfg.log_every == 0:
            def mean(xs):
                return sum(xs) / max(len(xs), 1)

            now = time.time()
            term = sum(stats["term"]) / max(sum(stats["done_n"]), 1.0)
            row = [
                iteration + 1,
                (now - started) / 3600,
                mean(agg["wm"]),
                mean(agg["err"]),
                mean(agg["pi"]),
                mean(agg["v"]),
                mean(agg["score"]),
                mean(agg["surv"]),
                mean(stats["r"]),
                mean(stats["ep"]),
                term,
                cfg.log_every / (now - logged),
                lr,
            ]
            log_f.write(",".join(f"{x:.6g}" for x in row) + "\n")
            log_f.flush()
            print(
                f"it {iteration + 1} | {row[1]:.3f} h | wm {row[2]:.4f} e8 {row[3]:.4f} | "
                f"pi {row[4]:.3f} v {row[5]:.4f} imag {row[6]:.3f} surv {row[7]:.3f} | "
                f"sim {row[8]:.3f} ep {row[9]:.3f} term {term:.2f} | {row[11]:.2f} it/s",
                flush=True,
            )
            logged = now
            stats = {"r": [], "ep": [], "term": [], "done_n": []}
            agg = {key: [] for key in agg}
        if (iteration + 1) % cfg.ckpt_every == 0 or iteration + 1 == cfg.iterations:
            ckpt = {
                "wm": wm.state_dict(),
                "pi": policy.state_dict(),
                "critic": critic.state_dict(),
                "norms": norms.state_dict(),
                "opt_wm": opt_wm.state_dict(),
                "opt_pi": opt_pi.state_dict(),
                "opt_v": opt_v.state_dict(),
                "iteration": iteration + 1,
                "cfg": cfg.__dict__,
            }
            path = log_dir / f"ckpt_{iteration + 1:06d}.pt"
            torch.save(ckpt, path)
            torch.save(ckpt, log_dir / "last.pt")
    log_f.close()
