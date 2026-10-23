"""SuperMimic training loop: collect, fit the world model, BPTT the policy, fit the critic."""

from __future__ import annotations

import contextlib
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
    CONTROL_DT,
    GOAL_DIM,
    GOAL_FRAMES,
    HIST,
    INV_DIM,
    PRIV_DIM,
    NUM_J,
    REWARD_TERM_NAMES,
    REWARD_TERM_WEIGHT,
    XI_DIM,
    invariant_features,
    local_state_error,
    nominal_error,
    normalize_xi,
    unflatten_state,
)
from . import network
from .network import Critic, InputNorms, MLP, Policy, WorldModel, WorldModelState
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
    wm_lr: float = 2e-4
    pi_lr: float = 5e-4
    # Actor stays at 0 until the world model has a usable 8-step error (~200 iters).
    pi_warmup: int = 250
    v_lr: float = 2e-4
    gamma: float = 0.99
    lam: float = 0.95
    action_noise: float = 0.1
    critic_epochs: int = 5
    critic_minibatches: int = 4
    lr_decay_iters: int = 250_000
    wm_norm: str = "none"
    wm_residual: bool | float = 0.0
    stochastic_actor: bool = False
    # Encoder and decoder widths. Depth is the list length. Paper Table A.4 is four layers.
    policy_z_dim: int = 32
    policy_encoder: tuple[int, ...] = (2048, 1024, 512, 256)
    policy_decoder: tuple[int, ...] = (2048, 1024, 512, 256)
    # Encoder and decoder MLP norm: none, layer, or rms.
    policy_norm: str = "none"
    # ``split`` is the tanh encoder-decoder. ``trunk`` is one MLP on goal and history.
    policy_kind: str = "split"
    policy_trunk: tuple[int, ...] = (512, 512, 512)
    anchor: str = "torso"
    obs_terms: tuple[str, ...] = ("joint_pos", "joint_vel", "base_ang_vel", "projected_gravity", "last_action")
    # ``gaussian`` | ``cauchy`` | ``gaussian_linear``.
    reward_style: str = "gaussian"
    swanlab_project: str = "rethinking-supermimic"
    swanlab_mode: str = "offline"
    swanlab_experiment: str = "lafan1"
    device: str = "cuda"
    # None / <=0 disables. Grad norms are always logged.
    grad_clip_norm: float | None = None
    grad_clip_value: float | None = None  # policy only; None disables
    # Scale policy/critic rewards by 1/max(√Var(G), G_r_max/G_max). Stats from sim collect.
    # G_max = max_step_reward(style)/(1-γ) is fixed by the MDP.
    normalize_reward: bool = False
    # torch.compile the world-model window and one imagination step. False runs both eager.
    compile: bool = True
    # MLP matmuls. ``bf16`` casts back to fp32 before integration. ``none`` is fp32.
    amp: str = "bf16"
    # Reward FK. ``warp`` (default) or ``torch``.
    fk: str = "warp"
    # LSGAN generator weight added to the world-model MSE. 0 leaves MSE unchanged.
    adv_coef: float = 0.0
    # L1 reward-match weight added to the world-model MSE. 0 leaves MSE unchanged.
    reward_coef: float = 0.0
    log_every: int = 50
    ckpt_every: int = 2000
    seed: int = 0
    # Continue from this checkpoint. Missing modules (older runs) are left at init.
    resume: str | None = None
    # Checkpoint to load the world model from. null trains it from scratch.
    # Also loads the state, action, and ξ normalizers that model was trained with.
    wm_ckpt: str | None = None
    # frozen: the loaded model and those normalizers stay fixed.
    # tunable: keep training them. Ignored when wm_ckpt is null.
    wm_tune: str = "frozen"
    # adam | muon. Muon covers hidden 2D weights of the world model and the policy.
    # The first Linear, the output head, biases, and the critic stay Adam.
    optim: str = "muon"


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


def pi_lr_at(iteration: int, base: float, decay_iters: int, warmup: int) -> float:
    """Linear warmup from 0, then the same cosine decay as the other optimizers."""
    if warmup > 0 and iteration < warmup:
        return base * iteration / warmup
    return lr_at(iteration, base, decay_iters)


@contextmanager
def hold_out_net(network: nn.Module):
    """Hold ``network`` params out of the graph (TorchRL-style).

    Gradients still flow through the module w.r.t. its *inputs*, so the policy
    can BPTT through WM dynamics / critic ∂V/∂s without updating θ_WM / θ_V.
    """
    mode = next((p.requires_grad for p in network.parameters()), True)
    try:
        if mode:
            network.requires_grad_(False)
        yield
    finally:
        if mode:
            network.requires_grad_(True)


def _stack(xs: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(xs).reshape(-1, *xs[0].shape[1:])


def _flatten_time(states: list[tuple]) -> tuple:
    return tuple(torch.stack([s[i] for s in states]).reshape(-1, states[0][i].shape[-1]) for i in range(6))


def _unflatten_time(flat: tuple, steps: int) -> tuple:
    return tuple(t.view(steps, -1, t.shape[-1]).reshape(-1, t.shape[-1]) for t in flat)


def _mean(xs: list) -> float:
    """Host mean. CUDA tensors are reduced on-device and synced once here (log time)."""
    if not xs:
        return 0.0
    if isinstance(xs[0], torch.Tensor):
        return float(torch.stack([x.detach().reshape(()) for x in xs]).mean())
    return sum(xs) / len(xs)


def _term_score(
    name: str, value: torch.Tensor | float, style: str = "gaussian"
) -> torch.Tensor | float:
    """Normalize a reward term for logging.

    Tracking: value / (weight · dt), about (0, 1] at the peak. Action-rate and
    joint-limit are the paper penalties in every style, logged as the raw mean / dt
    (negative). Keeps tensors on-device; ``_mean`` syncs at log time.
    """
    if name in ("action_rate", "joint_limit"):
        return value / CONTROL_DT
    return value / (REWARD_TERM_WEIGHT[style][name] * CONTROL_DT)


def _grad_norm(params) -> float:
    grads = [p.grad.detach() for p in params if p.grad is not None]
    if not grads:
        return 0.0
    return float(nn.utils.get_total_norm(grads))


def _grads_norm(grads: list[torch.Tensor | None]) -> float:
    """Total norm of an ``autograd.grad`` tuple (skips unused ``None`` slots)."""
    used = [g.detach() for g in grads if g is not None]
    if not used:
        return 0.0
    return float(nn.utils.get_total_norm(used))


def _flat_grad(grads: list[torch.Tensor | None]) -> torch.Tensor:
    used = [g.detach().reshape(-1) for g in grads if g is not None]
    if not used:
        return torch.zeros(1)
    return torch.cat(used)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = a.norm() * b.norm()
    if float(denom) < 1e-12:
        return 0.0
    return float(torch.dot(a, b) / denom)


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
    """LSGAN transition discriminator on invariant features.

    Discriminates real vs predicted next-state features, conditioned on
    ``(inv(s), a, ξ)``. Labels: real → +1, fake → −1; the generator fools D
    toward +1 (Mao et al. LSGAN / AMP). The world-model objective is MSE plus
    ``adv_coef`` times the generator loss. ``adv_coef=0`` leaves MSE unchanged.
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


class RewardL1Loss(nn.Module):
    """L1 match of analytic rewards between predicted and true next states.

    State components have different units and scales. This term ignores that and
    trains the world model so the predicted state scores the same reward as the
    true next state, at the same reference frame and actions. Gradients flow only
    through the predicted state. The true-state reward is always detached.
    """

    def __init__(self, mdp):
        super().__init__()
        self.mdp = mdp

    def forward(self, preds: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """``preds`` is ``(K, B, STATE)`` from one world-model window."""
        steps, batch_size = preds.shape[0], preds.shape[1]
        pred = preds.reshape(steps * batch_size, -1)
        true = batch["s1"].transpose(0, 1).reshape(steps * batch_size, -1)
        frame = batch["f"].transpose(0, 1).reshape(steps * batch_size)
        last = self.mdp.motion.clip_end[self.mdp.motion.clip_id[frame]] - 1
        frame = torch.minimum(frame + 1, last)
        action = batch["a"].transpose(0, 1).reshape(steps * batch_size, -1)
        a_prev = batch["a_prev"].transpose(0, 1).reshape(steps * batch_size, -1)
        reward_pred = self.mdp.reward(*unflatten_state(pred), frame, action, a_prev)
        with torch.no_grad():
            reward_true = self.mdp.reward(*unflatten_state(true), frame, action, a_prev)
        return (reward_pred - reward_true).abs().mean()


def world_model_window(
    wm: WorldModel,
    norms: InputNorms,
    s: torch.Tensor,
    s1: torch.Tensor,
    actions: torch.Tensor,
    xi: torch.Tensor,
    detach_pred: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Autoregressive WM window compiled as one graph.

    Returns MSE errors ``(B, K, D)``, detached start states, and predicted states
    ``(K, B, STATE)``. Predicted states stay on the graph when ``detach_pred`` is
    false so the generator loss can train the world model. The K loop is unrolled.
    """
    state = WorldModelState(*unflatten_state(s[:, 0]))
    xi_n = normalize_xi(xi[:, 0])
    errors = []
    starts = []
    preds = []
    for j in range(actions.shape[1]):
        starts.append(torch.cat(state, dim=-1))
        state = wm.step(state, actions[:, j], xi_n, norms)
        true = unflatten_state(s1[:, j])
        errors.append(local_state_error(*true, *state))
        preds.append(torch.cat(state, dim=-1))
    pred = torch.stack(preds, dim=0)
    if detach_pred:
        pred = pred.detach()
    return torch.stack(errors, dim=1), torch.stack(starts, dim=0).detach(), pred


def imagine_step(
    policy: Policy,
    wm: WorldModel,
    norms: InputNorms,
    mdp,
    obs_hist: torch.Tensor,
    goal: torch.Tensor,
    p: torch.Tensor,
    quat: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    q: torch.Tensor,
    qd: torch.Tensor,
    a_prev: torch.Tensor,
    xi_n: torch.Tensor,
    xi: torch.Tensor,
    frame: torch.Tensor,
    deterministic: bool,
):
    """One imagination step. This is the compiled policy graph; the horizon loop stays eager."""
    action = policy(obs_hist, goal, norms, deterministic=deterministic)
    p, quat, v, w, q, qd = wm.step(WorldModelState(p, quat, v, w, q, qd), action, xi_n, norms)
    frame = frame + 1
    goal = mdp.motion.goal(frame, GOAL_FRAMES)
    obs = mdp.add_noise(mdp.obs(p, quat, v, w, q, qd, action, xi, frame))
    obs_hist = torch.cat((obs_hist[:, 1:], obs.unsqueeze(1)), dim=1)
    return obs_hist, goal, p, quat, v, w, q, qd, frame, action


def imagine_horizon(
    policy: Policy,
    wm: WorldModel,
    critic: Critic,
    norms: InputNorms,
    mdp,
    hist_s: torch.Tensor,
    hist_a_prev: torch.Tensor,
    hist_xi: torch.Tensor,
    hist_f: torch.Tensor,
    s: torch.Tensor,
    a_prev: torch.Tensor,
    xi: torch.Tensor,
    frame: torch.Tensor,
    reward_scale: torch.Tensor,
    horizon: int,
    gamma: float,
    deterministic: bool,
    step=imagine_step,
):
    """H-step imagination. ``step`` may be a compiled single step.

    Returns the policy loss and the tensors the critic and the logs need.
    ``reward_scale`` is 1 when reward normalization is off.
    """
    batch = s.shape[0]
    device = s.device
    p, quat, v, w, q, qd = unflatten_state(hist_s)
    hist_n = p.shape[0] * p.shape[1]
    obs = mdp.obs(
        p.reshape(hist_n, 3),
        quat.reshape(hist_n, 4),
        v.reshape(hist_n, 3),
        w.reshape(hist_n, 3),
        q.reshape(hist_n, -1),
        qd.reshape(hist_n, -1),
        hist_a_prev.reshape(hist_n, -1),
        hist_xi.reshape(hist_n, -1),
        hist_f.reshape(hist_n),
    )
    obs_hist = mdp.add_noise(obs).view(batch, HIST, -1)
    p, quat, v, w, q, qd = unflatten_state(s)
    xi_n = normalize_xi(xi)
    goal = mdp.motion.goal(frame, GOAL_FRAMES)
    states = [(p, quat, v, w, q, qd)]
    frames = [frame]
    goals = [goal]
    actions = []
    prev_actions = []
    for _ in range(horizon):
        prev_actions.append(a_prev)
        obs_hist, goal, p, quat, v, w, q, qd, frame, action = step(
            policy,
            wm,
            norms,
            mdp,
            obs_hist,
            goal,
            p,
            quat,
            v,
            w,
            q,
            qd,
            a_prev,
            xi_n,
            xi,
            frame,
            deterministic,
        )
        actions.append(action)
        a_prev = action
        states.append((p, quat, v, w, q, qd))
        frames.append(frame)
        goals.append(goal)
    flat = _flatten_time(states[1:])
    reward, term_parts = mdp.reward(
        *flat, _stack(frames[1:]), _stack(actions), _stack(prev_actions), terms=True
    )
    done = mdp.soft_termination(flat[0], flat[1], _stack(frames[1:]))
    reward = reward.view(horizon, batch)
    done = done.view(horizon, batch)
    term_raw = torch.stack([term_parts[name] for name in REWARD_TERM_NAMES]).view(len(REWARD_TERM_NAMES), horizon, batch)
    term_mean = term_raw.detach().mean(dim=(1, 2))
    reward_mean = reward.detach().mean()
    reward = reward / reward_scale
    term_r = term_raw / reward_scale
    alive = torch.cumprod(1.0 - done, dim=0)
    survival = torch.cat((torch.ones(1, batch, device=device), alive[:-1]), dim=0)
    discount = gamma ** torch.arange(horizon, device=device, dtype=torch.float32)
    weighted = discount[:, None] * survival
    step_return = weighted * reward
    returns = step_return.sum(0)
    bootstrap = gamma**horizon * alive[-1] * critic(
        norms.critic(
            invariant_features(*states[-1]),
            goals[-1],
            xi_n,
            mdp.privileged(*states[-1], frames[-1]),
        )
    )
    term_loss = -(weighted * term_r).sum(dim=1).mean(dim=-1)
    policy_loss = -(returns + bootstrap).mean()
    for action in actions:
        if action.requires_grad:
            action.retain_grad()
    critic_states = tuple(t.detach() for t in _flatten_time(states))
    return (
        policy_loss,
        step_return,
        reward,
        weighted.unsqueeze(0) * term_r,
        bootstrap,
        term_loss,
        reward.detach(),
        done.detach(),
        alive[-1].detach(),
        reward_mean.detach(),
        term_mean,
        critic_states,
        _stack(frames).detach(),
        _stack(goals).detach(),
        xi.detach(),
        actions,
        survival.detach().mean(-1),
    )


class Trainer:
    """Owns the simulator, buffer, networks, and the Algorithm A.1 loop."""

    def __init__(self, cfg: TrainConfig, log_dir: Path | str):
        from .kinematics import compile_g1, make_kinematics
        from .motion import MotionLib
        from .sim import TrackingSim

        self.cfg = cfg
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.device = cfg.device

        torch.manual_seed(cfg.seed)
        if cfg.amp not in ("none", "bf16"):
            raise ValueError(f"amp must be 'none' or 'bf16', got {cfg.amp!r}")
        network.AMP_DTYPE = torch.bfloat16 if cfg.amp == "bf16" else None
        # bf16 GEMMs do not use TF32. Compiled graphs donate buffers, which forbids
        # the retain_graph backwards used by the reward-grad diagnostic.
        torch.backends.cuda.matmul.allow_tf32 = cfg.amp == "none"
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high" if cfg.amp == "none" else "highest")
        if cfg.compile:
            import torch._functorch.config as functorch_config

            functorch_config.donated_buffer = False

        self.motion = MotionLib.load(device=self.device)
        model, names = compile_g1()
        self.kin = make_kinematics(model, names, cfg.fk, self.device)
        self.sim = TrackingSim(
            cfg.num_envs,
            self.motion,
            self.kin,
            self.device,
            gamma=cfg.gamma,
            anchor=cfg.anchor,
            obs_terms=cfg.obs_terms,
            reward_style=cfg.reward_style,
        )
        self.r_max = self.sim.mdp.max_step_reward
        obs_dim = self.sim.mdp.obs_dim
        self.wm = WorldModel(cfg.wm_norm, cfg.wm_residual).to(self.device)
        # The world-model window is one graph. The policy compiles a single imagination
        # step (mode="default"); the horizon loop stays eager. Both are eager when compile is off.
        if cfg.compile:
            self._wm_window = torch.compile(world_model_window, dynamic=False)
            self._imagine_step = torch.compile(imagine_step, mode="default", dynamic=False)
        else:
            self._wm_window = world_model_window
            self._imagine_step = imagine_step
        self.policy = Policy(
            obs_dim,
            GOAL_DIM,
            HIST,
            z_dim=cfg.policy_z_dim,
            stochastic=cfg.stochastic_actor,
            encoder_hidden=cfg.policy_encoder,
            decoder_hidden=cfg.policy_decoder,
            trunk_hidden=cfg.policy_trunk,
            norm=cfg.policy_norm,
            kind=cfg.policy_kind,
        ).to(self.device)
        self.critic = Critic(INV_DIM + GOAL_DIM + XI_DIM + PRIV_DIM).to(self.device)
        self.norms = InputNorms(obs_dim).to(self.device)
        from .optim import build_optimizer

        self.opt_wm = build_optimizer(self.wm, cfg.wm_lr, cfg.optim)
        self.opt_pi = build_optimizer(self.policy, cfg.pi_lr, cfg.optim, adam_betas=(0.9, 0.95))
        self.opt_v = torch.optim.Adam(self.critic.parameters(), lr=cfg.v_lr)
        self.buffer = ReplayBuffer(cfg.buffer_steps, cfg.num_envs, self.device)
        self.mse_loss = MSELoss(self.device).to(self.device)
        self.reward_l1 = RewardL1Loss(self.sim.mdp)
        # Discriminator probe only — never added into the WM objective.
        self.adv_loss = AdversarialLoss().to(self.device)
        self.opt_disc = torch.optim.Adam(self.adv_loss.parameters(), lr=cfg.wm_lr)
        self.reward_norm = (
            RewardNormalizer(
                cfg.num_envs,
                cfg.gamma,
                g_max=self.r_max / (1.0 - cfg.gamma),
                device=self.device,
            )
            if cfg.normalize_reward
            else None
        )
        self.obs_hist = torch.zeros(cfg.num_envs, HIST, obs_dim, device=self.device)

        self.stats = {
            "r": [],
            "ep": [],
            "term": [],
            "done_n": [],
            "sim_terms": {name: [] for name in REWARD_TERM_NAMES},
            "ep_terms": {name: [] for name in REWARD_TERM_NAMES},
        }
        self.agg = {
            "wm": [],
            "wm_mse": [],
            "wm_adv": [],
            "wm_reward": [],
            "err": [],
            "pi": [],
            "v": [],
            "score": [],
            "surv": [],
            "gn_wm": [],
            "gn_pi": [],
            "gn_v": [],
            "disc": [],
            "disc_real": [],
            "disc_fake": [],
            "disc_acc": [],
            "imag_terms": {name: [] for name in REWARD_TERM_NAMES},
            "reward_grads": {name: [] for name in REWARD_TERM_NAMES},
            "reward_grad_boot": [],
        }
        self.timing = {"collect": [], "wm": [], "imagine": [], "policy": [], "critic": []}
        self.step_acc = 0.0
        self.started = 0.0
        self.logged = 0.0
        self.start_iteration = 0
        self.wm_frozen = False
        self._log_f = None
        if cfg.resume:
            self.load_checkpoint(cfg.resume)
        if cfg.wm_ckpt:
            self.load_pretrained_wm(cfg.wm_ckpt)

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
        term_cols = ",".join(
            f"sim_{n},ep_{n},imag_{n}" for n in REWARD_TERM_NAMES
        )
        grad_r_cols = ",".join(f"grad_r_{n}" for n in REWARD_TERM_NAMES)
        self._log_f.write(
            "iteration,wall_h,wm_loss,wm_err8,pi_loss,v_loss,imag_score,imag_survival,"
            "sim_score,ep_score,term_frac,it_per_s,lr,pi_lr,entropy,"
            f"{term_cols},"
            f"{grad_r_cols},grad_r_boot,"
            "disc_loss,disc_real,disc_fake,disc_acc,wm_loss_mse,wm_loss_adv,wm_loss_reward,"
            "grad_wm,grad_pi,grad_v,"
            "t_collect,t_wm,t_imagine,t_policy,t_critic\n"
        )
        self.started = time.time()
        self.logged = self.started
        try:
            for iteration in range(self.start_iteration, cfg.iterations):
                self._step(iteration)
        finally:
            swanlab.finish()
            if self._log_f is not None:
                self._log_f.close()

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

        with timed(self.device):
            wm_loss, err8, gn_wm, disc_stats = self.update_world_model()
        self.timing["wm"].append(timed.last)

        # Freeze WM/V params through imagine + policy backward (not torch.no_grad:
        # π must still differentiate through dynamics and ∂V/∂s).
        diagnose_reward = (iteration + 1) % cfg.log_every == 0
        with hold_out_net(self.wm), hold_out_net(self.critic):
            with timed(self.device):
                start = self.sample_policy_starts()
                pi_loss, critic_batch, imag_score, survival, imag_terms, term_losses = self.imagine(
                    start
                )
            self.timing["imagine"].append(timed.last)

            with timed(self.device):
                gn_pi = self.update_policy(pi_loss, term_losses, diagnose_reward=diagnose_reward)
            self.timing["policy"].append(timed.last)

        with timed(self.device):
            v_loss, gn_v = self.update_critic(critic_batch)
        self.timing["critic"].append(timed.last)

        # Keep CUDA scalars on-device; ``_mean`` syncs once at log time.
        # Keep CUDA scalars on-device; ``_mean`` syncs once at log time.
        self.agg["wm"].append(wm_loss.detach())
        self.agg["wm_mse"].append(disc_stats["mse"])
        self.agg["wm_adv"].append(disc_stats["adv"])
        self.agg["wm_reward"].append(disc_stats["reward"])
        self.agg["err"].append(err8.detach())
        self.agg["pi"].append(pi_loss.detach())
        self.agg["v"].append(v_loss.detach() if isinstance(v_loss, torch.Tensor) else v_loss)
        self.agg["score"].append(imag_score.detach() if isinstance(imag_score, torch.Tensor) else imag_score)
        self.agg["surv"].append(survival.detach() if isinstance(survival, torch.Tensor) else survival)
        self.agg["gn_wm"].append(gn_wm)
        self.agg["gn_pi"].append(gn_pi)
        self.agg["gn_v"].append(gn_v)
        self.agg["disc"].append(disc_stats["loss"])
        self.agg["disc_real"].append(disc_stats["real"])
        self.agg["disc_fake"].append(disc_stats["fake"])
        self.agg["disc_acc"].append(disc_stats["acc"])
        for name, value in imag_terms.items():
            self.agg["imag_terms"][name].append(value)

        if (iteration + 1) % cfg.log_every == 0:
            self._log(iteration + 1, lr, pi_lr)
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
            self.norms.obs.update(obs)
            self.norms.goal.update(goal)
            # A frozen world model keeps the normalizers it was loaded with.
            if not self.wm_frozen:
                self.norms.state.update(invariant_features(p, quat, v, w, q, qd))
                self.norms.xi.update(xi_n)
            with torch.no_grad():
                action = self.policy(
                    self.obs_hist, goal, self.norms, deterministic=not cfg.stochastic_actor
                )
            executed = action + cfg.action_noise * torch.randn_like(action)
            if not self.wm_frozen:
                self.norms.action.update(executed)
            out = self.sim.step(executed)
            self.buffer.add(out)
            if self.reward_norm is not None:
                self.reward_norm.update(out["r"], out["done"])
            style = self.cfg.reward_style
            self.stats["r"].append((out["r"].mean() / self.r_max).detach())
            for name, value in out["terms"].items():
                self.stats["sim_terms"][name].append(
                    _term_score(name, value.mean().detach(), style)
                )
            done = out["done"]
            # One intentional sync for the Python branch.
            if bool(done.any().item()):
                length = out["ep_len"][done].float().clamp(min=1)
                self.stats["ep"].append(
                    (out["ep_ret"][done] / (self.r_max * length)).mean().detach()
                )
                for name, value in out["ep_term"].items():
                    self.stats["ep_terms"][name].append(
                        _term_score(name, (value[done] / length).mean().detach(), style)
                    )
                # Already synced above; host floats are fine for rare episode counts.
                self.stats["term"].append(float(out["d"][done].float().sum()))
                self.stats["done_n"].append(float(done.sum()))

    def update_world_model(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, float, dict[str, torch.Tensor]]:
        """Fit the WM on MSE plus optional adversarial and reward-L1 terms, then step D."""
        cfg = self.cfg
        loss = err8 = None
        norms: list[float] = []
        mse_losses: list[torch.Tensor] = []
        adv_losses: list[torch.Tensor] = []
        reward_losses: list[torch.Tensor] = []
        disc_losses: list[torch.Tensor] = []
        disc_reals: list[torch.Tensor] = []
        disc_fakes: list[torch.Tensor] = []
        disc_accs: list[torch.Tensor] = []
        use_adv = cfg.adv_coef > 0 and not self.wm_frozen
        use_reward = cfg.reward_coef > 0 and not self.wm_frozen
        for _ in range(cfg.wm_batches):
            batch = self.buffer.sample_windows(cfg.wm_batch, cfg.wm_window)
            grad_ctx = torch.no_grad() if self.wm_frozen else contextlib.nullcontext()
            with grad_ctx:
                errors, saved, preds = self.world_model_rollout(
                    batch, detach_pred=not (use_adv or use_reward)
                )
                mse, err8 = self.mse_loss(errors)
            if use_reward:
                reward_l1 = self.reward_l1(preds, batch)
            else:
                with torch.no_grad():
                    reward_l1 = self.reward_l1(preds.detach(), batch)
            self.opt_wm.zero_grad(set_to_none=True)
            loss = mse
            if use_adv:
                # D is held fixed so the generator step does not write θ_D grads.
                with hold_out_net(self.adv_loss):
                    adv = self._generator_loss(saved, preds)
                    loss = loss + cfg.adv_coef * adv
            else:
                adv = mse.new_zeros(())
            if use_reward:
                loss = loss + cfg.reward_coef * reward_l1
            if self.wm_frozen:
                norms.append(0.0)
            else:
                loss.backward()
                norms.append(_clip_grads(self.wm.parameters(), cfg.grad_clip_norm))
                self.opt_wm.step()
            mse_losses.append(mse.detach())
            reward_losses.append(reward_l1.detach())

            # Features for D are detached. With coef 0 this is also the logged generator loss.
            d_loss, d_real, d_fake, d_acc, adv_det = self.update_discriminator(saved)
            adv_losses.append(adv.detach() if use_adv else adv_det)
            disc_losses.append(d_loss)
            disc_reals.append(d_real)
            disc_fakes.append(d_fake)
            disc_accs.append(d_acc)
        assert loss is not None and err8 is not None
        mse_mean = torch.stack(mse_losses).mean()
        adv_mean = torch.stack(adv_losses).mean()
        reward_mean = torch.stack(reward_losses).mean()
        disc_stats = {
            "loss": torch.stack(disc_losses).mean(),
            "real": torch.stack(disc_reals).mean(),
            "fake": torch.stack(disc_fakes).mean(),
            "acc": torch.stack(disc_accs).mean(),
            "mse": mse_mean,
            "adv": adv_mean,
            "reward": reward_mean,
        }
        total = mse_mean + cfg.adv_coef * adv_mean + cfg.reward_coef * reward_mean
        return total, err8, _mean(norms), disc_stats

    def update_discriminator(
        self, saved: list[dict[str, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One LSGAN D step. Also returns the detached generator loss for logging."""
        with torch.no_grad():
            inv_s = torch.cat([invariant_features(*s["s"]) for s in saved])
            inv_pred = torch.cat([invariant_features(*s["s_pred"]) for s in saved])
            inv_true = torch.cat([invariant_features(*s["s_true"]) for s in saved])
        action = torch.cat([s["action"] for s in saved])
        xi_n = torch.cat([s["xi_n"] for s in saved])
        real = self.adv_loss.score(inv_s, action, xi_n, inv_true)
        fake = self.adv_loss.score(inv_s, action, xi_n, inv_pred)
        d_loss = self.adv_loss.discriminator_loss(real, fake)
        self.opt_disc.zero_grad(set_to_none=True)
        d_loss.backward()
        _clip_grads(self.adv_loss.parameters(), self.cfg.grad_clip_norm)
        self.opt_disc.step()
        # Accuracy under the LSGAN sign convention: real > 0, fake < 0.
        with torch.no_grad():
            acc = 0.5 * ((real > 0).float().mean() + (fake < 0).float().mean())
            adv = self.adv_loss.generator_loss(fake.detach())
        return d_loss.detach(), real.detach().mean(), fake.detach().mean(), acc.detach(), adv.detach()

    def _generator_loss(self, saved: list[dict[str, torch.Tensor]], preds: torch.Tensor) -> torch.Tensor:
        """LSGAN generator loss through predicted states. Caller holds D's parameters fixed."""
        inv_s = []
        inv_pred = []
        for j, step in enumerate(saved):
            start = step["s"] if j == 0 else unflatten_state(preds[j - 1])
            inv_s.append(invariant_features(*start))
            inv_pred.append(invariant_features(*unflatten_state(preds[j])))
        fake = self.adv_loss.score(
            torch.cat(inv_s),
            torch.cat([step["action"] for step in saved]),
            torch.cat([step["xi_n"] for step in saved]),
            torch.cat(inv_pred),
        )
        return self.adv_loss.generator_loss(fake)

    def world_model_rollout(
        self, batch: dict[str, torch.Tensor], detach_pred: bool = True
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]], torch.Tensor]:
        """Autoregressive WM window.

        Returns MSE errors, detached states for D, and predicted states. Predicted
        states require grad only when ``detach_pred`` is false.
        """
        errors, starts, preds = self._wm_window(
            self.wm,
            self.norms,
            batch["s"],
            batch["s1"],
            batch["a"],
            batch["xi"],
            detach_pred,
        )
        xi_n = normalize_xi(batch["xi"][:, 0]).detach()
        saved = []
        for j in range(starts.shape[0]):
            true = unflatten_state(batch["s1"][:, j])
            saved.append(
                {
                    "s": unflatten_state(starts[j]),
                    "s_pred": unflatten_state(preds[j].detach()),
                    "s_true": tuple(t.detach() for t in true),
                    "action": batch["a"][:, j].detach(),
                    "xi_n": xi_n,
                }
            )
        return errors, saved, preds

    def world_model_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        errors, _saved, _preds = self.world_model_rollout(batch)
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

    def _imagine_graph(self, start: dict[str, torch.Tensor]) -> dict:
        """Differentiable imagination rollout and return pieces (shared by train / diagnostics)."""
        cfg = self.cfg
        scale = (
            self.reward_norm.scale().detach()
            if self.reward_norm is not None
            else start["s"].new_ones(())
        )
        (
            policy_loss,
            step_return,
            step_reward,
            term_weighted,
            bootstrap,
            term_loss,
            reward,
            done,
            alive_last,
            reward_mean,
            term_mean,
            critic_states,
            frames,
            goals,
            xi,
            actions,
            survival,
        ) = imagine_horizon(
            self.policy,
            self.wm,
            self.critic,
            self.norms,
            self.sim.mdp,
            start["hist_s"],
            start["hist_a_prev"],
            start["hist_xi"],
            start["hist_f"],
            start["s"],
            start["a_prev"],
            start["xi"],
            start["f"],
            scale,
            cfg.horizon,
            cfg.gamma,
            not cfg.stochastic_actor,
            self._imagine_step,
        )
        term_losses = {name: term_loss[i] for i, name in enumerate(REWARD_TERM_NAMES)}
        term_losses["bootstrap"] = -bootstrap.mean()
        imag_terms = {
            name: _term_score(name, term_mean[i], cfg.reward_style) for i, name in enumerate(REWARD_TERM_NAMES)
        }
        critic_batch = {
            "states": critic_states,
            "frames": frames,
            "goals": goals,
            "xi": xi.unsqueeze(0).expand(cfg.horizon + 1, -1, -1).reshape(-1, xi.shape[-1]),
            "reward": reward,
            "done": done,
        }
        return {
            "policy_loss": policy_loss,
            "critic_batch": critic_batch,
            "raw_score": reward_mean / self.r_max,
            "survival_last": alive_last.mean(),
            "imag_terms": imag_terms,
            "term_losses": term_losses,
            "step_return": step_return,
            "step_reward": step_reward,
            "term_weighted": term_weighted,
            "bootstrap": bootstrap,
            "actions": actions,
            "survival": survival,
            "horizon": cfg.horizon,
        }

    def imagine(self, start: dict[str, torch.Tensor]):
        """One differentiable rollout. Returns the policy loss and detached critic data."""
        g = self._imagine_graph(start)
        return (
            g["policy_loss"],
            g["critic_batch"],
            g["raw_score"],
            g["survival_last"],
            g["imag_terms"],
            g["term_losses"],
        )

    def diagnose_imagination_step_grads(
        self, start: dict[str, torch.Tensor] | None = None
    ) -> dict[str, list[float] | float]:
        """Actor-gradient diagnostics along one imagination horizon.

        Not called from training. Each scalar backward retains the BPTT graph.

        ``marginal`` / ``raw`` are ``||∇_θ r_t||`` with and without ``γ^t S_t``.
        ``cosine`` is the alignment of that weighted step with the full update.
        ``encoder`` / ``decoder`` split the weighted step. ``action`` is
        ``||∂G/∂a_t||`` of the full return. ``terms`` is per reward term.
        """
        if start is None:
            start = self.sample_policy_starts()
        if self.policy.encoder is None:
            encoder = []
            decoder = [p for p in self.policy.trunk.parameters() if p.requires_grad]
        else:
            encoder = [p for p in self.policy.encoder.parameters() if p.requires_grad]
            decoder = [p for p in self.policy.decoder.parameters() if p.requires_grad]
        extra = [self.policy.log_std] if self.policy.log_std is not None else []
        params = encoder + decoder + extra
        n_enc = len(encoder)
        with hold_out_net(self.wm), hold_out_net(self.critic):
            g = self._imagine_graph(start)
            step_return = g["step_return"]
            step_reward = g["step_reward"]
            term_weighted = g["term_weighted"]
            bootstrap = g["bootstrap"]
            horizon = int(g["horizon"])

            def grads_of(loss: torch.Tensor) -> list[torch.Tensor | None]:
                return list(torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True))

            full = grads_of(g["policy_loss"])
            full_flat = _flat_grad(full)
            full_norm = _grads_norm(full)
            reward_only = grads_of(-step_return.sum(0).mean())
            boot = grads_of(-bootstrap.mean())
            boot_norm = _grads_norm(boot)
            boot_cos = _cosine(_flat_grad(boot), _flat_grad(reward_only))

            marginal, raw, cosine, enc_norm, dec_norm = [], [], [], [], []
            for t in range(horizon):
                weighted = grads_of(-step_return[t].mean())
                unweighted = grads_of(-step_reward[t].mean())
                marginal.append(_grads_norm(weighted))
                raw.append(_grads_norm(unweighted))
                cosine.append(_cosine(_flat_grad(weighted), full_flat))
                enc_norm.append(_grads_norm(weighted[:n_enc]))
                dec_norm.append(_grads_norm(weighted[n_enc : n_enc + len(decoder)]))
            terms = {name: [] for name in REWARD_TERM_NAMES}
            for index, name in enumerate(REWARD_TERM_NAMES):
                for t in range(horizon):
                    terms[name].append(_grads_norm(grads_of(-term_weighted[index, t].mean())))
            g["policy_loss"].backward()
            action_norm = [float(action.grad.detach().norm()) for action in g["actions"]]
            survival = [float(x) for x in g["survival"].detach().cpu()]
        return {
            "marginal": marginal,
            "raw": raw,
            "cosine": cosine,
            "encoder": enc_norm,
            "decoder": dec_norm,
            "action": action_norm,
            "terms": terms,
            "survival": survival,
            "bootstrap": boot_norm,
            "bootstrap_cosine": boot_cos,
            "full": full_norm,
            "reward": _grads_norm(reward_only),
        }

    def update_policy(
        self,
        pi_loss: torch.Tensor,
        term_losses: dict[str, torch.Tensor] | None = None,
        diagnose_reward: bool = False,
    ) -> float:
        """Policy step. On log iterations, also measure ||∇_θ L_k|| per reward term."""
        cfg = self.cfg
        params = [p for p in self.policy.parameters() if p.requires_grad]
        if diagnose_reward and term_losses is not None:
            for name in REWARD_TERM_NAMES:
                grads = torch.autograd.grad(
                    term_losses[name], params, retain_graph=True, allow_unused=True
                )
                self.agg["reward_grads"][name].append(_grads_norm(list(grads)))
            boot_grads = torch.autograd.grad(
                term_losses["bootstrap"], params, retain_graph=True, allow_unused=True
            )
            self.agg["reward_grad_boot"].append(_grads_norm(list(boot_grads)))
        self.opt_pi.zero_grad(set_to_none=True)
        pi_loss.backward()
        grad_norm = _clip_grads(params, cfg.grad_clip_norm, cfg.grad_clip_value)
        if math.isfinite(grad_norm):
            self.opt_pi.step()
        return grad_norm

    def update_critic(self, batch: dict) -> tuple[torch.Tensor, float]:
        cfg = self.cfg
        horizon, n = batch["reward"].shape
        flat = _unflatten_time(batch["states"], horizon + 1)
        priv = self.sim.mdp.privileged(*flat, batch["frames"])
        self.norms.priv.update(priv.detach())
        features = self.norms.critic(
            invariant_features(*flat), batch["goals"], normalize_xi(batch["xi"]), priv
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

    def _log(self, iteration: int, lr: float, pi_lr: float) -> None:
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
        sim_terms = {name: _mean(self.stats["sim_terms"][name]) for name in REWARD_TERM_NAMES}
        ep_terms = {name: _mean(self.stats["ep_terms"][name]) for name in REWARD_TERM_NAMES}
        imag_terms = {name: _mean(self.agg["imag_terms"][name]) for name in REWARD_TERM_NAMES}
        reward_grads = {name: _mean(self.agg["reward_grads"][name]) for name in REWARD_TERM_NAMES}
        reward_grad_boot = _mean(self.agg["reward_grad_boot"])
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
            pi_lr,
            entropy,
        ]
        for name in REWARD_TERM_NAMES:
            row.extend((sim_terms[name], ep_terms[name], imag_terms[name]))
        for name in REWARD_TERM_NAMES:
            row.append(reward_grads[name])
        row.append(reward_grad_boot)
        disc_loss = _mean(self.agg["disc"])
        disc_real = _mean(self.agg["disc_real"])
        disc_fake = _mean(self.agg["disc_fake"])
        disc_acc = _mean(self.agg["disc_acc"])
        wm_mse = _mean(self.agg["wm_mse"])
        wm_adv = _mean(self.agg["wm_adv"])
        wm_reward = _mean(self.agg["wm_reward"])
        row.extend((disc_loss, disc_real, disc_fake, disc_acc, wm_mse, wm_adv, wm_reward))
        row.extend(
            (
                gn_wm,
                gn_pi,
                gn_v,
                seconds["collect"],
                seconds["wm"],
                seconds["imagine"],
                seconds["policy"],
                seconds["critic"],
            )
        )
        assert self._log_f is not None
        self._log_f.write(",".join(f"{x:.6g}" for x in row) + "\n")
        self._log_f.flush()
        metrics = {
            "train/wm_loss": row[2],
            "train/wm_loss_mse": wm_mse,
            "train/wm_loss_adv": wm_adv,
            "train/wm_loss_reward": wm_reward,
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
            "train/pi_lr": pi_lr,
            "grad/wm": gn_wm,
            "grad/pi": gn_pi,
            "grad/v": gn_v,
            "time/collect_s": seconds["collect"],
            "time/wm_s": seconds["wm"],
            "time/imagine_s": seconds["imagine"],
            "time/policy_s": seconds["policy"],
            "time/critic_s": seconds["critic"],
        }
        for name in REWARD_TERM_NAMES:
            metrics[f"score/sim_{name}"] = sim_terms[name]
            metrics[f"score/ep_{name}"] = ep_terms[name]
            metrics[f"score/imag_{name}"] = imag_terms[name]
            metrics[f"grad/reward_{name}"] = reward_grads[name]
        metrics["grad/reward_bootstrap"] = reward_grad_boot
        metrics["train/disc_loss"] = disc_loss
        metrics["train/disc_real"] = disc_real
        metrics["train/disc_fake"] = disc_fake
        metrics["train/disc_acc"] = disc_acc
        if math.isfinite(entropy):
            metrics["train/entropy"] = entropy
        if self.reward_norm is not None:
            metrics["train/reward_scale"] = float(self.reward_norm.scale())
        swanlab.log(metrics, step=iteration)
        ent_s = f" H {entropy:.3f} |" if math.isfinite(entropy) else ""
        track = " ".join(
            f"{n[0]}{sim_terms[n]:.2f}"
            for n in ("body_pos", "body_ori", "body_lin_vel", "body_ang_vel", "root_pos", "root_ori")
        )
        # Share of Σ_k ||∇L_k|| (bootstrap excluded) — flags saturated exp terms (~0).
        grad_sum = sum(reward_grads.values()) + 1e-12
        grad_share = " ".join(
            f"{n[:3]}{reward_grads[n] / grad_sum:.2f}"
            for n in ("body_pos", "body_ori", "body_lin_vel", "body_ang_vel", "root_pos", "root_ori")
        )
        print(
            f"it {iteration} | {row[1]:.3f} h | wm {row[2]:.4f} mse {wm_mse:.4f} adv {wm_adv:.4f} r {wm_reward:.4f} e8 {row[3]:.4f} | "
            f"pi {row[4]:.3f} v {row[5]:.4f} imag {row[6]:.3f} surv {row[7]:.3f} | "
            f"sim {row[8]:.3f} ep {row[9]:.3f} term {term:.2f} | "
            f"[{track}] rate {sim_terms['action_rate']:.3g} lim {sim_terms['joint_limit']:.3g} |"
            f"{ent_s} {row[11]:.2f} it/s | pi_lr {pi_lr:.2e} | "
            f"gn wm {gn_wm:.3g} pi {gn_pi:.3g} v {gn_v:.3g} | "
            f"∇r [{grad_share}] boot {reward_grad_boot:.3g} | "
            f"D loss {disc_loss:.3f} real {disc_real:.2f} fake {disc_fake:.2f} acc {disc_acc:.2f} | "
            f"collect {seconds['collect']:.3f}s wm {seconds['wm']:.3f}s "
            f"imagine {seconds['imagine']:.3f}s policy {seconds['policy']:.3f}s critic {seconds['critic']:.3f}s",
            flush=True,
        )
        self.logged = now
        self.stats = {
            "r": [],
            "ep": [],
            "term": [],
            "done_n": [],
            "sim_terms": {name: [] for name in REWARD_TERM_NAMES},
            "ep_terms": {name: [] for name in REWARD_TERM_NAMES},
        }
        self.agg = {
            "wm": [],
            "wm_mse": [],
            "wm_adv": [],
            "wm_reward": [],
            "err": [],
            "pi": [],
            "v": [],
            "score": [],
            "surv": [],
            "gn_wm": [],
            "gn_pi": [],
            "gn_v": [],
            "disc": [],
            "disc_real": [],
            "disc_fake": [],
            "disc_acc": [],
            "imag_terms": {name: [] for name in REWARD_TERM_NAMES},
            "reward_grads": {name: [] for name in REWARD_TERM_NAMES},
            "reward_grad_boot": [],
        }
        self.timing = {key: [] for key in self.timing}

    def load_pretrained_wm(self, path: str | Path) -> None:
        """Load a world model and the normalizers it reads. Policy and critic stay fresh."""
        if self.cfg.wm_tune not in ("frozen", "tunable"):
            raise ValueError(f"wm_tune must be 'frozen' or 'tunable', got {self.cfg.wm_tune!r}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if "wm" not in ckpt:
            raise KeyError(f"{path} has no world model ('wm').")
        self.wm.load_state_dict(ckpt["wm"])
        norms = ckpt.get("norms")
        if norms is not None:
            own = self.norms.state_dict()
            kept = [
                key
                for key in own
                if key.startswith(("state.", "action.", "xi.")) and key in norms
            ]
            for key in kept:
                own[key] = norms[key]
            self.norms.load_state_dict(own)
        else:
            kept = []
        self.wm_frozen = self.cfg.wm_tune == "frozen"
        self.wm.requires_grad_(not self.wm_frozen)
        print(
            f"loaded world model from {path} ({self.cfg.wm_tune}); normalizers {kept or 'absent'}",
            flush=True,
        )

    def load_checkpoint(self, path: str | Path) -> None:
        """Load whatever this checkpoint contains. Absent modules stay freshly initialized."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        modules = {
            "wm": self.wm,
            "pi": self.policy,
            "critic": self.critic,
            "disc": self.adv_loss,
            "norms": self.norms,
        }
        optimizers = {
            "opt_wm": self.opt_wm,
            "opt_pi": self.opt_pi,
            "opt_v": self.opt_v,
            "opt_disc": self.opt_disc,
        }
        loaded: list[str] = []
        absent: list[str] = []
        for key, module in modules.items():
            if key not in ckpt:
                absent.append(key)
                continue
            try:
                missing, unexpected = module.load_state_dict(ckpt[key], strict=False)
            except (ValueError, RuntimeError) as exc:
                absent.append(key)
                print(f"resume skip {key}: {exc}", flush=True)
                continue
            loaded.append(key)
            if missing or unexpected:
                print(f"resume {key}: missing={list(missing)} unexpected={list(unexpected)}", flush=True)
        for key, opt in optimizers.items():
            if key not in ckpt:
                absent.append(key)
                continue
            try:
                opt.load_state_dict(ckpt[key])
            except (ValueError, RuntimeError) as exc:
                absent.append(key)
                print(f"resume skip {key}: {exc}", flush=True)
                continue
            loaded.append(key)
        if self.reward_norm is not None and "reward_norm" in ckpt:
            try:
                self.reward_norm.load_state_dict(ckpt["reward_norm"])
                loaded.append("reward_norm")
            except (KeyError, RuntimeError, ValueError) as exc:
                absent.append("reward_norm")
                print(f"resume skip reward_norm: {exc}", flush=True)
        elif "reward_norm" not in ckpt:
            absent.append("reward_norm")
        self.start_iteration = int(ckpt.get("iteration", 0))
        print(
            f"resumed {path} at iteration {self.start_iteration} | loaded {loaded} | absent {absent}",
            flush=True,
        )

    def save_checkpoint(self, iteration: int) -> None:
        ckpt = {
            "wm": self.wm.state_dict(),
            "pi": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "disc": self.adv_loss.state_dict(),
            "norms": self.norms.state_dict(),
            "opt_wm": self.opt_wm.state_dict(),
            "opt_pi": self.opt_pi.state_dict(),
            "opt_v": self.opt_v.state_dict(),
            "opt_disc": self.opt_disc.state_dict(),
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
