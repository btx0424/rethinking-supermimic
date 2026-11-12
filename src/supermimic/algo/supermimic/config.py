from dataclasses import dataclass, field
from supermimic.env.mdp.config import TaskConfig


@dataclass
class WandbConfig:
    project: str = "rethinking-supermimic"
    mode: str = "online"
    name: str | None = None


@dataclass
class TrainConfig:
    task: TaskConfig = field(default_factory=TaskConfig)
    run_dir: str | None = None
    wm_kind: str = "explicit"
    wm_hidden: tuple[int, ...] = (512, 512, 512)
    latent_dim: int = 256
    simnorm_dim: int = 8
    actor_hidden: tuple[int, ...] = (512, 512, 512)
    consistency_weight: float = 1.0
    num_envs: int = 4096
    iterations: int = 40000
    buffer_steps: int = 1000
    collect_steps: int = 4
    wm_window: int = 8
    wm_batches: int = 10
    wm_batch: int = 2048
    pi_batch: int = 4096
    horizon: int = 32
    pi_warmup: int = 50
    wm_lr: float = 2e-4
    pi_lr: float = 5e-4
    v_lr: float = 5e-4
    gamma: float = 0.99
    lam: float = 0.95
    action_noise: float = 0.1
    grad_clip_norm: float = 1.0
    critic_heads: int = 2
    critic_epochs: int = 5
    critic_minibatches: int = 4
    normalize_reward: bool = True
    vecnorm_decay: float = 0.9999
    optim: str = "muon"
    cuda_graphs: bool = True
    device: str = "cuda"
    seed: int = 0
    log_every: int = 10
    ckpt_every: int = 100
    resume: str | None = None
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def validate(self):
        self.task.validate()
        if self.wm_kind not in ("explicit", "latent"):
            raise ValueError("Unknown world model")
        for key in (
            "num_envs",
            "iterations",
            "buffer_steps",
            "collect_steps",
            "wm_window",
            "wm_batches",
            "wm_batch",
            "pi_batch",
            "horizon",
            "critic_heads",
            "critic_epochs",
            "critic_minibatches",
            "log_every",
            "ckpt_every",
        ):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if self.buffer_steps < self.wm_window:
            raise ValueError("Replay shorter than WM window")
        if not 0 <= self.gamma < 1 or not 0 <= self.lam <= 1:
            raise ValueError("Invalid return discount")
        if (
            self.pi_warmup < 0
            or self.action_noise < 0
            or self.grad_clip_norm <= 0
            or self.consistency_weight < 0
        ):
            raise ValueError("Invalid learning scale")
        if self.wandb.mode not in ("online", "disabled"):
            raise ValueError("Use online logging, or disabled for smoke tests")
