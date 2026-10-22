"""Train a SuperMimic tracker on the 40 LAFAN1 clips."""

from __future__ import annotations

from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from .algo import TrainConfig, run
from .mdp import resolve_obs_terms


@hydra.main(config_path="conf", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    OmegaConf.save(cfg, run_dir / "config.yaml")
    print(f"log_dir={run_dir}", flush=True)

    terms = resolve_obs_terms([name for name, enabled in cfg.obs.items() if enabled])
    train_cfg = TrainConfig(
        num_envs=int(cfg.num_envs),
        iterations=int(cfg.iterations),
        buffer_steps=int(cfg.buffer_steps),
        utd=float(cfg.utd),
        horizon=int(cfg.horizon),
        wm_window=int(cfg.wm_window),
        wm_batches=int(cfg.wm_batches),
        wm_batch=int(cfg.wm_batch),
        pi_batch=int(cfg.pi_batch),
        lr=float(cfg.lr),
        gamma=float(cfg.gamma),
        lam=float(cfg.lam),
        action_noise=float(cfg.action_noise),
        critic_epochs=int(cfg.critic_epochs),
        critic_minibatches=int(cfg.critic_minibatches),
        lr_decay_iters=int(cfg.lr_decay_iters),
        wm_norm=str(cfg.wm_norm),
        wm_residual=bool(cfg.wm_residual),
        stochastic_actor=bool(cfg.stochastic_actor),
        anchor=str(cfg.anchor),
        obs_terms=terms,
        swanlab_project=str(cfg.swanlab.project),
        swanlab_mode=str(cfg.swanlab.mode),
        swanlab_experiment="/".join(run_dir.parts[-2:]),
        grad_clip_norm=None if cfg.grad_clip_norm is None else float(cfg.grad_clip_norm),
        grad_clip_value=None if cfg.grad_clip_value is None else float(cfg.grad_clip_value),
        normalize_reward=bool(cfg.normalize_reward),
        device=str(cfg.device),
        log_every=int(cfg.log_every),
        ckpt_every=int(cfg.ckpt_every),
        seed=int(cfg.seed),
    )
    run(train_cfg, run_dir)


if __name__ == "__main__":
    main()
