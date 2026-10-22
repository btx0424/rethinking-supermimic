"""Train a SuperMimic tracker on the 40 LAFAN1 clips."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from .algo import TrainConfig, run
from .mdp import resolve_obs_terms

REPO = Path(__file__).resolve().parents[2]


@hydra.main(config_path="conf", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
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
        swanlab_experiment=str(cfg.swanlab.experiment),
        device=str(cfg.device),
        log_every=int(cfg.log_every),
        ckpt_every=int(cfg.ckpt_every),
        seed=int(cfg.seed),
    )
    run(train_cfg, REPO / str(cfg.log_dir))


if __name__ == "__main__":
    main()
