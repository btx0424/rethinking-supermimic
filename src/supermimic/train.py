"""Train SuperMimic or SAC on the LAFAN1 clips."""

from __future__ import annotations

from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from supermimic.mdp import resolve_obs_terms


def _obs_terms(cfg: DictConfig) -> tuple[str, ...]:
    return resolve_obs_terms([name for name, enabled in cfg.obs.items() if enabled])


def _experiment_name(run_dir: Path) -> str:
    return "/".join(run_dir.parts[-2:])


def _build_supermimic_cfg(cfg: DictConfig, run_dir: Path):
    from supermimic.algos.supermimic import TrainConfig

    return TrainConfig(
        num_envs=int(cfg.num_envs),
        iterations=int(cfg.iterations),
        buffer_steps=int(cfg.buffer_steps),
        utd=float(cfg.utd),
        horizon=int(cfg.horizon),
        wm_window=int(cfg.wm_window),
        wm_batches=int(cfg.wm_batches),
        wm_batch=int(cfg.wm_batch),
        pi_batch=int(cfg.pi_batch),
        wm_lr=float(cfg.wm_lr),
        pi_lr=float(cfg.pi_lr),
        pi_warmup=int(cfg.pi_warmup),
        v_lr=float(cfg.v_lr),
        gamma=float(cfg.gamma),
        lam=float(cfg.lam),
        action_noise=float(cfg.action_noise),
        critic_epochs=int(cfg.critic_epochs),
        critic_minibatches=int(cfg.critic_minibatches),
        critic_ensemble=int(cfg.critic_ensemble),
        critic_kind=str(cfg.critic_kind),
        critic_atoms=int(cfg.critic_atoms),
        critic_v_min=float(cfg.critic_v_min),
        critic_v_max=float(cfg.critic_v_max),
        critic_sigma_ratio=float(cfg.critic_sigma_ratio),
        lr_decay_iters=int(cfg.lr_decay_iters),
        wm_norm=str(cfg.wm_norm),
        wm_residual=bool(cfg.wm_residual),
        wm_frame=str(cfg.wm_frame),
        stochastic_actor=bool(cfg.stochastic_actor),
        policy_kind=str(cfg.actor.kind),
        policy_z_dim=int(cfg.actor.get("z_dim", 32)),
        policy_encoder=tuple(int(width) for width in cfg.actor.get("encoder", [2048, 1024, 512, 256])),
        policy_decoder=tuple(int(width) for width in cfg.actor.get("decoder", [2048, 1024, 512, 256])),
        policy_trunk=tuple(int(width) for width in cfg.actor.get("hidden", [512, 512, 512])),
        policy_norm=str(cfg.actor.norm),
        anchor=str(cfg.anchor),
        obs_terms=_obs_terms(cfg),
        reward_style=str(cfg.reward_style),
        swanlab_project=str(cfg.swanlab.project),
        swanlab_mode=str(cfg.swanlab.mode),
        swanlab_experiment=_experiment_name(run_dir),
        grad_clip_norm=None if cfg.grad_clip_norm is None else float(cfg.grad_clip_norm),
        grad_clip_value=None if cfg.grad_clip_value is None else float(cfg.grad_clip_value),
        normalize_reward=bool(cfg.get("normalize_reward", False)),
        compile=bool(cfg.compile),
        compile_wm=str(cfg.compile_wm),
        amp=str(cfg.amp),
        fk=str(cfg.fk),
        adv_coef=float(cfg.adv_coef),
        reward_coef=float(cfg.reward_coef),
        contact_coef=float(cfg.contact_coef),
        device=str(cfg.device),
        log_every=int(cfg.log_every),
        ckpt_every=int(cfg.ckpt_every),
        seed=int(cfg.seed),
        resume=None if cfg.resume is None else str(cfg.resume),
        wm_ckpt=None if cfg.wm_ckpt is None else str(cfg.wm_ckpt),
        wm_tune=str(cfg.wm_tune),
        optim=str(cfg.optim),
        policy_bootstrap=bool(cfg.policy_bootstrap),
        pi_ema=_parse_pi_ema(cfg),
    )


def _parse_pi_ema(cfg: DictConfig) -> float | None:
    raw = cfg.get("pi_ema", 0.995)
    return None if raw is None else float(raw)


def _build_sac_cfg(cfg: DictConfig, run_dir: Path):
    from supermimic.sac.sac import TrainConfig

    sigma = cfg.get("target_entropy_sigma")
    return TrainConfig(
        num_envs=int(cfg.num_envs),
        iterations=int(cfg.iterations),
        buffer_steps=int(cfg.buffer_steps),
        utd=float(cfg.utd),
        actor_tau=float(cfg.actor_tau),
        critic_tau=float(cfg.critic_tau),
        critic_kind=str(cfg.critic_kind),
        critic_ensemble=int(cfg.critic_ensemble),
        critic_atoms=int(cfg.critic_atoms),
        critic_v_min=float(cfg.critic_v_min),
        critic_v_max=float(cfg.critic_v_max),
        critic_sigma_ratio=float(cfg.critic_sigma_ratio),
        normalize_reward=bool(cfg.normalize_reward),
        gamma=float(cfg.gamma),
        n_steps=int(cfg.n_steps),
        lr=float(cfg.lr),
        lr_alpha=float(cfg.lr_alpha),
        optim=str(cfg.optim),
        weight_decay=float(cfg.weight_decay),
        warm_up_steps=int(cfg.warm_up_steps),
        actor_lr_warmup_ops=int(cfg.get("actor_lr_warmup_ops", 100)),
        train_every=int(cfg.train_every),
        utd_ratio=int(cfg.utd_ratio),
        critic_batch_size=int(cfg.critic_batch_size),
        actor_batch_size=int(cfg.actor_batch_size),
        alpha_init=float(cfg.alpha_init),
        target_entropy_sigma=None if sigma is None else float(sigma),
        entropy_bonus=float(cfg.entropy_bonus),
        target_action_noise=float(cfg.target_action_noise),
        max_grad_norm=float(cfg.max_grad_norm),
        actor_init=str(cfg.actor_init),
        vecnorm=bool(cfg.vecnorm),
        use_amp=bool(cfg.use_amp),
        anchor=str(cfg.anchor),
        obs_terms=_obs_terms(cfg),
        reward_style=str(cfg.reward_style),
        device=str(cfg.device),
        seed=int(cfg.seed),
        log_every=int(cfg.log_every),
        ckpt_every=int(cfg.get("ckpt_every", 1000)),
        swanlab_project=str(cfg.swanlab.project),
        swanlab_mode=str(cfg.swanlab.mode),
        swanlab_experiment=_experiment_name(run_dir),
        pi_ema=_parse_pi_ema(cfg),
    )


@hydra.main(config_path="conf", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    OmegaConf.save(cfg, run_dir / "config.yaml")
    print(f"log_dir={run_dir} algo={cfg.algo}", flush=True)

    algo = str(cfg.algo)
    if algo == "supermimic":
        from supermimic.algos.supermimic import run

        run(_build_supermimic_cfg(cfg, run_dir), run_dir)
    elif algo == "sac":
        from supermimic.sac import run

        run(_build_sac_cfg(cfg, run_dir), run_dir)
    else:
        raise ValueError(f"Unknown algo={algo!r}. Expected 'supermimic' or 'sac'.")


if __name__ == "__main__":
    main()
