"""Train SuperMimic, NablaMimic, or SAC on the LAFAN1 clips."""

from __future__ import annotations

from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from supermimic.mdp import resolve_obs_terms
from supermimic.tasks import TaskSpec, make_task, make_task_from_config


def _task_run_name(
    task: str,
    object_name: str | None,
    split: str | None,
    has_object: bool | None = None,
    simple_hands: bool | None = None,
) -> str:
    """Run label of a task spec, used for the log directory and the logger."""
    return make_task(
        str(task),
        object_name=None if object_name is None else str(object_name),
        split=None if split is None else str(split),
        has_object=None if has_object is None else bool(has_object),
        simple_hands=None if simple_hands is None else bool(simple_hands),
    ).run_name


# Registered at import time, before ``@hydra.main`` resolves the config.
OmegaConf.register_new_resolver("task_run_name", _task_run_name, replace=True)


def _task(cfg: DictConfig) -> TaskSpec:
    """Task spec from the ``task`` config node (kept as an opaque node)."""
    return make_task_from_config(cfg.task)


def _obs_terms(cfg: DictConfig) -> tuple[str, ...]:
    task = _task(cfg)
    chosen = [name for name, enabled in cfg.obs.items() if enabled]
    if task.has_object:
        for name in ("object_pos", "object_ori"):
            if name not in chosen:
                chosen.append(name)
    return resolve_obs_terms(chosen)


def _experiment_name(run_dir: Path) -> str:
    return "/".join(run_dir.parts[-2:])


def _supermimic_kwargs(cfg: DictConfig, run_dir: Path) -> dict:
    return dict(
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
        ref_residual=float(cfg.get("ref_residual", 0.0)),
        no_host_sync=bool(cfg.get("no_host_sync", True)),
        compile_mdp=bool(cfg.get("compile_mdp", True)),
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
        save_optim=bool(cfg.get("save_optim", False)),
        seed=int(cfg.seed),
        resume=None if cfg.resume is None else str(cfg.resume),
        wm_ckpt=None if cfg.wm_ckpt is None else str(cfg.wm_ckpt),
        wm_tune=str(cfg.wm_tune),
        optim=str(cfg.optim),
        policy_bootstrap=bool(cfg.policy_bootstrap),
        pi_ema=_parse_ema(cfg, "pi_ema", 0.99),
        wm_ema=_parse_ema(cfg, "wm_ema", 0.99),
        imagine_wm=str(cfg.get("imagine_wm", "live")),
    )


def _build_supermimic_cfg(cfg: DictConfig, run_dir: Path):
    from supermimic.algos.supermimic import TrainConfig

    return TrainConfig(**_supermimic_kwargs(cfg, run_dir))


def _build_nablamimic_cfg(cfg: DictConfig, run_dir: Path):
    from supermimic.algos.nablamimic import NablaTrainConfig

    return NablaTrainConfig(
        **_supermimic_kwargs(cfg, run_dir),
        act_cot_clip=None if cfg.get("act_cot_clip") is None else float(cfg.act_cot_clip),
        action_target_steps=int(cfg.get("action_target_steps", 8)),
        action_target_windows=int(cfg.get("action_target_windows", 1)),
        action_target_delta=float(cfg.get("action_target_delta", 0.03)),
        action_target_delta_floor=float(cfg.get("action_target_delta_floor", 0.3)),
        action_target_batch=int(cfg.get("action_target_batch", 16384)),
        action_target_lr=float(cfg.get("action_target_lr", cfg.pi_lr)),
    )


def _parse_ema(cfg: DictConfig, key: str, default: float) -> float | None:
    raw = cfg.get(key, default)
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
        pi_ema=_parse_ema(cfg, "pi_ema", 0.995),
        prior_data=None if cfg.get("prior_data") is None else str(cfg.prior_data),
        prior_data_ratio=float(cfg.get("prior_data_ratio", 0.5)),
        prior_fraction=float(cfg.get("prior_fraction", 1.0)),
    )


@hydra.main(config_path="conf", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    # Resolve interpolations (e.g. the task-derived ``swanlab.experiment``) so the
    # saved config reloads standalone instead of needing our resolvers.
    OmegaConf.save(OmegaConf.to_container(cfg, resolve=True), run_dir / "config.yaml")
    print(f"log_dir={run_dir} algo={cfg.algo}", flush=True)

    algo = str(cfg.algo)
    if algo == "supermimic":
        from supermimic.algos.supermimic import run

        run(_build_supermimic_cfg(cfg, run_dir), cfg.task, run_dir)
    elif algo == "nablamimic":
        from supermimic.algos.nablamimic import run

        run(_build_nablamimic_cfg(cfg, run_dir), cfg.task, run_dir)
    elif algo == "sac":
        from supermimic.sac import run

        run(_build_sac_cfg(cfg, run_dir), cfg.task, run_dir)
    else:
        raise ValueError(f"Unknown algo={algo!r}. Expected 'supermimic', 'nablamimic', or 'sac'.")


if __name__ == "__main__":
    main()
