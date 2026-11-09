"""Train SuperMimic, NablaMimic, ScoreMimic, or SAC."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
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


def _trainer_cfg(cls, cfg: DictConfig, run_dir: Path):
    """Build ``cls`` from the composed config.

    Hydra already checked the schema. The observation list and the run name are
    derived here, and an ``actor`` group overrides the policy-shape fields.
    """
    names = {item.name for item in fields(cls)}
    raw = {name: cfg[name] for name in names if name in cfg}
    raw["obs_terms"] = _obs_terms(cfg)
    raw["swanlab_project"] = str(cfg.swanlab.project)
    raw["swanlab_mode"] = str(cfg.swanlab.mode)
    raw["swanlab_experiment"] = _experiment_name(run_dir)
    if "actor" in cfg and "policy_kind" in names:
        actor = cfg.actor
        raw["policy_kind"] = str(actor.kind)
        raw["policy_norm"] = str(actor.norm)
        if "z_dim" in actor:
            raw["policy_z_dim"] = int(actor.z_dim)
        if "encoder" in actor:
            raw["policy_encoder"] = [int(width) for width in actor.encoder]
        if "decoder" in actor:
            raw["policy_decoder"] = [int(width) for width in actor.decoder]
        if "hidden" in actor:
            raw["policy_trunk"] = [int(width) for width in actor.hidden]
    return OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(cls), raw))


def _register_configs() -> None:
    """Register each trainer's config as an ``algo`` group, merged at the root."""
    from supermimic.sac.sac import TrainConfig as SACTrainConfig
    from supermimic.supermimic.nablamimic import NablaTrainConfig
    from supermimic.supermimic.scoremimic import ScoreTrainConfig
    from supermimic.supermimic.supermimic import TrainConfig
    from supermimic.supermimic.supermimic_latent import LatentTrainConfig
    from supermimic.supermimic.supermimic_vfirst import VFirstTrainConfig

    store = ConfigStore.instance()
    for name, node in (
        ("supermimic", TrainConfig),
        ("supermimic_vfirst", VFirstTrainConfig),
        ("supermimic_latent", LatentTrainConfig),
        ("nablamimic", NablaTrainConfig),
        ("scoremimic", ScoreTrainConfig),
        ("sac", SACTrainConfig),
    ):
        # Not struct: train.yaml still adds task, obs, and swanlab beside these fields.
        schema = OmegaConf.structured(node)
        OmegaConf.set_struct(schema, False)
        store.store(group="algo", name=name, node=schema, package="_global_")


_register_configs()


@hydra.main(config_path="conf", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    # Resolve interpolations (e.g. the task-derived ``swanlab.experiment``) so the
    # saved config reloads standalone instead of needing our resolvers.
    OmegaConf.save(OmegaConf.to_container(cfg, resolve=True), run_dir / "config.yaml")
    print(f"log_dir={run_dir} algo={cfg.algo}", flush=True)

    from supermimic.sac.sac import TrainConfig as SACTrainConfig
    from supermimic.supermimic.nablamimic import NablaTrainConfig
    from supermimic.supermimic.scoremimic import ScoreTrainConfig
    from supermimic.supermimic.supermimic import TrainConfig
    from supermimic.supermimic.supermimic_latent import LatentTrainConfig
    from supermimic.supermimic.supermimic_vfirst import VFirstTrainConfig

    trainers = {
        "supermimic": (TrainConfig, "supermimic.algos.supermimic"),
        "supermimic_vfirst": (VFirstTrainConfig, "supermimic.algos.supermimic_vfirst"),
        "supermimic_latent": (LatentTrainConfig, "supermimic.algos.supermimic_latent"),
        "nablamimic": (NablaTrainConfig, "supermimic.algos.nablamimic"),
        "scoremimic": (ScoreTrainConfig, "supermimic.algos.scoremimic"),
        "sac": (SACTrainConfig, "supermimic.sac"),
    }
    algo = str(cfg.algo)
    if algo not in trainers:
        known = ", ".join(trainers)
        raise ValueError(f"Unknown algo={algo!r}. Expected one of: {known}.")
    cls, module = trainers[algo]
    from importlib import import_module

    import_module(module).run(_trainer_cfg(cls, cfg, run_dir), cfg.task, run_dir)


if __name__ == "__main__":
    main()
