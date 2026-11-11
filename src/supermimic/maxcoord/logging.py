"""Explicit online logger selection for standalone MaxCoord experiments."""
from __future__ import annotations

import json
from pathlib import Path


class ExperimentLogger:
    def __init__(self, backend: str, mode: str, project: str, name: str, config: dict, directory: Path):
        if backend not in ("swanlab", "wandb"):
            raise ValueError("log_backend must be swanlab or wandb")
        self.backend = backend
        self.run = None
        if mode == "disabled":
            return
        if backend == "wandb":
            import wandb

            self.run = wandb.init(project=project, name=name, config=config, mode=mode,
                                  dir=str(directory))
            metadata = {"backend": backend, "id": self.run.id, "url": self.run.url, "mode": mode}
        else:
            import swanlab

            self.run = swanlab.init(project=project, name=name, config=config, mode=mode)
            metadata = {"backend": backend, "mode": mode}
        (directory / "online_run.json").write_text(json.dumps(metadata, indent=2) + "\n")

    def log(self, metrics, step):
        if self.run is None:
            return
        if self.backend == "wandb":
            self.run.log(metrics, step=step)
        else:
            import swanlab

            swanlab.log(metrics, step=step)

    def finish(self):
        if self.run is None:
            return
        if self.backend == "wandb":
            self.run.finish()
        else:
            import swanlab

            swanlab.finish()


def grouped_metrics(metrics):
    """Slash prefixes create separate panels, as in mimic-lite's train logger.

    Keep CSV's flat schema for monitoring/backward-compatible analysis. Reward
    terms are weighted means BEFORE multiplication by CONTROL_DT. Training
    rewards/returns retain dt; normalized_total also uses pre-dt display units.
    """
    fixed = {"sim_reward": "SimReward/total", "sim_terminated": "SimReward/terminated_fraction",
             "episode_length": "SimReward/episode_length", "imag_reward": "WMReward/normalized_total",
             "imag_raw_reward": "WMReward/total", "imag_survival": "WMReward/survival",
             "grad_wm": "WMLearning/grad_norm", "pi_loss": "ActorCriticLearning/actor_loss",
             "v_loss": "ActorCriticLearning/critic_loss", "grad_pi": "ActorCriticLearning/actor_grad_norm",
             "grad_v": "ActorCriticLearning/critic_grad_norm",
             "collect_time": "Performance/collect_time", "train_time": "Performance/train_time",
             "compute_time": "Performance/compute_time",
             "iteration": "Performance/iteration", "seconds": "Performance/seconds"}
    out = {}
    for key, value in metrics.items():
        if key in fixed: name = fixed[key]
        elif key.startswith("sim_term_"): name = "SimReward/" + key.removeprefix("sim_term_")
        elif key.startswith("imag_term_"): name = "WMReward/" + key.removeprefix("imag_term_")
        elif key.startswith("wm_"): name = "WMLearning/" + key.removeprefix("wm_")
        else: raise ValueError(f"Unclassified MaxCoord metric: {key}")
        out[name] = value
    return out
