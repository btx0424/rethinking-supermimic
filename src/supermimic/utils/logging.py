"""Namespaced online metrics plus local CSV. No silent offline fallback."""

import csv
import json
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, cfg, directory, config):
        self.directory = Path(directory)
        self.run = None
        if cfg.mode == "online":
            import wandb

            self.run = wandb.init(
                project=cfg.project,
                mode="online",
                name=cfg.name or self.directory.name,
                config=config,
                dir=str(directory),
            )
            (self.directory / "online_run.json").write_text(
                json.dumps({"id": self.run.id, "url": self.run.url, "mode": "online"}, indent=2)
            )
        self.tb = SummaryWriter(str(self.directory / "tb"))
        self.metric_file = (self.directory / "train_log.csv").open("w")
        self.metric_writer = None
        self.perf_file = (self.directory / "performance.csv").open("w")
        self.perf_writer = csv.DictWriter(
            self.perf_file, fieldnames=["iteration", "collect_time", "train_time", "compute_time"]
        )
        self.perf_writer.writeheader()

    def log(self, step, timing, metrics=None):
        self.perf_writer.writerow({"iteration": step, **timing})
        self.perf_file.flush()
        payload = {
            "Performance/iteration": step,
            **{f"Performance/{k}": v for k, v in timing.items()},
        }
        if metrics is not None:
            row = {"iteration": step, **{k: float(v) for k, v in metrics.items()}}
            if self.metric_writer is None:
                self.metric_writer = csv.DictWriter(self.metric_file, fieldnames=list(row))
                self.metric_writer.writeheader()
            self.metric_writer.writerow(row)
            self.metric_file.flush()
            print(json.dumps(row), flush=True)
            payload.update({k: v for k, v in row.items() if k != "iteration"})
        for key, value in payload.items():
            self.tb.add_scalar(key, value, step)
        if self.run is not None:
            self.run.log(payload, step=step)

    def close(self):
        self.metric_file.close()
        self.perf_file.close()
        self.tb.close()
        if self.run is not None:
            self.run.finish()
