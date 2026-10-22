"""Train a SuperMimic tracker on the 40 LAFAN1 clips."""

from __future__ import annotations

import argparse
from pathlib import Path

from .algo import TrainConfig, run

LOG_ROOT = Path(__file__).resolve().parents[2] / "logs"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=320_000)
    parser.add_argument("--log-dir", type=Path, default=LOG_ROOT / "lafan1")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wm-norm", choices=("none", "layer", "rms"), default="none")
    parser.add_argument("--wm-residual", action="store_true")
    parser.add_argument("--stochastic-actor", action="store_true")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--horizon", type=int, default=32)
    args = parser.parse_args()
    cfg = TrainConfig(
        num_envs=args.num_envs,
        iterations=args.iterations,
        device=args.device,
        seed=args.seed,
        wm_norm=args.wm_norm,
        wm_residual=args.wm_residual,
        stochastic_actor=args.stochastic_actor,
        lr=args.lr,
        horizon=args.horizon,
    )
    run(cfg, args.log_dir)


if __name__ == "__main__":
    main()
