"""G1 LAFAN training. Overrides use key=value; explicit and latent share one algo."""

import argparse
import torch
from omegaconf import OmegaConf
from .algo.supermimic.config import TrainConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Optional YAML config")
    parser.add_argument(
        "overrides", nargs="*", help="For example wm_kind=latent wandb.mode=disabled"
    )
    args = parser.parse_args()
    config = OmegaConf.structured(TrainConfig)
    if args.config:
        config = OmegaConf.merge(config, OmegaConf.load(args.config))
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.overrides))
    cfg = OmegaConf.to_object(config)
    torch.set_num_threads(8)
    from .algo.supermimic.trainer import Trainer

    Trainer(cfg).fit()


if __name__ == "__main__":
    main()
