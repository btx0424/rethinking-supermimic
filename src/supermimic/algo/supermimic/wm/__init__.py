from .interface import WorldModel, WMState, WMBatch, WMTrajectory, LossOutput
from supermimic.env.mdp.geometry.representation import MaxCoordRepresentation


def build_world_model(cfg, representation, action_dim):
    if cfg.wm_kind == "explicit":
        from .explicit import ExplicitWorldModel

        return ExplicitWorldModel(representation, action_dim, cfg.wm_hidden)
    if cfg.wm_kind == "latent":
        from .latent import LatentWorldModel

        return LatentWorldModel(
            representation, action_dim, cfg.wm_hidden, cfg.latent_dim, cfg.simnorm_dim
        )
    raise ValueError("wm_kind must be explicit or latent")


__all__ = [
    "WorldModel",
    "WMState",
    "WMBatch",
    "WMTrajectory",
    "LossOutput",
    "MaxCoordRepresentation",
    "build_world_model",
]
