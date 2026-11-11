"""Independent maximal-coordinate world models and rollout supervision."""
from .representation import MaxCoordRepresentation

from .interface import WorldModel, WMState, RolloutState, WMBatch, WMTargets, WMTrajectory, LossOutput
