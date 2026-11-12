"""Physical state exposed to shared MDP terms; contains no latent variables."""

from typing import NamedTuple
import torch


class PhysicalState(NamedTuple):
    aux: torch.Tensor  # world root position and quaternion
    f: torch.Tensor  # raw MaxCoord body features
    xi_n: torch.Tensor
