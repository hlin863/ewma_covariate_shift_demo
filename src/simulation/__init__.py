"""Synthetic-data generation helpers used by CSE experiments."""

from src.simulation.gaussian import generate_gaussian_mean_shift
from src.simulation.jumping_mean import (
    JumpingMeanConfig,
    generate_d2_jumping_mean,
)

__all__ = [
    "JumpingMeanConfig",
    "generate_gaussian_mean_shift",
    "generate_d2_jumping_mean",
]
