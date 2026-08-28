"""Synthetic-data generation helpers used by CSE experiments."""

from src.simulation.gaussian import generate_gaussian_mean_shift
from src.simulation.jumping_mean import generate_d2_jumping_mean

__all__ = [
    "generate_gaussian_mean_shift",
    "generate_d2_jumping_mean",
]
