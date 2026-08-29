"""Synthetic-data generation helpers used by CSE experiments."""

from src.simulation.gaussian import generate_gaussian_mean_shift
from src.simulation.jumping_mean import (
    JumpingMeanConfig,
    generate_d2_jumping_mean,
)
from src.simulation.multivariate_shift import (
    MultivariateShiftConfig,
    generate_d3_multivariate_normal,
    generate_d4_multivariate_t,
    paper_covariance_matrix,
)

__all__ = [
    "JumpingMeanConfig",
    "MultivariateShiftConfig",
    "generate_gaussian_mean_shift",
    "generate_d2_jumping_mean",
    "generate_d3_multivariate_normal",
    "generate_d4_multivariate_t",
    "paper_covariance_matrix",
]
