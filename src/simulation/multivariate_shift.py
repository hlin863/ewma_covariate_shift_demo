"""Paper-aligned D3 and D4 multivariate mean-shift generators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MultivariateShiftConfig:
    """Configuration shared by the paper's D3 and D4 datasets."""

    random_seed: int = 42
    n_samples: int = 300
    n_features: int = 10
    first_shift_time: int = 101
    second_shift_time: int = 201
    initial_mean: float = 0.0
    shifted_mean: float = 1.0
    variance: float = 0.45
    covariance: float = 0.30
    degrees_of_freedom: float = 10.0

    def __post_init__(self) -> None:
        if self.n_features < 2:
            raise ValueError("n_features must be at least 2.")
        if not 1 < self.first_shift_time < self.second_shift_time <= self.n_samples:
            raise ValueError("shift times must be ordered within the stream.")
        if self.variance <= 0.0:
            raise ValueError("variance must be positive.")
        if not -self.variance / (self.n_features - 1) < self.covariance < self.variance:
            raise ValueError("variance/covariance must define a positive-definite matrix.")
        if self.degrees_of_freedom <= 2.0:
            raise ValueError("degrees_of_freedom must exceed 2 for finite covariance.")


def paper_covariance_matrix(
    config: MultivariateShiftConfig | None = None,
) -> np.ndarray:
    """Return the compound-symmetry covariance specified for D3/D4."""

    effective = config or MultivariateShiftConfig()
    matrix = np.full(
        (effective.n_features, effective.n_features),
        effective.covariance,
        dtype=float,
    )
    np.fill_diagonal(matrix, effective.variance)
    return matrix


def _mean_schedule(config: MultivariateShiftConfig) -> np.ndarray:
    means = np.full(
        (config.n_samples, config.n_features),
        config.initial_mean,
        dtype=float,
    )
    means[config.first_shift_time - 1 : config.second_shift_time - 1] = (
        config.shifted_mean
    )
    return means


def _as_stream(
    values: np.ndarray,
    means: np.ndarray,
    config: MultivariateShiftConfig,
    distribution: str,
) -> pd.DataFrame:
    times = np.arange(1, config.n_samples + 1, dtype=int)
    regimes = np.ones(config.n_samples, dtype=int)
    regimes[config.first_shift_time - 1 : config.second_shift_time - 1] = 2
    regimes[config.second_shift_time - 1 :] = 3
    true_shift = np.zeros(config.n_samples, dtype=int)
    true_shift[[config.first_shift_time - 1, config.second_shift_time - 1]] = 1

    feature_columns = {
        f"x{index + 1}": values[:, index]
        for index in range(config.n_features)
    }
    return pd.DataFrame(
        {
            "time": times,
            **feature_columns,
            "target_mean": means[:, 0],
            "regime": regimes,
            "true_shift": true_shift,
            "distribution": distribution,
        }
    )


def generate_d3_multivariate_normal(
    config: MultivariateShiftConfig | None = None,
) -> pd.DataFrame:
    """Generate D3: correlated Gaussian data with a 0 -> 1 -> 0 mean."""

    effective = config or MultivariateShiftConfig()
    rng = np.random.default_rng(effective.random_seed)
    means = _mean_schedule(effective)
    centered = rng.multivariate_normal(
        mean=np.zeros(effective.n_features),
        cov=paper_covariance_matrix(effective),
        size=effective.n_samples,
    )
    return _as_stream(centered + means, means, effective, "normal")


def generate_d4_multivariate_t(
    config: MultivariateShiftConfig | None = None,
) -> pd.DataFrame:
    """Generate D4: correlated multivariate-t data with the D3 covariance.

    NumPy's construction uses a Gaussian scale matrix.  A t distribution with
    ``nu`` degrees of freedom has covariance ``nu / (nu - 2) * scale``;
    therefore the Gaussian scale is adjusted so the resulting covariance is
    the paper's stated Sigma rather than an inflated proxy.
    """

    effective = config or MultivariateShiftConfig()
    rng = np.random.default_rng(effective.random_seed)
    means = _mean_schedule(effective)
    covariance = paper_covariance_matrix(effective)
    scale = covariance * (
        (effective.degrees_of_freedom - 2.0) / effective.degrees_of_freedom
    )
    gaussian = rng.multivariate_normal(
        mean=np.zeros(effective.n_features),
        cov=scale,
        size=effective.n_samples,
    )
    chi_squared = rng.chisquare(
        effective.degrees_of_freedom,
        size=effective.n_samples,
    )
    centered = gaussian / np.sqrt(
        chi_squared[:, np.newaxis] / effective.degrees_of_freedom
    )
    return _as_stream(centered + means, means, effective, "student_t")
