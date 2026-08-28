"""Paper-aligned generator for the D2 jumping-mean time series."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class JumpingMeanConfig:
    """Configuration for the synthetic D2 jumping-mean dataset.

    The paper defines 5,000 samples in 100-observation regimes. Its printed
    mean schedule ends at regime 49, which leaves the final 100 observations
    undefined. This implementation completes all ``ceil(n_samples /
    shift_interval)`` regimes using the same recurrence.
    """

    random_seed: int = 42
    n_samples: int = 5000
    shift_interval: int = 100
    ar_coefficient_1: float = 0.6
    ar_coefficient_2: float = -0.5
    noise_std: float = 1.5
    initial_x1: float = 0.0
    initial_x2: float = 0.0

    def __post_init__(self) -> None:
        if self.n_samples < 2:
            raise ValueError("n_samples must be at least 2.")
        if self.shift_interval <= 0:
            raise ValueError("shift_interval must be positive.")
        if self.noise_std <= 0.0 or not np.isfinite(self.noise_std):
            raise ValueError("noise_std must be positive and finite.")

        finite_parameters = (
            self.ar_coefficient_1,
            self.ar_coefficient_2,
            self.initial_x1,
            self.initial_x2,
        )
        if not np.isfinite(finite_parameters).all():
            raise ValueError("AR coefficients and initial values must be finite.")


def _noise_means(config: JumpingMeanConfig) -> np.ndarray:
    """Return the cumulative paper mean schedule for every regime."""

    regime_count = ceil(config.n_samples / config.shift_interval)
    regime_means = np.zeros(regime_count, dtype=float)
    for regime_number in range(2, regime_count + 1):
        regime_means[regime_number - 1] = (
            regime_means[regime_number - 2]
            + regime_number / 16.0
        )
    return regime_means


def generate_d2_jumping_mean(
    config: JumpingMeanConfig | None = None,
) -> pd.DataFrame:
    """Generate the 2015 paper's D2 jumping-mean stream.

    Paper time is one-based. Consequently, a regime covering observations
    1--100 changes at observation 101, not at zero-based array index 100.
    The returned truth columns make this boundary explicit for evaluation.
    """

    effective_config = config or JumpingMeanConfig()
    rng = np.random.default_rng(effective_config.random_seed)

    times = np.arange(1, effective_config.n_samples + 1, dtype=int)
    regimes = (
        (times - 1) // effective_config.shift_interval
        + 1
    )
    regime_means = _noise_means(effective_config)
    observation_noise_means = regime_means[regimes - 1]

    innovations = np.zeros(effective_config.n_samples, dtype=float)
    if effective_config.n_samples > 2:
        innovations[2:] = rng.normal(
            loc=observation_noise_means[2:],
            scale=effective_config.noise_std,
        )

    values = np.empty(effective_config.n_samples, dtype=float)
    values[0] = effective_config.initial_x1
    values[1] = effective_config.initial_x2
    for index in range(2, effective_config.n_samples):
        values[index] = (
            effective_config.ar_coefficient_1 * values[index - 1]
            + effective_config.ar_coefficient_2 * values[index - 2]
            + innovations[index]
        )

    true_shift = np.zeros(effective_config.n_samples, dtype=int)
    true_shift[1:] = (regimes[1:] != regimes[:-1]).astype(int)

    return pd.DataFrame(
        {
            "time": times,
            "x": values,
            "innovation": innovations,
            "noise_mean": observation_noise_means,
            "regime": regimes,
            "true_shift": true_shift,
        }
    )
