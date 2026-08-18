"""Synthetic Gaussian mean-shift data generation."""

import numpy as np
import pandas as pd

from src.config import GaussianConfig


def generate_gaussian_mean_shift(config: GaussianConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.random_seed)
    training = rng.normal(config.mean_before, config.std_before, config.n_train)
    normal_test = rng.normal(
        config.mean_before,
        config.std_before,
        config.n_normal_test,
    )
    shifted_test = rng.normal(
        config.mean_after,
        config.std_after,
        config.n_shifted_test,
    )
    values = np.concatenate([training, normal_test, shifted_test])
    time = np.arange(values.size)
    regimes = np.select(
        [time < config.train_end, time < config.shift_point],
        ["training", "normal_test"],
        default="shifted_test",
    )
    return pd.DataFrame(
        {
            "time": time,
            "x": values,
            "true_regime": regimes,
            "true_shift": (time == config.shift_point).astype(int),
        }
    )
