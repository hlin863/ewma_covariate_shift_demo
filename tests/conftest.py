import numpy as np
import pytest

from src.ewma import SD_EWMA_Config


@pytest.fixture
def default_config() -> SD_EWMA_Config:
    return SD_EWMA_Config(
        lambda_value=0.2,
        variance_smoothing=0.05,
        control_limit_multiplier=3.0,
    )


@pytest.fixture
def stationary_values() -> np.ndarray:
    return np.full(50, 2.0)


@pytest.fixture
def abrupt_shift_values() -> np.ndarray:
    return np.concatenate([
        np.full(30, 1.0),
        np.full(30, 6.0),
    ])


@pytest.fixture
def d1_generator():
    def generate(seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)

        return np.concatenate([
            rng.normal(1.0, 1.0, 1000),
            rng.normal(3.0, 1.0, 1000),
        ])

    return generate