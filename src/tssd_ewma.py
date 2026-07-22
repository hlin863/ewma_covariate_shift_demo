"""Combined runner for the two-stage TSSD-EWMA detector."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd

from src.ewma import (
    EWMATrainingResult,
    SD_EWMA_Config,
    fit_sd_ewma,
    run_sd_ewma,
)
from src.stage_2 import (
    Stage2Config,
    validate_stage_1_alarms,
)


@dataclass(frozen=True)
class TSSDEWMAConfig:
    """Combined configuration for the two-stage detector."""

    stage_1: SD_EWMA_Config
    stage_2: Stage2Config


@dataclass(frozen=True)
class TSSDEWMAResult:
    """Complete output from one TSSD-EWMA run."""

    training_result: EWMATrainingResult
    stage_1_results: pd.DataFrame
    stage_2_results: pd.DataFrame
    computation_time_seconds: float


def run_tssd_ewma(
    training_values: np.ndarray,
    testing_values: np.ndarray,
    testing_times: np.ndarray,
    config: TSSDEWMAConfig,
) -> TSSDEWMAResult:
    """Run Stage I and Stage II as one TSSD-EWMA pipeline."""

    training_observations = np.asarray(
        training_values,
        dtype=float,
    )
    testing_observations = np.asarray(
        testing_values,
        dtype=float,
    )
    time_values = np.asarray(testing_times)

    if training_observations.ndim != 1:
        raise ValueError(
            "training_values must be one-dimensional."
        )

    if testing_observations.ndim != 1:
        raise ValueError(
            "testing_values must be one-dimensional."
        )

    if time_values.ndim != 1:
        raise ValueError(
            "testing_times must be one-dimensional."
        )

    if training_observations.size == 0:
        raise ValueError(
            "training_values must not be empty."
        )

    if testing_observations.size == 0:
        raise ValueError(
            "testing_values must not be empty."
        )

    if testing_observations.size != time_values.size:
        raise ValueError(
            "testing_values and testing_times "
            "must have equal length."
        )

    if not np.isfinite(training_observations).all():
        raise ValueError(
            "training_values must contain only finite values."
        )

    if not np.isfinite(testing_observations).all():
        raise ValueError(
            "testing_values must contain only finite values."
        )

    start_time = perf_counter()

    training_result = fit_sd_ewma(
        training_values=training_observations,
    )

    stage_1_config = SD_EWMA_Config(
        lambda_value=training_result.lambda_value,
        variance_smoothing=(
            config.stage_1.variance_smoothing
        ),
        control_limit_multiplier=(
            config.stage_1.control_limit_multiplier
        ),
    )

    stage_1_results = run_sd_ewma(
        values=testing_observations,
        times=time_values,
        initial_z=training_result.final_z,
        initial_error_variance=(
            training_result.error_variance
        ),
        config=stage_1_config,
    )

    stage_2_results = validate_stage_1_alarms(
        stage_1_results=stage_1_results,
        values=testing_observations,
        times=time_values,
        config=config.stage_2,
    )

    elapsed = perf_counter() - start_time

    return TSSDEWMAResult(
        training_result=training_result,
        stage_1_results=stage_1_results,
        stage_2_results=stage_2_results,
        computation_time_seconds=elapsed,
    )