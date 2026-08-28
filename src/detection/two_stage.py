"""Combined runner for the univariate two-stage TSSD-EWMA detector."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd

from src.detection.stage1.sd_ewma import (
    EWMATrainingResult,
    SD_EWMA_Config,
    fit_sd_ewma,
    run_sd_ewma,
)
from src.detection.stage2.ks import Stage2Config, validate_stage_1_alarms


@dataclass(frozen=True)
class TSSDEWMAConfig:
    stage_1: SD_EWMA_Config
    stage_2: Stage2Config
    lambda_mode: str = "estimate"

    def __post_init__(self) -> None:
        if self.lambda_mode not in {"estimate", "configured"}:
            raise ValueError("lambda_mode must be 'estimate' or 'configured'.")


@dataclass(frozen=True)
class TSSDEWMAResult:
    training_result: EWMATrainingResult
    stage_1_results: pd.DataFrame
    stage_2_results: pd.DataFrame
    stage_1_computation_time_seconds: float
    stage_2_computation_time_seconds: float
    computation_time_seconds: float


def run_tssd_ewma(training_values, testing_values, testing_times, config):
    training = np.asarray(training_values, dtype=float)
    testing = np.asarray(testing_values, dtype=float)
    times = np.asarray(testing_times)
    if training.ndim != 1 or testing.ndim != 1 or times.ndim != 1:
        raise ValueError("training_values, testing_values and testing_times must be one-dimensional.")
    if testing.size != times.size:
        raise ValueError("testing_values and testing_times must have equal length.")
    if training.size == 0 or testing.size == 0:
        raise ValueError("training and testing values must not be empty.")
    if not np.isfinite(training).all() or not np.isfinite(testing).all():
        raise ValueError("training and testing values must contain only finite values.")

    start = perf_counter()
    lambda_override = (
        config.stage_1.lambda_value
        if config.lambda_mode == "configured"
        else None
    )
    training_result = fit_sd_ewma(training, lambda_override=lambda_override)
    effective_stage_1 = SD_EWMA_Config(
        lambda_value=training_result.lambda_value,
        variance_smoothing=config.stage_1.variance_smoothing,
        control_limit_multiplier=config.stage_1.control_limit_multiplier,
        variance_update_mode=config.stage_1.variance_update_mode,
    )
    stage_1_results = run_sd_ewma(
        testing,
        times,
        training_result.final_z,
        training_result.error_variance,
        effective_stage_1,
    )
    stage_1_finished = perf_counter()
    stage_2_results = validate_stage_1_alarms(
        stage_1_results, testing, times, config.stage_2
    )
    finished = perf_counter()
    return TSSDEWMAResult(
        training_result=training_result,
        stage_1_results=stage_1_results,
        stage_2_results=stage_2_results,
        stage_1_computation_time_seconds=stage_1_finished - start,
        stage_2_computation_time_seconds=finished - stage_1_finished,
        computation_time_seconds=finished - start,
    )
