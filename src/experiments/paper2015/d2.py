"""End-to-end experiment orchestration for the paper's D2 dataset."""

from dataclasses import dataclass, field

import pandas as pd

from src.detection.stage1.sd_ewma import SD_EWMA_Config
from src.detection.stage2.ks import Stage2Config
from src.detection.two_stage import (
    TSSDEWMAConfig,
    TSSDEWMAResult,
    run_tssd_ewma,
)
from src.reporting.metrics import (
    RepeatedShiftEvaluation,
    evaluate_repeated_shift_detection,
)
from src.simulation.jumping_mean import (
    JumpingMeanConfig,
    generate_d2_jumping_mean,
)


PAPER_D2_REFERENCE = {
    "lambda": 0.40,
    "sd_ewma": {"fp_percent": 0.3, "fn_percent": 0.0, "rci": 0},
    "tssd_ewma": {"fp_percent": 0.0, "fn_percent": 11.1, "rci": 10},
}


@dataclass(frozen=True)
class D2ExperimentConfig:
    dataset: JumpingMeanConfig = field(default_factory=JumpingMeanConfig)
    train_size: int = 500
    variance_smoothing: float = 0.05
    control_limit_multiplier: float = 1.96
    variance_update_mode: str = "always"
    stage_2_window_size: int = 10
    stage_2_alpha: float = 0.05
    minimum_alarm_gap: int = 10
    lambda_mode: str = "estimate"
    configured_lambda: float = 0.40

    def __post_init__(self) -> None:
        if not 2 <= self.train_size < self.dataset.n_samples:
            raise ValueError("train_size must leave at least one testing observation.")
        if self.stage_2_window_size <= 0:
            raise ValueError("stage_2_window_size must be positive.")
        if self.minimum_alarm_gap < 0:
            raise ValueError("minimum_alarm_gap must be non-negative.")
        if self.lambda_mode not in {"estimate", "configured"}:
            raise ValueError("lambda_mode must be 'estimate' or 'configured'.")


@dataclass(frozen=True)
class D2ExperimentResult:
    stream: pd.DataFrame
    detector: TSSDEWMAResult
    stage_1_evaluation: RepeatedShiftEvaluation
    stage_2_evaluation: RepeatedShiftEvaluation


def run_d2_experiment(
    config: D2ExperimentConfig | None = None,
) -> D2ExperimentResult:
    """Generate D2, run SD/TSSD-EWMA and score every testing shift.

    This evaluates the full testing stream. It deliberately does not claim to
    reproduce Table III's undocumented nine-shift scoring subset.
    """

    effective = config or D2ExperimentConfig()
    stream = generate_d2_jumping_mean(effective.dataset)
    training = stream.iloc[: effective.train_size]
    testing = stream.iloc[effective.train_size :]
    true_shift_times = testing.loc[testing["true_shift"] == 1, "time"].to_numpy()
    testing_times = testing["time"].to_numpy()

    detector_config = TSSDEWMAConfig(
        stage_1=SD_EWMA_Config(
            lambda_value=effective.configured_lambda,
            variance_smoothing=effective.variance_smoothing,
            control_limit_multiplier=effective.control_limit_multiplier,
            variance_update_mode=effective.variance_update_mode,
        ),
        stage_2=Stage2Config(
            before_size=effective.stage_2_window_size,
            after_size=effective.stage_2_window_size,
            alpha=effective.stage_2_alpha,
            minimum_alarm_gap=effective.minimum_alarm_gap,
        ),
        lambda_mode=effective.lambda_mode,
    )
    detector = run_tssd_ewma(
        training_values=training["x"].to_numpy(),
        testing_values=testing["x"].to_numpy(),
        testing_times=testing_times,
        config=detector_config,
    )

    stage_1_evaluation = evaluate_repeated_shift_detection(
        detections=detector.stage_1_results,
        true_shift_times=true_shift_times,
        observation_times=testing_times,
        computation_time_seconds=detector.stage_1_computation_time_seconds,
    )
    stage_2_evaluation = evaluate_repeated_shift_detection(
        detections=detector.stage_2_results,
        true_shift_times=true_shift_times,
        observation_times=testing_times,
        computation_time_seconds=detector.computation_time_seconds,
        detection_time_column="validation_time",
        detection_flag_column="confirmed_shift",
    )
    return D2ExperimentResult(
        stream=stream,
        detector=detector,
        stage_1_evaluation=stage_1_evaluation,
        stage_2_evaluation=stage_2_evaluation,
    )
