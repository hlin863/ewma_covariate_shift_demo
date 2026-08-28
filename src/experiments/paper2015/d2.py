"""End-to-end experiment orchestration for the paper's D2 dataset."""

from dataclasses import dataclass, field

import pandas as pd

from src.detection.baselines.ici_cdt import (
    ICICDTConfig,
    ICICDTResult,
    run_ici_cdt,
)
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
    "sd_ewma": {
        "fp_percent": 0.3,
        "fn_percent": 0.0,
        "rci": 0,
        "ct_seconds": 0.189,
    },
    "tssd_ewma": {
        "fp_percent": 0.0,
        "fn_percent": 11.1,
        "rci": 10,
        "ct_seconds": 0.224,
    },
    "ici_cdt": {
        "fp_percent": 0.0,
        "fn_percent": 55.4,
        "rci": 40,
        "ct_seconds": 0.196,
    },
}

TABLE_3_D2_SHIFT_COUNT = 9


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
    ici_block_size: int = 10
    ici_confidence_parameter: float = 2.0

    def __post_init__(self) -> None:
        if not 2 <= self.train_size < self.dataset.n_samples:
            raise ValueError("train_size must leave at least one testing observation.")
        if self.stage_2_window_size <= 0:
            raise ValueError("stage_2_window_size must be positive.")
        if self.minimum_alarm_gap < 0:
            raise ValueError("minimum_alarm_gap must be non-negative.")
        if self.lambda_mode not in {"estimate", "configured"}:
            raise ValueError("lambda_mode must be 'estimate' or 'configured'.")
        if self.ici_block_size <= 0:
            raise ValueError("ici_block_size must be positive.")
        if self.ici_confidence_parameter <= 0.0:
            raise ValueError("ici_confidence_parameter must be positive.")


@dataclass(frozen=True)
class D2ExperimentResult:
    stream: pd.DataFrame
    detector: TSSDEWMAResult
    stage_1_evaluation: RepeatedShiftEvaluation
    stage_2_evaluation: RepeatedShiftEvaluation


@dataclass(frozen=True)
class D2Table3Result:
    """Computed D2 Table III rows plus their published reference values."""

    experiment: D2ExperimentResult
    ici_cdt: ICICDTResult
    table_3_shift_times: tuple[int, ...]
    evaluations: dict[str, RepeatedShiftEvaluation]
    computed: pd.DataFrame
    comparison: pd.DataFrame


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


def _table_3_scope(
    stream: pd.DataFrame,
    shift_count: int = TABLE_3_D2_SHIFT_COUNT,
) -> tuple[pd.DataFrame, tuple[int, ...]]:
    """Return the final 1,000-point D2 window containing nine shifts.

    Table III states that D2 contains nine evaluated shifts but does not list
    their indices.  Figure 5(b) plots a high-mean late-stream window.  The
    repository therefore makes the inference auditable by selecting the last
    nine truth markers and the stationary 100-point regime immediately before
    the first selected marker.
    """

    all_shift_times = stream.loc[stream["true_shift"] == 1, "time"].to_numpy()
    if all_shift_times.size < shift_count:
        raise ValueError("The generated stream does not contain enough shifts.")
    selected = tuple(int(value) for value in all_shift_times[-shift_count:])
    shift_interval = selected[1] - selected[0] if len(selected) > 1 else 1
    start_time = selected[0] - shift_interval
    scope = stream.loc[stream["time"] >= start_time].copy()
    return scope, selected


def _score_table_3_method(
    detections: pd.DataFrame,
    scope: pd.DataFrame,
    shift_times: tuple[int, ...],
    computation_time_seconds: float,
    *,
    detection_time_column: str,
    detection_flag_column: str,
) -> RepeatedShiftEvaluation:
    start_time = int(scope["time"].iloc[0])
    end_time = int(scope["time"].iloc[-1])
    detection_times = pd.to_numeric(
        detections[detection_time_column], errors="coerce"
    )
    scoped_detections = detections.loc[
        detection_times.between(start_time, end_time, inclusive="both")
    ].copy()
    return evaluate_repeated_shift_detection(
        detections=scoped_detections,
        true_shift_times=shift_times,
        observation_times=scope["time"].to_numpy(),
        computation_time_seconds=computation_time_seconds,
        detection_time_column=detection_time_column,
        detection_flag_column=detection_flag_column,
    )


def _computed_table(evaluations: dict[str, RepeatedShiftEvaluation]) -> pd.DataFrame:
    labels = {
        "sd_ewma": "SD-EWMA",
        "tssd_ewma": "TSSD-EWMA",
        "ici_cdt": "ICI-CDT",
    }
    records = []
    for method, evaluation in evaluations.items():
        metrics = evaluation.metrics
        records.append(
            {
                "method": labels[method],
                "fp_percent": 100.0 * metrics.false_positive_rate,
                "fn_percent": 100.0 * metrics.false_negative_rate,
                "rci": metrics.mean_recognition_capability_index,
                "ct_seconds": metrics.computation_time_seconds,
            }
        )
    return pd.DataFrame.from_records(records).set_index("method")


def _comparison_table(computed: pd.DataFrame) -> pd.DataFrame:
    published = pd.DataFrame.from_dict(
        {
            "SD-EWMA": PAPER_D2_REFERENCE["sd_ewma"],
            "TSSD-EWMA": PAPER_D2_REFERENCE["tssd_ewma"],
            "ICI-CDT": PAPER_D2_REFERENCE["ici_cdt"],
        },
        orient="index",
    )
    published.index.name = "method"
    comparison = computed.add_prefix("computed_").join(
        published.add_prefix("published_")
    )
    for metric in ("fp_percent", "fn_percent", "rci", "ct_seconds"):
        comparison[f"difference_{metric}"] = (
            comparison[f"computed_{metric}"]
            - comparison[f"published_{metric}"]
        )
    return comparison


def run_d2_table_3_experiment(
    config: D2ExperimentConfig | None = None,
) -> D2Table3Result:
    """Run all three D2 Table III methods under one explicit scoring scope."""

    effective = config or D2ExperimentConfig()
    experiment = run_d2_experiment(effective)
    scope, shift_times = _table_3_scope(experiment.stream)
    training = experiment.stream.iloc[: effective.train_size]
    testing = experiment.stream.iloc[effective.train_size :]
    ici_cdt = run_ici_cdt(
        training_values=training["x"].to_numpy(),
        testing_values=testing["x"].to_numpy(),
        testing_times=testing["time"].to_numpy(),
        config=ICICDTConfig(
            block_size=effective.ici_block_size,
            confidence_parameter=effective.ici_confidence_parameter,
        ),
    )

    evaluations = {
        "sd_ewma": _score_table_3_method(
            experiment.detector.stage_1_results,
            scope,
            shift_times,
            experiment.detector.stage_1_computation_time_seconds,
            detection_time_column="time",
            detection_flag_column="stage_1_alarm",
        ),
        "tssd_ewma": _score_table_3_method(
            experiment.detector.stage_2_results,
            scope,
            shift_times,
            experiment.detector.computation_time_seconds,
            detection_time_column="validation_time",
            detection_flag_column="confirmed_shift",
        ),
        "ici_cdt": _score_table_3_method(
            ici_cdt.trace,
            scope,
            shift_times,
            ici_cdt.computation_time_seconds,
            detection_time_column="time",
            detection_flag_column="ici_alarm",
        ),
    }
    computed = _computed_table(evaluations)
    return D2Table3Result(
        experiment=experiment,
        ici_cdt=ici_cdt,
        table_3_shift_times=shift_times,
        evaluations=evaluations,
        computed=computed,
        comparison=_comparison_table(computed),
    )
