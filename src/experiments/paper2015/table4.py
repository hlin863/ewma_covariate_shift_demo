"""D3/D4 orchestration for reproducing the paper's Table IV structure."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from time import perf_counter

import numpy as np
import pandas as pd

from src.detection.multivariate_preprocessing import fit_paper_ica, fit_paper_pca
from src.detection.stage1.paper_msd_ewma import (
    fit_paper_msd_ewma,
    run_paper_msd_ewma,
)
from src.detection.stage2.paper_hotelling import (
    PaperHotellingConfig,
    validate_paper_hotelling_alarms,
)
from src.simulation.multivariate_shift import (
    MultivariateShiftConfig,
    generate_d3_multivariate_normal,
    generate_d4_multivariate_t,
)


TABLE_4_METHODS = (
    "MSD-EWMA",
    "MSD-EWMA-PCA",
    "MSD-EWMA-ICA",
    "TSMSD-EWMA",
    "TSMSD-EWMA-ICA",
)

PAPER_TABLE_4_REFERENCE = {
    "D3": {
        "MSD-EWMA": (5.0, 8.0, 9.0, 0.228),
        "MSD-EWMA-PCA": (5.5, 6.0, 8.0, 0.272),
        "MSD-EWMA-ICA": (5.0, 8.0, 8.0, 0.528),
        "TSMSD-EWMA": (2.0, 5.0, 25.0, 0.201),
        "TSMSD-EWMA-ICA": (2.5, 5.0, 25.0, 0.594),
    },
    "D4": {
        "MSD-EWMA": (4.0, 15.0, 7.0, 0.213),
        "MSD-EWMA-PCA": (5.5, 5.0, 8.0, 0.251),
        "MSD-EWMA-ICA": (5.0, 15.0, 7.0, 0.677),
        "TSMSD-EWMA": (5.0, 7.0, 25.0, 0.202),
        "TSMSD-EWMA-ICA": (4.5, 9.0, 25.0, 0.647),
    },
}


@dataclass(frozen=True)
class Table4ExperimentConfig:
    dataset: MultivariateShiftConfig = field(default_factory=MultivariateShiftConfig)
    repetitions: int = 100
    lambda_value: float = 0.10
    raw_control_limit: float = 22.67
    pca_control_limit: float = 10.58
    pca_components: int = 2
    stage_2_window_size: int = 25
    stage_2_alpha: float = 0.05
    covariance_regularization: float = 1e-6

    def __post_init__(self) -> None:
        if self.repetitions < 1:
            raise ValueError("repetitions must be positive.")
        if self.dataset.first_shift_time != 101 or self.dataset.second_shift_time != 201:
            raise ValueError("Table IV requires paper shift times 101 and 201.")
        if not 0.0 < self.lambda_value <= 1.0:
            raise ValueError("lambda_value must be in (0, 1].")
        if self.pca_components != 2:
            raise ValueError("Table IV specifies the first two PCA components.")


@dataclass(frozen=True)
class Table4Result:
    computed: pd.DataFrame
    comparison: pd.DataFrame
    representative_streams: dict[str, pd.DataFrame]
    representative_traces: dict[str, pd.DataFrame]
    metadata: dict[str, object]


@dataclass(frozen=True)
class _Table4RunScore:
    false_positive_count: int
    false_negative_count: int
    recognition_delay: int | None
    pre_shift_observation_count: int
    computation_time_seconds: float


def _feature_columns(stream: pd.DataFrame) -> list[str]:
    return [column for column in stream.columns if column.startswith("x")]


def _score(
    detections: pd.DataFrame,
    stream: pd.DataFrame,
    elapsed: float,
    *,
    time_column: str,
    flag_column: str,
) -> _Table4RunScore:
    """Score the first 0 -> 1 departure used in the paper's Figure 6.

    The generated stream still contains the return to zero at observation 201,
    but the paper describes/plots the detected shift after observation 100.
    Sustained alarms while the process remains in its shifted state are not new
    false shift events.  FP is therefore measured in the in-control prefix;
    FN and RCI use the first report from 101 through 200.
    """

    first_shift = int(stream.loc[stream["true_shift"] == 1, "time"].iloc[0])
    return_time = int(stream.loc[stream["true_shift"] == 1, "time"].iloc[1])
    reported = pd.to_numeric(
        detections.loc[detections[flag_column].fillna(False).astype(bool), time_column],
        errors="coerce",
    ).dropna()
    false_positive_count = int((reported < first_shift).sum())
    post_shift = reported[(reported >= first_shift) & (reported < return_time)]
    first_detection = int(post_shift.min()) if not post_shift.empty else None
    return _Table4RunScore(
        false_positive_count=false_positive_count,
        false_negative_count=int(first_detection is None),
        recognition_delay=(
            first_detection - first_shift if first_detection is not None else None
        ),
        pre_shift_observation_count=first_shift - 1,
        computation_time_seconds=elapsed,
    )


def _run_dataset_once(
    dataset_name: str,
    config: Table4ExperimentConfig,
    seed: int,
) -> tuple[dict[str, object], pd.DataFrame, dict[str, pd.DataFrame]]:
    dataset_config = replace(config.dataset, random_seed=seed)
    generator = (
        generate_d3_multivariate_normal
        if dataset_name == "D3"
        else generate_d4_multivariate_t
    )
    stream = generator(dataset_config)
    values = stream[_feature_columns(stream)].to_numpy()
    training_size = dataset_config.first_shift_time - 1
    training = values[:training_size]
    times = stream["time"].to_numpy()

    transform_start = perf_counter()
    pca = fit_paper_pca(training, values, config.pca_components)
    pca_transform_time = perf_counter() - transform_start
    transform_start = perf_counter()
    ica = fit_paper_ica(training, values, random_seed=seed)
    ica_transform_time = perf_counter() - transform_start

    feature_sets = {
        "raw": (training, values, 0.0, config.raw_control_limit),
        "pca": (
            pca.training_transformed,
            pca.full_transformed,
            pca_transform_time,
            config.pca_control_limit,
        ),
        "ica": (
            ica.training_transformed,
            ica.full_transformed,
            ica_transform_time,
            config.raw_control_limit,
        ),
    }
    stage_1: dict[str, pd.DataFrame] = {}
    stage_1_times: dict[str, float] = {}
    for name, (training_values, full_values, transform_time, limit) in feature_sets.items():
        started = perf_counter()
        fitted = fit_paper_msd_ewma(
            training_values,
            lambda_value=config.lambda_value,
            regularization=config.covariance_regularization,
        )
        stage_1[name] = run_paper_msd_ewma(full_values, times, fitted, limit)
        stage_1_times[name] = transform_time + perf_counter() - started

    hotelling_config = PaperHotellingConfig(
        window_size=config.stage_2_window_size,
        alpha=config.stage_2_alpha,
        regularization=config.covariance_regularization,
    )
    stage_2: dict[str, pd.DataFrame] = {}
    stage_2_times: dict[str, float] = {}
    for name in ("raw", "ica"):
        started = perf_counter()
        stage_2[name] = validate_paper_hotelling_alarms(
            feature_sets[name][1],
            times,
            stage_1[name],
            hotelling_config,
        )
        stage_2_times[name] = stage_1_times[name] + perf_counter() - started

    evaluations = {
        "MSD-EWMA": _score(stage_1["raw"], stream, stage_1_times["raw"], time_column="time", flag_column="stage_1_alarm"),
        "MSD-EWMA-PCA": _score(stage_1["pca"], stream, stage_1_times["pca"], time_column="time", flag_column="stage_1_alarm"),
        "MSD-EWMA-ICA": _score(stage_1["ica"], stream, stage_1_times["ica"], time_column="time", flag_column="stage_1_alarm"),
        "TSMSD-EWMA": _score(stage_2["raw"], stream, stage_2_times["raw"], time_column="validation_time", flag_column="confirmed_shift"),
        "TSMSD-EWMA-ICA": _score(stage_2["ica"], stream, stage_2_times["ica"], time_column="validation_time", flag_column="confirmed_shift"),
    }
    traces = {
        "MSD-EWMA": stage_1["raw"],
        "MSD-EWMA-PCA": stage_1["pca"],
        "MSD-EWMA-ICA": stage_1["ica"],
        "TSMSD-EWMA": stage_2["raw"],
        "TSMSD-EWMA-ICA": stage_2["ica"],
    }
    return evaluations, stream, traces


def _published_frame() -> pd.DataFrame:
    rows = []
    for dataset_name, methods in PAPER_TABLE_4_REFERENCE.items():
        for method, values in methods.items():
            rows.append((dataset_name, method, *values))
    return pd.DataFrame(
        rows,
        columns=["dataset", "method", "fp_percent", "fn_percent", "rci", "ct_seconds"],
    ).set_index(["dataset", "method"])


def run_table_4_experiment(
    config: Table4ExperimentConfig | None = None,
    datasets: tuple[str, ...] = ("D3", "D4"),
) -> Table4Result:
    """Run D3/D4 Monte Carlo trials and compare with published Table IV."""

    effective = config or Table4ExperimentConfig()
    invalid = set(datasets).difference({"D3", "D4"})
    if invalid:
        raise ValueError(f"Unsupported datasets: {sorted(invalid)}")

    accumulators: dict[tuple[str, str], dict[str, object]] = {}
    representative_streams: dict[str, pd.DataFrame] = {}
    representative_traces: dict[str, pd.DataFrame] = {}
    for dataset_offset, dataset_name in enumerate(datasets):
        for repetition in range(effective.repetitions):
            seed = effective.dataset.random_seed + 10_000 * dataset_offset + repetition
            evaluations, stream, traces = _run_dataset_once(dataset_name, effective, seed)
            if repetition == 0:
                representative_streams[dataset_name] = stream
                for method, trace in traces.items():
                    representative_traces[f"{dataset_name}_{method}"] = trace
            for method, evaluation in evaluations.items():
                key = (dataset_name, method)
                bucket = accumulators.setdefault(
                    key,
                    {"fp": 0, "fn": 0, "shifts": 0, "non_shifts": 0, "delays": [], "ct": []},
                )
                bucket["fp"] += evaluation.false_positive_count
                bucket["fn"] += evaluation.false_negative_count
                bucket["shifts"] += 1
                bucket["non_shifts"] += evaluation.pre_shift_observation_count
                if evaluation.recognition_delay is not None:
                    bucket["delays"].append(evaluation.recognition_delay)
                bucket["ct"].append(evaluation.computation_time_seconds)

    rows = []
    for dataset_name in datasets:
        for method in TABLE_4_METHODS:
            bucket = accumulators[(dataset_name, method)]
            delays = bucket["delays"]
            rows.append(
                {
                    "dataset": dataset_name,
                    "method": method,
                    "fp_percent": 100.0 * bucket["fp"] / bucket["non_shifts"],
                    "fn_percent": 100.0 * bucket["fn"] / bucket["shifts"],
                    "rci": float(np.mean(delays)) if delays else np.nan,
                    "ct_seconds": float(np.mean(bucket["ct"])),
                }
            )
    computed = pd.DataFrame(rows).set_index(["dataset", "method"])
    published = _published_frame().loc[computed.index]
    comparison = published.add_prefix("published_").join(
        computed.add_prefix("computed_")
    )
    for metric in ("fp_percent", "fn_percent", "rci", "ct_seconds"):
        comparison[f"difference_{metric}"] = (
            comparison[f"computed_{metric}"] - comparison[f"published_{metric}"]
        )

    return Table4Result(
        computed=computed,
        comparison=comparison,
        representative_streams=representative_streams,
        representative_traces=representative_traces,
        metadata={
            "repetitions": effective.repetitions,
            "lambda": effective.lambda_value,
            "raw_control_limit": effective.raw_control_limit,
            "pca_control_limit": effective.pca_control_limit,
            "pca_components": effective.pca_components,
            "stage_2_window_size": effective.stage_2_window_size,
            "ica_implementation": "scikit-learn FastICA",
            "d4_parameterization": "scale adjusted so t covariance equals paper Sigma",
            "scoring_scope": "first departure at 101; return at 201 retained in stream",
        },
    )
