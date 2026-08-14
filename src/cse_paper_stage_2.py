"""Paper-aligned Stage-II validation for the CSE detector.

The 2019 CSE-UAEL paper describes Stage II as a multivariate two-sample
Hotelling T-squared test using two distinct samples with equal numbers of
observations.  Its cited 2015 CSE method makes the construction explicit:
partition the feature stream around a Stage-I warning into two disjoint,
equal-length subsequences and test whether their multivariate means remain
stationary.

This module keeps that paper-reproduction path separate from the existing
single-vector/training-reference interpretation and from the more general
retrospective validator.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import f

from src.multivariate_stage_2 import calculate_hotelling_t_squared


@dataclass(frozen=True)
class PaperTwoSampleHotellingConfig:
    """Configuration for the paper-aligned equal-window Stage-II test.

    ``window_size`` is intentionally configurable because the papers define an
    equal-length subsequence size but the 2019 Algorithm 1 excerpt does not fix
    one universal numerical value for all datasets.
    """

    window_size: int = 50
    alpha: float = 0.05
    covariance_method: str = "empirical"
    regularization: float = 1e-6
    minimum_alarm_gap: int | None = None

    def __post_init__(self) -> None:
        if self.window_size < 2:
            raise ValueError("window_size must be at least 2.")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0, 1).")
        if self.covariance_method not in {"empirical", "shrinkage"}:
            raise ValueError(
                "covariance_method must be 'empirical' or 'shrinkage'."
            )
        if self.regularization < 0.0:
            raise ValueError("regularization must not be negative.")
        if self.minimum_alarm_gap is not None and self.minimum_alarm_gap < 0:
            raise ValueError("minimum_alarm_gap must not be negative.")


def calculate_paper_two_sample_hotelling(
    current_sample: np.ndarray,
    reference_sample: np.ndarray,
    *,
    covariance_method: str = "empirical",
    regularization: float = 1e-6,
) -> tuple[float, float, int, int, float]:
    """Calculate the paper's two-sample Hotelling statistic.

    Both inputs must be true multivariate samples with shape ``(n, D)``.  The
    paper specifies equal numbers of observations, so unequal sample lengths
    are rejected rather than silently accepted or reshaped from a single
    feature vector.
    """

    current = np.asarray(current_sample, dtype=float)
    reference = np.asarray(reference_sample, dtype=float)

    if current.ndim != 2 or reference.ndim != 2:
        raise ValueError(
            "current_sample and reference_sample must both have shape (n, D)."
        )
    if current.shape[0] != reference.shape[0]:
        raise ValueError(
            "paper_two_sample requires equal numbers of observations."
        )
    if current.shape[1] != reference.shape[1]:
        raise ValueError("both samples must have the same feature count.")

    return calculate_hotelling_t_squared(
        reference,
        current,
        covariance_method=covariance_method,
        regularization=regularization,
    )


def validate_paper_two_sample_alarms(
    features: np.ndarray,
    times: np.ndarray,
    stage_1_results: pd.DataFrame,
    config: PaperTwoSampleHotellingConfig | None = None,
) -> pd.DataFrame:
    """Validate Stage-I warnings with equal disjoint subsequences.

    For a warning at index ``i`` and window length ``H``:

    * reference sample: ``features[i-H+1 : i+1]``
    * current sample: ``features[i+1 : i+1+H]``

    This mirrors the two-subsequence construction described by the 2015 CSE
    method cited by the 2019 CSE-UAEL paper.  Validation is therefore
    retrospective and becomes available once the second subsequence exists.
    """

    validation_config = config or PaperTwoSampleHotellingConfig()
    observations = np.asarray(features, dtype=float)
    time_values = np.asarray(times)

    if observations.ndim != 2 or observations.shape[0] == 0:
        raise ValueError("features must have shape (n_observations, n_features).")
    if observations.shape[1] == 0:
        raise ValueError("features must contain at least one feature column.")
    if time_values.ndim != 1 or time_values.size != observations.shape[0]:
        raise ValueError("times must be 1D and match the feature row count.")
    if not np.isfinite(observations).all():
        raise ValueError("features must contain only finite values.")
    if np.unique(time_values).size != time_values.size:
        raise ValueError("times must uniquely identify observations.")

    required_columns = {"time", "stage_1_alarm"}
    missing = required_columns.difference(stage_1_results.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    columns = [
        "alarm_time",
        "validation_time",
        "reference_start_time",
        "reference_end_time",
        "current_start_time",
        "current_end_time",
        "sample_size",
        "n_features",
        "hotelling_t_squared",
        "f_statistic",
        "degrees_of_freedom_1",
        "degrees_of_freedom_2",
        "critical_value",
        "p_value",
        "status",
        "confirmed_shift",
    ]

    alarm_times = (
        stage_1_results.loc[
            stage_1_results["stage_1_alarm"].astype(bool), "time"
        ]
        .drop_duplicates()
        .to_numpy()
    )

    records: list[dict[str, object]] = []
    last_evaluated_index: int | None = None
    minimum_gap = (
        validation_config.window_size
        if validation_config.minimum_alarm_gap is None
        else validation_config.minimum_alarm_gap
    )

    for alarm_time in alarm_times:
        positions = np.flatnonzero(time_values == alarm_time)
        if positions.size != 1:
            raise ValueError(
                "Each alarm time must identify exactly one testing observation."
            )

        alarm_index = int(positions[0])
        window_size = validation_config.window_size
        reference_start = alarm_index - window_size + 1
        current_start = alarm_index + 1
        current_end = current_start + window_size

        record: dict[str, object] = {
            "alarm_time": int(alarm_time),
            "validation_time": None,
            "reference_start_time": None,
            "reference_end_time": None,
            "current_start_time": None,
            "current_end_time": None,
            "sample_size": window_size,
            "n_features": int(observations.shape[1]),
            "hotelling_t_squared": np.nan,
            "f_statistic": np.nan,
            "degrees_of_freedom_1": int(observations.shape[1]),
            "degrees_of_freedom_2": np.nan,
            "critical_value": np.nan,
            "p_value": np.nan,
            "status": None,
            "confirmed_shift": False,
        }

        if (
            last_evaluated_index is not None
            and alarm_index - last_evaluated_index < minimum_gap
        ):
            record["status"] = "skipped_nearby_alarm"
            records.append(record)
            continue

        if reference_start < 0:
            record["status"] = "insufficient_reference_window"
            records.append(record)
            continue
        if current_end > observations.shape[0]:
            record["status"] = "pending_current_window"
            records.append(record)
            continue

        reference_sample = observations[reference_start : alarm_index + 1]
        current_sample = observations[current_start:current_end]

        try:
            (
                t_squared,
                f_statistic,
                df1,
                df2,
                p_value,
            ) = calculate_paper_two_sample_hotelling(
                current_sample=current_sample,
                reference_sample=reference_sample,
                covariance_method=validation_config.covariance_method,
                regularization=validation_config.regularization,
            )
        except ValueError as error:
            if "too small for the number of features" not in str(error):
                raise
            record["status"] = "insufficient_degrees_of_freedom"
            records.append(record)
            continue

        critical_value = float(
            f.ppf(1.0 - validation_config.alpha, df1, df2)
        )
        confirmed = bool(p_value < validation_config.alpha)

        record.update({
            "validation_time": int(time_values[current_end - 1]),
            "reference_start_time": int(time_values[reference_start]),
            "reference_end_time": int(time_values[alarm_index]),
            "current_start_time": int(time_values[current_start]),
            "current_end_time": int(time_values[current_end - 1]),
            "hotelling_t_squared": t_squared,
            "f_statistic": f_statistic,
            "degrees_of_freedom_1": df1,
            "degrees_of_freedom_2": df2,
            "critical_value": critical_value,
            "p_value": p_value,
            "status": "confirmed" if confirmed else "rejected",
            "confirmed_shift": confirmed,
        })
        records.append(record)
        last_evaluated_index = alarm_index

    return pd.DataFrame.from_records(records, columns=columns)
