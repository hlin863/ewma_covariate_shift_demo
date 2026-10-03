"""Retrospective multivariate Stage-II validation using Hotelling's T-squared."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import f
from sklearn.covariance import LedoitWolf


@dataclass(frozen=True)
class HotellingConfig:
    """Configuration for retrospective Hotelling validation."""

    before_size: int = 50
    after_size: int = 50
    alpha: float = 0.05
    covariance_method: str = "shrinkage"
    regularization: float = 1e-6
    minimum_alarm_gap: int | None = None

    def __post_init__(self) -> None:
        if self.before_size <= 0 or self.after_size <= 0:
            raise ValueError("window sizes must be positive.")

        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0, 1).")

        if self.covariance_method not in {"empirical", "shrinkage"}:
            raise ValueError(
                "covariance_method must be 'empirical' or 'shrinkage'."
            )

        if self.regularization < 0.0:
            raise ValueError("regularization must not be negative.")

        if (
            self.minimum_alarm_gap is not None
            and self.minimum_alarm_gap < 0
        ):
            raise ValueError("minimum_alarm_gap must not be negative.")


@dataclass(frozen=True)
class HotellingValidationResult:
    """Result of validating one multivariate Stage-I warning."""

    alarm_time: int
    validation_time: int
    before_start_time: int
    before_end_time: int
    after_start_time: int
    after_end_time: int
    before_size: int
    after_size: int
    n_features: int
    hotelling_t_squared: float
    f_statistic: float
    degrees_of_freedom_1: int
    degrees_of_freedom_2: int
    critical_value: float
    p_value: float
    confirmed_shift: bool


def _validate_multivariate_stream_inputs(
    features: np.ndarray,
    times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate a multivariate feature stream and its time index."""

    observations = np.asarray(features, dtype=float)
    time_values = np.asarray(times)

    if observations.ndim != 2:
        raise ValueError(
            "features must have shape (n_observations, n_features)."
        )

    if observations.shape[0] == 0:
        raise ValueError("features and times must not be empty.")

    if observations.shape[1] == 0:
        raise ValueError(
            "features must contain at least one feature column."
        )

    if time_values.ndim != 1:
        raise ValueError("times must be one-dimensional.")

    if observations.shape[0] != time_values.size:
        raise ValueError(
            "features and times must contain the same number of observations."
        )

    if not np.isfinite(observations).all():
        raise ValueError("features must contain only finite values.")

    try:
        finite_times = np.isfinite(time_values).all()
    except TypeError as error:
        raise ValueError("times must contain finite numeric values.") from error

    if not finite_times:
        raise ValueError("times must contain only finite values.")

    if np.unique(time_values).size != time_values.size:
        raise ValueError("times must uniquely identify observations.")

    if time_values.size > 1 and np.any(np.diff(time_values) <= 0):
        raise ValueError("times must be strictly increasing.")

    return observations, time_values


def _estimate_group_covariances(
    before_values: np.ndarray,
    after_values: np.ndarray,
    *,
    method: str,
    regularization: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate covariance matrices for the two retrospective windows."""

    n_features = before_values.shape[1]

    if method == "empirical":
        before_covariance = np.atleast_2d(
            np.cov(before_values, rowvar=False, ddof=1)
        )
        after_covariance = np.atleast_2d(
            np.cov(after_values, rowvar=False, ddof=1)
        )
    else:
        before_covariance = LedoitWolf().fit(before_values).covariance_
        after_covariance = LedoitWolf().fit(after_values).covariance_

    identity = np.eye(n_features)

    return (
        np.asarray(before_covariance, dtype=float)
        + regularization * identity,
        np.asarray(after_covariance, dtype=float)
        + regularization * identity,
    )


def calculate_hotelling_t_squared(
    before_values: np.ndarray,
    after_values: np.ndarray,
    *,
    covariance_method: str = "shrinkage",
    regularization: float = 1e-6,
) -> tuple[float, float, int, int, float]:
    """Calculate a two-sample Hotelling statistic and F-test p-value.

    The statistic follows the paper's two-window structure:

        HT^2 = (mu_1 - mu_2)^T
               (Sigma_1 / n_1 + Sigma_2 / n_2)^(-1)
               (mu_1 - mu_2)

    The returned F statistic uses the standard two-sample Hotelling
    degrees-of-freedom conversion as an operational approximation.
    """

    first = np.asarray(before_values, dtype=float)
    second = np.asarray(after_values, dtype=float)

    if first.ndim != 2 or second.ndim != 2:
        raise ValueError("both samples must be two-dimensional matrices.")

    if first.shape[1] != second.shape[1]:
        raise ValueError("both samples must have the same feature count.")

    if first.shape[0] < 2 or second.shape[0] < 2:
        raise ValueError("each sample must contain at least two observations.")

    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("samples must contain only finite values.")

    if covariance_method not in {"empirical", "shrinkage"}:
        raise ValueError(
            "covariance_method must be 'empirical' or 'shrinkage'."
        )

    if regularization < 0.0:
        raise ValueError("regularization must not be negative.")

    n_before, n_features = first.shape
    n_after = second.shape[0]

    degrees_of_freedom_1 = int(n_features)
    degrees_of_freedom_2 = int(
        n_before + n_after - n_features - 1
    )

    if degrees_of_freedom_2 <= 0:
        raise ValueError(
            "window sizes are too small for the number of features."
        )

    before_covariance, after_covariance = _estimate_group_covariances(
        first,
        second,
        method=covariance_method,
        regularization=regularization,
    )

    mean_difference = first.mean(axis=0) - second.mean(axis=0)
    covariance_of_difference = (
        before_covariance / n_before
        + after_covariance / n_after
    )

    inverse_covariance = np.linalg.pinv(covariance_of_difference)

    hotelling_t_squared = float(
        mean_difference.T
        @ inverse_covariance
        @ mean_difference
    )

    # Standard Hotelling T-squared to F conversion. This provides a
    # stable operational p-value while preserving the paper's HT^2
    # comparison of the two retrospective multivariate windows.
    conversion = (
        degrees_of_freedom_2
        / (
            degrees_of_freedom_1
            * (n_before + n_after - 2)
        )
    )
    f_statistic = float(conversion * hotelling_t_squared)
    p_value = float(
        f.sf(
            f_statistic,
            degrees_of_freedom_1,
            degrees_of_freedom_2,
        )
    )

    return (
        hotelling_t_squared,
        f_statistic,
        degrees_of_freedom_1,
        degrees_of_freedom_2,
        p_value,
    )


def validate_multivariate_alarm(
    features: np.ndarray,
    times: np.ndarray,
    alarm_time: int,
    config: HotellingConfig | None = None,
) -> HotellingValidationResult:
    """Validate one Stage-I warning using disjoint multivariate windows."""

    observations, time_values = _validate_multivariate_stream_inputs(
        features=features,
        times=times,
    )
    hotelling_config = config or HotellingConfig()

    alarm_positions = np.flatnonzero(time_values == alarm_time)

    if alarm_positions.size != 1:
        raise ValueError(
            "alarm_time must identify exactly one observation."
        )

    alarm_index = int(alarm_positions[0])
    before_start = (
        alarm_index - hotelling_config.before_size + 1
    )
    after_start = alarm_index + 1
    after_end = after_start + hotelling_config.after_size

    if before_start < 0:
        raise ValueError(
            "Insufficient observations before the alarm."
        )

    if after_end > observations.shape[0]:
        raise ValueError(
            "Insufficient observations after the alarm."
        )

    before_values = observations[
        before_start:alarm_index + 1
    ]
    after_values = observations[
        after_start:after_end
    ]

    (
        hotelling_t_squared,
        f_statistic,
        degrees_of_freedom_1,
        degrees_of_freedom_2,
        p_value,
    ) = calculate_hotelling_t_squared(
        before_values,
        after_values,
        covariance_method=hotelling_config.covariance_method,
        regularization=hotelling_config.regularization,
    )

    critical_value = float(
        f.ppf(
            1.0 - hotelling_config.alpha,
            degrees_of_freedom_1,
            degrees_of_freedom_2,
        )
    )

    return HotellingValidationResult(
        alarm_time=int(alarm_time),
        validation_time=int(time_values[after_end - 1]),
        before_start_time=int(time_values[before_start]),
        before_end_time=int(time_values[alarm_index]),
        after_start_time=int(time_values[after_start]),
        after_end_time=int(time_values[after_end - 1]),
        before_size=int(before_values.shape[0]),
        after_size=int(after_values.shape[0]),
        n_features=int(observations.shape[1]),
        hotelling_t_squared=hotelling_t_squared,
        f_statistic=f_statistic,
        degrees_of_freedom_1=degrees_of_freedom_1,
        degrees_of_freedom_2=degrees_of_freedom_2,
        critical_value=critical_value,
        p_value=p_value,
        confirmed_shift=bool(
            p_value < hotelling_config.alpha
        ),
    )


def validate_multivariate_alarms(
    features: np.ndarray,
    times: np.ndarray,
    stage_1_results: pd.DataFrame,
    config: HotellingConfig | None = None,
) -> pd.DataFrame:
    """Validate all eligible Stage-I warnings in chronological order."""

    required_columns = {"time", "stage_1_alarm"}
    missing_columns = required_columns.difference(
        stage_1_results.columns
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    observations, time_values = _validate_multivariate_stream_inputs(
        features=features,
        times=times,
    )
    hotelling_config = config or HotellingConfig()

    minimum_alarm_gap = (
        hotelling_config.after_size
        if hotelling_config.minimum_alarm_gap is None
        else hotelling_config.minimum_alarm_gap
    )

    alarm_times = (
        stage_1_results.loc[
            stage_1_results["stage_1_alarm"] == 1,
            "time",
        ]
        .drop_duplicates()
        .to_numpy()
    )

    alarm_positions: list[tuple[int, int]] = []

    for alarm_time in alarm_times:
        positions = np.flatnonzero(time_values == alarm_time)

        if positions.size != 1:
            raise ValueError(
                "Each alarm time must identify exactly one observation."
            )

        alarm_positions.append(
            (int(positions[0]), int(alarm_time))
        )

    alarm_positions.sort(key=lambda item: item[0])

    columns = [
        "alarm_time",
        "validation_time",
        "before_start_time",
        "before_end_time",
        "after_start_time",
        "after_end_time",
        "before_size",
        "after_size",
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

    records: list[dict[str, object]] = []
    last_evaluated_index: int | None = None

    for alarm_index, alarm_time in alarm_positions:
        before_start = (
            alarm_index - hotelling_config.before_size + 1
        )
        after_start = alarm_index + 1
        after_end = after_start + hotelling_config.after_size

        record: dict[str, object] = {
            "alarm_time": alarm_time,
            "validation_time": None,
            "before_start_time": None,
            "before_end_time": None,
            "after_start_time": None,
            "after_end_time": None,
            "before_size": hotelling_config.before_size,
            "after_size": hotelling_config.after_size,
            "n_features": observations.shape[1],
            "hotelling_t_squared": np.nan,
            "f_statistic": np.nan,
            "degrees_of_freedom_1": observations.shape[1],
            "degrees_of_freedom_2": np.nan,
            "critical_value": np.nan,
            "p_value": np.nan,
            "status": None,
            "confirmed_shift": False,
        }

        if (
            last_evaluated_index is not None
            and alarm_index - last_evaluated_index
            < minimum_alarm_gap
        ):
            record["status"] = "skipped_nearby_alarm"
            records.append(record)
            continue

        if before_start < 0:
            record["status"] = "insufficient_before_window"
            records.append(record)
            continue

        if after_end > observations.shape[0]:
            record["status"] = "pending_after_window"
            records.append(record)
            continue

        try:
            result = validate_multivariate_alarm(
                features=observations,
                times=time_values,
                alarm_time=alarm_time,
                config=hotelling_config,
            )
        except ValueError as error:
            if "too small for the number of features" not in str(error):
                raise

            record["status"] = "insufficient_degrees_of_freedom"
            records.append(record)
            continue

        record.update({
            "validation_time": result.validation_time,
            "before_start_time": result.before_start_time,
            "before_end_time": result.before_end_time,
            "after_start_time": result.after_start_time,
            "after_end_time": result.after_end_time,
            "before_size": result.before_size,
            "after_size": result.after_size,
            "n_features": result.n_features,
            "hotelling_t_squared": result.hotelling_t_squared,
            "f_statistic": result.f_statistic,
            "degrees_of_freedom_1": result.degrees_of_freedom_1,
            "degrees_of_freedom_2": result.degrees_of_freedom_2,
            "critical_value": result.critical_value,
            "p_value": result.p_value,
            "status": (
                "confirmed"
                if result.confirmed_shift
                else "rejected"
            ),
            "confirmed_shift": result.confirmed_shift,
        })

        records.append(record)
        last_evaluated_index = alarm_index

    return pd.DataFrame.from_records(
        records,
        columns=columns,
    )
