"""Equal-window Hotelling validation for the paper's TSMSD-EWMA."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import f


@dataclass(frozen=True)
class PaperHotellingConfig:
    window_size: int = 25
    alpha: float = 0.05
    regularization: float = 1e-6
    minimum_alarm_gap: int | None = None

    def __post_init__(self) -> None:
        if self.window_size < 2:
            raise ValueError("window_size must be at least 2.")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0, 1).")
        if self.regularization < 0.0:
            raise ValueError("regularization must be non-negative.")


def calculate_two_sample_hotelling(
    before: np.ndarray,
    after: np.ndarray,
    regularization: float = 1e-6,
) -> tuple[float, float, int, int, float]:
    """Return T-squared, F, df1, df2 and p-value for equal windows."""

    first = np.asarray(before, dtype=float)
    second = np.asarray(after, dtype=float)
    if first.ndim != 2 or second.ndim != 2 or first.shape != second.shape:
        raise ValueError("before and after must be equal-shaped 2D samples.")
    n, dimensions = first.shape
    df2 = 2 * n - dimensions - 1
    if n < 2 or df2 <= 0:
        raise ValueError("window size is too small for the feature count.")

    covariance_1 = np.atleast_2d(np.cov(first, rowvar=False, ddof=1))
    covariance_2 = np.atleast_2d(np.cov(second, rowvar=False, ddof=1))
    covariance_difference = (
        covariance_1 / n
        + covariance_2 / n
        + regularization * np.eye(dimensions)
    )
    mean_difference = first.mean(axis=0) - second.mean(axis=0)
    t_squared = float(
        mean_difference.T @ np.linalg.pinv(covariance_difference) @ mean_difference
    )
    f_statistic = float(df2 / (dimensions * (2 * n - 2)) * t_squared)
    p_value = float(f.sf(f_statistic, dimensions, df2))
    return t_squared, f_statistic, dimensions, df2, p_value


def validate_paper_hotelling_alarms(
    features: np.ndarray,
    times: np.ndarray,
    stage_1_results: pd.DataFrame,
    config: PaperHotellingConfig | None = None,
) -> pd.DataFrame:
    """Validate Stage-I candidates with ``m`` samples before and after."""

    effective = config or PaperHotellingConfig()
    values = np.asarray(features, dtype=float)
    time_values = np.asarray(times)
    if values.ndim != 2 or values.shape[0] != time_values.size:
        raise ValueError("features and times must describe the same 2D stream.")
    required = {"time", "stage_1_alarm"}
    if missing := required.difference(stage_1_results.columns):
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    columns = [
        "alarm_time", "validation_time", "sample_size", "n_features",
        "hotelling_t_squared", "f_statistic", "degrees_of_freedom_1",
        "degrees_of_freedom_2", "p_value", "status", "confirmed_shift",
    ]
    alarm_times = stage_1_results.loc[
        stage_1_results["stage_1_alarm"].astype(bool), "time"
    ].drop_duplicates()
    records: list[dict[str, object]] = []
    last_evaluated: int | None = None
    gap = effective.window_size if effective.minimum_alarm_gap is None else effective.minimum_alarm_gap

    for alarm_time in alarm_times:
        positions = np.flatnonzero(time_values == alarm_time)
        if positions.size != 1:
            raise ValueError("Every alarm time must identify one observation.")
        index = int(positions[0])
        record: dict[str, object] = {
            "alarm_time": int(alarm_time), "validation_time": np.nan,
            "sample_size": effective.window_size, "n_features": values.shape[1],
            "hotelling_t_squared": np.nan, "f_statistic": np.nan,
            "degrees_of_freedom_1": values.shape[1],
            "degrees_of_freedom_2": np.nan, "p_value": np.nan,
            "status": None, "confirmed_shift": False,
        }
        if last_evaluated is not None and index - last_evaluated < gap:
            record["status"] = "skipped_nearby_alarm"
            records.append(record)
            continue
        start = index - effective.window_size + 1
        stop = index + 1 + effective.window_size
        if start < 0:
            record["status"] = "insufficient_reference_window"
            records.append(record)
            continue
        if stop > values.shape[0]:
            record["status"] = "pending_current_window"
            records.append(record)
            continue
        statistics = calculate_two_sample_hotelling(
            values[start : index + 1],
            values[index + 1 : stop],
            effective.regularization,
        )
        t_squared, f_statistic, df1, df2, p_value = statistics
        confirmed = p_value < effective.alpha
        record.update(
            {
                "validation_time": int(time_values[stop - 1]),
                "hotelling_t_squared": t_squared,
                "f_statistic": f_statistic,
                "degrees_of_freedom_1": df1,
                "degrees_of_freedom_2": df2,
                "p_value": p_value,
                "status": "confirmed" if confirmed else "rejected",
                "confirmed_shift": bool(confirmed),
            }
        )
        records.append(record)
        last_evaluated = index
    return pd.DataFrame.from_records(records, columns=columns)
