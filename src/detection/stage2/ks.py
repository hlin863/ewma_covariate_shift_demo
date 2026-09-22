"""Univariate Stage-II validation using two-sample K-S tests."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


@dataclass(frozen=True)
class Stage2Config:
    before_size: int = 50
    after_size: int = 50
    alpha: float = 0.05
    minimum_alarm_gap: int | None = None

    def __post_init__(self) -> None:
        if self.before_size <= 0 or self.after_size <= 0:
            raise ValueError("window sizes must be positive.")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0, 1).")
        if self.minimum_alarm_gap is not None and self.minimum_alarm_gap < 0:
            raise ValueError("minimum_alarm_gap must not be negative.")


@dataclass(frozen=True)
class Stage2ValidationResult:
    alarm_time: int
    before_start_time: int
    before_end_time: int
    after_start_time: int
    after_end_time: int
    before_size: int
    after_size: int
    ks_statistic: float
    p_value: float
    confirmed_shift: bool


def _validate_stream_inputs(values, times):
    observations = np.asarray(values, dtype=float)
    time_values = np.asarray(times)
    if observations.ndim != 1:
        raise ValueError("values must be one-dimensional.")
    if time_values.ndim != 1:
        raise ValueError("times must be one-dimensional.")
    if observations.size != time_values.size:
        raise ValueError("values and times must have equal length.")
    if observations.size == 0:
        raise ValueError("values and times must not be empty.")
    if not np.isfinite(observations).all():
        raise ValueError("values must contain only finite observations.")
    return observations, time_values


def _ks_critical_constant(alpha: float) -> float:
    critical_values = {
        0.10: 1.22,
        0.05: 1.36,
        0.025: 1.48,
        0.01: 1.63,
        0.005: 1.73,
        0.001: 1.95,
    }

    if alpha not in critical_values:
        raise ValueError(
            "Unsupported alpha for the Raza et al. (2015) "
            "K-S critical-value table. "
            f"Supported values: {sorted(critical_values)}"
        )

    return critical_values[alpha]


def validate_stage_1_alarm(
    values,
    times,
    alarm_time,
    before_size=50,
    after_size=50,
    alpha=0.05,
):
    observations, time_values = _validate_stream_inputs(values, times)
    config = Stage2Config(before_size=before_size, after_size=after_size, alpha=alpha)
    alarm_positions = np.flatnonzero(time_values == alarm_time)
    if alarm_positions.size != 1:
        raise ValueError("alarm_time must identify exactly one observation.")
    alarm_index = int(alarm_positions[0])
    before_start = alarm_index - config.before_size + 1
    after_start = alarm_index + 1
    after_end = after_start + config.after_size
    if before_start < 0:
        raise ValueError("Insufficient observations before the alarm.")
    if after_end > observations.size:
        raise ValueError("Insufficient observations after the alarm.")
    before_values = observations[before_start : alarm_index + 1]
    after_values = observations[after_start:after_end]

    n1 = before_values.shape[0]
    n2 = after_values.shape[0]

    result = ks_2samp(
        before_values, after_values, alternative="two-sided", method="auto"
    )

    scaled_statistic = (np.sqrt(n1 * n2 / (n1 + n2))) * result.statistic

    k_alpha = _ks_critical_constant(config.alpha)

    return Stage2ValidationResult(
        alarm_time=int(alarm_time),
        before_start_time=int(time_values[before_start]),
        before_end_time=int(time_values[alarm_index]),
        after_start_time=int(time_values[after_start]),
        after_end_time=int(time_values[after_end - 1]),
        before_size=int(before_values.size),
        after_size=int(after_values.size),
        ks_statistic=float(result.statistic),
        p_value=float(result.pvalue),
        confirmed_shift=bool(scaled_statistic > k_alpha),
    )


def validate_stage_1_alarms(stage_1_results, values, times, config=None):
    required = {"time", "stage_1_alarm"}
    missing = required.difference(stage_1_results.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    observations, time_values = _validate_stream_inputs(values, times)
    effective = config or Stage2Config()
    minimum_gap = (
        effective.after_size
        if effective.minimum_alarm_gap is None
        else effective.minimum_alarm_gap
    )
    alarm_times = (
        stage_1_results.loc[stage_1_results["stage_1_alarm"] == 1, "time"]
        .drop_duplicates()
        .to_numpy()
    )
    alarm_positions = []
    for alarm_time in alarm_times:
        positions = np.flatnonzero(time_values == alarm_time)
        if positions.size != 1:
            raise ValueError("Each alarm time must identify exactly one observation.")
        alarm_positions.append((int(positions[0]), int(alarm_time)))
    alarm_positions.sort()
    records = []
    last_evaluated_index = None
    for alarm_index, alarm_time in alarm_positions:
        before_start = alarm_index - effective.before_size + 1
        after_start = alarm_index + 1
        after_end = after_start + effective.after_size
        record = {
            "alarm_time": alarm_time,
            "validation_time": None,
            "before_start_time": None,
            "before_end_time": None,
            "after_start_time": None,
            "after_end_time": None,
            "before_size": effective.before_size,
            "after_size": effective.after_size,
            "ks_statistic": np.nan,
            "p_value": np.nan,
            "status": None,
            "confirmed_shift": False,
        }
        if (
            last_evaluated_index is not None
            and alarm_index - last_evaluated_index < minimum_gap
        ):
            record["status"] = "skipped_nearby_alarm"
        elif before_start < 0:
            record["status"] = "insufficient_before_window"
        elif after_end > observations.size:
            record["status"] = "pending_after_window"
        else:
            result = validate_stage_1_alarm(
                observations,
                time_values,
                alarm_time,
                effective.before_size,
                effective.after_size,
                effective.alpha,
            )
            record.update(
                {
                    "validation_time": result.after_end_time,
                    "before_start_time": result.before_start_time,
                    "before_end_time": result.before_end_time,
                    "after_start_time": result.after_start_time,
                    "after_end_time": result.after_end_time,
                    "ks_statistic": result.ks_statistic,
                    "p_value": result.p_value,
                    "status": "confirmed" if result.confirmed_shift else "rejected",
                    "confirmed_shift": result.confirmed_shift,
                }
            )
            last_evaluated_index = alarm_index
        records.append(record)
    return pd.DataFrame.from_records(
        records,
        columns=[
            "alarm_time",
            "validation_time",
            "before_start_time",
            "before_end_time",
            "after_start_time",
            "after_end_time",
            "before_size",
            "after_size",
            "ks_statistic",
            "p_value",
            "status",
            "confirmed_shift",
        ],
    )
