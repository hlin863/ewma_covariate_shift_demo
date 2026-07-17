from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


@dataclass(frozen=True)
class Stage2Config:
    """Configuration for retrospective Stage-II validation."""

    before_size: int = 50
    after_size: int = 50
    alpha: float = 0.05
    minimum_alarm_gap: int | None = None

    def __post_init__(self) -> None:
        if self.before_size <= 0 or self.after_size <= 0:
            raise ValueError("window sizes must be positive.")

        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0, 1).")

        if (
            self.minimum_alarm_gap is not None
            and self.minimum_alarm_gap < 0
        ):
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


def _validate_stream_inputs(
    values: np.ndarray,
    times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
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


def validate_stage_1_alarm(
    values: np.ndarray,
    times: np.ndarray,
    alarm_time: int,
    before_size: int = 50,
    after_size: int = 50,
    alpha: float = 0.05,
) -> Stage2ValidationResult:
    """Validate one Stage-I alarm using two disjoint K-S windows."""

    observations, time_values = _validate_stream_inputs(
        values=values,
        times=times,
    )

    config = Stage2Config(
        before_size=before_size,
        after_size=after_size,
        alpha=alpha,
    )

    alarm_positions = np.flatnonzero(
        time_values == alarm_time
    )

    if alarm_positions.size != 1:
        raise ValueError(
            "alarm_time must identify exactly one observation."
        )

    alarm_index = int(alarm_positions[0])

    before_start = alarm_index - config.before_size + 1
    after_start = alarm_index + 1
    after_end = after_start + config.after_size

    if before_start < 0:
        raise ValueError(
            "Insufficient observations before the alarm."
        )

    if after_end > observations.size:
        raise ValueError(
            "Insufficient observations after the alarm."
        )

    before_values = observations[
        before_start:alarm_index + 1
    ]

    after_values = observations[
        after_start:after_end
    ]

    ks_result = ks_2samp(
        before_values,
        after_values,
        alternative="two-sided",
        method="auto",
    )

    return Stage2ValidationResult(
        alarm_time=int(alarm_time),
        before_start_time=int(time_values[before_start]),
        before_end_time=int(time_values[alarm_index]),
        after_start_time=int(time_values[after_start]),
        after_end_time=int(time_values[after_end - 1]),
        before_size=int(before_values.size),
        after_size=int(after_values.size),
        ks_statistic=float(ks_result.statistic),
        p_value=float(ks_result.pvalue),
        confirmed_shift=bool(
            ks_result.pvalue < config.alpha
        ),
    )


def validate_stage_1_alarms(
    stage_1_results: pd.DataFrame,
    values: np.ndarray,
    times: np.ndarray,
    config: Stage2Config | None = None,
) -> pd.DataFrame:
    """Validate all eligible Stage-I alarms in chronological order.

    Recent alarms without a complete following window are marked as
    ``pending_after_window``. Alarms too near a previously evaluated
    candidate are marked as ``skipped_nearby_alarm``.
    """

    required_columns = {"time", "stage_1_alarm"}
    missing_columns = required_columns.difference(
        stage_1_results.columns
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    observations, time_values = _validate_stream_inputs(
        values=values,
        times=times,
    )

    stage_2_config = config or Stage2Config()
    minimum_alarm_gap = (
        stage_2_config.after_size
        if stage_2_config.minimum_alarm_gap is None
        else stage_2_config.minimum_alarm_gap
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

    records: list[dict[str, object]] = []
    last_evaluated_index: int | None = None

    for alarm_index, alarm_time in alarm_positions:
        before_start = (
            alarm_index
            - stage_2_config.before_size
            + 1
        )
        after_start = alarm_index + 1
        after_end = (
            after_start
            + stage_2_config.after_size
        )

        base_record: dict[str, object] = {
            "alarm_time": alarm_time,
            "validation_time": None,
            "before_start_time": None,
            "before_end_time": None,
            "after_start_time": None,
            "after_end_time": None,
            "before_size": stage_2_config.before_size,
            "after_size": stage_2_config.after_size,
            "ks_statistic": np.nan,
            "p_value": np.nan,
            "status": None,
            "confirmed_shift": False,
        }

        if (
            last_evaluated_index is not None
            and alarm_index - last_evaluated_index
            < minimum_alarm_gap
        ):
            base_record["status"] = "skipped_nearby_alarm"
            records.append(base_record)
            continue

        if before_start < 0:
            base_record["status"] = (
                "insufficient_before_window"
            )
            records.append(base_record)
            continue

        if after_end > observations.size:
            base_record["status"] = "pending_after_window"
            records.append(base_record)
            continue

        result = validate_stage_1_alarm(
            values=observations,
            times=time_values,
            alarm_time=alarm_time,
            before_size=stage_2_config.before_size,
            after_size=stage_2_config.after_size,
            alpha=stage_2_config.alpha,
        )

        status = (
            "confirmed"
            if result.confirmed_shift
            else "rejected"
        )

        base_record.update({
            "validation_time": result.after_end_time,
            "before_start_time": result.before_start_time,
            "before_end_time": result.before_end_time,
            "after_start_time": result.after_start_time,
            "after_end_time": result.after_end_time,
            "ks_statistic": result.ks_statistic,
            "p_value": result.p_value,
            "status": status,
            "confirmed_shift": result.confirmed_shift,
        })

        records.append(base_record)
        last_evaluated_index = alarm_index

    columns = [
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
    ]

    return pd.DataFrame.from_records(
        records,
        columns=columns,
    )
