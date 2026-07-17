from dataclasses import dataclass

import numpy as np
from scipy.stats import ks_2samp

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

def validate_stage_1_alarm(
    values: np.ndarray,
    times: np.ndarray,
    alarm_time: int,
    before_size: int = 50,
    after_size: int = 50,
    alpha: float = 0.05,
) -> Stage2ValidationResult:
    observations = np.asarray(values, dtype=float)
    time_values = np.asarray(times)

    if observations.ndim != 1:
        raise ValueError("values must be one-dimensional.")

    if observations.size != time_values.size:
        raise ValueError("values and times must have equal length.")

    if before_size <= 0 or after_size <= 0:
        raise ValueError("window sizes must be positive.")

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1).")

    alarm_positions = np.flatnonzero(
        time_values == alarm_time
    )

    if alarm_positions.size != 1:
        raise ValueError(
            "alarm_time must identify exactly one observation."
        )

    alarm_index = int(alarm_positions[0])

    before_start = alarm_index - before_size + 1
    after_start = alarm_index + 1
    after_end = after_start + after_size

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
            ks_result.pvalue < alpha
        ),
    )