from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DetectionMetrics:
    false_positive_count: int
    false_positive_rate: float
    false_negative_count: int
    shift_detected: bool
    first_detection_time: int | None
    recognition_capability_index: int | None
    computation_time_seconds: float


def evaluate_stage_1_detection(
    results: pd.DataFrame,
    true_shift_time: int,
    computation_time_seconds: float,
) -> DetectionMetrics:
    """Evaluate Stage-I SD-EWMA against one known shift point."""

    required_columns = {
        "time",
        "stage_1_alarm",
    }

    missing_columns = required_columns.difference(results.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    alarm_times = results.loc[
        results["stage_1_alarm"] == 1,
        "time",
    ].to_numpy()

    pre_shift_results = results[
        results["time"] < true_shift_time
    ]

    false_alarm_times = alarm_times[
        alarm_times < true_shift_time
    ]

    post_shift_alarm_times = alarm_times[
        alarm_times >= true_shift_time
    ]

    first_detection_time = (
        int(post_shift_alarm_times[0])
        if post_shift_alarm_times.size > 0
        else None
    )

    shift_detected = first_detection_time is not None

    false_positive_count = int(
        false_alarm_times.size
    )

    false_positive_rate = (
        false_positive_count / len(pre_shift_results)
        if len(pre_shift_results) > 0
        else float("nan")
    )

    false_negative_count = int(
        not shift_detected
    )

    recognition_capability_index = (
        first_detection_time - true_shift_time
        if first_detection_time is not None
        else None
    )

    return DetectionMetrics(
        false_positive_count=false_positive_count,
        false_positive_rate=false_positive_rate,
        false_negative_count=false_negative_count,
        shift_detected=shift_detected,
        first_detection_time=first_detection_time,
        recognition_capability_index=recognition_capability_index,
        computation_time_seconds=computation_time_seconds,
    )