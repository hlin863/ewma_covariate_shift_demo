"""Evaluation metrics for synthetic and staged CSE experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

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


@dataclass(frozen=True)
class RepeatedShiftMetrics:
    true_shift_count: int
    detected_shift_count: int
    false_positive_count: int
    false_positive_rate: float
    false_negative_count: int
    false_negative_rate: float
    mean_recognition_capability_index: float | None
    median_recognition_capability_index: float | None
    maximum_recognition_capability_index: int | None
    computation_time_seconds: float


@dataclass(frozen=True)
class RepeatedShiftEvaluation:
    metrics: RepeatedShiftMetrics
    events: pd.DataFrame


def evaluate_stage_1_detection(
    results: pd.DataFrame,
    true_shift_time: int,
    computation_time_seconds: float,
) -> DetectionMetrics:
    required_columns = {"time", "stage_1_alarm"}
    missing_columns = required_columns.difference(results.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    alarm_times = results.loc[results["stage_1_alarm"] == 1, "time"].to_numpy()
    pre_shift_results = results[results["time"] < true_shift_time]
    false_alarm_times = alarm_times[alarm_times < true_shift_time]
    post_shift_alarm_times = alarm_times[alarm_times >= true_shift_time]
    first_detection_time = (
        int(post_shift_alarm_times[0]) if post_shift_alarm_times.size > 0 else None
    )
    shift_detected = first_detection_time is not None
    false_positive_count = int(false_alarm_times.size)
    false_positive_rate = (
        false_positive_count / len(pre_shift_results)
        if len(pre_shift_results) > 0
        else float("nan")
    )
    false_negative_count = int(not shift_detected)
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


def _sorted_unique_integers(values: Iterable[int], name: str) -> np.ndarray:
    numeric = np.asarray(list(values))
    if numeric.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if numeric.size == 0:
        raise ValueError(f"{name} must not be empty.")
    try:
        numeric_float = numeric.astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numeric values.") from error
    if not np.isfinite(numeric_float).all():
        raise ValueError(f"{name} must contain only finite values.")
    if not np.equal(numeric_float, np.floor(numeric_float)).all():
        raise ValueError(f"{name} must contain integer-valued times.")
    return np.unique(numeric_float.astype(int))


def evaluate_repeated_shift_detection(
    detections: pd.DataFrame,
    true_shift_times: Iterable[int],
    observation_times: Iterable[int],
    computation_time_seconds: float,
    *,
    detection_time_column: str = "time",
    detection_flag_column: str = "stage_1_alarm",
    maximum_delay: int | None = None,
) -> RepeatedShiftEvaluation:
    """Match alarms one-to-one against a sequence of true shift events.

    Each event owns the half-open interval from its shift time to the next
    shift. The earliest unused alarm in that interval is its detection; all
    remaining alarms are false positives. ``maximum_delay`` can additionally
    cap an event's eligible detection interval.
    """

    required_columns = {detection_time_column, detection_flag_column}
    missing_columns = required_columns.difference(detections.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    if computation_time_seconds < 0.0 or not np.isfinite(computation_time_seconds):
        raise ValueError("computation_time_seconds must be finite and non-negative.")
    if maximum_delay is not None and maximum_delay < 0:
        raise ValueError("maximum_delay must be non-negative.")

    shifts = _sorted_unique_integers(true_shift_times, "true_shift_times")
    observations = _sorted_unique_integers(observation_times, "observation_times")
    if not np.isin(shifts, observations).all():
        raise ValueError("Every true shift time must appear in observation_times.")

    flagged = detections.loc[
        detections[detection_flag_column].fillna(False).astype(bool),
        detection_time_column,
    ]
    if flagged.empty:
        alarm_times = np.array([], dtype=int)
    else:
        alarm_times = _sorted_unique_integers(flagged.tolist(), "detection times")

    used = np.zeros(alarm_times.size, dtype=bool)
    event_records: list[dict[str, int | bool | None]] = []
    delays: list[int] = []

    for index, shift_time in enumerate(shifts):
        before_next_shift = (
            alarm_times < shifts[index + 1]
            if index + 1 < shifts.size
            else np.ones(alarm_times.size, dtype=bool)
        )
        eligible = (~used) & (alarm_times >= shift_time) & before_next_shift
        if maximum_delay is not None:
            eligible &= alarm_times <= shift_time + maximum_delay
        eligible_indices = np.flatnonzero(eligible)

        if eligible_indices.size == 0:
            detection_time = None
            delay = None
            detected = False
        else:
            matched_index = int(eligible_indices[0])
            used[matched_index] = True
            detection_time = int(alarm_times[matched_index])
            delay = detection_time - int(shift_time)
            delays.append(delay)
            detected = True

        event_records.append(
            {
                "shift_time": int(shift_time),
                "detection_time": detection_time,
                "detected": detected,
                "recognition_capability_index": delay,
            }
        )

    detected_shift_count = int(used.sum())
    false_negative_count = int(shifts.size - detected_shift_count)
    false_positive_count = int((~used).sum())
    non_shift_observation_count = int(observations.size - shifts.size)
    false_positive_rate = (
        false_positive_count / non_shift_observation_count
        if non_shift_observation_count > 0
        else float("nan")
    )

    metrics = RepeatedShiftMetrics(
        true_shift_count=int(shifts.size),
        detected_shift_count=detected_shift_count,
        false_positive_count=false_positive_count,
        false_positive_rate=float(false_positive_rate),
        false_negative_count=false_negative_count,
        false_negative_rate=false_negative_count / int(shifts.size),
        mean_recognition_capability_index=(float(np.mean(delays)) if delays else None),
        median_recognition_capability_index=(
            float(np.median(delays)) if delays else None
        ),
        maximum_recognition_capability_index=(max(delays) if delays else None),
        computation_time_seconds=float(computation_time_seconds),
    )
    events = pd.DataFrame.from_records(
        event_records,
        columns=[
            "shift_time",
            "detection_time",
            "detected",
            "recognition_capability_index",
        ],
    )
    return RepeatedShiftEvaluation(metrics=metrics, events=events)
