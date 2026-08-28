"""Mean-feature Intersection-of-Confidence-Intervals change detector.

The implementation follows the minimum-variance, zeroth-order ICI rule used by
Alippi, Boracchi and Roveri (2011): observations are reduced to means of
disjoint subsequences, cumulative constant fits are converted to confidence
intervals, and a change is raised when their running intersection becomes
empty.  D2 changes the innovation mean, so the mean feature is the relevant
ICI-CDT feature for the Table III comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ICICDTConfig:
    """Configuration of the sequential mean-feature ICI-CDT baseline."""

    block_size: int = 10
    confidence_parameter: float = 2.0
    reset_after_alarm: bool = True

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError("block_size must be positive.")
        if (
            self.confidence_parameter <= 0.0
            or not np.isfinite(self.confidence_parameter)
        ):
            raise ValueError("confidence_parameter must be positive and finite.")


@dataclass(frozen=True)
class ICICDTTrainingResult:
    block_size: int
    block_count: int
    feature_mean: float
    feature_standard_deviation: float
    intersection_lower: float
    intersection_upper: float


@dataclass(frozen=True)
class ICICDTResult:
    training_result: ICICDTTrainingResult
    trace: pd.DataFrame
    computation_time_seconds: float


def _validated_vector(values, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence.")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values.")
    return vector


def _complete_block_means(values: np.ndarray, block_size: int) -> np.ndarray:
    block_count = values.size // block_size
    if block_count == 0:
        raise ValueError("At least one complete block is required.")
    return values[: block_count * block_size].reshape(block_count, block_size).mean(axis=1)


def fit_ici_cdt(
    training_values,
    config: ICICDTConfig | None = None,
) -> ICICDTTrainingResult:
    """Fit the initial ICI interval from stationary training observations."""

    effective = config or ICICDTConfig()
    training = _validated_vector(training_values, "training_values")
    block_means = _complete_block_means(training, effective.block_size)
    if block_means.size < 2:
        raise ValueError("ICI-CDT training requires at least two complete blocks.")

    feature_mean = float(block_means.mean())
    feature_standard_deviation = float(block_means.std(ddof=1))
    if feature_standard_deviation == 0.0:
        feature_standard_deviation = float(np.finfo(float).eps)
    estimator_standard_deviation = feature_standard_deviation / np.sqrt(
        block_means.size
    )
    half_width = effective.confidence_parameter * estimator_standard_deviation
    return ICICDTTrainingResult(
        block_size=effective.block_size,
        block_count=int(block_means.size),
        feature_mean=feature_mean,
        feature_standard_deviation=feature_standard_deviation,
        intersection_lower=feature_mean - half_width,
        intersection_upper=feature_mean + half_width,
    )


def run_ici_cdt(
    training_values,
    testing_values,
    testing_times,
    config: ICICDTConfig | None = None,
) -> ICICDTResult:
    """Run the sequential ICI rule and emit one record per complete test block.

    A repeated-shift stream needs monitoring to continue after a detection.  If
    ``reset_after_alarm`` is enabled, the alarm block becomes the first block of
    a new ICI epoch.  This operational reset is explicit because Algorithm 2 in
    the cited ICI paper stops after its first detection.
    """

    effective = config or ICICDTConfig()
    training = _validated_vector(training_values, "training_values")
    testing = _validated_vector(testing_values, "testing_values")
    times = np.asarray(testing_times)
    if times.ndim != 1 or times.size != testing.size:
        raise ValueError("testing_times must be one-dimensional and match testing_values.")
    if not np.isfinite(times.astype(float)).all():
        raise ValueError("testing_times must contain only finite values.")

    started = perf_counter()
    fitted = fit_ici_cdt(training, effective)
    complete_size = (testing.size // effective.block_size) * effective.block_size
    test_blocks = testing[:complete_size].reshape(-1, effective.block_size)
    time_blocks = times[:complete_size].reshape(-1, effective.block_size)

    epoch_sum = fitted.feature_mean * fitted.block_count
    epoch_count = fitted.block_count
    intersection_lower = fitted.intersection_lower
    intersection_upper = fitted.intersection_upper
    epoch = 0
    records: list[dict[str, int | float | bool]] = []

    for block_index, (block, block_times) in enumerate(zip(test_blocks, time_blocks), start=1):
        block_mean = float(block.mean())
        candidate_count = epoch_count + 1
        candidate_mean = (epoch_sum + block_mean) / candidate_count
        estimator_std = fitted.feature_standard_deviation / np.sqrt(candidate_count)
        half_width = effective.confidence_parameter * estimator_std
        interval_lower = candidate_mean - half_width
        interval_upper = candidate_mean + half_width
        next_lower = max(intersection_lower, interval_lower)
        next_upper = min(intersection_upper, interval_upper)
        alarm = bool(next_lower > next_upper)

        records.append(
            {
                "time": int(block_times[-1]),
                "block_index": block_index,
                "block_start_time": int(block_times[0]),
                "block_end_time": int(block_times[-1]),
                "block_mean": block_mean,
                "estimate": candidate_mean,
                "interval_lower": interval_lower,
                "interval_upper": interval_upper,
                "intersection_lower": next_lower,
                "intersection_upper": next_upper,
                "ici_alarm": int(alarm),
                "epoch": epoch,
            }
        )

        if alarm and effective.reset_after_alarm:
            epoch += 1
            epoch_sum = block_mean
            epoch_count = 1
            reset_half_width = (
                effective.confidence_parameter * fitted.feature_standard_deviation
            )
            intersection_lower = block_mean - reset_half_width
            intersection_upper = block_mean + reset_half_width
        else:
            epoch_sum += block_mean
            epoch_count = candidate_count
            intersection_lower = next_lower
            intersection_upper = next_upper

    trace = pd.DataFrame.from_records(
        records,
        columns=[
            "time",
            "block_index",
            "block_start_time",
            "block_end_time",
            "block_mean",
            "estimate",
            "interval_lower",
            "interval_upper",
            "intersection_lower",
            "intersection_upper",
            "ici_alarm",
            "epoch",
        ],
    )
    return ICICDTResult(
        training_result=fitted,
        trace=trace,
        computation_time_seconds=perf_counter() - started,
    )
