import numpy as np
import pandas as pd
import pytest

from src.evaluation import evaluate_stage_1_detection


def test_pre_shift_alarm_is_false_positive():
    results = pd.DataFrame({
        "time": [10, 11, 12, 13],
        "stage_1_alarm": [0, 1, 0, 1],
    })

    metrics = evaluate_stage_1_detection(
        results=results,
        true_shift_time=13,
        computation_time_seconds=0.01,
    )

    assert metrics.false_positive_count == 1
    assert np.isclose(
        metrics.false_positive_rate,
        1 / 3,
    )


def test_missing_shift_is_false_negative():
    results = pd.DataFrame({
        "time": [10, 11, 12, 13],
        "stage_1_alarm": [0, 0, 0, 0],
    })

    metrics = evaluate_stage_1_detection(
        results=results,
        true_shift_time=12,
        computation_time_seconds=0.01,
    )

    assert metrics.shift_detected is False
    assert metrics.false_negative_count == 1
    assert metrics.first_detection_time is None
    assert metrics.recognition_capability_index is None


@pytest.mark.parametrize(
    (
        "first_alarm",
        "shift_time",
        "expected_rci",
    ),
    [
        (100, 100, 0),
        (101, 100, 1),
        (110, 100, 10),
    ],
)
def test_rci_equals_detection_delay(
    first_alarm,
    shift_time,
    expected_rci,
):
    results = pd.DataFrame({
        "time": [
            shift_time - 1,
            first_alarm,
        ],
        "stage_1_alarm": [0, 1],
    })

    metrics = evaluate_stage_1_detection(
        results=results,
        true_shift_time=shift_time,
        computation_time_seconds=0.01,
    )

    assert (
        metrics.recognition_capability_index
        == expected_rci
    )


def test_fp_rate_is_nan_without_pre_shift_results():
    results = pd.DataFrame({
        "time": [500, 501, 502],
        "stage_1_alarm": [1, 0, 0],
    })

    metrics = evaluate_stage_1_detection(
        results=results,
        true_shift_time=500,
        computation_time_seconds=0.01,
    )

    assert np.isnan(metrics.false_positive_rate)