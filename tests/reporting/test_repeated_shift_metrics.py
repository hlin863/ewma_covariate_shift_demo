import numpy as np
import pandas as pd
import pytest

from src.reporting.metrics import evaluate_repeated_shift_detection


def test_repeated_shifts_are_matched_one_to_one() -> None:
    results = pd.DataFrame(
        {
            "time": np.arange(90, 401),
            "stage_1_alarm": np.isin(
                np.arange(90, 401),
                [99, 101, 105, 202, 350, 400],
            ).astype(int),
        }
    )

    evaluation = evaluate_repeated_shift_detection(
        detections=results,
        true_shift_times=[100, 200, 300, 400],
        observation_times=np.arange(90, 401),
        computation_time_seconds=0.1,
        maximum_delay=10,
    )

    assert evaluation.events["detection_time"].tolist()[:2] == [101.0, 202.0]
    assert np.isnan(evaluation.events.loc[2, "detection_time"])
    assert evaluation.events.loc[3, "detection_time"] == 400
    assert evaluation.metrics.detected_shift_count == 3
    assert evaluation.metrics.false_negative_count == 1
    assert evaluation.metrics.false_positive_count == 3
    assert evaluation.metrics.mean_recognition_capability_index == 1.0


def test_stage_2_can_be_scored_by_validation_time() -> None:
    stage_2 = pd.DataFrame(
        {
            "alarm_time": [100, 200],
            "validation_time": [110, 210],
            "confirmed_shift": [True, True],
        }
    )

    evaluation = evaluate_repeated_shift_detection(
        detections=stage_2,
        true_shift_times=[100, 200],
        observation_times=np.arange(100, 300),
        computation_time_seconds=0.2,
        detection_time_column="validation_time",
        detection_flag_column="confirmed_shift",
    )

    assert evaluation.events["recognition_capability_index"].tolist() == [10, 10]
    assert evaluation.metrics.false_positive_count == 0


def test_maximum_delay_turns_late_alarm_into_false_positive() -> None:
    results = pd.DataFrame({"time": [120], "stage_1_alarm": [1]})

    evaluation = evaluate_repeated_shift_detection(
        detections=results,
        true_shift_times=[100],
        observation_times=np.arange(100, 151),
        computation_time_seconds=0.0,
        maximum_delay=10,
    )

    assert evaluation.metrics.false_negative_count == 1
    assert evaluation.metrics.false_positive_count == 1


def test_missing_columns_raise() -> None:
    with pytest.raises(ValueError, match="Missing required columns"):
        evaluate_repeated_shift_detection(
            detections=pd.DataFrame({"time": [1]}),
            true_shift_times=[1],
            observation_times=[1, 2],
            computation_time_seconds=0.0,
        )
