"""Run in the real repository: python -m pytest tests/detection/stage2/test_adaptive_window_hotelling.py -q"""
import numpy as np
import pandas as pd
import pytest

from src.detection.stage2.adaptive_window_hotelling import (
    AdaptiveHotellingConfig,
    validate_adaptive_window_alarms,
)
from src.detection.stage2.window_feasibility import (
    WindowSelectionPolicy,
    minimum_equal_window_size,
    select_validation_window,
)


def _one_alarm(times: np.ndarray, warning_index: int) -> pd.DataFrame:
    alarms = np.zeros(len(times), dtype=int)
    alarms[warning_index] = 1
    return pd.DataFrame({"time": times, "stage_1_alarm": alarms})


def test_dimension_feasibility_mathematics() -> None:
    assert minimum_equal_window_size(20, 10) == 16
    assert minimum_equal_window_size(3, 10) == 7
    assert 2 * 10 - 20 - 1 == -1


def test_selection_rejects_missing_history_and_insufficient_policy() -> None:
    rng = np.random.default_rng(22)
    history = rng.normal(size=(10, 20))
    assert select_validation_window(history, WindowSelectionPolicy()).status == (
        "insufficient_reference_window"
    )
    assert select_validation_window(
        history,
        WindowSelectionPolicy(max_window=10),
    ).status == "infeasible_policy"


def test_selection_uses_only_history_before_warning() -> None:
    rng = np.random.default_rng(23)
    before = rng.normal(size=(30, 3))
    after_a = rng.normal(size=(25, 3))
    after_b = rng.normal(loc=40, size=(25, 3))
    times = np.arange(55)
    stage1 = _one_alarm(times, 29)
    policy = WindowSelectionPolicy(min_window=7, max_window=20)
    cfg = AdaptiveHotellingConfig(window_policy=policy)
    left = validate_adaptive_window_alarms(
        np.vstack((before, after_a)), times, stage1, cfg
    ).iloc[0]
    right = validate_adaptive_window_alarms(
        np.vstack((before, after_b)), times, stage1, cfg
    ).iloc[0]
    assert left["sample_size"] == right["sample_size"]
    assert left["reference_condition_number"] == pytest.approx(
        right["reference_condition_number"]
    )
    assert left["degrees_of_freedom_2"] >= policy.target_df2


def test_large_shift_is_confirmed() -> None:
    rng = np.random.default_rng(24)
    before = rng.normal(0, .2, size=(30, 3))
    after = rng.normal(3, .2, size=(30, 3))
    features = np.vstack((before, after))
    times = np.arange(len(features))
    result = validate_adaptive_window_alarms(
        features, times, _one_alarm(times, 29)
    ).iloc[0]
    assert result["status"] == "confirmed"
    assert bool(result["confirmed_shift"])
    assert result["validation_time"] == 29 + result["sample_size"]


def test_incomplete_post_warning_window_is_pending() -> None:
    rng = np.random.default_rng(25)
    features = rng.normal(size=(33, 3))
    times = np.arange(33)
    result = validate_adaptive_window_alarms(
        features, times, _one_alarm(times, 29)
    ).iloc[0]
    assert result["status"] == "pending_current_window"
    assert not bool(result["confirmed_shift"])


def test_false_alarm_fails_to_imply_confirmed_shift() -> None:
    rng = np.random.default_rng(26)
    features = rng.normal(size=(60, 3))
    times = np.arange(60)
    results = validate_adaptive_window_alarms(
        features, times, _one_alarm(times, 29)
    )
    assert results["status"].isin(["confirmed", "rejected"]).all()
    # A null stream can occasionally produce a significant result. Do NOT
    # assert that a single random run always rejects; calibrate via Monte Carlo.
