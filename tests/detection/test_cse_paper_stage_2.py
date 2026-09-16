import numpy as np
import pandas as pd
import pytest

from src.cse import CSEConfig
from src.cse_paper_stage_2 import (
    PaperTwoSampleHotellingConfig,
    calculate_paper_two_sample_hotelling,
    validate_paper_two_sample_alarms,
)


def test_paper_hotelling_requires_two_true_multivariate_samples() -> None:
    reference = np.zeros((20, 2))
    current_vector = np.zeros(2)

    with pytest.raises(ValueError, match="shape \\(n, D\\)"):
        calculate_paper_two_sample_hotelling(
            current_sample=current_vector,
            reference_sample=reference,
        )


def test_paper_hotelling_requires_equal_sample_lengths() -> None:
    reference = np.zeros((20, 2))
    current = np.zeros((19, 2))

    with pytest.raises(ValueError, match="equal numbers of observations"):
        calculate_paper_two_sample_hotelling(
            current_sample=current,
            reference_sample=reference,
        )


def test_paper_hotelling_detects_large_multivariate_mean_shift() -> None:
    rng = np.random.default_rng(101)
    reference = rng.normal(0.0, 0.3, size=(40, 3))
    current = rng.normal(2.5, 0.3, size=(40, 3))

    _, _, _, _, p_value = calculate_paper_two_sample_hotelling(
        current_sample=current,
        reference_sample=reference,
    )

    assert p_value < 0.05


def test_paper_validator_builds_equal_disjoint_windows_around_warning() -> None:
    rng = np.random.default_rng(102)
    before = rng.normal(0.0, 0.2, size=(10, 2))
    after = rng.normal(3.0, 0.2, size=(10, 2))
    features = np.vstack([before, after])
    times = np.arange(features.shape[0])
    stage_1 = pd.DataFrame(
        {
            "time": times,
            "stage_1_alarm": [0] * 9 + [1] + [0] * 10,
        }
    )

    results = validate_paper_two_sample_alarms(
        features=features,
        times=times,
        stage_1_results=stage_1,
        config=PaperTwoSampleHotellingConfig(
            window_size=10,
            covariance_method="empirical",
            minimum_alarm_gap=0,
        ),
    )

    row = results.iloc[0]
    assert row["sample_size"] == 10
    assert row["reference_start_time"] == 0
    assert row["reference_end_time"] == 9
    assert row["current_start_time"] == 10
    assert row["current_end_time"] == 19
    assert row["validation_time"] == 19
    assert bool(row["confirmed_shift"])


def test_paper_validator_marks_warning_pending_without_second_window() -> None:
    features = np.zeros((15, 2))
    times = np.arange(15)
    stage_1 = pd.DataFrame(
        {
            "time": times,
            "stage_1_alarm": [0] * 9 + [1] + [0] * 5,
        }
    )

    results = validate_paper_two_sample_alarms(
        features=features,
        times=times,
        stage_1_results=stage_1,
        config=PaperTwoSampleHotellingConfig(window_size=10),
    )

    assert results.iloc[0]["status"] == "pending_current_window"
    assert not bool(results.iloc[0]["confirmed_shift"])


def test_cse_config_requires_equal_windows_for_paper_mode() -> None:
    with pytest.raises(ValueError, match="equal validation window sizes"):
        CSEConfig(
            validation_mode="paper_two_sample",
            validation_before_size=20,
            validation_after_size=10,
        )
