
import numpy as np
import pytest

from src.detection.stage1.park_cpss import (
    calculate_cpss_score,
)


def test_large_peak_gap_rejected():
    """Park criterion must reject gap=20 when T=100."""
    score = calculate_cpss_score(
        maximum_values=[0],
        average_values=[1],
        b_max_values=[40],
        b_avg_values=[60],
        T=100,
        gamma=1.0,
    )
    assert score == 0.0


def test_zero_peak_separation():
    """Perfect agreement has exponential weight 1."""
    score = calculate_cpss_score(
        maximum_values=[0],
        average_values=[2],
        b_max_values=[50],
        b_avg_values=[50],
        T=100,
        gamma=1.0,
    )
    assert score == 1.0


def test_gamma_zero_recovers_binary_pass_fraction():
    """When gamma=0, valid decisions contribute 1."""
    score = calculate_cpss_score(
        maximum_values=[0, 1, 0],
        average_values=[2, 1, 1],
        b_max_values=[50, 50, 50],
        b_avg_values=[52, 51, 55],
        T=100,
        gamma=0.0,
    )

    # Only the first run passes both conditions.
    assert score == pytest.approx(1 / 3)


def test_cpss_score_is_bounded():
    score = calculate_cpss_score(
        maximum_values=[0, 1, 0],
        average_values=[1, 0, 2],
        b_max_values=[50, 48, 52],
        b_avg_values=[51, 49, 53],
        T=100,
        gamma=2.0,
    )
    assert 0.0 <= score <= 1.0
