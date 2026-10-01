import numpy as np
import pandas as pd
import pytest

from src.stage_2 import (
    Stage2Config,
    validate_stage_1_alarm,
    validate_stage_1_alarms,
)


def test_equal_windows_do_not_confirm_shift():
    """Identical distributions should not be confirmed as a shift."""

    values = np.ones(101)
    times = np.arange(values.size)

    result = validate_stage_1_alarm(
        values=values,
        times=times,
        alarm_time=50,
        before_size=25,
        after_size=25,
    )

    assert result.confirmed_shift is False
    assert result.p_value >= 0.05
    assert result.before_size == 25
    assert result.after_size == 25


def test_large_mean_change_confirms_shift():
    """Clearly different retrospective windows should confirm a shift."""

    values = np.concatenate([
        np.zeros(51),
        np.full(50, 5.0),
    ])

    times = np.arange(values.size)

    result = validate_stage_1_alarm(
        values=values,
        times=times,
        alarm_time=50,
        before_size=25,
        after_size=25,
    )

    assert result.confirmed_shift is True
    assert result.p_value < 0.05
    assert result.ks_statistic > 0.0
    assert result.before_size == 25
    assert result.after_size == 25


def test_alarm_observation_belongs_only_to_before_window():
    """The alarm observation should not be duplicated across both windows."""

    values = np.arange(101, dtype=float)
    times = np.arange(values.size)

    result = validate_stage_1_alarm(
        values=values,
        times=times,
        alarm_time=50,
        before_size=25,
        after_size=25,
    )

    assert result.before_start_time == 26
    assert result.before_end_time == 50
    assert result.after_start_time == 51
    assert result.after_end_time == 75
    assert result.before_end_time < result.after_start_time


def test_insufficient_before_window_raises():
    """Stage II should fail clearly when insufficient earlier data exist."""

    values = np.arange(20, dtype=float)
    times = np.arange(values.size)

    with pytest.raises(
        ValueError,
        match="Insufficient observations before",
    ):
        validate_stage_1_alarm(
            values=values,
            times=times,
            alarm_time=5,
            before_size=10,
            after_size=5,
        )


def test_insufficient_after_window_raises():
    """Stage II should fail clearly when insufficient later data exist."""

    values = np.arange(20, dtype=float)
    times = np.arange(values.size)

    with pytest.raises(
        ValueError,
        match="Insufficient observations after",
    ):
        validate_stage_1_alarm(
            values=values,
            times=times,
            alarm_time=15,
            before_size=5,
            after_size=10,
        )


def test_unknown_alarm_time_raises():
    """The alarm time must identify an observation in the stream."""

    values = np.arange(20, dtype=float)
    times = np.arange(values.size)

    with pytest.raises(
        ValueError,
        match="alarm_time must identify exactly one observation",
    ):
        validate_stage_1_alarm(
            values=values,
            times=times,
            alarm_time=100,
            before_size=5,
            after_size=5,
        )


@pytest.mark.parametrize(
    ("before_size", "after_size"),
    [
        (0, 5),
        (-1, 5),
        (5, 0),
        (5, -1),
    ],
)
def test_invalid_window_sizes_raise(
    before_size,
    after_size,
):
    values = np.arange(20, dtype=float)
    times = np.arange(values.size)

    with pytest.raises(
        ValueError,
        match="window sizes must be positive",
    ):
        validate_stage_1_alarm(
            values=values,
            times=times,
            alarm_time=10,
            before_size=before_size,
            after_size=after_size,
        )


@pytest.mark.parametrize(
    "alpha",
    [0.0, -0.1, 1.0, 1.1],
)
def test_invalid_alpha_raises(alpha):
    values = np.arange(20, dtype=float)
    times = np.arange(values.size)

    with pytest.raises(
        ValueError,
        match="alpha must be in",
    ):
        validate_stage_1_alarm(
            values=values,
            times=times,
            alarm_time=10,
            before_size=5,
            after_size=5,
            alpha=alpha,
        )


def test_mismatched_values_and_times_raise():
    with pytest.raises(
        ValueError,
        match="values and times must have equal length",
    ):
        validate_stage_1_alarm(
            values=np.arange(10, dtype=float),
            times=np.arange(9),
            alarm_time=5,
            before_size=3,
            after_size=3,
        )


def test_multidimensional_values_raise():
    values = np.ones((10, 2))
    times = np.arange(10)

    with pytest.raises(
        ValueError,
        match="values must be one-dimensional",
    ):
        validate_stage_1_alarm(
            values=values,
            times=times,
            alarm_time=5,
            before_size=3,
            after_size=3,
        )


def test_non_finite_values_raise():
    values = np.arange(20, dtype=float)
    values[10] = np.nan

    with pytest.raises(
        ValueError,
        match="finite observations",
    ):
        validate_stage_1_alarm(
            values=values,
            times=np.arange(values.size),
            alarm_time=10,
            before_size=5,
            after_size=5,
        )


def test_isolated_outlier_is_rejected():
    values = np.zeros(101)
    values[50] = 8.0

    result = validate_stage_1_alarm(
        values=values,
        times=np.arange(values.size),
        alarm_time=50,
        before_size=25,
        after_size=25,
    )

    assert result.confirmed_shift is False
    assert result.p_value >= 0.05


def test_stage_2_config_rejects_negative_alarm_gap():
    with pytest.raises(
        ValueError,
        match="minimum_alarm_gap",
    ):
        Stage2Config(minimum_alarm_gap=-1)


def test_multi_alarm_runner_confirms_and_rejects_candidates():
    values = np.concatenate([
        np.zeros(51),
        np.full(50, 5.0),
        np.full(50, 5.0),
    ])
    times = np.arange(values.size)

    stage_1_results = pd.DataFrame({
        "time": times,
        "stage_1_alarm": np.isin(
            times,
            [25, 50],
        ).astype(int),
    })

    results = validate_stage_1_alarms(
        stage_1_results=stage_1_results,
        values=values,
        times=times,
        config=Stage2Config(
            before_size=20,
            after_size=20,
            minimum_alarm_gap=0,
        ),
    )

    assert list(results["alarm_time"]) == [25, 50]
    assert list(results["status"]) == [
        "rejected",
        "confirmed",
    ]
    assert list(results["confirmed_shift"]) == [
        False,
        True,
    ]


def test_multi_alarm_runner_marks_recent_alarm_pending():
    values = np.zeros(100)
    times = np.arange(values.size)
    stage_1_results = pd.DataFrame({
        "time": times,
        "stage_1_alarm": (times == 90).astype(int),
    })

    results = validate_stage_1_alarms(
        stage_1_results=stage_1_results,
        values=values,
        times=times,
        config=Stage2Config(
            before_size=20,
            after_size=20,
        ),
    )

    assert results.loc[0, "status"] == "pending_after_window"
    assert bool(results.loc[0, "confirmed_shift"]) is False
    assert np.isnan(results.loc[0, "p_value"])


def test_multi_alarm_runner_marks_early_alarm_insufficient():
    values = np.zeros(100)
    times = np.arange(values.size)
    stage_1_results = pd.DataFrame({
        "time": times,
        "stage_1_alarm": (times == 5).astype(int),
    })

    results = validate_stage_1_alarms(
        stage_1_results=stage_1_results,
        values=values,
        times=times,
        config=Stage2Config(
            before_size=20,
            after_size=20,
        ),
    )

    assert (
        results.loc[0, "status"]
        == "insufficient_before_window"
    )


def test_multi_alarm_runner_skips_nearby_alarm():
    values = np.concatenate([
        np.zeros(51),
        np.full(100, 5.0),
    ])
    times = np.arange(values.size)
    stage_1_results = pd.DataFrame({
        "time": times,
        "stage_1_alarm": np.isin(
            times,
            [50, 55],
        ).astype(int),
    })

    results = validate_stage_1_alarms(
        stage_1_results=stage_1_results,
        values=values,
        times=times,
        config=Stage2Config(
            before_size=20,
            after_size=20,
            minimum_alarm_gap=20,
        ),
    )

    assert results.loc[0, "status"] == "confirmed"
    assert results.loc[1, "status"] == "skipped_nearby_alarm"


def test_multi_alarm_runner_returns_empty_table_without_alarms():
    values = np.zeros(50)
    times = np.arange(values.size)
    stage_1_results = pd.DataFrame({
        "time": times,
        "stage_1_alarm": np.zeros(values.size, dtype=int),
    })

    results = validate_stage_1_alarms(
        stage_1_results=stage_1_results,
        values=values,
        times=times,
    )

    assert results.empty
    assert "status" in results.columns


def test_multi_alarm_runner_requires_stage_1_columns():
    stage_1_results = pd.DataFrame({
        "time": [0, 1, 2],
    })

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        validate_stage_1_alarms(
            stage_1_results=stage_1_results,
            values=np.zeros(3),
            times=np.arange(3),
        )
