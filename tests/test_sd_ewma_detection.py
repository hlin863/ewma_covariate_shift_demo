import numpy as np
import pytest

from src.ewma import (
    SD_EWMA_Config,
    run_sd_ewma,
)


def test_first_limits_use_initial_state_and_variance():
    results = run_sd_ewma(
        values=np.array([1.5]),
        times=np.array([10]),
        initial_z=1.0,
        initial_error_variance=4.0,
        config=SD_EWMA_Config(
            lambda_value=0.5,
            variance_smoothing=0.05,
            control_limit_multiplier=3.0,
        ),
    )

    assert np.isclose(
        results.loc[0, "prediction"],
        1.0,
    )

    assert np.isclose(
        results.loc[0, "lcl"],
        -5.0,
    )

    assert np.isclose(
        results.loc[0, "ucl"],
        7.0,
    )


@pytest.mark.parametrize(
    ("observation", "expected_alarm"),
    [
        (-2.1, 1),
        (-2.0, 0),
        (0.0, 0),
        (4.0, 0),
        (4.1, 1),
    ],
)
def test_alarm_boundary_rule(
    observation,
    expected_alarm,
):
    results = run_sd_ewma(
        values=np.array([observation]),
        times=np.array([0]),
        initial_z=1.0,
        initial_error_variance=1.0,
        config=SD_EWMA_Config(
            lambda_value=0.5,
            variance_smoothing=0.05,
            control_limit_multiplier=3.0,
        ),
    )

    assert (
        results.loc[0, "stage_1_alarm"]
        == expected_alarm
    )


def test_constant_stationary_stream_has_no_alarms(
    stationary_values,
    default_config,
):
    results = run_sd_ewma(
        values=stationary_values,
        times=np.arange(stationary_values.size),
        initial_z=2.0,
        initial_error_variance=1.0,
        config=default_config,
    )

    assert results["stage_1_alarm"].sum() == 0


def test_clear_abrupt_shift_is_detected(
    abrupt_shift_values,
    default_config,
):
    results = run_sd_ewma(
        values=abrupt_shift_values,
        times=np.arange(abrupt_shift_values.size),
        initial_z=1.0,
        initial_error_variance=0.25,
        config=default_config,
    )

    post_shift_alarms = results.loc[
        (results["time"] >= 30)
        & (results["stage_1_alarm"] == 1)
    ]

    assert not post_shift_alarms.empty
    assert int(post_shift_alarms.iloc[0]["time"]) == 30