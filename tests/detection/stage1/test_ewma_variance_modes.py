import numpy as np
import pytest

from src.ewma import SD_EWMA_Config, run_sd_ewma


def _run(values, *, mode="always", multiplier=1.96):
    values = np.asarray(values, dtype=float)
    return run_sd_ewma(
        values=values,
        times=np.arange(values.size),
        initial_z=0.0,
        initial_error_variance=1.0,
        config=SD_EWMA_Config(
            lambda_value=0.2,
            variance_smoothing=0.5,
            control_limit_multiplier=multiplier,
            variance_update_mode=mode,
        ),
    )


def test_frozen_variance_never_changes():
    result = _run([0.5, 1.0, 3.0], mode="frozen")
    assert np.allclose(result["error_variance_before"], 1.0)
    assert np.allclose(result["error_variance_after"], 1.0)


def test_always_mode_updates_variance():
    result = _run([0.5, 1.0], mode="always")
    assert result.iloc[0]["error_variance_after"] != pytest.approx(1.0)
    assert result.iloc[1]["error_variance_before"] == pytest.approx(
        result.iloc[0]["error_variance_after"]
    )


def test_non_alarm_mode_updates_without_alarm():
    result = _run([0.5], mode="non_alarm")
    assert result.iloc[0]["stage_1_alarm"] == 0
    assert result.iloc[0]["error_variance_after"] != pytest.approx(1.0)


def test_non_alarm_mode_freezes_on_alarm():
    result = _run([5.0], mode="non_alarm")
    assert result.iloc[0]["stage_1_alarm"] == 1
    assert result.iloc[0]["error_variance_after"] == pytest.approx(1.0)


def test_results_include_control_limit_diagnostics():
    result = _run([0.5])
    required = {
        "error_variance_before",
        "error_std_before",
        "control_limit_multiplier",
        "control_limit_half_width",
        "error_variance_after",
        "variance_update_mode",
    }
    assert required.issubset(result.columns)
    row = result.iloc[0]
    assert row["control_limit_half_width"] == pytest.approx(
        row["control_limit_multiplier"] * row["error_std_before"]
    )


def test_larger_l_produces_no_more_alarms():
    values = [0.0, 1.7, -1.7, 2.2, -2.2]
    small = _run(values, mode="frozen", multiplier=1.5)
    large = _run(values, mode="frozen", multiplier=3.0)
    assert large["stage_1_alarm"].sum() <= small["stage_1_alarm"].sum()


@pytest.mark.parametrize("mode", ["bad", "conditional", "none"])
def test_invalid_variance_update_mode_is_rejected(mode):
    with pytest.raises(ValueError, match="variance_update_mode"):
        SD_EWMA_Config(lambda_value=0.2, variance_update_mode=mode)
