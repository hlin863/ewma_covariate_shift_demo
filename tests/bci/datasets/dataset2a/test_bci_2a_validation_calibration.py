import numpy as np
import pytest

from src.bci.datasets.dataset2a.validation_calibration import (
    calibrate_stage1_control_limit_from_validation,
)


def _features(seed: int, n_rows: int, n_features: int = 6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(n_rows, n_features))
    values[:, 1] += 0.6 * values[:, 0]
    values[:, 2] -= 0.3 * values[:, 0]
    return values


def test_validation_calibration_uses_held_out_errors_to_produce_finite_l() -> None:
    training = _features(10, 80)
    validation = _features(11, 30)

    result = calibrate_stage1_control_limit_from_validation(
        training,
        validation,
        lambda_value=0.50,
        pca_components=None,
        target_false_alarm_rate=0.05,
    )

    assert result.control_limit_multiplier > 0.0
    assert np.isfinite(result.control_limit_multiplier)
    assert result.validation_rmse > 0.0
    assert result.n_validation_observations == 30
    assert 0.0 <= result.observed_exceedance_rate <= 0.05
    assert 0.0 < result.pc1_explained_variance_ratio <= 1.0


def test_stricter_validation_tail_rate_does_not_reduce_control_limit() -> None:
    training = _features(20, 100)
    validation = _features(21, 40)

    loose = calibrate_stage1_control_limit_from_validation(
        training,
        validation,
        lambda_value=0.55,
        target_false_alarm_rate=0.20,
    )
    strict = calibrate_stage1_control_limit_from_validation(
        training,
        validation,
        lambda_value=0.55,
        target_false_alarm_rate=0.05,
    )

    assert strict.control_limit_multiplier >= loose.control_limit_multiplier


def test_validation_calibration_rejects_non_alarm_variance_mode() -> None:
    training = _features(30, 60)
    validation = _features(31, 20)

    with pytest.raises(ValueError, match="non_alarm"):
        calibrate_stage1_control_limit_from_validation(
            training,
            validation,
            lambda_value=0.60,
            variance_update_mode="non_alarm",
        )
