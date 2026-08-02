import numpy as np
import pytest

from src.cse import CSEConfig, run_cse


def _make_small_shift_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    covariance = np.array([
        [1.0, 0.25],
        [0.25, 1.0],
    ])

    training = rng.multivariate_normal(
        mean=np.zeros(2),
        cov=covariance,
        size=120,
    )
    testing = np.vstack([
        rng.multivariate_normal(
            mean=np.zeros(2),
            cov=covariance,
            size=60,
        ),
        rng.multivariate_normal(
            mean=np.full(2, 4.0),
            cov=covariance,
            size=60,
        ),
    ])

    return training, testing, np.arange(testing.shape[0])


def test_cse_uses_lambda_override() -> None:
    training, testing, times = _make_small_shift_data()

    result = run_cse(
        training_features=training,
        testing_features=testing,
        testing_times=times,
        config=CSEConfig(
            pca_components=2,
            lambda_override=0.30,
            validation_before_size=10,
            validation_after_size=10,
            minimum_alarm_gap=10,
        ),
    )

    assert result.effective_lambda == pytest.approx(0.30)
    assert result.ewma_training_result.lambda_value > 0.0


def test_cse_uses_fitted_lambda_without_override() -> None:
    training, testing, times = _make_small_shift_data()

    result = run_cse(
        training_features=training,
        testing_features=testing,
        testing_times=times,
        config=CSEConfig(
            pca_components=2,
            validation_before_size=10,
            validation_after_size=10,
            minimum_alarm_gap=10,
        ),
    )

    assert result.effective_lambda == pytest.approx(
        result.ewma_training_result.lambda_value
    )


@pytest.mark.parametrize("lambda_value", [0.0, -0.1, 1.1])
def test_cse_rejects_invalid_lambda_override(
    lambda_value: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="lambda_override must be in",
    ):
        CSEConfig(lambda_override=lambda_value)
