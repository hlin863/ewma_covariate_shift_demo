import numpy as np
import pytest

from src.ewma import (
    calculate_ewma_training_path,
    estimate_lambda,
    fit_sd_ewma,
)


def test_training_path_matches_manual_calculation():
    values = np.array([2.0, 4.0, 6.0])

    ewma_values, errors = calculate_ewma_training_path(
        values=values,
        lambda_value=0.5,
        initial_z=2.0,
    )

    np.testing.assert_allclose(
        errors,
        np.array([0.0, 2.0, 3.0]),
    )

    np.testing.assert_allclose(
        ewma_values,
        np.array([2.0, 3.0, 4.5]),
    )


def test_estimate_lambda_returns_lowest_sse_candidate():
    values = np.array([
        0.0,
        0.2,
        0.1,
        0.3,
        0.2,
    ])

    candidates = np.array([0.1, 0.5, 1.0])

    best_lambda, results = estimate_lambda(
        values=values,
        candidates=candidates,
    )

    expected_lambda = float(
        results.loc[
            results["sse"].idxmin(),
            "lambda",
        ]
    )

    assert best_lambda == expected_lambda


def test_fit_uses_training_mean_as_initial_state():
    values = np.array([1.0, 2.0, 3.0, 4.0])

    result = fit_sd_ewma(values)

    assert np.isclose(
        result.initial_z,
        np.mean(values),
    )


def test_fit_uses_mean_squared_prediction_error():
    values = np.array([1.0, 2.0, 3.0, 4.0])

    result = fit_sd_ewma(values)

    _, errors = calculate_ewma_training_path(
        values=values,
        lambda_value=result.lambda_value,
        initial_z=result.initial_z,
    )

    assert np.isclose(
        result.error_variance,
        np.mean(errors**2),
    )


@pytest.mark.parametrize(
    "lambda_value",
    [0.0, -0.1, 1.1],
)
def test_invalid_lambda_raises(lambda_value):
    with pytest.raises(ValueError):
        calculate_ewma_training_path(
            values=np.array([1.0, 2.0]),
            lambda_value=lambda_value,
            initial_z=1.0,
        )