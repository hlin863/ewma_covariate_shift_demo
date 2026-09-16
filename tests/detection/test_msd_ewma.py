import numpy as np
import pytest

from src.msd_ewma import (
    calculate_multivariate_ewma_path,
)


def test_multivariate_ewma_path_returns_expected_shapes():
    values = np.array([
        [1.0, 10.0],
        [2.0, 12.0],
        [3.0, 14.0],
    ])

    states, errors = calculate_multivariate_ewma_path(
        values=values,
        lambda_value=0.5,
        initial_z=np.array([0.0, 8.0]),
    )

    assert states.shape == values.shape
    assert errors.shape == values.shape


def test_multivariate_ewma_path_calculates_first_state():
    values = np.array([
        [2.0, 10.0],
        [4.0, 14.0],
    ])

    states, errors = calculate_multivariate_ewma_path(
        values=values,
        lambda_value=0.5,
        initial_z=np.array([0.0, 6.0]),
    )

    np.testing.assert_allclose(
        errors[0],
        np.array([2.0, 4.0]),
    )

    np.testing.assert_allclose(
        states[0],
        np.array([1.0, 8.0]),
    )


def test_multivariate_ewma_path_rejects_1d_values():
    with pytest.raises(
        ValueError,
        match="n_samples, n_features",
    ):
        calculate_multivariate_ewma_path(
            values=np.array([1.0, 2.0, 3.0]),
            lambda_value=0.5,
            initial_z=np.array([0.0]),
        )


def test_initial_state_requires_one_value_per_feature():
    values = np.ones((10, 3))

    with pytest.raises(
        ValueError,
        match="one value per feature",
    ):
        calculate_multivariate_ewma_path(
            values=values,
            lambda_value=0.5,
            initial_z=np.zeros(2),
        )


@pytest.mark.parametrize(
    "lambda_value",
    [0.0, -0.1, 1.1],
)
def test_invalid_lambda_raises(lambda_value):
    with pytest.raises(
        ValueError,
        match="lambda_value must be in",
    ):
        calculate_multivariate_ewma_path(
            values=np.ones((10, 2)),
            lambda_value=lambda_value,
            initial_z=np.zeros(2),
        )