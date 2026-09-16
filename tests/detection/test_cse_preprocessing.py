import numpy as np
import pytest

from src.cse_preprocessing import (
    extract_first_component,
    fit_cse_pca,
    transform_cse_features,
)


def test_fit_cse_pca_returns_expected_dimensions() -> None:
    training_features = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 5.0, 7.0],
        [4.0, 6.0, 8.0],
        [5.0, 8.0, 10.0],
    ])

    result = fit_cse_pca(
        training_features,
        n_components=2,
    )

    assert result.training_transformed.shape == (5, 2)
    assert result.n_input_features == 3
    assert result.n_components == 2
    assert result.explained_variance_ratio.shape == (2,)


def test_first_component_property_returns_one_dimension() -> None:
    training_features = np.array([
        [1.0, 2.0],
        [2.0, 4.0],
        [3.0, 5.0],
        [4.0, 8.0],
    ])

    result = fit_cse_pca(
        training_features,
        n_components=2,
    )

    assert result.first_component.shape == (4,)
    np.testing.assert_allclose(
        result.first_component,
        result.training_transformed[:, 0],
    )


def test_transform_reuses_fitted_pca_model() -> None:
    training_features = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 5.0, 6.0],
        [4.0, 7.0, 8.0],
    ])

    testing_features = np.array([
        [5.0, 8.0, 9.0],
        [6.0, 9.0, 11.0],
    ])

    result = fit_cse_pca(
        training_features,
        n_components=2,
    )

    transformed = transform_cse_features(
        testing_features,
        fitted_result=result,
    )

    expected = result.model.transform(testing_features)

    assert transformed.shape == (2, 2)
    np.testing.assert_allclose(
        transformed,
        expected,
    )


def test_extract_first_component() -> None:
    transformed_features = np.array([
        [2.5, 0.4],
        [3.1, -0.2],
        [4.0, 0.1],
    ])

    first_component = extract_first_component(
        transformed_features,
    )

    np.testing.assert_allclose(
        first_component,
        np.array([2.5, 3.1, 4.0]),
    )


def test_fit_rejects_one_dimensional_input() -> None:
    with pytest.raises(
        ValueError,
        match="two-dimensional",
    ):
        fit_cse_pca(
            np.array([1.0, 2.0, 3.0]),
        )


def test_fit_rejects_nonfinite_values() -> None:
    training_features = np.array([
        [1.0, 2.0],
        [np.nan, 3.0],
    ])

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        fit_cse_pca(training_features)


def test_fit_requires_at_least_two_observations() -> None:
    with pytest.raises(
        ValueError,
        match="at least 2 observations",
    ):
        fit_cse_pca(
            np.array([[1.0, 2.0]]),
        )


def test_fit_rejects_too_many_components() -> None:
    training_features = np.array([
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
    ])

    with pytest.raises(
        ValueError,
        match="must not exceed",
    ):
        fit_cse_pca(
            training_features,
            n_components=3,
        )


def test_transform_rejects_wrong_feature_count() -> None:
    training_features = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0],
    ])

    result = fit_cse_pca(
        training_features,
        n_components=2,
    )

    testing_features = np.array([
        [4.0, 5.0],
    ])

    with pytest.raises(
        ValueError,
        match="expected 3",
    ):
        transform_cse_features(
            testing_features,
            fitted_result=result,
        )