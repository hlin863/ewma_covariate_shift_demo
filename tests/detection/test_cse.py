import numpy as np
import pandas as pd
import pytest

from src.cse import CSEConfig, CSEResult, run_cse


def _make_cse_config() -> CSEConfig:
    """Return a compact, deterministic configuration for CSE tests."""

    return CSEConfig(
        pca_components=2,
        variance_smoothing=0.05,
        control_limit_multiplier=3.0,
        validation_before_size=20,
        validation_after_size=20,
        validation_alpha=0.05,
        covariance_method="shrinkage",
        covariance_regularization=1e-6,
        minimum_alarm_gap=20,
    )


def _make_mean_shift_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a paper-inspired multivariate mean-shift experiment."""

    rng = np.random.default_rng(42)

    covariance = np.array([
        [1.0, 0.35, 0.20],
        [0.35, 1.0, 0.25],
        [0.20, 0.25, 1.0],
    ])

    training_features = rng.multivariate_normal(
        mean=np.zeros(3),
        cov=covariance,
        size=300,
    )

    before_shift = rng.multivariate_normal(
        mean=np.zeros(3),
        cov=covariance,
        size=120,
    )

    after_shift = rng.multivariate_normal(
        mean=np.full(3, 5.0),
        cov=covariance,
        size=120,
    )

    testing_features = np.vstack([
        before_shift,
        after_shift,
    ])
    testing_times = np.arange(testing_features.shape[0])

    return training_features, testing_features, testing_times


def test_run_cse_returns_complete_result() -> None:
    training, testing, times = _make_mean_shift_data()

    result = run_cse(
        training_features=training,
        testing_features=testing,
        testing_times=times,
        config=_make_cse_config(),
    )

    assert isinstance(result, CSEResult)
    assert result.training_signal.shape == (training.shape[0],)
    assert result.testing_signal.shape == (testing.shape[0],)
    assert result.testing_transformed.shape == (testing.shape[0], 2)
    assert isinstance(result.warning_results, pd.DataFrame)
    assert isinstance(result.validation_results, pd.DataFrame)

    expected_warning_columns = {
        "time",
        "x",
        "prediction",
        "ewma",
        "error",
        "lcl",
        "ucl",
        "stage_1_alarm",
    }
    expected_validation_columns = {
        "alarm_time",
        "validation_time",
        "before_start_time",
        "before_end_time",
        "after_start_time",
        "after_end_time",
        "hotelling_t_squared",
        "f_statistic",
        "p_value",
        "status",
        "confirmed_shift",
    }

    assert expected_warning_columns.issubset(
        result.warning_results.columns
    )
    assert expected_validation_columns.issubset(
        result.validation_results.columns
    )


def test_cse_reuses_training_pca_and_monitors_first_component() -> None:
    training, testing, times = _make_mean_shift_data()

    result = run_cse(
        training_features=training,
        testing_features=testing,
        testing_times=times,
        config=_make_cse_config(),
    )

    expected_testing_transformed = (
        result.pca_result.model.transform(testing)
    )

    np.testing.assert_allclose(
        result.testing_transformed,
        expected_testing_transformed,
    )
    np.testing.assert_allclose(
        result.training_signal,
        result.pca_result.training_transformed[:, 0],
    )
    np.testing.assert_allclose(
        result.testing_signal,
        result.testing_transformed[:, 0],
    )


def test_sustained_multivariate_mean_shift_is_confirmed() -> None:
    training, testing, times = _make_mean_shift_data()
    config = _make_cse_config()

    result = run_cse(
        training_features=training,
        testing_features=testing,
        testing_times=times,
        config=config,
    )

    assert result.warning_results["stage_1_alarm"].sum() > 0

    confirmed = result.validation_results.loc[
        result.validation_results["confirmed_shift"]
    ]

    assert not confirmed.empty
    assert (confirmed["status"] == "confirmed").all()
    assert (confirmed["p_value"] < config.validation_alpha).all()


def test_evaluated_cse_windows_and_decisions_are_consistent() -> None:
    training, testing, times = _make_mean_shift_data()
    config = _make_cse_config()

    result = run_cse(
        training_features=training,
        testing_features=testing,
        testing_times=times,
        config=config,
    )

    evaluated = result.validation_results.loc[
        result.validation_results["status"].isin(
            ["confirmed", "rejected"]
        )
    ]

    assert not evaluated.empty
    assert (
        evaluated["before_end_time"]
        == evaluated["alarm_time"]
    ).all()
    assert (
        evaluated["after_start_time"]
        > evaluated["before_end_time"]
    ).all()
    assert (
        evaluated["validation_time"]
        == evaluated["after_end_time"]
    ).all()
    assert (evaluated["hotelling_t_squared"] >= 0.0).all()
    assert (evaluated["f_statistic"] >= 0.0).all()
    assert evaluated["p_value"].between(0.0, 1.0).all()

    expected_decisions = (
        evaluated["p_value"] < config.validation_alpha
    )
    np.testing.assert_array_equal(
        evaluated["confirmed_shift"].to_numpy(),
        expected_decisions.to_numpy(),
    )


def test_cse_rejects_mismatched_feature_counts() -> None:
    training = np.zeros((20, 3))
    testing = np.zeros((10, 2))

    with pytest.raises(
        ValueError,
        match="same number of feature columns",
    ):
        run_cse(
            training_features=training,
            testing_features=testing,
            testing_times=np.arange(testing.shape[0]),
        )


def test_cse_rejects_duplicate_testing_times() -> None:
    training = np.zeros((20, 2))
    testing = np.zeros((10, 2))
    times = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 8])

    with pytest.raises(
        ValueError,
        match="uniquely identify observations",
    ):
        run_cse(
            training_features=training,
            testing_features=testing,
            testing_times=times,
        )


def test_cse_rejects_nonfinite_features() -> None:
    training = np.zeros((20, 2))
    testing = np.zeros((10, 2))
    testing[4, 1] = np.nan

    with pytest.raises(
        ValueError,
        match="testing_features must contain only finite values",
    ):
        run_cse(
            training_features=training,
            testing_features=testing,
            testing_times=np.arange(testing.shape[0]),
        )
