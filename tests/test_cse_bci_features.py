"""Integration tests for loading processed BCI features into CSE.

These tests use small deterministic ``.npz`` files with the same contract as
future BCI Competition IV Dataset 2A feature exports. They do not require the
large raw GDF archive during normal pytest runs.
"""

from pathlib import Path

import numpy as np

from src.bci_data import load_cse_feature_file
from src.cse import CSEConfig, run_cse


def _covariance() -> np.ndarray:
    return np.array(
        [
            [1.00, 0.35, 0.15, 0.10],
            [0.35, 1.00, 0.30, 0.15],
            [0.15, 0.30, 1.00, 0.25],
            [0.10, 0.15, 0.25, 1.00],
        ]
    )


def _write_feature_file(
    path: Path,
    features: np.ndarray,
    *,
    start_time: int = 0,
) -> None:
    times = np.arange(start_time, start_time + features.shape[0])
    labels = np.resize(np.array([1, 2, 3, 4]), features.shape[0])
    np.savez(path, features=features, times=times, labels=labels)


def _make_bci_like_feature_files(tmp_path: Path) -> tuple[Path, Path]:
    """Create training and evaluation files with a session-level shift."""

    rng = np.random.default_rng(2026)
    covariance = _covariance()

    training_features = rng.multivariate_normal(
        mean=np.zeros(4),
        cov=covariance,
        size=300,
    )

    evaluation_before = rng.multivariate_normal(
        mean=np.zeros(4),
        cov=covariance,
        size=120,
    )
    evaluation_after = rng.multivariate_normal(
        mean=np.array([2.5, 2.0, 1.5, 1.0]),
        cov=covariance,
        size=120,
    )
    evaluation_features = np.vstack(
        [evaluation_before, evaluation_after]
    )

    training_path = tmp_path / "A01T_features.npz"
    evaluation_path = tmp_path / "A01E_features.npz"

    _write_feature_file(training_path, training_features)
    _write_feature_file(evaluation_path, evaluation_features)

    return training_path, evaluation_path


def _config() -> CSEConfig:
    return CSEConfig(
        pca_components=3,
        variance_smoothing=0.05,
        control_limit_multiplier=3.0,
        validation_before_size=20,
        validation_after_size=20,
        validation_alpha=0.05,
        covariance_method="shrinkage",
        covariance_regularization=1e-6,
        minimum_alarm_gap=20,
    )


def test_processed_bci_feature_files_preserve_distribution_shapes(
    tmp_path: Path,
) -> None:
    training_path, evaluation_path = _make_bci_like_feature_files(tmp_path)

    training, training_times, training_labels = load_cse_feature_file(
        training_path
    )
    evaluation, evaluation_times, evaluation_labels = load_cse_feature_file(
        evaluation_path
    )

    assert training.shape == (300, 4)
    assert evaluation.shape == (240, 4)
    assert training_times.shape == (300,)
    assert evaluation_times.shape == (240,)
    assert training_labels is not None
    assert evaluation_labels is not None

    np.testing.assert_allclose(
        training.mean(axis=0),
        np.zeros(4),
        atol=0.20,
    )

    before_mean = evaluation[:120].mean(axis=0)
    after_mean = evaluation[120:].mean(axis=0)

    assert np.linalg.norm(after_mean - before_mean) > 2.5


def test_cse_accepts_loaded_bci_feature_files(tmp_path: Path) -> None:
    training_path, evaluation_path = _make_bci_like_feature_files(tmp_path)

    training, _, _ = load_cse_feature_file(training_path)
    evaluation, evaluation_times, _ = load_cse_feature_file(
        evaluation_path
    )

    result = run_cse(
        training_features=training,
        testing_features=evaluation,
        testing_times=evaluation_times,
        config=_config(),
    )

    assert result.testing_transformed.shape == (240, 3)
    assert result.warning_results.shape[0] == 240
    assert result.warning_results["stage_1_alarm"].sum() > 0
    assert result.validation_results["confirmed_shift"].any()


def test_cse_transformed_distribution_changes_after_session_shift(
    tmp_path: Path,
) -> None:
    training_path, evaluation_path = _make_bci_like_feature_files(tmp_path)

    training, _, _ = load_cse_feature_file(training_path)
    evaluation, evaluation_times, _ = load_cse_feature_file(
        evaluation_path
    )

    result = run_cse(
        training_features=training,
        testing_features=evaluation,
        testing_times=evaluation_times,
        config=_config(),
    )

    transformed_before = result.testing_transformed[:120]
    transformed_after = result.testing_transformed[120:]

    mean_distance = np.linalg.norm(
        transformed_after.mean(axis=0)
        - transformed_before.mean(axis=0)
    )

    assert mean_distance > 2.0
    assert abs(result.testing_signal[120:].mean()) > (
        abs(result.testing_signal[:120].mean()) + 1.0
    )


def test_confirmed_shift_is_associated_with_shifted_half(
    tmp_path: Path,
) -> None:
    training_path, evaluation_path = _make_bci_like_feature_files(tmp_path)

    training, _, _ = load_cse_feature_file(training_path)
    evaluation, evaluation_times, _ = load_cse_feature_file(
        evaluation_path
    )

    result = run_cse(
        training_features=training,
        testing_features=evaluation,
        testing_times=evaluation_times,
        config=_config(),
    )

    confirmed = result.validation_results.loc[
        result.validation_results["confirmed_shift"]
    ]

    assert not confirmed.empty
    assert (confirmed["p_value"] < 0.05).all()
    assert (confirmed["validation_time"] >= 120).any()
