"""Integration tests for loading processed BCI features into CSE.

These tests use small deterministic ``.npz`` files with the same contract as
future BCI Competition IV Dataset 2A feature exports. They do not require the
large raw GDF archive during normal pytest runs.

The full CSE pipeline test checks data loading and end-to-end execution. A
separate Stage-II test places a known warning at the synthetic distribution
boundary, because an EWMA warning may occur after the true boundary; when that
happens, both retrospective windows can contain post-shift observations and a
correct Hotelling test may reject the warning.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.bci_data import load_cse_feature_file
from src.cse import CSEConfig, run_cse
from src.multivariate_stage_2 import (
    HotellingConfig,
    validate_multivariate_alarms,
)


SHIFT_INDEX = 120


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
        size=SHIFT_INDEX,
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

    before_mean = evaluation[:SHIFT_INDEX].mean(axis=0)
    after_mean = evaluation[SHIFT_INDEX:].mean(axis=0)

    assert np.linalg.norm(after_mean - before_mean) > 2.5


def test_cse_accepts_loaded_bci_feature_files(tmp_path: Path) -> None:
    """The loaded feature contract must run through the full CSE pipeline."""

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
    assert not result.validation_results.empty
    assert set(result.validation_results["status"]).issubset(
        {
            "confirmed",
            "rejected",
            "pending_after_window",
            "insufficient_before_window",
            "insufficient_degrees_of_freedom",
            "skipped_nearby_alarm",
        }
    )


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

    transformed_before = result.testing_transformed[:SHIFT_INDEX]
    transformed_after = result.testing_transformed[SHIFT_INDEX:]

    mean_distance = np.linalg.norm(
        transformed_after.mean(axis=0)
        - transformed_before.mean(axis=0)
    )

    assert mean_distance > 2.0
    assert abs(result.testing_signal[SHIFT_INDEX:].mean()) > (
        abs(result.testing_signal[:SHIFT_INDEX].mean()) + 1.0
    )


def test_stage_two_confirms_warning_at_known_distribution_boundary(
    tmp_path: Path,
) -> None:
    """A boundary-aligned warning must compare pre- and post-shift windows."""

    training_path, evaluation_path = _make_bci_like_feature_files(tmp_path)

    training, _, _ = load_cse_feature_file(training_path)
    evaluation, evaluation_times, _ = load_cse_feature_file(
        evaluation_path
    )

    cse_result = run_cse(
        training_features=training,
        testing_features=evaluation,
        testing_times=evaluation_times,
        config=_config(),
    )

    boundary_warning = pd.DataFrame(
        {
            "time": evaluation_times,
            "stage_1_alarm": (
                evaluation_times == SHIFT_INDEX - 1
            ).astype(int),
        }
    )

    validation = validate_multivariate_alarms(
        features=cse_result.testing_transformed,
        times=evaluation_times,
        stage_1_results=boundary_warning,
        config=HotellingConfig(
            before_size=20,
            after_size=20,
            alpha=0.05,
            covariance_method="shrinkage",
            regularization=1e-6,
            minimum_alarm_gap=20,
        ),
    )

    assert validation.shape[0] == 1
    row = validation.iloc[0]
    assert row["alarm_time"] == SHIFT_INDEX - 1
    assert row["before_end_time"] == SHIFT_INDEX - 1
    assert row["after_start_time"] == SHIFT_INDEX
    assert row["status"] == "confirmed"
    assert bool(row["confirmed_shift"])
    assert row["p_value"] < 0.05
