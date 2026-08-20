"""Real-data Figure 1 helpers for Dataset 2A subject A07.

The visualisation deliberately reuses the Dataset 2A GDF loader, trial extractor,
band-pass filtering and CSP implementation used by the reproduction code.  It
never fabricates scatter points or ellipse locations.

When the separately released Session-II labels are unavailable, all 783-cued
Session-II trials are shown as one unlabelled test distribution.  That mode is
useful for inspecting the real marginal covariate shift, but it must not be
presented as an exact class-aware reproduction of the paper's Figure 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.bci.data import load_bci_competition_iv_2a_session
from src.bci.datasets.dataset2a import extract_dataset_2a_trials
from src.bci.datasets.dataset2a.evaluation_labels import (
    load_dataset_2a_evaluation_labels,
)
from src.bci.fbcsp import bandpass_trials, fit_binary_csp


@dataclass(frozen=True)
class Figure1BandData:
    name: str
    low_hz: float
    high_hz: float
    training_points: np.ndarray
    testing_points: np.ndarray
    training_labels: np.ndarray
    testing_labels: np.ndarray | None
    training_ellipse: np.ndarray
    testing_ellipse: np.ndarray
    training_class_ellipses: tuple[np.ndarray, ...]
    testing_class_ellipses: tuple[np.ndarray, ...]
    training_boundary: np.ndarray | None
    testing_boundary: np.ndarray | None
    feature_indices: tuple[int, int]


@dataclass(frozen=True)
class Figure1Data:
    subject: str
    training_session: str
    testing_session: str
    evaluation_mode: str
    bands: tuple[Figure1BandData, ...]


def _log_variance_features(projected_trials: np.ndarray) -> np.ndarray:
    variances = np.var(np.asarray(projected_trials, dtype=float), axis=-1, ddof=0)
    totals = variances.sum(axis=1, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError("CSP projected trials must have positive total variance.")
    return np.log(np.maximum(variances / totals, 1e-20))


def _fisher_scores(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=int)
    classes = np.unique(y)
    if classes.size != 2:
        raise ValueError("Figure 1 CSP feature ranking requires two training classes.")
    first = values[y == classes[0]]
    second = values[y == classes[1]]
    numerator = (first.mean(axis=0) - second.mean(axis=0)) ** 2
    denominator = first.var(axis=0) + second.var(axis=0) + 1e-12
    return numerator / denominator


def _ellipse_points(points: np.ndarray, *, scale: float = 2.0, count: int = 120) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] < 2:
        return np.empty((0, 2), dtype=float)
    centre = values.mean(axis=0)
    covariance = np.cov(values, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=True)
    circle = np.stack((np.cos(angles), np.sin(angles)), axis=0)
    transform = eigenvectors @ np.diag(scale * np.sqrt(eigenvalues))
    return (centre[:, None] + transform @ circle).T


def _linear_boundary(points: np.ndarray, labels: np.ndarray) -> np.ndarray | None:
    """Return a two-point equal-covariance LDA boundary in feature coordinates."""

    x = np.asarray(points, dtype=float)
    y = np.asarray(labels, dtype=int)
    classes = np.unique(y)
    if x.shape[0] < 4 or classes.size != 2:
        return None
    a = x[y == classes[0]]
    b = x[y == classes[1]]
    mean_a = a.mean(axis=0)
    mean_b = b.mean(axis=0)
    centred = np.vstack((a - mean_a, b - mean_b))
    covariance = centred.T @ centred / max(centred.shape[0] - 2, 1)
    weight = np.linalg.pinv(covariance + 1e-9 * np.eye(2)) @ (mean_b - mean_a)
    midpoint = 0.5 * (mean_a + mean_b)
    intercept = -float(weight @ midpoint)
    x_min, x_max = float(x[:, 0].min()), float(x[:, 0].max())
    padding = 0.08 * max(x_max - x_min, 1e-6)
    x_values = np.asarray([x_min - padding, x_max + padding], dtype=float)
    if abs(weight[1]) < 1e-12:
        x0 = -intercept / weight[0] if abs(weight[0]) >= 1e-12 else midpoint[0]
        y_min, y_max = float(x[:, 1].min()), float(x[:, 1].max())
        return np.asarray([[x0, y_min], [x0, y_max]], dtype=float)
    y_values = -(weight[0] * x_values + intercept) / weight[1]
    return np.column_stack((x_values, y_values))


def _class_ellipses(points: np.ndarray, labels: np.ndarray | None) -> tuple[np.ndarray, ...]:
    if labels is None:
        return ()
    y = np.asarray(labels, dtype=int)
    return tuple(_ellipse_points(points[y == label]) for label in np.unique(y))


def _band_features(
    training_signals: np.ndarray,
    training_labels: np.ndarray,
    testing_signals: np.ndarray,
    sampling_frequency: float,
    low_hz: float,
    high_hz: float,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    filtered_training = bandpass_trials(
        training_signals, sampling_frequency, low_hz, high_hz
    )
    filters = fit_binary_csp(filtered_training, training_labels)
    filtered_testing = bandpass_trials(
        testing_signals, sampling_frequency, low_hz, high_hz
    )
    projected_training = np.einsum("kc,tcs->tks", filters, filtered_training)
    projected_testing = np.einsum("kc,tcs->tks", filters, filtered_testing)
    training_features = _log_variance_features(projected_training)
    testing_features = _log_variance_features(projected_testing)
    scores = _fisher_scores(training_features, training_labels)
    best = np.argsort(scores)[::-1][:2]
    if best.size != 2:
        raise RuntimeError("Figure 1 requires two CSP feature dimensions.")
    indices = (int(best[0]), int(best[1]))
    return training_features[:, best], testing_features[:, best], indices


def build_dataset_2a_figure1(
    data_directory: str | Path,
    *,
    subject: int = 7,
    labels_directory: str | Path | None = None,
) -> Figure1Data:
    """Build the A07 mu/beta plot data from the real GDF feature pipeline."""

    data_dir = Path(data_directory)
    training_session = load_bci_competition_iv_2a_session(data_dir, subject, "T")
    testing_session = load_bci_competition_iv_2a_session(data_dir, subject, "E")
    training_trials = extract_dataset_2a_trials(training_session)

    testing_labels: np.ndarray | None = None
    evaluation_mode = "GDF-only: real 288-trial Session-II marginal distribution"
    label_path: Path | None = None
    if labels_directory is not None:
        candidate = Path(labels_directory) / f"A{subject:02d}E.mat"
        if candidate.is_file():
            label_path = candidate
    if label_path is None:
        candidate = data_dir / f"A{subject:02d}E.mat"
        if candidate.is_file():
            label_path = candidate

    if label_path is not None:
        official = load_dataset_2a_evaluation_labels(label_path)
        testing_trials = extract_dataset_2a_trials(
            testing_session, evaluation_labels=official
        )
        testing_labels = np.asarray(testing_trials.labels, dtype=int)
        evaluation_mode = "Released-label left/right Session-II distribution"
    else:
        testing_trials = extract_dataset_2a_trials(testing_session)

    bands: list[Figure1BandData] = []
    for name, low_hz, high_hz in (("μ", 8.0, 12.0), ("β", 14.0, 30.0)):
        train_points, test_points, feature_indices = _band_features(
            training_trials.signals,
            training_trials.labels,
            testing_trials.signals,
            training_trials.sampling_frequency,
            low_hz,
            high_hz,
        )
        bands.append(
            Figure1BandData(
                name=name,
                low_hz=low_hz,
                high_hz=high_hz,
                training_points=train_points,
                testing_points=test_points,
                training_labels=np.asarray(training_trials.labels, dtype=int),
                testing_labels=testing_labels,
                training_ellipse=_ellipse_points(train_points),
                testing_ellipse=_ellipse_points(test_points),
                training_class_ellipses=_class_ellipses(
                    train_points, training_trials.labels
                ),
                testing_class_ellipses=_class_ellipses(
                    test_points, testing_labels
                ),
                training_boundary=_linear_boundary(
                    train_points, training_trials.labels
                ),
                testing_boundary=(
                    _linear_boundary(test_points, testing_labels)
                    if testing_labels is not None
                    else None
                ),
                feature_indices=feature_indices,
            )
        )

    return Figure1Data(
        subject=f"A{subject:02d}",
        training_session="Session I",
        testing_session="Session II",
        evaluation_mode=evaluation_mode,
        bands=tuple(bands),
    )


def serialise_figure1(data: Figure1Data) -> dict[str, object]:
    def array(values: np.ndarray | None) -> list[list[float]] | None:
        if values is None:
            return None
        return np.asarray(values, dtype=float).round(8).tolist()

    return {
        "subject": data.subject,
        "training_session": data.training_session,
        "testing_session": data.testing_session,
        "evaluation_mode": data.evaluation_mode,
        "bands": [
            {
                "name": band.name,
                "low_hz": band.low_hz,
                "high_hz": band.high_hz,
                "training_points": array(band.training_points),
                "testing_points": array(band.testing_points),
                "training_ellipse": array(band.training_ellipse),
                "testing_ellipse": array(band.testing_ellipse),
                "training_class_ellipses": [array(item) for item in band.training_class_ellipses],
                "testing_class_ellipses": [array(item) for item in band.testing_class_ellipses],
                "training_boundary": array(band.training_boundary),
                "testing_boundary": array(band.testing_boundary),
                "feature_indices": list(band.feature_indices),
            }
            for band in data.bands
        ],
    }
