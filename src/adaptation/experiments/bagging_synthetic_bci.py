"""Synthetic Dataset 2A/2B integration experiment for bagging.

The deterministic sessions mirror the small synthetic BCI Competition IV
scenarios already used by the repository's Dataset 2A and Dataset 2B tests.
They are intentionally lightweight: the goal is to verify that the committed
BaggingClassifier can consume the same FBCSP feature pipelines as the existing
single-classifier experiments before moving to real EEG data.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from src.adaptation.classifier import LinearSVMClassifier
from src.adaptation.ensemble.bagging import BaggingClassifier
from src.bci.data import BCISessionData
from src.bci.datasets.dataset2a.experiment import (
    DATASET_2A_CHANNELS,
    build_dataset_2a_fbcsp_features,
    extract_dataset_2a_trials,
)
from src.bci.datasets.dataset2b.experiment import (
    build_dataset_2b_fbcsp_features,
    concatenate_trial_signals,
    extract_dataset_2b_trials,
)


def _dataset_2a_session(
    session: str,
    *,
    n_trials: int,
) -> BCISessionData:
    """Build the deterministic Dataset 2A-like signal used for integration."""

    rng = np.random.default_rng(70 if session == "T" else 71)
    sampling_frequency = 100.0
    first_onset = 2.0
    spacing = 5.0
    last_onset = first_onset + (n_trials - 1) * spacing
    n_samples = int(np.ceil((last_onset + 4.0) * sampling_frequency))

    signals = rng.normal(
        scale=0.25,
        size=(n_samples, len(DATASET_2A_CHANNELS)),
    )
    annotations: list[tuple[float, float, str]] = []

    for index in range(n_trials):
        onset = first_onset + index * spacing
        code = "783" if session == "E" else ("769" if index % 2 == 0 else "770")
        start = int(onset * sampling_frequency)
        stop = start + int(3 * sampling_frequency)
        time = np.arange(stop - start) / sampling_frequency
        channel = 0 if index % 2 == 0 else min(5, len(DATASET_2A_CHANNELS) - 1)
        signals[start:stop, channel] += 1.5 * np.sin(2 * np.pi * 10.0 * time)
        annotations.append((float(onset), 0.0, code))

    return BCISessionData(
        subject=1,
        session=session,
        file_path=Path(f"A01{session}.gdf"),
        signals=signals,
        times=np.arange(signals.shape[0]) / sampling_frequency,
        channel_names=DATASET_2A_CHANNELS,
        sampling_frequency=sampling_frequency,
        annotations=tuple(annotations),
    )


def _dataset_2a_evaluation_labels() -> np.ndarray:
    """Return the 288-class official-format labels used by the 2A test fixture."""

    return np.tile(np.asarray([1, 2, 3, 4], dtype=int), 72)


def _dataset_2b_session(
    session: str,
    *,
    labelled: bool,
) -> BCISessionData:
    """Build the deterministic Dataset 2B-like signal used for integration."""

    rng = np.random.default_rng(7 + int(session[:2]))
    sampling_frequency = 100.0
    signals = rng.normal(scale=0.4, size=(5000, 3))
    annotations: list[tuple[float, float, str]] = []

    for index, onset in enumerate((2, 7, 12, 17, 22, 27, 32, 37)):
        code = ("769" if index % 2 == 0 else "770") if labelled else "783"
        start = int(onset * sampling_frequency)
        stop = start + int(3 * sampling_frequency)
        time = np.arange(stop - start) / sampling_frequency
        channel = 0 if index % 2 == 0 else 2
        signals[start:stop, channel] += 2.0 * np.sin(2 * np.pi * 10.0 * time)
        annotations.append((float(onset), 0.0, code))

    return BCISessionData(
        subject=1,
        session=session,
        file_path=Path(f"B01{session}.gdf"),
        signals=signals,
        times=np.arange(signals.shape[0]) / sampling_frequency,
        channel_names=("C3", "Cz", "C4"),
        sampling_frequency=sampling_frequency,
        annotations=tuple(annotations),
    )


def _evaluate_models(
    *,
    dataset: str,
    training_features: np.ndarray,
    training_labels: np.ndarray,
    testing_features: np.ndarray,
    testing_labels: np.ndarray,
    random_state: int,
    n_estimators: int,
    sample_fraction: float,
) -> list[dict[str, object]]:
    """Compare the repository's linear SVM with its bagged counterpart."""

    records: list[dict[str, object]] = []

    single = LinearSVMClassifier(random_state=random_state)
    started = perf_counter()
    single.fit(training_features, training_labels)
    fit_seconds = perf_counter() - started
    started = perf_counter()
    single_predictions = single.predict(testing_features)
    predict_seconds = perf_counter() - started

    records.append(
        {
            "dataset": dataset,
            "method": "single_linear_svm",
            "training_trials": int(training_features.shape[0]),
            "testing_trials": int(testing_features.shape[0]),
            "correct_predictions": int(
                np.count_nonzero(single_predictions == testing_labels)
            ),
            "accuracy": float(np.mean(single_predictions == testing_labels)),
            "fit_seconds": float(fit_seconds),
            "predict_seconds": float(predict_seconds),
            "requested_estimators": 1,
            "fitted_estimators": 1,
            "sample_fraction": 1.0,
            "mean_member_disagreement": 0.0,
        }
    )

    ensemble = BaggingClassifier(
        estimator_factory=lambda: LinearSVMClassifier(random_state=random_state),
        n_estimators=n_estimators,
        sample_fraction=sample_fraction,
        bootstrap=True,
        random_state=random_state,
    )
    started = perf_counter()
    ensemble.fit(training_features, training_labels)
    fit_seconds = perf_counter() - started
    started = perf_counter()
    ensemble_predictions = ensemble.predict(testing_features)
    predict_seconds = perf_counter() - started

    member_predictions = np.stack(
        [estimator.predict(testing_features) for estimator in ensemble.estimators_],
        axis=0,
    )
    disagreement = np.mean(
        member_predictions != ensemble_predictions.reshape(1, -1),
        axis=0,
    )

    records.append(
        {
            "dataset": dataset,
            "method": "bagged_linear_svm",
            "training_trials": int(training_features.shape[0]),
            "testing_trials": int(testing_features.shape[0]),
            "correct_predictions": int(
                np.count_nonzero(ensemble_predictions == testing_labels)
            ),
            "accuracy": float(np.mean(ensemble_predictions == testing_labels)),
            "fit_seconds": float(fit_seconds),
            "predict_seconds": float(predict_seconds),
            "requested_estimators": int(n_estimators),
            "fitted_estimators": int(len(ensemble.estimators_)),
            "sample_fraction": float(sample_fraction),
            "mean_member_disagreement": float(np.mean(disagreement)),
        }
    )

    return records


def run_bagging_synthetic_bci_experiment(
    *,
    random_state: int = 42,
    n_estimators: int = 30,
    sample_fraction: float = 0.8,
) -> pd.DataFrame:
    """Run bagging against the repository's deterministic 2A/2B-like streams."""

    training_2a = extract_dataset_2a_trials(_dataset_2a_session("T", n_trials=48))
    testing_2a = extract_dataset_2a_trials(
        _dataset_2a_session("E", n_trials=288),
        evaluation_labels=_dataset_2a_evaluation_labels(),
    )
    pipeline_2a = build_dataset_2a_fbcsp_features(training_2a, testing_2a)

    training_2b = concatenate_trial_signals(
        [
            extract_dataset_2b_trials(_dataset_2b_session("01T", labelled=True)),
            extract_dataset_2b_trials(_dataset_2b_session("02T", labelled=True)),
            extract_dataset_2b_trials(_dataset_2b_session("03T", labelled=True)),
        ]
    )
    testing_2b = concatenate_trial_signals(
        [
            extract_dataset_2b_trials(_dataset_2b_session("04E", labelled=True)),
            extract_dataset_2b_trials(_dataset_2b_session("05E", labelled=True)),
        ]
    )
    pipeline_2b = build_dataset_2b_fbcsp_features(training_2b, testing_2b)

    records = []
    records.extend(
        _evaluate_models(
            dataset="2A",
            training_features=pipeline_2a.training.features,
            training_labels=training_2a.labels,
            testing_features=pipeline_2a.testing.features,
            testing_labels=testing_2a.labels,
            random_state=random_state,
            n_estimators=n_estimators,
            sample_fraction=sample_fraction,
        )
    )
    records.extend(
        _evaluate_models(
            dataset="2B",
            training_features=pipeline_2b.training.features,
            training_labels=training_2b.labels,
            testing_features=pipeline_2b.testing.features,
            testing_labels=testing_2b.labels,
            random_state=random_state,
            n_estimators=n_estimators,
            sample_fraction=sample_fraction,
        )
    )

    return pd.DataFrame.from_records(records)
