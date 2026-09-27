"""Real BCI Competition IV Dataset 2A/2B bagging evaluation.

This module deliberately uses the repository's GDF/MAT loaders. Synthetic
sessions remain available in bagging_synthetic_bci.py only as a fast smoke
test for the software integration.

Protocol
--------
Dataset 2A:
    Session I (T) supplies left/right training trials. Session II (E) supplies
    left/right evaluation trials using the official AxxE.mat labels.

Dataset 2B:
    The default GDF-only benchmark uses Sessions I-II for training and the
    labelled Session III GDF for evaluation. This keeps the experiment runnable
    with the Dataset 2B files already supported by the repository.

    If an explicit labels_directory is supplied, the optional paper-evaluation
    mode instead retains Session III as calibration and evaluates Sessions IV-V
    using released Bxx04E.mat/Bxx05E.mat labels.

The comparison is classifier-focused: one LinearSVMClassifier versus a
BaggingClassifier wrapping the same linear SVM. It does not reproduce IWLDA
or CSE-UAEL.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from src.adaptation.classifier import LinearSVMClassifier
from src.adaptation.ensemble.bagging import BaggingClassifier
from src.bci.data import (
    load_bci_competition_iv_2a_session,
    load_bci_competition_iv_2b_session,
)
from src.bci.datasets.dataset2a import (
    build_dataset_2a_fbcsp_features,
    extract_dataset_2a_trials,
)
from src.bci.datasets.dataset2a.evaluation_labels import (
    load_dataset_2a_evaluation_labels,
    resolve_dataset_2a_evaluation_label_path,
)
from src.bci.datasets.dataset2b import (
    build_dataset_2b_fbcsp_features,
    concatenate_trial_signals,
    extract_dataset_2b_trials,
)
from src.bci.datasets.dataset2b.evaluation_labels import (
    load_dataset_2b_evaluation_labels,
    resolve_dataset_2b_evaluation_label_path,
)


def _evaluate_models(
    *,
    dataset: str,
    subject: str,
    evaluation_scope: str,
    training_features: np.ndarray,
    training_labels: np.ndarray,
    testing_features: np.ndarray,
    testing_labels: np.ndarray,
    calibration_trials: int,
    random_state: int,
    n_estimators: int,
    sample_fraction: float,
) -> list[dict[str, object]]:
    """Fit single and bagged classifiers on identical feature matrices."""

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
            "subject": subject,
            "evaluation_scope": evaluation_scope,
            "method": "single_linear_svm",
            "training_trials": int(training_features.shape[0]),
            "calibration_trials": int(calibration_trials),
            "testing_trials": int(testing_features.shape[0]),
            "correct_predictions": int(np.count_nonzero(single_predictions == testing_labels)),
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
            "subject": subject,
            "evaluation_scope": evaluation_scope,
            "method": "bagged_linear_svm",
            "training_trials": int(training_features.shape[0]),
            "calibration_trials": int(calibration_trials),
            "testing_trials": int(testing_features.shape[0]),
            "correct_predictions": int(np.count_nonzero(ensemble_predictions == testing_labels)),
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


def run_dataset_2a_bagging(
    *,
    data_directory: str | Path,
    labels_directory: str | Path,
    subject: int,
    random_state: int = 42,
    n_estimators: int = 30,
    sample_fraction: float = 0.8,
) -> list[dict[str, object]]:
    """Evaluate single versus bagged SVM on one real Dataset 2A subject."""

    training_session = load_bci_competition_iv_2a_session(
        data_directory, subject, "T"
    )
    evaluation_session = load_bci_competition_iv_2a_session(
        data_directory, subject, "E"
    )
    label_path = resolve_dataset_2a_evaluation_label_path(
        subject,
        data_directory=Path(data_directory),
        labels_directory=Path(labels_directory),
    )
    official_labels = load_dataset_2a_evaluation_labels(label_path)

    training = extract_dataset_2a_trials(training_session)
    testing = extract_dataset_2a_trials(
        evaluation_session,
        evaluation_labels=official_labels,
    )
    pipeline = build_dataset_2a_fbcsp_features(training, testing)

    if len(testing.labels) != 144:
        raise ValueError(
            f"A{subject:02d}: expected 144 left/right Session-II evaluation trials; "
            f"received {len(testing.labels)}."
        )

    return _evaluate_models(
        dataset="2A",
        subject=f"A{subject:02d}",
        evaluation_scope="Session II",
        training_features=pipeline.training.features,
        training_labels=training.labels,
        testing_features=pipeline.testing.features,
        testing_labels=testing.labels,
        calibration_trials=0,
        random_state=random_state,
        n_estimators=n_estimators,
        sample_fraction=sample_fraction,
    )


def _label_dataset_2b_evaluation_trials(
    *,
    data_directory: str | Path,
    labels_directory: str | Path | None,
    subject: int,
    session: int,
):
    session_data = load_bci_competition_iv_2b_session(
        data_directory, subject, session
    )
    trials = extract_dataset_2b_trials(session_data)
    label_path = resolve_dataset_2b_evaluation_label_path(
        subject,
        session,
        data_directory=data_directory,
        labels_directory=labels_directory,
    )
    labels = load_dataset_2b_evaluation_labels(label_path)
    if labels.size != len(trials.labels):
        raise ValueError(
            f"B{subject:02d} session {session}: extracted {len(trials.labels)} "
            f"trials but loaded {labels.size} official labels."
        )
    return replace(trials, labels=np.asarray(labels, dtype=int))


def run_dataset_2b_bagging(
    *,
    data_directory: str | Path,
    labels_directory: str | Path | None = None,
    subject: int,
    random_state: int = 42,
    n_estimators: int = 30,
    sample_fraction: float = 0.8,
) -> list[dict[str, object]]:
    """Evaluate single versus bagged SVM on one real Dataset 2B subject.

    With no labels_directory, the experiment is deliberately GDF-only:
    Sessions I-II train the feature extractor/classifiers and labelled Session
    III is the held-out evaluation set. This is a real-data bagging benchmark,
    not a reproduction of the paper's Session IV-V classification table.

    Supplying labels_directory opts into the paper-style evaluation split:
    Sessions I-II train, Session III is retained as calibration, and Sessions
    IV-V are evaluated using the separately released official labels.
    """

    sessions = {
        index: extract_dataset_2b_trials(
            load_bci_competition_iv_2b_session(data_directory, subject, index)
        )
        for index in (1, 2, 3)
    }
    training = concatenate_trial_signals([sessions[1], sessions[2]])

    if labels_directory is None:
        testing = sessions[3]
        if np.any(testing.labels < 0):
            raise ValueError(
                f"B{subject:02d}: Session III must contain labelled 769/770 "
                "motor-imagery cues for the GDF-only bagging benchmark."
            )
        pipeline = build_dataset_2b_fbcsp_features(training, testing)
        return _evaluate_models(
            dataset="2B",
            subject=f"B{subject:02d}",
            evaluation_scope="Session III (GDF-only holdout)",
            training_features=pipeline.training.features,
            training_labels=training.labels,
            testing_features=pipeline.testing.features,
            testing_labels=testing.labels,
            calibration_trials=0,
            random_state=random_state,
            n_estimators=n_estimators,
            sample_fraction=sample_fraction,
        )

    evaluation_iv = _label_dataset_2b_evaluation_trials(
        data_directory=data_directory,
        labels_directory=labels_directory,
        subject=subject,
        session=4,
    )
    evaluation_v = _label_dataset_2b_evaluation_trials(
        data_directory=data_directory,
        labels_directory=labels_directory,
        subject=subject,
        session=5,
    )
    testing = concatenate_trial_signals([evaluation_iv, evaluation_v])
    pipeline = build_dataset_2b_fbcsp_features(training, testing)

    if len(testing.labels) != 320:
        raise ValueError(
            f"B{subject:02d}: expected 320 Session-IV/V evaluation trials; "
            f"received {len(testing.labels)}."
        )

    return _evaluate_models(
        dataset="2B",
        subject=f"B{subject:02d}",
        evaluation_scope="Sessions IV-V (official labels)",
        training_features=pipeline.training.features,
        training_labels=training.labels,
        testing_features=pipeline.testing.features,
        testing_labels=testing.labels,
        calibration_trials=len(sessions[3].labels),
        random_state=random_state,
        n_estimators=n_estimators,
        sample_fraction=sample_fraction,
    )


def run_bagging_bci_experiment(
    *,
    data_2a: str | Path,
    labels_2a: str | Path,
    data_2b: str | Path,
    labels_2b: str | Path | None = None,
    datasets: tuple[str, ...] = ("2a", "2b"),
    subjects: tuple[int, ...] = tuple(range(1, 10)),
    random_state: int = 42,
    n_estimators: int = 30,
    sample_fraction: float = 0.8,
) -> pd.DataFrame:
    """Run the real-data bagging evaluation for requested datasets/subjects."""

    records: list[dict[str, object]] = []
    for subject in subjects:
        if "2a" in datasets:
            records.extend(
                run_dataset_2a_bagging(
                    data_directory=data_2a,
                    labels_directory=labels_2a,
                    subject=subject,
                    random_state=random_state,
                    n_estimators=n_estimators,
                    sample_fraction=sample_fraction,
                )
            )
        if "2b" in datasets:
            records.extend(
                run_dataset_2b_bagging(
                    data_directory=data_2b,
                    labels_directory=labels_2b,
                    subject=subject,
                    random_state=random_state,
                    n_estimators=n_estimators,
                    sample_fraction=sample_fraction,
                )
            )

    return pd.DataFrame.from_records(records)
