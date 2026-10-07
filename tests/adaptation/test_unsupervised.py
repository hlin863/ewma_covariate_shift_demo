import inspect

import numpy as np
import pandas as pd
import pytest

from src.adaptation.policies import RetrainOnValidatedShift
from src.adaptation.evaluation import evaluate_unsupervised_adaptation
from src.adaptation.pseudo_labelling import PWKNNPseudoLabeler
from src.adaptation.transductive import (
    UnsupervisedAdaptationConfig,
    run_unsupervised_adaptation,
)


def _confirmed(at: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "alarm_time": at - 1,
                "validation_time": at,
                "confirmed_shift": True,
                "status": "confirmed",
                "p_value": 0.01,
            }
        ]
    )


def test_unsupervised_api_does_not_accept_evaluation_ground_truth():
    parameters = inspect.signature(run_unsupervised_adaptation).parameters
    assert "evaluation_labels" not in parameters


def test_pwknn_prefers_nearby_class_and_returns_confidence():
    labeler = PWKNNPseudoLabeler(n_neighbors=3, rbf_sigma=1.0)
    result = labeler.predict_with_confidence(
        reference_features=np.array([[-2.0], [-1.0], [1.0], [2.0]]),
        reference_labels=np.array([0, 0, 1, 1]),
        query_features=np.array([[-1.5], [1.5]]),
    )

    assert result.labels.tolist() == [0, 1]
    assert np.all(result.confidence > 0.5)
    assert np.all(result.confidence <= 1.0)


def test_validated_shift_pseudo_labels_seen_trials_and_retrains():
    result = run_unsupervised_adaptation(
        calibration_features=np.array([[-2.0], [-1.0], [1.0], [2.0]]),
        calibration_labels=np.array([0, 0, 1, 1]),
        evaluation_features=np.array([[-1.7], [1.7], [1.8]]),
        evaluation_times=np.arange(3),
        validation_results=_confirmed(1),
        policy=RetrainOnValidatedShift(),
        config=UnsupervisedAdaptationConfig(
            update_scope="all_seen",
            n_neighbors=3,
            rbf_sigma=1.0,
            confidence_threshold=0.60,
        ),
    )

    assert result.update_count == 1
    assert result.update_events["accepted_samples"].tolist() == [2]
    assert result.update_events["trials_since_update"].tolist() == [2]
    assert result.trial_results["trials_since_update"].tolist() == [1, 2, 1]
    assert result.final_classifier.training_size == 6
    assert result.pseudo_label_events["accepted"].all()
    assert "true_label" not in result.trial_results.columns
    assert "correct" not in result.trial_results.columns


def test_low_confidence_pseudo_label_is_rejected_without_retraining():
    result = run_unsupervised_adaptation(
        calibration_features=np.array([[-1.0], [1.0]]),
        calibration_labels=np.array([0, 1]),
        evaluation_features=np.array([[0.0]]),
        evaluation_times=np.array([0]),
        validation_results=_confirmed(0),
        config=UnsupervisedAdaptationConfig(
            update_scope="current_trial",
            n_neighbors=2,
            rbf_sigma=1.0,
            confidence_threshold=0.90,
        ),
    )

    assert result.update_count == 0
    assert result.final_classifier.training_size == 2
    assert result.pseudo_label_attempt_count == 1
    assert result.accepted_pseudo_label_count == 0
    assert not bool(result.pseudo_label_events["accepted"].iloc[0])


def test_prediction_occurs_before_unsupervised_update():
    class Spy:
        def __init__(self):
            self.training_size = 0
            self.version = 0

        def fit(self, x, y):
            self.training_size = len(x)
            return self

        def predict(self, x):
            return np.asarray([self.version])

        def append_and_retrain(self, x, y):
            self.version += 1
            self.training_size += len(x)
            return self

    result = run_unsupervised_adaptation(
        calibration_features=np.array([[-2.0], [-1.0], [1.0], [2.0]]),
        calibration_labels=np.array([0, 0, 1, 1]),
        evaluation_features=np.array([[-1.5], [1.5], [1.7]]),
        evaluation_times=np.arange(3),
        validation_results=_confirmed(1),
        classifier=Spy(),
        config=UnsupervisedAdaptationConfig(
            update_scope="current_trial",
            n_neighbors=3,
            confidence_threshold=0.55,
        ),
    )

    assert result.trial_results["predicted_label"].tolist() == [0, 0, 1]
    assert result.trial_results["classifier_version"].tolist() == [0, 0, 1]
    assert result.trial_results["retrained_after_trial"].tolist() == [
        False,
        True,
        False,
    ]


def test_all_seen_scope_does_not_append_accepted_trial_twice():
    validations = pd.DataFrame(
        [
            {
                "validation_time": 1,
                "confirmed_shift": True,
                "status": "confirmed",
                "p_value": 0.01,
            },
            {
                "validation_time": 2,
                "confirmed_shift": True,
                "status": "confirmed",
                "p_value": 0.01,
            },
        ]
    )

    result = run_unsupervised_adaptation(
        calibration_features=np.array([[-2.0], [-1.0], [1.0], [2.0]]),
        calibration_labels=np.array([0, 0, 1, 1]),
        evaluation_features=np.array([[-1.8], [1.8], [1.9]]),
        evaluation_times=np.arange(3),
        validation_results=validations,
        config=UnsupervisedAdaptationConfig(
            update_scope="all_seen",
            n_neighbors=3,
            confidence_threshold=0.55,
        ),
    )

    assert result.update_count == 2
    assert result.update_events["accepted_samples"].tolist() == [2, 1]
    assert result.final_classifier.training_size == 7


def test_ground_truth_scoring_is_separate_from_online_adaptation():
    result = run_unsupervised_adaptation(
        calibration_features=np.array([[-2.0], [-1.0], [1.0], [2.0]]),
        calibration_labels=np.array([0, 0, 1, 1]),
        evaluation_features=np.array([[-1.7], [1.7], [1.8]]),
        evaluation_times=np.arange(3),
        validation_results=_confirmed(1),
        config=UnsupervisedAdaptationConfig(
            update_scope="all_seen",
            n_neighbors=3,
            confidence_threshold=0.60,
        ),
    )

    scored = evaluate_unsupervised_adaptation(
        result,
        evaluation_labels=np.array([0, 1, 1]),
    )

    assert scored.accuracy == pytest.approx(1.0)
    assert scored.pseudo_label_accuracy == pytest.approx(1.0)
    assert scored.accepted_pseudo_label_accuracy == pytest.approx(1.0)
    assert "true_label" in scored.trial_results.columns
    assert "true_label" not in result.trial_results.columns


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_neighbors", 0),
        ("rbf_sigma", 0.0),
        ("confidence_threshold", 0.49),
        ("confidence_threshold", 1.01),
        ("min_accepted_samples", 0),
    ],
)
def test_unsupervised_config_rejects_invalid_values(field, value):
    with pytest.raises(ValueError):
        UnsupervisedAdaptationConfig(**{field: value})
