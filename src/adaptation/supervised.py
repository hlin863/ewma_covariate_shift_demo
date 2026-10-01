"""Supervised and first-step transductive adaptation loops for CSD-triggered BCI experiments.

The supervised path reconstructs the computational adaptation pattern used by
the 2018 EEG-CSAC study: classify incoming trials sequentially, decide whether
an update is required, append labelled observations, and retrain the classifier
for subsequent trials.

The unsupervised path is the first extension toward the 2019 CSE-UAEL setting.
It still assumes a labelled calibration/training set, but evaluation labels are
not exposed to the learner.  Candidate evaluation trials are pseudo-labelled
with a probabilistic weighted K-nearest-neighbour (PWKNN) rule and are admitted
to the training knowledge base only when their confidence exceeds a threshold.

Both paths are deliberately detector-agnostic. They consume Stage-I and
Stage-II outputs produced elsewhere in the repository rather than embedding
EWMA or Hotelling logic here.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import numpy as np
import pandas as pd

from src.adaptation.classifier import LinearSVMClassifier, RetrainableClassifier
from src.adaptation.policies import (
    AdaptationContext,
    AdaptationPolicy,
    RetrainOnValidatedShift,
)


@dataclass(frozen=True)
class SupervisedAdaptationConfig:
    """Controls which labelled evaluation trials are added at an update."""

    update_scope: str = "since_last_update"
    performance_window: int = 20

    def __post_init__(self) -> None:
        if self.performance_window < 1:
            raise ValueError("performance_window must be positive.")
        if self.update_scope not in {"current_trial", "since_last_update"}:
            raise ValueError(
                "update_scope must be 'current_trial' or 'since_last_update'."
            )


@dataclass(frozen=True)
class UnsupervisedAdaptationConfig:
    """Configuration for transductive pseudo-labelled adaptation.

    ``all_seen`` is the closest option here to the CSE-UAEL pseudocode, where
    all evaluation observations seen up to a confirmed shift are considered for
    transductive labelling.  Trials already admitted to the knowledge base are
    never appended twice.

    The PWKNN defaults are engineering defaults for experimentation, not
    published subject-specific CSE-UAEL parameter values.  Raza et al. (2019)
    selected K and the confidence threshold on validation data.
    """

    update_scope: str = "all_seen"
    n_neighbors: int = 5
    rbf_sigma: float = 1.0
    confidence_threshold: float = 0.80
    min_accepted_samples: int = 1

    def __post_init__(self) -> None:
        if self.update_scope not in {
            "current_trial",
            "since_last_update",
            "all_seen",
        }:
            raise ValueError(
                "update_scope must be 'current_trial', 'since_last_update', "
                "or 'all_seen'."
            )
        if self.n_neighbors < 1:
            raise ValueError("n_neighbors must be positive.")
        if not np.isfinite(self.rbf_sigma) or self.rbf_sigma <= 0.0:
            raise ValueError("rbf_sigma must be a finite positive value.")
        if (
            not np.isfinite(self.confidence_threshold)
            or not 0.5 <= self.confidence_threshold <= 1.0
        ):
            raise ValueError("confidence_threshold must be finite and in [0.5, 1.0].")
        if self.min_accepted_samples < 1:
            raise ValueError("min_accepted_samples must be positive.")


@dataclass(frozen=True)
class PseudoLabelBatch:
    """Pseudo-labels and confidence ratios for a batch of unlabelled features."""

    labels: np.ndarray
    confidence: np.ndarray


class PseudoLabeler(Protocol):
    """Contract for transductive or pseudo-labelling mechanisms."""

    def predict_with_confidence(
        self,
        *,
        reference_features: np.ndarray,
        reference_labels: np.ndarray,
        query_features: np.ndarray,
    ) -> PseudoLabelBatch:
        """Estimate labels and confidence without accessing evaluation truth."""
        ...


@dataclass(frozen=True)
class PWKNNPseudoLabeler:
    """RBF-weighted KNN pseudo-labeller inspired by Raza et al. (2019).

    For each query observation, the K nearest observations in the labelled
    knowledge base are selected. Their class votes are weighted with an RBF
    kernel:

        kappa(x_p, x_q) = exp(-||x_p - x_q||^2 / (2 * sigma^2))

    The pseudo-label is the class with the largest weighted posterior estimate,
    and confidence is that maximum normalized weight.

    The implementation supports any finite set of calibration labels, although
    the current BCI experiments are binary.
    """

    n_neighbors: int = 5
    rbf_sigma: float = 1.0

    def __post_init__(self) -> None:
        if self.n_neighbors < 1:
            raise ValueError("n_neighbors must be positive.")
        if not np.isfinite(self.rbf_sigma) or self.rbf_sigma <= 0.0:
            raise ValueError("rbf_sigma must be a finite positive value.")

    def predict_with_confidence(
        self,
        *,
        reference_features: np.ndarray,
        reference_labels: np.ndarray,
        query_features: np.ndarray,
    ) -> PseudoLabelBatch:
        x_ref = np.asarray(reference_features, dtype=float)
        y_ref = np.asarray(reference_labels)
        x_query = np.asarray(query_features, dtype=float)

        if x_ref.ndim != 2 or x_query.ndim != 2:
            raise ValueError(
                "reference_features and query_features must be two-dimensional."
            )
        if y_ref.ndim != 1 or x_ref.shape[0] != y_ref.size:
            raise ValueError(
                "reference_labels must be one-dimensional and match "
                "reference_features."
            )
        if x_ref.shape[0] < 1:
            raise ValueError("reference_features must not be empty.")
        if x_ref.shape[1] != x_query.shape[1]:
            raise ValueError("Reference and query feature dimensions must match.")
        if not np.isfinite(x_ref).all() or not np.isfinite(x_query).all():
            raise ValueError("PWKNN features must contain only finite values.")

        classes = np.unique(y_ref)
        if classes.size < 2:
            raise ValueError("PWKNN requires at least two reference classes.")

        if x_query.shape[0] == 0:
            return PseudoLabelBatch(
                labels=np.asarray([], dtype=y_ref.dtype),
                confidence=np.asarray([], dtype=float),
            )

        k = min(self.n_neighbors, x_ref.shape[0])
        estimated_labels: list[object] = []
        confidence_ratios: list[float] = []

        for query in x_query:
            squared_distance = np.sum((x_ref - query) ** 2, axis=1)
            nearest = np.argpartition(squared_distance, k - 1)[:k]
            nearest_distance = squared_distance[nearest]
            nearest_labels = y_ref[nearest]

            # Shift by the minimum before exponentiation.  This multiplies all
            # RBF weights by the same constant, preserving normalized class
            # probabilities while reducing underflow for distant observations.
            stabilized_distance = nearest_distance - nearest_distance.min()
            weights = np.exp(-stabilized_distance / (2.0 * self.rbf_sigma**2))
            total_weight = float(np.sum(weights))
            if not np.isfinite(total_weight) or total_weight <= 0.0:
                raise RuntimeError("PWKNN produced an invalid weight sum.")

            class_scores = np.asarray(
                [
                    np.sum(weights[nearest_labels == label]) / total_weight
                    for label in classes
                ],
                dtype=float,
            )
            best = int(np.argmax(class_scores))
            estimated_labels.append(classes[best])
            confidence_ratios.append(float(class_scores[best]))

        return PseudoLabelBatch(
            labels=np.asarray(estimated_labels, dtype=y_ref.dtype),
            confidence=np.asarray(confidence_ratios, dtype=float),
        )


@dataclass(frozen=True)
class UnsupervisedAdaptationResult:
    """Outputs of a transductive run without evaluation ground-truth labels."""

    trial_results: pd.DataFrame
    pseudo_label_events: pd.DataFrame
    update_events: pd.DataFrame
    final_classifier: RetrainableClassifier
    initial_fit_seconds: float = 0.0

    @property
    def update_count(self) -> int:
        return int(self.update_events.shape[0])

    @property
    def pseudo_label_attempt_count(self) -> int:
        return int(self.pseudo_label_events.shape[0])

    @property
    def accepted_pseudo_label_count(self) -> int:
        if self.pseudo_label_events.empty:
            return 0
        return int(self.pseudo_label_events["accepted"].astype(bool).sum())

    @property
    def pseudo_label_acceptance_rate(self) -> float:
        if self.pseudo_label_events.empty:
            return float("nan")
        return float(self.pseudo_label_events["accepted"].astype(bool).mean())


@dataclass(frozen=True)
class UnsupervisedEvaluationResult:
    """Offline scoring kept separate from the label-free learner interface."""

    trial_results: pd.DataFrame
    pseudo_label_events: pd.DataFrame
    accuracy: float
    pseudo_label_accuracy: float
    accepted_pseudo_label_accuracy: float


@dataclass(frozen=True)
class SupervisedAdaptationResult:
    """Outputs of a sequential adaptive-classification run."""

    trial_results: pd.DataFrame
    update_events: pd.DataFrame
    final_classifier: RetrainableClassifier
    initial_fit_seconds: float = 0.0

    @property
    def accuracy(self) -> float:
        if self.trial_results.empty:
            return float("nan")
        return float(self.trial_results["correct"].mean())

    @property
    def update_count(self) -> int:
        return int(self.update_events.shape[0])


def _validate_inputs(
    calibration_features: np.ndarray,
    calibration_labels: np.ndarray,
    evaluation_features: np.ndarray,
    evaluation_labels: np.ndarray,
    evaluation_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_cal = np.asarray(calibration_features, dtype=float)
    y_cal = np.asarray(calibration_labels)
    x_eval = np.asarray(evaluation_features, dtype=float)
    y_eval = np.asarray(evaluation_labels)
    times = np.asarray(evaluation_times)

    if x_cal.ndim != 2 or x_eval.ndim != 2:
        raise ValueError("calibration_features and evaluation_features must be 2-D.")
    if y_cal.ndim != 1 or y_eval.ndim != 1 or times.ndim != 1:
        raise ValueError("labels and evaluation_times must be one-dimensional.")
    if x_cal.shape[0] != y_cal.size:
        raise ValueError("Calibration features and labels must have equal length.")
    if x_eval.shape[0] != y_eval.size or x_eval.shape[0] != times.size:
        raise ValueError(
            "Evaluation features, labels, and times must contain equal numbers of trials."
        )
    if x_cal.shape[1] != x_eval.shape[1]:
        raise ValueError("Calibration and evaluation feature dimensions must match.")
    if not np.isfinite(x_cal).all() or not np.isfinite(x_eval).all():
        raise ValueError("Feature matrices must contain only finite values.")
    if np.unique(times).size != times.size:
        raise ValueError("evaluation_times must uniquely identify trials.")
    return x_cal, y_cal, x_eval, y_eval, times


def _validate_unsupervised_inputs(
    calibration_features: np.ndarray,
    calibration_labels: np.ndarray,
    evaluation_features: np.ndarray,
    evaluation_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate the labelled-calibration / unlabelled-evaluation regime."""

    x_cal = np.asarray(calibration_features, dtype=float)
    y_cal = np.asarray(calibration_labels)
    x_eval = np.asarray(evaluation_features, dtype=float)
    times = np.asarray(evaluation_times)

    if x_cal.ndim != 2 or x_eval.ndim != 2:
        raise ValueError("calibration_features and evaluation_features must be 2-D.")
    if y_cal.ndim != 1 or times.ndim != 1:
        raise ValueError(
            "calibration_labels and evaluation_times must be one-dimensional."
        )
    if x_cal.shape[0] != y_cal.size:
        raise ValueError("Calibration features and labels must have equal length.")
    if x_eval.shape[0] != times.size:
        raise ValueError(
            "Evaluation features and evaluation_times must contain equal "
            "numbers of trials."
        )
    if x_cal.shape[1] != x_eval.shape[1]:
        raise ValueError("Calibration and evaluation feature dimensions must match.")
    if x_cal.shape[0] < 2:
        raise ValueError("Calibration data must contain at least two observations.")
    if np.unique(y_cal).size < 2:
        raise ValueError("Calibration labels must contain at least two classes.")
    if not np.isfinite(x_cal).all() or not np.isfinite(x_eval).all():
        raise ValueError("Feature matrices must contain only finite values.")
    if np.unique(times).size != times.size:
        raise ValueError("evaluation_times must uniquely identify trials.")

    return x_cal, y_cal, x_eval, times


def _validation_by_time(
    validation_results: pd.DataFrame,
) -> dict[object, dict[str, object]]:
    if validation_results.empty:
        return {}

    if "validation_time" in validation_results.columns:
        time_column = "validation_time"
    elif "alarm_time" in validation_results.columns:
        time_column = "alarm_time"
    else:
        raise ValueError(
            "validation_results must contain 'validation_time' or 'alarm_time'."
        )

    records: dict[object, dict[str, object]] = {}
    for row in validation_results.to_dict(orient="records"):
        value = row.get(time_column)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        records[value] = row
    return records


def _warning_times(warning_results: pd.DataFrame | None) -> set[object]:
    if warning_results is None or warning_results.empty:
        return set()
    required = {"time", "stage_1_alarm"}
    if not required.issubset(warning_results.columns):
        raise ValueError(
            "warning_results must contain 'time' and 'stage_1_alarm' columns."
        )
    alarms = warning_results.loc[warning_results["stage_1_alarm"].astype(bool), "time"]
    return set(alarms.tolist())


def _python_scalar(value: object) -> object:
    return value.item() if hasattr(value, "item") else value


def _candidate_indices(
    *,
    index: int,
    update_scope: str,
    last_update_end: int,
    already_added: set[int],
) -> np.ndarray:
    stop = index + 1

    if update_scope == "current_trial":
        raw = np.asarray([index], dtype=int)
    elif update_scope == "since_last_update":
        raw = np.arange(last_update_end, stop, dtype=int)
    elif update_scope == "all_seen":
        raw = np.arange(0, stop, dtype=int)
    else:
        raise ValueError(f"Unsupported update_scope: {update_scope!r}")

    if not already_added:
        return raw

    return np.asarray(
        [trial_index for trial_index in raw if trial_index not in already_added],
        dtype=int,
    )


def run_supervised_adaptation(
    *,
    calibration_features: np.ndarray,
    calibration_labels: np.ndarray,
    evaluation_features: np.ndarray,
    evaluation_labels: np.ndarray,
    evaluation_times: np.ndarray,
    validation_results: pd.DataFrame,
    warning_results: pd.DataFrame | None = None,
    classifier: RetrainableClassifier | None = None,
    policy: AdaptationPolicy | None = None,
    config: SupervisedAdaptationConfig | None = None,
) -> SupervisedAdaptationResult:
    """Run sequential classification and supervised append-and-retrain updates.

    Prediction for trial ``t`` is made before any adaptation triggered at that
    same time. Therefore, a validated shift changes the model used for future
    trials rather than retroactively changing the current prediction.

    ``evaluation_labels`` are intentionally required because this function
    models the supervised synchronous setting of the 2018 online BCI study.
    Unsupervised or pseudo-labelled adaptation is implemented as a separate
    pathway below rather than silently reusing true evaluation labels.
    """

    x_cal, y_cal, x_eval, y_eval, times = _validate_inputs(
        calibration_features,
        calibration_labels,
        evaluation_features,
        evaluation_labels,
        evaluation_times,
    )

    model = classifier or LinearSVMClassifier()
    started = perf_counter()
    model.fit(x_cal, y_cal)
    initial_fit_seconds = perf_counter() - started
    trigger_policy = policy or RetrainOnValidatedShift()
    adaptation_config = config or SupervisedAdaptationConfig()

    validations = _validation_by_time(validation_results)
    warning_time_set = _warning_times(warning_results)

    trial_records: list[dict[str, object]] = []
    update_records: list[dict[str, object]] = []
    last_update_end = 0
    classifier_version = 0
    correctness: list[bool] = []

    for index, (features, label, time_value) in enumerate(zip(x_eval, y_eval, times)):
        started = perf_counter()
        prediction = model.predict(features.reshape(1, -1))[0]
        prediction_seconds = perf_counter() - started
        correctness.append(bool(prediction == label))
        window = adaptation_config.performance_window
        recent_accuracy = (
            float(np.mean(correctness[-window:]))
            if len(correctness) >= window
            else None
        )
        validation_record = validations.get(time_value)
        stage1_warning = time_value in warning_time_set

        context = AdaptationContext(
            trial_index=index,
            time=_python_scalar(time_value),
            stage1_warning=stage1_warning,
            validation_record=validation_record,
            recent_accuracy=recent_accuracy,
            trials_since_update=index + 1 - last_update_end,
        )
        should_update = bool(trigger_policy.should_update(context))

        trial_records.append(
            {
                "trial_index": index,
                "time": context.time,
                "true_label": _python_scalar(label),
                "predicted_label": _python_scalar(prediction),
                "correct": bool(prediction == label),
                "rolling_accuracy": recent_accuracy,
                "prediction_seconds": prediction_seconds,
                "stage1_warning": stage1_warning,
                "stage2_status": (
                    validation_record.get("status")
                    if validation_record is not None
                    else None
                ),
                "stage2_p_value": (
                    validation_record.get("p_value")
                    if validation_record is not None
                    else None
                ),
                "validated_shift": bool(
                    validation_record.get("confirmed_shift", False)
                    if validation_record is not None
                    else False
                ),
                "classifier_version": classifier_version,
                "training_size_before": model.training_size,
                "retrained_after_trial": should_update,
            }
        )

        if not should_update:
            continue

        if adaptation_config.update_scope == "current_trial":
            start = index
        else:
            start = last_update_end
        stop = index + 1

        x_new = x_eval[start:stop]
        y_new = y_eval[start:stop]
        size_before = model.training_size
        started = perf_counter()
        model.append_and_retrain(x_new, y_new)
        retrain_seconds = perf_counter() - started
        classifier_version += 1
        last_update_end = stop

        update_records.append(
            {
                "update_index": len(update_records),
                "retrain_seconds": retrain_seconds,
                "recent_accuracy": recent_accuracy,
                "trigger_trial_index": index,
                "trigger_time": context.time,
                "stage1_warning": stage1_warning,
                "stage2_status": (
                    validation_record.get("status")
                    if validation_record is not None
                    else None
                ),
                "validated_shift": bool(
                    validation_record.get("confirmed_shift", False)
                    if validation_record is not None
                    else False
                ),
                "added_start_trial": start,
                "added_end_trial": stop - 1,
                "added_samples": int(stop - start),
                "training_size_before": size_before,
                "training_size_after": model.training_size,
                "classifier_version": classifier_version,
            }
        )

    return SupervisedAdaptationResult(
        trial_results=pd.DataFrame(trial_records),
        update_events=pd.DataFrame(update_records),
        final_classifier=model,
        initial_fit_seconds=initial_fit_seconds,
    )


def run_unsupervised_adaptation(
    *,
    calibration_features: np.ndarray,
    calibration_labels: np.ndarray,
    evaluation_features: np.ndarray,
    evaluation_times: np.ndarray,
    validation_results: pd.DataFrame,
    warning_results: pd.DataFrame | None = None,
    classifier: RetrainableClassifier | None = None,
    policy: AdaptationPolicy | None = None,
    config: UnsupervisedAdaptationConfig | None = None,
    pseudo_labeler: PseudoLabeler | None = None,
) -> UnsupervisedAdaptationResult:
    """Run labelled-calibration / unlabelled-evaluation adaptation.

    This is the first step toward the CSE-UAEL adaptation regime.  The learner
    receives true labels for the calibration set only.  Evaluation labels are
    deliberately absent from the API and therefore cannot leak into online
    adaptation.

    Per trial:
      1. predict with the current inductive classifier;
      2. construct an adaptation context from Stage-I/Stage-II evidence;
      3. if the policy triggers, select only evaluation observations that have
         already arrived;
      4. pseudo-label those candidates from the current labelled knowledge base;
      5. accept candidates whose confidence is strictly greater than the
         configured threshold;
      6. append accepted pseudo-labelled observations and retrain;
      7. use the new classifier only for subsequent trials.

    This function intentionally does *not* implement the full 2019 CSE-UAEL
    ensemble-growth and dynamic weighted-voting stages.  It implements the
    transductive knowledge-base update needed before those stages can be added.

    Ground-truth evaluation labels, if available for offline scoring, should be
    supplied to a separate evaluation function rather than to this learner.
    """

    x_cal, y_cal, x_eval, times = _validate_unsupervised_inputs(
        calibration_features,
        calibration_labels,
        evaluation_features,
        evaluation_times,
    )

    adaptation_config = config or UnsupervisedAdaptationConfig()
    transductive_labeler = pseudo_labeler or PWKNNPseudoLabeler(
        n_neighbors=adaptation_config.n_neighbors,
        rbf_sigma=adaptation_config.rbf_sigma,
    )
    trigger_policy = policy or RetrainOnValidatedShift()

    model = classifier or LinearSVMClassifier()
    started = perf_counter()
    model.fit(x_cal, y_cal)
    initial_fit_seconds = perf_counter() - started

    # Keep the pseudo-labelling knowledge base independent of classifier
    # internals.  This lets any RetrainableClassifier satisfy the contract.
    knowledge_features = x_cal.copy()
    knowledge_labels = y_cal.copy()

    validations = _validation_by_time(validation_results)
    warning_time_set = _warning_times(warning_results)

    trial_records: list[dict[str, object]] = []
    pseudo_label_records: list[dict[str, object]] = []
    update_records: list[dict[str, object]] = []

    classifier_version = 0
    last_update_end = 0
    accepted_trial_indices: set[int] = set()
    adaptation_attempt_index = 0

    for index, (features, time_value) in enumerate(zip(x_eval, times)):
        # Prediction always happens before any update triggered at this time.
        started = perf_counter()
        prediction = model.predict(features.reshape(1, -1))[0]
        prediction_seconds = perf_counter() - started

        validation_record = validations.get(time_value)
        stage1_warning = time_value in warning_time_set
        context = AdaptationContext(
            trial_index=index,
            time=_python_scalar(time_value),
            stage1_warning=stage1_warning,
            validation_record=validation_record,
            recent_accuracy=None,
            trials_since_update=index + 1 - last_update_end,
        )
        adaptation_triggered = bool(trigger_policy.should_update(context))

        training_size_before = model.training_size
        retrained_after_trial = False
        candidate_count = 0
        accepted_count = 0

        if adaptation_triggered:
            candidate_index = _candidate_indices(
                index=index,
                update_scope=adaptation_config.update_scope,
                last_update_end=last_update_end,
                already_added=accepted_trial_indices,
            )
            candidate_count = int(candidate_index.size)

            if candidate_count > 0:
                candidate_features = x_eval[candidate_index]
                estimates = transductive_labeler.predict_with_confidence(
                    reference_features=knowledge_features,
                    reference_labels=knowledge_labels,
                    query_features=candidate_features,
                )

                if (
                    estimates.labels.ndim != 1
                    or estimates.confidence.ndim != 1
                    or estimates.labels.size != candidate_count
                    or estimates.confidence.size != candidate_count
                ):
                    raise ValueError(
                        "pseudo_labeler must return one label and confidence "
                        "value per query observation."
                    )
                if (
                    not np.isfinite(estimates.confidence).all()
                    or np.any(estimates.confidence < 0.0)
                    or np.any(estimates.confidence > 1.0)
                ):
                    raise ValueError(
                        "Pseudo-label confidence values must be finite and in [0, 1]."
                    )

                accepted_mask = (
                    estimates.confidence > adaptation_config.confidence_threshold
                )
                accepted_count = int(np.sum(accepted_mask))

                for local_index, trial_index in enumerate(candidate_index):
                    pseudo_label_records.append(
                        {
                            "attempt_index": adaptation_attempt_index,
                            "trigger_trial_index": index,
                            "trigger_time": context.time,
                            "candidate_trial_index": int(trial_index),
                            "candidate_time": _python_scalar(times[trial_index]),
                            "pseudo_label": _python_scalar(
                                estimates.labels[local_index]
                            ),
                            "confidence": float(estimates.confidence[local_index]),
                            "confidence_threshold": (
                                adaptation_config.confidence_threshold
                            ),
                            "accepted": bool(accepted_mask[local_index]),
                            "classifier_version_before": classifier_version,
                        }
                    )

                if accepted_count >= adaptation_config.min_accepted_samples:
                    accepted_index = candidate_index[accepted_mask]
                    accepted_features = candidate_features[accepted_mask]
                    accepted_labels = estimates.labels[accepted_mask]

                    size_before = model.training_size
                    started = perf_counter()
                    model.append_and_retrain(
                        accepted_features,
                        accepted_labels,
                    )
                    retrain_seconds = perf_counter() - started

                    # The same accepted pseudo-labelled observations become
                    # reference knowledge for later PWKNN decisions.
                    knowledge_features = np.vstack(
                        [knowledge_features, accepted_features]
                    )
                    knowledge_labels = np.concatenate(
                        [knowledge_labels, accepted_labels]
                    )
                    accepted_trial_indices.update(
                        int(value) for value in accepted_index.tolist()
                    )

                    classifier_version += 1
                    last_update_end = index + 1
                    retrained_after_trial = True

                    update_records.append(
                        {
                            "update_index": len(update_records),
                            "attempt_index": adaptation_attempt_index,
                            "trigger_trial_index": index,
                            "trigger_time": context.time,
                            "stage1_warning": stage1_warning,
                            "stage2_status": (
                                validation_record.get("status")
                                if validation_record is not None
                                else None
                            ),
                            "stage2_p_value": (
                                validation_record.get("p_value")
                                if validation_record is not None
                                else None
                            ),
                            "validated_shift": bool(
                                validation_record.get("confirmed_shift", False)
                                if validation_record is not None
                                else False
                            ),
                            "candidate_samples": candidate_count,
                            "accepted_samples": accepted_count,
                            "rejected_samples": (candidate_count - accepted_count),
                            "confidence_threshold": (
                                adaptation_config.confidence_threshold
                            ),
                            "mean_accepted_confidence": float(
                                np.mean(estimates.confidence[accepted_mask])
                            ),
                            "training_size_before": size_before,
                            "training_size_after": model.training_size,
                            "knowledge_base_size_after": int(
                                knowledge_features.shape[0]
                            ),
                            "retrain_seconds": retrain_seconds,
                            "classifier_version": classifier_version,
                        }
                    )

            adaptation_attempt_index += 1

        trial_records.append(
            {
                "trial_index": index,
                "time": context.time,
                "predicted_label": _python_scalar(prediction),
                "prediction_seconds": prediction_seconds,
                "stage1_warning": stage1_warning,
                "stage2_status": (
                    validation_record.get("status")
                    if validation_record is not None
                    else None
                ),
                "stage2_p_value": (
                    validation_record.get("p_value")
                    if validation_record is not None
                    else None
                ),
                "validated_shift": bool(
                    validation_record.get("confirmed_shift", False)
                    if validation_record is not None
                    else False
                ),
                "classifier_version": (
                    classifier_version - 1
                    if retrained_after_trial
                    else classifier_version
                ),
                "training_size_before": training_size_before,
                "adaptation_triggered": adaptation_triggered,
                "pseudo_label_candidates": candidate_count,
                "pseudo_labels_accepted": accepted_count,
                "retrained_after_trial": retrained_after_trial,
            }
        )

    return UnsupervisedAdaptationResult(
        trial_results=pd.DataFrame(trial_records),
        pseudo_label_events=pd.DataFrame(pseudo_label_records),
        update_events=pd.DataFrame(update_records),
        final_classifier=model,
        initial_fit_seconds=initial_fit_seconds,
    )


def evaluate_unsupervised_adaptation(
    result: UnsupervisedAdaptationResult,
    evaluation_labels: np.ndarray,
) -> UnsupervisedEvaluationResult:
    """Score an unsupervised run *after* adaptation has finished.

    Ground-truth evaluation labels are deliberately accepted here rather than by
    ``run_unsupervised_adaptation``.  This keeps experimental scoring separate
    from the information available to the online learner and makes accidental
    evaluation-label leakage much harder.

    The returned pseudo-label accuracy is event based.  If a rejected trial is
    reconsidered at a later confirmed shift, each pseudo-labelling decision is
    scored separately because the knowledge base may have changed.
    """

    y_eval = np.asarray(evaluation_labels)
    if y_eval.ndim != 1:
        raise ValueError("evaluation_labels must be one-dimensional.")
    if y_eval.size != result.trial_results.shape[0]:
        raise ValueError(
            "evaluation_labels must contain one label per evaluation trial."
        )

    trial_results = result.trial_results.copy()
    trial_indices = trial_results["trial_index"].to_numpy(dtype=int)
    if (
        np.any(trial_indices < 0)
        or np.any(trial_indices >= y_eval.size)
        or np.unique(trial_indices).size != trial_indices.size
    ):
        raise ValueError("trial_results contains invalid trial indices.")

    true_labels = y_eval[trial_indices]
    predicted = trial_results["predicted_label"].to_numpy()
    trial_results["true_label"] = true_labels
    trial_results["correct"] = predicted == true_labels
    accuracy = (
        float(trial_results["correct"].mean())
        if not trial_results.empty
        else float("nan")
    )

    pseudo_events = result.pseudo_label_events.copy()
    if pseudo_events.empty:
        pseudo_label_accuracy = float("nan")
        accepted_pseudo_label_accuracy = float("nan")
    else:
        candidate_indices = pseudo_events["candidate_trial_index"].to_numpy(dtype=int)
        if np.any(candidate_indices < 0) or np.any(candidate_indices >= y_eval.size):
            raise ValueError(
                "pseudo_label_events contains invalid candidate trial indices."
            )

        pseudo_truth = y_eval[candidate_indices]
        pseudo_predicted = pseudo_events["pseudo_label"].to_numpy()
        pseudo_events["true_label"] = pseudo_truth
        pseudo_events["pseudo_label_correct"] = pseudo_predicted == pseudo_truth
        pseudo_label_accuracy = float(pseudo_events["pseudo_label_correct"].mean())

        accepted = pseudo_events["accepted"].astype(bool)
        accepted_pseudo_label_accuracy = (
            float(pseudo_events.loc[accepted, "pseudo_label_correct"].mean())
            if accepted.any()
            else float("nan")
        )

    return UnsupervisedEvaluationResult(
        trial_results=trial_results,
        pseudo_label_events=pseudo_events,
        accuracy=accuracy,
        pseudo_label_accuracy=pseudo_label_accuracy,
        accepted_pseudo_label_accuracy=accepted_pseudo_label_accuracy,
    )
