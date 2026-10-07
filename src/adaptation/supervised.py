"""Supervised synchronous adaptation for CSD-triggered BCI experiments.

This preserves the Chowdhury et al. (2018) labelled-adaptation lineage. Later
PWKNN/transductive functionality has dedicated canonical modules, but is
re-exported here so historical imports continue to work.
"""
from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
import numpy as np
import pandas as pd
from src.adaptation._events import _python_scalar, _validation_by_time, _warning_times
from src.adaptation.classifier import LinearSVMClassifier, RetrainableClassifier
from src.adaptation.policies import AdaptationContext, AdaptationPolicy, RetrainOnValidatedShift

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
                "trials_since_update": context.trials_since_update,
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
                "trials_since_update": context.trials_since_update,
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


from src.adaptation.evaluation import (  # noqa: E402
    TransductiveEvaluationResult,
    UnsupervisedEvaluationResult,
    evaluate_transductive_adaptation,
    evaluate_unsupervised_adaptation,
)
from src.adaptation.pseudo_labelling import PseudoLabelBatch, PseudoLabeler, PWKNNPseudoLabeler  # noqa: E402
from src.adaptation.transductive import (  # noqa: E402
    TransductiveAdaptationConfig,
    TransductiveAdaptationResult,
    UnsupervisedAdaptationConfig,
    UnsupervisedAdaptationResult,
    run_transductive_adaptation,
    run_unsupervised_adaptation,
)

__all__ = [
    "SupervisedAdaptationConfig", "SupervisedAdaptationResult", "run_supervised_adaptation",
    "PseudoLabelBatch", "PseudoLabeler", "PWKNNPseudoLabeler",
    "TransductiveAdaptationConfig", "TransductiveAdaptationResult",
    "UnsupervisedAdaptationConfig", "UnsupervisedAdaptationResult",
    "run_transductive_adaptation", "run_unsupervised_adaptation",
    "TransductiveEvaluationResult", "UnsupervisedEvaluationResult",
    "evaluate_transductive_adaptation", "evaluate_unsupervised_adaptation",
]
