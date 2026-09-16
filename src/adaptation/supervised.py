"""Supervised online adaptation loop for CSD-triggered BCI experiments.

This module reconstructs the computational adaptation pattern used by the
2018 EEG-CSAC study: classify incoming trials sequentially, decide whether an
update is required, append labelled observations, and retrain the classifier
for subsequent trials.

The implementation is deliberately detector-agnostic.  It consumes Stage-I
and Stage-II outputs produced elsewhere in the repository rather than
embedding EWMA or Hotelling logic here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.adaptation.classifier import LinearSVMClassifier
from src.adaptation.policies import (
    AdaptationContext,
    AdaptationPolicy,
    RetrainOnValidatedShift,
)


@dataclass(frozen=True)
class SupervisedAdaptationConfig:
    """Controls which labelled evaluation trials are added at an update."""

    update_scope: str = "since_last_update"

    def __post_init__(self) -> None:
        if self.update_scope not in {"current_trial", "since_last_update"}:
            raise ValueError(
                "update_scope must be 'current_trial' or 'since_last_update'."
            )


@dataclass(frozen=True)
class SupervisedAdaptationResult:
    """Outputs of a sequential adaptive-classification run."""

    trial_results: pd.DataFrame
    update_events: pd.DataFrame
    final_classifier: LinearSVMClassifier

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


def _validation_by_time(validation_results: pd.DataFrame) -> dict[object, dict[str, object]]:
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
    alarms = warning_results.loc[
        warning_results["stage_1_alarm"].astype(bool), "time"
    ]
    return set(alarms.tolist())


def run_supervised_adaptation(
    *,
    calibration_features: np.ndarray,
    calibration_labels: np.ndarray,
    evaluation_features: np.ndarray,
    evaluation_labels: np.ndarray,
    evaluation_times: np.ndarray,
    validation_results: pd.DataFrame,
    warning_results: pd.DataFrame | None = None,
    classifier: LinearSVMClassifier | None = None,
    policy: AdaptationPolicy | None = None,
    config: SupervisedAdaptationConfig | None = None,
) -> SupervisedAdaptationResult:
    """Run sequential classification and supervised append-and-retrain updates.

    Prediction for trial ``t`` is made before any adaptation triggered at that
    same time.  Therefore, a validated shift changes the model used for future
    trials rather than retroactively changing the current prediction.

    ``evaluation_labels`` are intentionally required because this function
    models the supervised synchronous setting of the 2018 online BCI study.
    Unsupervised or pseudo-labelled adaptation should be implemented as a
    separate pathway rather than silently reusing true evaluation labels.
    """

    x_cal, y_cal, x_eval, y_eval, times = _validate_inputs(
        calibration_features,
        calibration_labels,
        evaluation_features,
        evaluation_labels,
        evaluation_times,
    )

    model = classifier or LinearSVMClassifier()
    model.fit(x_cal, y_cal)
    trigger_policy = policy or RetrainOnValidatedShift()
    adaptation_config = config or SupervisedAdaptationConfig()

    validations = _validation_by_time(validation_results)
    warning_time_set = _warning_times(warning_results)

    trial_records: list[dict[str, object]] = []
    update_records: list[dict[str, object]] = []
    last_update_end = 0
    classifier_version = 0

    for index, (features, label, time_value) in enumerate(zip(x_eval, y_eval, times)):
        prediction = model.predict(features.reshape(1, -1))[0]
        validation_record = validations.get(time_value)
        stage1_warning = time_value in warning_time_set

        context = AdaptationContext(
            trial_index=index,
            time=time_value.item() if hasattr(time_value, "item") else time_value,
            stage1_warning=stage1_warning,
            validation_record=validation_record,
        )
        should_update = bool(trigger_policy.should_update(context))

        trial_records.append(
            {
                "trial_index": index,
                "time": context.time,
                "true_label": label.item() if hasattr(label, "item") else label,
                "predicted_label": (
                    prediction.item() if hasattr(prediction, "item") else prediction
                ),
                "correct": bool(prediction == label),
                "stage1_warning": stage1_warning,
                "stage2_status": (
                    validation_record.get("status") if validation_record is not None else None
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
        model.append_and_retrain(x_new, y_new)
        classifier_version += 1
        last_update_end = stop

        update_records.append(
            {
                "update_index": len(update_records),
                "trigger_trial_index": index,
                "trigger_time": context.time,
                "stage1_warning": stage1_warning,
                "stage2_status": (
                    validation_record.get("status") if validation_record is not None else None
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
    )
