"""Post-run evaluation for label-hidden adaptation experiments."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from src.adaptation.transductive import UnsupervisedAdaptationResult

@dataclass(frozen=True)
class UnsupervisedEvaluationResult:
    """Offline scoring kept separate from the label-free learner interface."""

    trial_results: pd.DataFrame
    pseudo_label_events: pd.DataFrame
    accuracy: float
    pseudo_label_accuracy: float
    accepted_pseudo_label_accuracy: float


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


TransductiveEvaluationResult = UnsupervisedEvaluationResult
evaluate_transductive_adaptation = evaluate_unsupervised_adaptation

__all__ = [
    "TransductiveEvaluationResult",
    "UnsupervisedEvaluationResult",
    "evaluate_transductive_adaptation",
    "evaluate_unsupervised_adaptation",
]
