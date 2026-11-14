import numpy as np


def compare_candidates(result_a, result_b):
    """Compare two candidates on the same validation stream."""

    a = result_a.trials
    b = result_b.trials

    if not np.array_equal(a["trial_index"], b["trial_index"]):
        raise ValueError("Candidates must use identical validation trials")

    prediction_difference = np.mean(
        a["prediction"].to_numpy() != b["prediction"].to_numpy()
    )

    model_version_difference = np.mean(
        a["model_version"].to_numpy() != b["model_version"].to_numpy()
    )

    updates_a = set(result_a.updates["time"])
    updates_b = set(result_b.updates["time"])

    update_difference = len(updates_a.symmetric_difference(updates_b))

    return {
        "prediction_disagreement_rate": float(prediction_difference),
        "model_version_disagreement_rate": float(model_version_difference),
        "update_timing_disagreement": update_difference,
        "accuracy_difference": float(result_a.accuracy - result_b.accuracy),
    }
