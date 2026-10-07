import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class ActiveLearningConfig:
    alpha: float = 0.7
    query_budget: int = 10

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1.")

        if self.query_budget < 0:
            raise ValueError("query_budget must be non-negative.")


@dataclass(frozen=True)
class ActiveLearningSelection:
    indices: np.ndarray
    scores: np.ndarray
    uncertainty: np.ndarray
    diversity: np.ndarray


def svm_uncertainty(
    decision_values: np.ndarray,
) -> np.ndarray:
    """Convert linear-SVM decision values into uncertainty scores.

    For binary classification, observations close to the decision boundary
    have decision values close to zero and therefore receive higher
    uncertainty scores.

        U(x) = 1 / (1 + |f(x)|)

    Returns values in (0, 1], where larger values indicate greater
    uncertainty.
    """
    values = np.asarray(decision_values, dtype=float)

    if values.ndim != 1:
        raise ValueError(
            "decision_values must be a one-dimensional array "
            "for binary SVM classification."
        )

    if not np.isfinite(values).all():
        raise ValueError("decision_values must contain only finite values.")

    return 1.0 / (1.0 + np.abs(values))


def diversity_scores(
    candidate_features: np.ndarray,
    selected_features: np.ndarray,
) -> np.ndarray:
    """Calculate feature-space diversity for each candidate.

    Diversity is defined as the Euclidean distance from each candidate
    to its nearest already-selected sample:

        D(x) = min_{s in S} ||x - s||_2

    Larger values indicate candidates that are less similar to the
    samples already selected.

    If no samples have yet been selected, all candidates receive the
    same diversity score.
    """
    candidates = np.asarray(candidate_features, dtype=float)
    selected = np.asarray(selected_features, dtype=float)

    if candidates.ndim != 2:
        raise ValueError("candidate_features must be a 2-D array.")

    if selected.ndim != 2:
        raise ValueError("selected_features must be a 2-D array.")

    if candidates.shape[1] != selected.shape[1]:
        raise ValueError(
            "candidate_features and selected_features must have "
            "the same number of features."
        )

    if candidates.shape[0] == 0:
        return np.asarray([], dtype=float)

    if selected.shape[0] == 0:
        return np.ones(candidates.shape[0], dtype=float)

    differences = candidates[:, None, :] - selected[None, :, :]
    distances = np.linalg.norm(differences, axis=2)

    return np.min(distances, axis=1)


def committee_disagreement_scores(
    committee_predictions: np.ndarray,
) -> np.ndarray:
    """Calculate vote-disagreement scores for classifier committees.

    committee_predictions has shape:

        (n_candidates, n_classifiers)

    D(x) = 1 - max_c n_c(x) / M
    """
    predictions = np.asarray(committee_predictions)

    if predictions.ndim != 2:
        raise ValueError(
            "committee_predictions must have shape "
            "(n_candidates, n_classifiers)."
        )

    n_candidates, n_classifiers = predictions.shape

    if n_classifiers == 0:
        raise ValueError("At least one classifier is required.")

    scores = np.empty(n_candidates, dtype=float)

    for index, votes in enumerate(predictions):
        _, counts = np.unique(votes, return_counts=True)
        majority_votes = np.max(counts)

        scores[index] = 1.0 - majority_votes / n_classifiers

    return scores


def _normalise_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max normalise scores into [0, 1]."""
    values = np.asarray(scores, dtype=float)

    if values.ndim != 1:
        raise ValueError("scores must be one-dimensional.")

    if values.size == 0:
        return values.copy()

    if not np.isfinite(values).all():
        raise ValueError("scores must contain only finite values.")

    minimum = float(np.min(values))
    maximum = float(np.max(values))

    if np.isclose(minimum, maximum):
        return np.zeros_like(values)

    return (values - minimum) / (maximum - minimum)


def select_query_batch(
    candidate_features: np.ndarray,
    uncertainty: np.ndarray,
    *,
    config: ActiveLearningConfig,
) -> ActiveLearningSelection:
    """Select an informative and diverse batch for true-label querying.

    Candidates are selected greedily according to:

        Q(x) = alpha * U(x) + (1 - alpha) * D(x)

    where:
        U(x) is classifier uncertainty,
        D(x) is feature-space diversity relative to samples already
             selected for the current query batch.

    The first candidate is effectively selected by uncertainty because
    there is not yet a meaningful diversity preference.
    """
    candidates = np.asarray(candidate_features, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    if candidates.ndim != 2:
        raise ValueError("candidate_features must be a 2-D array.")

    if uncertainty.ndim != 1:
        raise ValueError("uncertainty must be a one-dimensional array.")

    if candidates.shape[0] != uncertainty.size:
        raise ValueError(
            "candidate_features and uncertainty must contain "
            "the same number of observations."
        )

    if not np.isfinite(candidates).all():
        raise ValueError("candidate_features must contain only finite values.")

    if not np.isfinite(uncertainty).all():
        raise ValueError("uncertainty must contain only finite values.")

    n_candidates = candidates.shape[0]

    if n_candidates == 0 or config.query_budget == 0:
        return ActiveLearningSelection(
            indices=np.asarray([], dtype=int),
            scores=np.asarray([], dtype=float),
            uncertainty=np.asarray([], dtype=float),
            diversity=np.asarray([], dtype=float),
        )

    budget = min(config.query_budget, n_candidates)

    normalised_uncertainty = _normalise_scores(uncertainty)

    available_indices = list(range(n_candidates))
    selected_indices: list[int] = []

    selected_scores: list[float] = []
    selected_uncertainty: list[float] = []
    selected_diversity: list[float] = []

    selected_features = np.empty(
        (0, candidates.shape[1]),
        dtype=float,
    )

    for _ in range(budget):
        remaining_indices = np.asarray(available_indices, dtype=int)
        remaining_features = candidates[remaining_indices]

        raw_diversity = diversity_scores(
            remaining_features,
            selected_features,
        )

        normalised_diversity = _normalise_scores(raw_diversity)

        remaining_uncertainty = normalised_uncertainty[
            remaining_indices
        ]

        query_scores = (
            config.alpha * remaining_uncertainty
            + (1.0 - config.alpha) * normalised_diversity
        )

        best_position = int(np.argmax(query_scores))
        best_index = int(remaining_indices[best_position])

        selected_indices.append(best_index)
        selected_scores.append(float(query_scores[best_position]))
        selected_uncertainty.append(
            float(remaining_uncertainty[best_position])
        )
        selected_diversity.append(
            float(normalised_diversity[best_position])
        )

        selected_features = np.vstack(
            [
                selected_features,
                candidates[best_index],
            ]
        )

        available_indices.remove(best_index)

    return ActiveLearningSelection(
        indices=np.asarray(selected_indices, dtype=int),
        scores=np.asarray(selected_scores, dtype=float),
        uncertainty=np.asarray(selected_uncertainty, dtype=float),
        diversity=np.asarray(selected_diversity, dtype=float),
    )