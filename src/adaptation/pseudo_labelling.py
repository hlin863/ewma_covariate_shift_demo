"""Pseudo-labelling mechanisms for transductive adaptation.

The PWKNN implementation preserves the Raza et al. (2019) CSE-UAEL lineage.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import numpy as np

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

