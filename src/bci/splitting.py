# src/bci/splitting.py

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class StratifiedSplitIndices:
    training: np.ndarray
    validation: np.ndarray


def stratified_split_indices(
    labels: np.ndarray,
    *,
    validation_fraction: float = 0.30,
    random_state: int = 42,
) -> StratifiedSplitIndices:
    labels = np.asarray(labels)

    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional.")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1).")

    classes = np.unique(labels)
    if classes.size != 2 or np.any(classes < 0):
        raise ValueError("stratified splitting requires two labelled classes.")

    rng = np.random.default_rng(random_state)

    training_indices = []
    validation_indices = []

    for label in classes:
        class_indices = np.flatnonzero(labels == label)

        if class_indices.size < 2:
            raise ValueError("each class must contain at least two observations.")

        shuffled = rng.permutation(class_indices)

        n_validation = int(round(class_indices.size * validation_fraction))
        n_validation = min(max(n_validation, 1), class_indices.size - 1)

        validation_indices.extend(shuffled[:n_validation])
        training_indices.extend(shuffled[n_validation:])

    training = np.sort(np.asarray(training_indices, dtype=int))
    validation = np.sort(np.asarray(validation_indices, dtype=int))

    if np.intersect1d(training, validation).size:
        raise RuntimeError("training and validation subsets must not overlap.")

    if training.size + validation.size != labels.size:
        raise RuntimeError("split must preserve every observation.")

    return StratifiedSplitIndices(training, validation)
