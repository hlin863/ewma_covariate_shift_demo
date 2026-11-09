"""Prequential within-batch accuracy comparison from Algorithm 1 lines 5–8."""
import numpy as np


def calculate_batch_accuracies(predictions: np.ndarray, labels: np.ndarray,
                               *, win_size: int) -> tuple[float, float]:
    pred, truth = np.asarray(predictions), np.asarray(labels)
    if pred.ndim != 1 or truth.ndim != 1 or pred.shape != truth.shape:
        raise ValueError("predictions and labels must be matching 1-D arrays")
    if not 0 < win_size < len(pred):
        raise ValueError("Require 0 < win_size < batch length")
    correct = pred == truth
    return float(correct[:-win_size].mean()), float(correct[-win_size:].mean())
