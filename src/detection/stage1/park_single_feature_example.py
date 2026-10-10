"""Auditable one-feature teaching example for Park et al. (2023), Eq. (2.2).

This retrospective candidate-split calculation is not a calibrated change test.
It is deliberately independent of the multivariate routine, which requires p >= 2.
"""
from math import sqrt
import numpy as np


def single_feature_cusum(values, *, split: int) -> dict:
    """Return the mean-normalised two-segment CUSUM contrast and its components.

    split counts observations in the first segment (1-based partition convention).
    A positive full-sequence mean is required by Park's normalisation.
    """
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or data.size < 3 or not np.isfinite(data).all():
        raise ValueError("Provide a finite one-dimensional sequence of at least three observations.")
    if isinstance(split, bool) or not isinstance(split, (int, np.integer)) or not 1 <= split < data.size:
        raise ValueError("The split must be an integer strictly between 0 and T.")
    overall = float(data.mean())
    if overall <= 1e-12:
        raise ValueError("The overall mean must be strictly positive for Park's normalisation.")
    n = int(data.size)
    before = float(data[:split].mean())
    after = float(data[split:].mean())
    weight = sqrt(split * (n - split) / n)
    contrast = abs(before - after)
    value = weight * contrast / overall
    return {
        "T": n, "split": int(split), "observations": data.tolist(),
        "first": data[:split].tolist(), "second": data[split:].tolist(),
        "mean_before": before, "mean_after": after, "overall_mean": overall,
        "weight": weight, "absolute_difference": contrast, "cusum": value,
    }


def worked_example() -> dict:
    """Fixed educational example; not a BCI dataset or empirical benchmark."""
    values = [2, 2, 2, 2, 6, 6, 6, 6]
    example = single_feature_cusum(values, split=4)
    example["curve"] = [
        {"split": t, "cusum": single_feature_cusum(values, split=t)["cusum"]}
        for t in range(1, len(values))
    ]
    example["expected_exact"] = "√2"
    example["passes_example"] = bool(np.isclose(example["cusum"], sqrt(2), rtol=0, atol=1e-12))
    return example
