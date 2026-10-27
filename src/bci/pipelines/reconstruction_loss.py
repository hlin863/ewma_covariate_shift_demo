"""Leak-free frozen-PCA reconstruction-loss monitoring for BCI feature streams.

This is a BCI integration of the loss proxy in Heyden et al. (2024), Eq. (1).
It is not the authors' full online ABCD algorithm or a calibrated hypothesis test.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.detection.preprocessing import fit_cse_pca, reconstruct_cse_features
from src.detection.baselines.abcd import scan_candidate_splits


@dataclass(frozen=True)
class ReconstructionLossResult:
    training_loss: np.ndarray
    evaluation_loss: np.ndarray
    evaluation_trace: pd.DataFrame
    feature_mse_before: np.ndarray
    feature_mse_after: np.ndarray
    pca_components: int
    explained_variance_ratio: np.ndarray
    batch_split: dict | None


def _matrix(value, name, *, columns=None):
    result = np.asarray(value, dtype=float)
    if result.ndim != 2 or result.shape[0] < 2 or result.shape[1] < 2:
        raise ValueError(f"{name} must be a 2D matrix with >=2 rows and >=2 columns")
    if columns is not None and result.shape[1] != columns:
        raise ValueError("Feature dimension mismatch between training and evaluation")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite values")
    return result


def evaluate_bci_reconstruction_loss(
    training_features,
    evaluation_features,
    *,
    times=None,
    n_components=1,
    feature_names=None,
    candidate_min_side=10,
    bernstein_M=None,
    delta=0.05,
):
    """Fit only on development training data; retain trial order in evaluation.

    Evaluation labels are never supplied to the monitor. Loss L = Q / d,
    where Q is the squared residual energy of the existing PCA CSE helper.
    The optional Bernstein scan is a batch teaching score, not an online
    detector; it scans evaluation observations only and cannot validate drift
    without known change-point ground truth.
    """
    training = _matrix(training_features, "training_features")
    evaluation = _matrix(evaluation_features, "evaluation_features", columns=training.shape[1])
    if not isinstance(n_components, int) or not 1 <= n_components < min(training.shape):
        raise ValueError("n_components must be an integer in [1, min(training.shape)-1]")
    if feature_names is not None and len(feature_names) != training.shape[1]:
        raise ValueError("feature_names length does not match feature dimensions")
    timestamps = np.arange(len(evaluation)) if times is None else np.asarray(times)
    if timestamps.ndim != 1 or len(timestamps) != len(evaluation):
        raise ValueError("times must align one-to-one with evaluation observations")
    if candidate_min_side < 2:
        raise ValueError("candidate_min_side must be >=2")

    fitted = fit_cse_pca(training, n_components=n_components)
    before = reconstruct_cse_features(training, fitted)
    after = reconstruct_cse_features(evaluation, fitted)
    d = training.shape[1]
    train_mse = before.squared_reconstruction_error / d
    eval_mse = after.squared_reconstruction_error / d
    residual_squared_train = before.residual_vectors ** 2
    residual_squared_eval = after.residual_vectors ** 2
    trace = pd.DataFrame({
        "trial_index": np.arange(len(evaluation), dtype=int),
        "time": timestamps,
        "reconstruction_mse": eval_mse,
        "residual_q": after.squared_reconstruction_error,
        "pc1_score": fitted.model.transform(evaluation)[:, 0],
    })
    for j in range(d):
        name = f"feature_{j + 1}" if feature_names is None else str(feature_names[j])
        trace[f"residual_sq_{j + 1}"] = residual_squared_eval[:, j]
    scan = None
    if bernstein_M is not None and len(eval_mse) >= 2 * candidate_min_side:
        scan = scan_candidate_splits(
            eval_mse, min_side=candidate_min_side, M=bernstein_M, delta=delta
        )
    return ReconstructionLossResult(
        training_loss=np.asarray(train_mse),
        evaluation_loss=np.asarray(eval_mse),
        evaluation_trace=trace,
        feature_mse_before=residual_squared_train.mean(axis=0),
        feature_mse_after=residual_squared_eval.mean(axis=0),
        pca_components=fitted.n_components,
        explained_variance_ratio=fitted.explained_variance_ratio,
        batch_split=scan,
    )
