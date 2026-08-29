"""Training-fitted PCA and ICA transforms for multivariate monitoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import FastICA, PCA


@dataclass(frozen=True)
class MultivariateTransformResult:
    method: str
    model: PCA | FastICA
    training_transformed: np.ndarray
    full_transformed: np.ndarray
    n_components: int


def _validate_pair(
    training_values: np.ndarray,
    full_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    training = np.asarray(training_values, dtype=float)
    full = np.asarray(full_values, dtype=float)
    if training.ndim != 2 or full.ndim != 2:
        raise ValueError("training_values and full_values must be 2D matrices.")
    if training.shape[0] < 2 or training.shape[1] < 1:
        raise ValueError("training_values must contain at least two rows.")
    if training.shape[1] != full.shape[1]:
        raise ValueError("training_values and full_values must share features.")
    if not np.isfinite(training).all() or not np.isfinite(full).all():
        raise ValueError("feature matrices must contain only finite values.")
    return training, full


def fit_paper_pca(
    training_values: np.ndarray,
    full_values: np.ndarray,
    n_components: int = 2,
) -> MultivariateTransformResult:
    """Fit PCA on the stationary section and transform the whole stream."""

    training, full = _validate_pair(training_values, full_values)
    if not 1 <= n_components <= min(training.shape):
        raise ValueError("n_components is outside the supported PCA range.")
    model = PCA(n_components=n_components, svd_solver="full")
    training_transformed = model.fit_transform(training)
    return MultivariateTransformResult(
        method="pca",
        model=model,
        training_transformed=np.asarray(training_transformed, dtype=float),
        full_transformed=np.asarray(model.transform(full), dtype=float),
        n_components=n_components,
    )


def fit_paper_ica(
    training_values: np.ndarray,
    full_values: np.ndarray,
    *,
    random_seed: int = 42,
) -> MultivariateTransformResult:
    """Fit a reproducible FastICA approximation to the paper's ICA branch.

    The 2015 paper discusses FastICA, Infomax and FBSS but labels the synthetic
    Table-IV column simply ``ICA``.  FastICA is the repository's deterministic,
    dependency-light implementation; the method is surfaced in result metadata
    so it is not mistaken for the paper's later EEG recommendation of Infomax.
    """

    training, full = _validate_pair(training_values, full_values)
    n_components = training.shape[1]
    model = FastICA(
        n_components=n_components,
        whiten="unit-variance",
        algorithm="deflation",
        fun="exp",
        max_iter=1000,
        tol=1e-3,
        random_state=random_seed,
    )
    training_transformed = model.fit_transform(training)
    return MultivariateTransformResult(
        method="fastica",
        model=model,
        training_transformed=np.asarray(training_transformed, dtype=float),
        full_transformed=np.asarray(model.transform(full), dtype=float),
        n_components=n_components,
    )
