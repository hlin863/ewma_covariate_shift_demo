"""One-observation interface for the existing fitted PCA preprocessing."""

import numpy as np

from src.detection.preprocessing import CSEPCAResult, transform_cse_features


def transform_cse_observation(
    features: np.ndarray,
    fitted_result: CSEPCAResult,
) -> np.ndarray:
    """Transform one (D,) feature vector without refitting the PCA model."""
    observation = np.asarray(features, dtype=float)
    if observation.ndim != 1 or observation.size != fitted_result.n_input_features:
        raise ValueError(
            "features must be one vector matching the fitted PCA input feature count."
        )
    if not np.isfinite(observation).all():
        raise ValueError("features must contain only finite values.")
    return transform_cse_features(
        observation.reshape(1, -1), fitted_result
    )[0].copy()
