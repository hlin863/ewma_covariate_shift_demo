"""Chronological online replay on precomputed BCI feature vectors.

FBCSP features are precomputed with the repository's offline trial pipeline.
Only the frozen PCA transform and SD-EWMA monitor run one observation at a time.
This is feature-level streaming replay, NOT causal real-time EEG filtering.
"""

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from src.detection.preprocessing import CSEPCAResult, fit_cse_pca
from src.detection.stage1.sd_ewma import (
    OnlineSDEWMA,
    SD_EWMA_Config,
    fit_sd_ewma,
)
from src.online.preprocessing import transform_cse_observation


@dataclass(frozen=True)
class OnlineFeatureResult:
    time: int
    features: np.ndarray
    projected_features: np.ndarray
    stage1_trace: dict[str, float | int | str]


def replay_feature_rows(
    features: np.ndarray, times: np.ndarray
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield one (D,) observation per call, preserving chronological order."""
    values = np.asarray(features, dtype=float)
    indices = np.asarray(times)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("features must be a nonempty (N, D) matrix.")
    if not np.isfinite(values).all():
        raise ValueError("features must be finite.")
    if indices.ndim != 1 or len(indices) != len(values):
        raise ValueError("times must be one-dimensional and match the feature rows.")
    if indices.dtype.kind not in "iu" or indices.dtype.kind == "b":
        raise ValueError("times must contain integer observation indices.")
    if np.any(np.diff(indices) <= 0):
        raise ValueError("times must be strictly increasing.")
    for time, vector in zip(indices, values):
        yield int(time), vector.copy()


class OnlinePC1Monitor:
    """Frozen PCA -> scalar PC1 -> one-step SD-EWMA, preserving full feature data."""

    def __init__(
        self, pca: CSEPCAResult, detector: OnlineSDEWMA
    ) -> None:
        self.pca = pca
        self.detector = detector

    @classmethod
    def from_training_features(
        cls,
        training_features: np.ndarray,
        *,
        pca_components: int | float | None = None,
        lambda_override: float | None = None,
        variance_smoothing: float = 0.05,
        control_limit_multiplier: float = 1.96,
        variance_update_mode: str = "always",
    ) -> "OnlinePC1Monitor":
        pca = fit_cse_pca(training_features, n_components=pca_components)
        training = fit_sd_ewma(
            pca.first_component, lambda_override=lambda_override
        )
        config = SD_EWMA_Config(
            lambda_value=training.lambda_value,
            variance_smoothing=variance_smoothing,
            control_limit_multiplier=control_limit_multiplier,
            variance_update_mode=variance_update_mode,
        )
        return cls(pca, OnlineSDEWMA.from_training(training, config))

    def update_one(self, features: np.ndarray, time: int) -> OnlineFeatureResult:
        """Transform one incoming feature vector and update Stage-I once."""
        vector = np.asarray(features, dtype=float)
        projected = transform_cse_observation(vector, self.pca)
        trace = self.detector.update_one(float(projected[0]), time)
        return OnlineFeatureResult(
            time=int(time),
            features=vector.copy(),
            projected_features=projected,
            stage1_trace=trace,
        )

    def replay(
        self, features: np.ndarray, times: np.ndarray
    ) -> Iterator[OnlineFeatureResult]:
        for time, vector in replay_feature_rows(features, times):
            yield self.update_one(vector, time)
