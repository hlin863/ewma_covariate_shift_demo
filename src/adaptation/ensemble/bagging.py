"""Bagging classifier for adaptive non-stationary learning experiments.

The implementation keeps ensemble construction separate from change
detection and adaptation policies. It can therefore be used as a passive
ensemble baseline or as the predictive model in later CSD-triggered
adaptation experiments.

The default ensemble size and sample fraction follow the configuration
reported by Li et al. (2010): 30 learners trained from samples containing
80% of the available training observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np


class BaseClassifier(Protocol):
    """Minimal interface required from each ensemble member."""

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> object: ...

    def predict(
        self,
        features: np.ndarray,
    ) -> np.ndarray: ...


EstimatorFactory = Callable[[], BaseClassifier]


@dataclass
class BaggingClassifier:
    """Bootstrap aggregation classifier.

    Parameters
    ----------
    estimator_factory:
        Callable that creates a fresh unfitted classifier for each
        ensemble member.

    n_estimators:
        Number of independently trained classifiers. The default of 30
        follows Li et al. (2010).

    sample_fraction:
        Fraction of the available training observations used for each
        ensemble member. The default 0.8 follows Li et al. (2010).

    bootstrap:
        If True, observations are sampled with replacement.

    random_state:
        Seed controlling ensemble resampling.
    """

    estimator_factory: EstimatorFactory
    n_estimators: int = 30
    sample_fraction: float = 0.8
    bootstrap: bool = True
    random_state: int | None = 42

    estimators_: list[BaseClassifier] = field(
        default_factory=list,
        init=False,
        repr=False,
    )

    _training_features: np.ndarray | None = field(
        default=None,
        init=False,
        repr=False,
    )

    _training_labels: np.ndarray | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be at least 1.")

        if not 0.0 < self.sample_fraction <= 1.0:
            raise ValueError("sample_fraction must be in the interval (0, 1].")

    @property
    def is_fitted(self) -> bool:
        return bool(self.estimators_)

    @property
    def training_size(self) -> int:
        if self._training_features is None:
            return 0

        return int(self._training_features.shape[0])

    @property
    def n_features(self) -> int | None:
        if self._training_features is None:
            return None

        return int(self._training_features.shape[1])

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> "BaggingClassifier":
        """Fit independent classifiers on resampled training data."""

        x = np.asarray(features, dtype=float)
        y = np.asarray(labels)

        if x.ndim != 2:
            raise ValueError("features must be a two-dimensional feature matrix.")

        if y.ndim != 1:
            raise ValueError("labels must be one-dimensional.")

        if x.shape[0] != y.size:
            raise ValueError(
                "features and labels must contain the same " "number of samples."
            )

        if x.shape[0] < 1 or x.shape[1] < 1:
            raise ValueError("features must contain at least one sample and feature.")

        if not np.isfinite(x).all():
            raise ValueError("features must contain only finite values.")

        if np.unique(y).size < 2:
            raise ValueError("Bagging classification requires at least two classes.")

        rng = np.random.default_rng(self.random_state)

        n_samples = x.shape[0]
        sample_size = max(
            1,
            int(round(self.sample_fraction * n_samples)),
        )

        self.estimators_ = []

        for _ in range(self.n_estimators):
            indices = rng.choice(
                n_samples,
                size=sample_size,
                replace=self.bootstrap,
            )

            x_sample = x[indices]
            y_sample = y[indices]

            # A bootstrap sample can occasionally contain only one class.
            if np.unique(y_sample).size < 2:
                continue

            estimator = self.estimator_factory()
            estimator.fit(x_sample, y_sample)
            self.estimators_.append(estimator)

        if not self.estimators_:
            raise RuntimeError("No valid ensemble members could be fitted.")

        self._training_features = x.copy()
        self._training_labels = y.copy()

        return self

    def predict(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        """Predict labels using majority voting."""

        self._require_fitted()

        x = np.asarray(features, dtype=float)

        if x.ndim != 2:
            raise ValueError("features must be a two-dimensional feature matrix.")

        if self.n_features is not None and x.shape[1] != self.n_features:
            raise ValueError(
                "Feature dimension does not match the fitted ensemble: "
                f"expected {self.n_features}, received {x.shape[1]}."
            )

        predictions = np.stack(
            [estimator.predict(x) for estimator in self.estimators_],
            axis=0,
        )

        return np.apply_along_axis(
            self._majority_vote,
            axis=0,
            arr=predictions,
        )

    @staticmethod
    def _majority_vote(values: np.ndarray):
        labels, counts = np.unique(
            values,
            return_counts=True,
        )

        return labels[np.argmax(counts)]

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("BaggingClassifier must be fitted before prediction.")
