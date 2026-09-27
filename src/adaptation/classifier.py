"""Classifier abstractions for CSD-triggered adaptation experiments.

The 2018 online BCI study used a linear-kernel SVM that was retrained after
validated covariate shifts.  This module keeps that classifier behaviour
separate from drift detection so the same adaptation experiments can later be
run with alternative models.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.svm import SVC


ArrayLike = np.ndarray


def _as_2d_features(values: ArrayLike, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional feature matrix.")
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one sample and feature.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _as_1d_labels(values: ArrayLike, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if array.size < 1:
        raise ValueError(f"{name} must not be empty.")
    return array


@dataclass
class LinearSVMClassifier:
    """Linear SVM with append-and-retrain support.

    This reproduces the classifier family used by the online EEG-CSAC study.
    The wrapper stores the effective training set so adaptation can append new
    labelled trials before rebuilding the SVM.  It intentionally does not know
    anything about EWMA, CSD, or Stage-II validation.
    """

    c: float = 1.0
    class_weight: str | dict | None = None
    random_state: int | None = 42
    _model: SVC = field(init=False, repr=False)
    _training_features: np.ndarray | None = field(default=None, init=False, repr=False)
    _training_labels: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.c <= 0.0:
            raise ValueError("c must be positive.")
        self._model = self._new_model()

    def _new_model(self) -> SVC:
        return SVC(
            kernel="linear",
            C=self.c,
            class_weight=self.class_weight,
            random_state=self.random_state,
        )

    @property
    def is_fitted(self) -> bool:
        return self._training_features is not None

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

    @property
    def training_features(self) -> np.ndarray:
        if self._training_features is None:
            raise RuntimeError("Classifier has not been fitted.")
        return self._training_features.copy()

    @property
    def training_labels(self) -> np.ndarray:
        if self._training_labels is None:
            raise RuntimeError("Classifier has not been fitted.")
        return self._training_labels.copy()

    def fit(self, features: ArrayLike, labels: ArrayLike) -> "LinearSVMClassifier":
        x = _as_2d_features(features, name="features")
        y = _as_1d_labels(labels, name="labels")
        if x.shape[0] != y.size:
            raise ValueError("features and labels must contain the same number of samples.")
        if np.unique(y).size < 2:
            raise ValueError("Linear SVM training requires at least two classes.")

        self._model = self._new_model()
        self._model.fit(x, y)
        self._training_features = x.copy()
        self._training_labels = y.copy()
        return self

    def predict(self, features: ArrayLike) -> np.ndarray:
        self._require_fitted()
        x = _as_2d_features(features, name="features")
        self._validate_feature_count(x)
        return self._model.predict(x)

    def decision_function(self, features: ArrayLike) -> np.ndarray:
        self._require_fitted()
        x = _as_2d_features(features, name="features")
        self._validate_feature_count(x)
        return np.asarray(self._model.decision_function(x))

    def append_and_retrain(
        self,
        new_features: ArrayLike,
        new_labels: ArrayLike,
    ) -> "LinearSVMClassifier":
        """Append labelled observations and rebuild the linear SVM."""

        self._require_fitted()
        x_new = _as_2d_features(new_features, name="new_features")
        y_new = _as_1d_labels(new_labels, name="new_labels")
        if x_new.shape[0] != y_new.size:
            raise ValueError(
                "new_features and new_labels must contain the same number of samples."
            )
        self._validate_feature_count(x_new)

        x = np.vstack([self._training_features, x_new])
        y = np.concatenate([self._training_labels, y_new])
        return self.fit(x, y)

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before prediction or adaptation.")

    def _validate_feature_count(self, features: np.ndarray) -> None:
        if self.n_features is not None and features.shape[1] != self.n_features:
            raise ValueError(
                "Feature dimension does not match the fitted classifier: "
                f"expected {self.n_features}, received {features.shape[1]}."
            )
