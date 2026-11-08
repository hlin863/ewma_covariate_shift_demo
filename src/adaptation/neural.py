"""Incremental neural classifier for chronological BCI feature replay.

Consumes precomputed FBCSP feature vectors. The model never sees evaluation
labels before the online controller releases them; it does not alter the
frozen-PCA EWMA/Hotelling monitoring path.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


@dataclass
class OnlineNeuralClassifier:
    """Small MLP with mini-batch incremental updates via partial_fit.

    Initial calibration fitting is distinct from subsequent update batches.
    The fitted scaler stays frozen throughout evaluation, making online
    learner changes attributable to the MLP rather than scaler drift.
    """

    hidden_layer_sizes: tuple[int, ...] = (32,)
    learning_rate_init: float = 0.001
    initial_epochs: int = 30
    update_epochs: int = 1
    random_state: int = 42
    _model: MLPClassifier | None = field(default=None, init=False, repr=False)
    _scaler: StandardScaler | None = field(default=None, init=False, repr=False)
    _classes: np.ndarray | None = field(default=None, init=False, repr=False)
    _n_features: int | None = field(default=None, init=False, repr=False)
    _training_size: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not self.hidden_layer_sizes or any(
            not isinstance(h, int) or isinstance(h, bool) or h <= 0
            for h in self.hidden_layer_sizes
        ):
            raise ValueError("hidden_layer_sizes must contain positive integers")
        if self.learning_rate_init <= 0:
            raise ValueError("learning_rate_init must be positive")
        if self.initial_epochs < 1 or self.update_epochs < 1:
            raise ValueError("epoch counts must be positive")

    @property
    def training_size(self) -> int:
        return self._training_size

    def _inputs(self, features: np.ndarray, labels: np.ndarray | None = None
                ) -> tuple[np.ndarray, np.ndarray | None]:
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all():
            raise ValueError("features must be a finite, nonempty 2D array")
        if self._n_features is not None and x.shape[1] != self._n_features:
            raise ValueError("feature dimension differs from calibration")
        if labels is None:
            return x, None
        y = np.asarray(labels)
        if y.ndim != 1 or y.shape[0] != x.shape[0]:
            raise ValueError("labels must be 1D and aligned with features")
        return x, y

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "OnlineNeuralClassifier":
        x, y = self._inputs(features, labels)
        classes = np.unique(y)
        if classes.size != 2:
            raise ValueError("binary motor-imagery calibration requires two classes")
        self._n_features = x.shape[1]
        self._classes = classes.copy()
        self._scaler = StandardScaler().fit(x)
        self._model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="relu", solver="adam",
            learning_rate_init=self.learning_rate_init,
            random_state=self.random_state,
        )
        z = self._scaler.transform(x)
        for epoch in range(self.initial_epochs):
            if epoch == 0:
                self._model.partial_fit(z, y, classes=self._classes)
            else:
                self._model.partial_fit(z, y)
        self._training_size = len(x)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        x, _ = self._inputs(features)
        if self._model is None or self._scaler is None:
            raise RuntimeError("fit must be called before prediction")
        return np.asarray(self._model.predict(self._scaler.transform(x)))

    def update(self, new_features: np.ndarray,
               new_labels: np.ndarray) -> "OnlineNeuralClassifier":
        if self._model is None or self._scaler is None or self._classes is None:
            raise RuntimeError("fit must be called before online update")
        x, y = self._inputs(new_features, new_labels)
        if not np.isin(y, self._classes).all():
            raise ValueError("online labels must belong to calibration classes")
        z = self._scaler.transform(x)
        for _ in range(self.update_epochs):
            self._model.partial_fit(z, y)
        self._training_size += len(x)
        return self

    def append_and_retrain(self, new_features: np.ndarray,
                           new_labels: np.ndarray) -> "OnlineNeuralClassifier":
        """Compatibility adapter for the existing online BCI controller.

        Unlike the historical SVM implementation, this updates incrementally;
        it does not append historical features or refit from scratch.
        """
        return self.update(new_features, new_labels)
