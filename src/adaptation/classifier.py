"""Classifier abstractions for CSD-triggered adaptation experiments.

The 2018 online BCI study used a linear-kernel SVM that was retrained after
validated covariate shifts.  This module keeps that classifier behaviour
separate from drift detection so the same adaptation experiments can later be
run with alternative models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from sklearn.neighbors import KNeighborsClassifier as SKKNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as SKDecisionTreeClassifier


ArrayLike = np.ndarray


class Classifier(Protocol):
    """Feature-matrix contract shared by single and ensemble classifiers."""

    def fit(self, features: np.ndarray, labels: np.ndarray) -> object: ...
    def predict(self, features: np.ndarray) -> np.ndarray: ...


class RetrainableClassifier(Classifier, Protocol):
    """Additional contract needed by supervised append-and-retrain policies."""

    @property
    def training_size(self) -> int: ...
    def append_and_retrain(
        self, new_features: np.ndarray, new_labels: np.ndarray
    ) -> object: ...


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

    @property
    def classes_(self) -> np.ndarray:
        self._require_fitted()
        return np.asarray(self._model.classes_).copy()

    @property
    def coef_(self) -> np.ndarray:
        self._require_fitted()
        return np.asarray(self._model.coef_, dtype=float).copy()

    @property
    def intercept_(self) -> np.ndarray:
        self._require_fitted()
        return np.asarray(self._model.intercept_, dtype=float).copy()

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



@dataclass
class KNNClassifier:
    """Standard k-nearest-neighbour classifier with append-and-retrain support.

    This is deliberately separate from PWKNN. KNN is a final supervised
    classifier; PWKNN in pseudo_labelling.py is a confidence-bearing
    pseudo-labeller used to acquire labels for transductive adaptation.
    """

    n_neighbors: int = 5
    weights: str = "uniform"
    p: int = 2
    _model: SKKNeighborsClassifier = field(init=False, repr=False)
    _training_features: np.ndarray | None = field(default=None, init=False, repr=False)
    _training_labels: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n_neighbors < 1:
            raise ValueError("n_neighbors must be positive.")
        if self.weights not in {"uniform", "distance"}:
            raise ValueError("weights must be 'uniform' or 'distance'.")
        if self.p < 1:
            raise ValueError("p must be positive.")
        self._model = self._new_model(self.n_neighbors)

    def _new_model(self, n_neighbors: int) -> SKKNeighborsClassifier:
        return SKKNeighborsClassifier(
            n_neighbors=int(n_neighbors), weights=self.weights, p=self.p
        )

    @property
    def is_fitted(self) -> bool:
        return self._training_features is not None

    @property
    def training_size(self) -> int:
        return 0 if self._training_features is None else int(self._training_features.shape[0])

    @property
    def n_features(self) -> int | None:
        return None if self._training_features is None else int(self._training_features.shape[1])

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

    @property
    def classes_(self) -> np.ndarray:
        self._require_fitted()
        return np.asarray(self._model.classes_).copy()

    def fit(self, features: ArrayLike, labels: ArrayLike) -> "KNNClassifier":
        x = _as_2d_features(features, name="features")
        y = _as_1d_labels(labels, name="labels")
        if x.shape[0] != y.size:
            raise ValueError("features and labels must contain the same number of samples.")
        if np.unique(y).size < 2:
            raise ValueError("KNN training requires at least two classes.")
        self._model = self._new_model(min(self.n_neighbors, int(x.shape[0])))
        self._model.fit(x, y)
        self._training_features = x.copy()
        self._training_labels = y.copy()
        return self

    def predict(self, features: ArrayLike) -> np.ndarray:
        self._require_fitted()
        x = _as_2d_features(features, name="features")
        self._validate_feature_count(x)
        return np.asarray(self._model.predict(x))

    def kneighbors(
        self, features: ArrayLike, *, n_neighbors: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        self._require_fitted()
        x = _as_2d_features(features, name="features")
        self._validate_feature_count(x)
        requested = self._model.n_neighbors if n_neighbors is None else int(n_neighbors)
        if requested < 1:
            raise ValueError("n_neighbors must be positive.")
        requested = min(requested, self.training_size)
        distances, indices = self._model.kneighbors(x, n_neighbors=requested)
        return np.asarray(distances, dtype=float), np.asarray(indices, dtype=int)

    def append_and_retrain(
        self, new_features: ArrayLike, new_labels: ArrayLike
    ) -> "KNNClassifier":
        self._require_fitted()
        x_new = _as_2d_features(new_features, name="new_features")
        y_new = _as_1d_labels(new_labels, name="new_labels")
        if x_new.shape[0] != y_new.size:
            raise ValueError("new_features and new_labels must contain the same number of samples.")
        self._validate_feature_count(x_new)
        return self.fit(
            np.vstack([self._training_features, x_new]),
            np.concatenate([self._training_labels, y_new]),
        )

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before prediction or adaptation.")

    def _validate_feature_count(self, features: np.ndarray) -> None:
        if self.n_features is not None and features.shape[1] != self.n_features:
            raise ValueError(
                "Feature dimension does not match the fitted classifier: "
                f"expected {self.n_features}, received {features.shape[1]}."
            )


@dataclass
class DecisionTreeClassifier:
    """Interpretable CART baseline for BCI feature-space decisions.

    The tree is intentionally a decision-making model, not a drift detector.
    It can be retrained after an adaptation policy fires, while Stage-I/II
    detector evidence remains owned by src.detection.
    """

    criterion: str = "gini"
    max_depth: int | None = 4
    min_samples_leaf: int = 5
    random_state: int | None = 42
    _model: SKDecisionTreeClassifier = field(init=False, repr=False)
    _training_features: np.ndarray | None = field(default=None, init=False, repr=False)
    _training_labels: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.criterion not in {"gini", "entropy", "log_loss"}:
            raise ValueError("criterion must be gini, entropy, or log_loss.")
        if self.max_depth is not None and self.max_depth < 1:
            raise ValueError("max_depth must be positive when supplied.")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be positive.")
        self._model = self._new_model()

    def _new_model(self) -> SKDecisionTreeClassifier:
        return SKDecisionTreeClassifier(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )

    @property
    def is_fitted(self) -> bool:
        return self._training_features is not None

    @property
    def training_size(self) -> int:
        return 0 if self._training_features is None else int(self._training_features.shape[0])

    @property
    def n_features(self) -> int | None:
        return None if self._training_features is None else int(self._training_features.shape[1])

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

    @property
    def classes_(self) -> np.ndarray:
        self._require_fitted()
        return np.asarray(self._model.classes_).copy()

    @property
    def tree_(self):
        self._require_fitted()
        return self._model.tree_

    @property
    def feature_importances_(self) -> np.ndarray:
        self._require_fitted()
        return np.asarray(self._model.feature_importances_, dtype=float).copy()

    def fit(self, features: ArrayLike, labels: ArrayLike) -> "DecisionTreeClassifier":
        x = _as_2d_features(features, name="features")
        y = _as_1d_labels(labels, name="labels")
        if x.shape[0] != y.size:
            raise ValueError("features and labels must contain the same number of samples.")
        if np.unique(y).size < 2:
            raise ValueError("Decision-tree training requires at least two classes.")
        self._model = self._new_model()
        self._model.fit(x, y)
        self._training_features = x.copy()
        self._training_labels = y.copy()
        return self

    def predict(self, features: ArrayLike) -> np.ndarray:
        self._require_fitted()
        x = _as_2d_features(features, name="features")
        self._validate_feature_count(x)
        return np.asarray(self._model.predict(x))

    def decision_path(self, features: ArrayLike):
        self._require_fitted()
        x = _as_2d_features(features, name="features")
        self._validate_feature_count(x)
        return self._model.decision_path(x)

    def apply(self, features: ArrayLike) -> np.ndarray:
        self._require_fitted()
        x = _as_2d_features(features, name="features")
        self._validate_feature_count(x)
        return np.asarray(self._model.apply(x), dtype=int)

    def append_and_retrain(
        self, new_features: ArrayLike, new_labels: ArrayLike
    ) -> "DecisionTreeClassifier":
        self._require_fitted()
        x_new = _as_2d_features(new_features, name="new_features")
        y_new = _as_1d_labels(new_labels, name="new_labels")
        if x_new.shape[0] != y_new.size:
            raise ValueError("new_features and new_labels must contain the same number of samples.")
        self._validate_feature_count(x_new)
        return self.fit(
            np.vstack([self._training_features, x_new]),
            np.concatenate([self._training_labels, y_new]),
        )

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before prediction or adaptation.")

    def _validate_feature_count(self, features: np.ndarray) -> None:
        if self.n_features is not None and features.shape[1] != self.n_features:
            raise ValueError(
                "Feature dimension does not match the fitted classifier: "
                f"expected {self.n_features}, received {features.shape[1]}."
            )
