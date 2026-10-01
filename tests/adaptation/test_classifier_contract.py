"""Both classifier families consume the same prepared feature matrices."""

import numpy as np

from src.adaptation import BaggingClassifier, LinearSVMClassifier


def test_linear_and_bagging_use_the_same_feature_matrix_contract() -> None:
    features = np.array([[-2.0, 0.1], [-1.0, 0.2], [1.0, 0.3], [2.0, 0.4]])
    labels = np.array([0, 0, 1, 1])
    svm = LinearSVMClassifier().fit(features, labels)
    bagging = BaggingClassifier(
        estimator_factory=LinearSVMClassifier, n_estimators=5, random_state=2,
    ).fit(features, labels)

    assert svm.predict(features).shape == bagging.predict(features).shape == (4,)
    assert svm.training_size == bagging.training_size == 4
    assert hasattr(svm, "append_and_retrain")
    assert not hasattr(bagging, "append_and_retrain")
