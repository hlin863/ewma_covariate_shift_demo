"""Numerical and split-isolation tests for the BCI reconstruction proxy."""
import numpy as np
import pytest

from src.bci.pipelines.reconstruction_loss import evaluate_bci_reconstruction_loss


def test_reconstruction_mse_equals_residual_q_divided_by_features():
    rng = np.random.default_rng(7)
    train = rng.normal(size=(60, 5))
    test = rng.normal(size=(40, 5))
    result = evaluate_bci_reconstruction_loss(train, test, n_components=2)
    np.testing.assert_allclose(
        result.evaluation_trace["residual_q"] / 5,
        result.evaluation_trace["reconstruction_mse"],
    )
    assert result.pca_components == 2
    assert len(result.evaluation_trace) == 40
    assert result.batch_split is None


def test_training_only_fit_and_retained_direction_invisibility():
    rng = np.random.default_rng(11)
    # One well-separated dominant direction; ensure train-centred reference.
    x = rng.normal(size=150)
    train = np.column_stack((10 * x, np.zeros_like(x)))
    test = np.column_stack((10 * x[:80] + 1000, np.zeros(80)))
    result = evaluate_bci_reconstruction_loss(train, test, n_components=1)
    np.testing.assert_allclose(result.evaluation_loss, 0, atol=1e-16)
    # This does not claim the original data distributions are equal.
    assert abs(test[:, 0].mean() - train[:, 0].mean()) > 100


def test_discarded_direction_shift_increases_loss():
    rng = np.random.default_rng(3)
    x = rng.normal(size=100)
    train = np.column_stack((10 * x, np.zeros_like(x)))
    eval_features = np.column_stack((10 * x, np.ones_like(x) * 4))
    result = evaluate_bci_reconstruction_loss(train, eval_features, n_components=1)
    np.testing.assert_allclose(result.evaluation_loss, 8, atol=1e-12)


def test_rejects_invalid_features_and_times():
    a = np.ones((20, 3))
    with pytest.raises(ValueError, match="dimension"):
        evaluate_bci_reconstruction_loss(a, np.ones((20, 4)))
    with pytest.raises(ValueError, match="times"):
        evaluate_bci_reconstruction_loss(a, a, times=[1, 2])
