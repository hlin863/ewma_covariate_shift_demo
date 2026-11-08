"""Neural online learner and existing chronological BCI controller."""
import numpy as np
import pytest

from src.adaptation.neural import OnlineNeuralClassifier
from src.adaptation.policies import NeverUpdate, PeriodicRetrain
from src.online.pipeline import OnlineAdaptationConfig, OnlineAdaptiveBCI


def _data():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(80, 4))
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    return x, y


def test_incremental_neural_model_preserves_calibration_space():
    x, y = _data()
    model = OnlineNeuralClassifier(initial_epochs=5, update_epochs=2)
    model.fit(x[:40], y[:40])
    assert model.predict(x[40:43]).shape == (3,)
    assert model.training_size == 40
    before_scaler = model._scaler.mean_.copy()
    original_model = model._model
    model.update(x[40:46], y[40:46])
    assert model._model is original_model
    assert model.training_size == 46
    np.testing.assert_array_equal(before_scaler, model._scaler.mean_)
    with pytest.raises(ValueError, match="dimension"):
        model.predict(np.ones((1, 3)))
    with pytest.raises(ValueError, match="calibration classes"):
        model.update(x[46:47], np.array([3]))


def test_neural_classifier_runs_with_never_update_online_replay():
    x, y = _data()
    classifier = OnlineNeuralClassifier(initial_epochs=4)
    pipeline = OnlineAdaptiveBCI.from_calibration(
        x[:40], y[:40], classifier=classifier, policy=NeverUpdate(),
        config=OnlineAdaptationConfig(window_size=4, label_delay=None),
        pca_components=2,
    )
    result = pipeline.run_replay(x[40:], np.arange(40), y[40:])
    assert result.trials.shape[0] == 40
    assert result.trials["model_version"].eq(0).all()
    assert result.updates.empty
    assert classifier.training_size == 40


def test_neural_classifier_updates_only_on_available_labels():
    x, y = _data()
    classifier = OnlineNeuralClassifier(initial_epochs=4)
    pipeline = OnlineAdaptiveBCI.from_calibration(
        x[:40], y[:40], classifier=classifier, policy=PeriodicRetrain(interval=4),
        config=OnlineAdaptationConfig(window_size=4, label_delay=2),
        pca_components=2,
    )
    result = pipeline.run_replay(x[40:], np.arange(40), y[40:])
    assert not result.updates.empty
    assert classifier.training_size > 40
    assert result.updates["model_version"].is_monotonic_increasing
    for event in result.updates.itertuples(index=False):
        assert all(int(t) + 2 <= int(event.time) for t in event.sample_times)
