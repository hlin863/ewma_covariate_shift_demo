"""End-to-end online CSE -> delayed Hotelling -> policy -> classifier adaptation."""
import numpy as np
import pandas as pd

from src.adaptation.policies import NeverUpdate, PeriodicRetrain, RetrainOnValidatedShift
from src.detection.stage2.paper_two_sample_hotelling import (
    PaperTwoSampleHotellingConfig,
    validate_paper_two_sample_alarms,
)
from src.online.pipeline import OnlineAdaptiveBCI, OnlineAdaptationConfig
from src.online.validation import OnlineHotellingValidator


def example_data():
    rng = np.random.default_rng(9)
    train = rng.normal(size=(80, 3))
    train_labels = (train[:, 0] + 0.3*train[:, 1] > 0).astype(int)
    eval_x = rng.normal(size=(44, 3))
    eval_x[22:, :] += np.array([3., 2., 2.])
    eval_labels = (eval_x[:, 0] + 0.3*eval_x[:, 1] > 0).astype(int)
    return train, train_labels, eval_x, eval_labels


def test_end_to_end_never_update_and_periodic_update():
    train, y, evaluation, truth = example_data()
    baseline = OnlineAdaptiveBCI.from_calibration(
        train, y, policy=NeverUpdate(),
        config=OnlineAdaptationConfig(window_size=8, label_delay=0),
        pca_components=2, control_limit_multiplier=1.3,
    )
    results = baseline.run_replay(evaluation, np.arange(44), truth)
    assert len(results.trials) == 44
    assert results.updates.empty
    assert results.trials["model_version"].eq(0).all()
    adapted = OnlineAdaptiveBCI.from_calibration(
        train, y, policy=PeriodicRetrain(interval=5),
        config=OnlineAdaptationConfig(window_size=8, label_delay=2),
        pca_components=2, control_limit_multiplier=1.3,
    )
    adaptive = adapted.run_replay(evaluation, np.arange(44), truth)
    assert not adaptive.updates.empty
    assert adaptive.updates["added_samples"].min() >= 1
    assert len(set(sum(adaptive.updates["sample_times"].to_list(), []))) == sum(adaptive.updates["added_samples"])


def test_delayed_validation_matches_offline_statistics():
    rng = np.random.default_rng(22)
    features = rng.normal(size=(40, 2))
    features[20:, :] += 2
    times = np.arange(40)
    warnings = pd.DataFrame({"time": [15], "stage_1_alarm": [1]})
    config = PaperTwoSampleHotellingConfig(
        window_size=6, covariance_method="shrinkage"
    )
    expected = validate_paper_two_sample_alarms(features, times, warnings, config)
    validator = OnlineHotellingValidator(config)
    events = []
    for time, row in zip(times, features):
        events.extend(validator.update_one(int(time), row, time == 15))
    confirmed = [e for e in events if e["status"] in ("confirmed", "rejected")]
    assert len(confirmed) == 1
    got = confirmed[0]
    assert got["validation_time"] == 21
    assert got["status"] == expected.iloc[0]["status"]
    np.testing.assert_allclose(got["p_value"], expected.iloc[0]["p_value"])
    assert all(e.get("validation_time") is None or e["validation_time"] >= 21 for e in events)


def test_validated_shift_can_retrain_only_after_validation_and_label_arrival():
    train, y, evaluation, truth = example_data()
    pipeline = OnlineAdaptiveBCI.from_calibration(
        train, y, policy=RetrainOnValidatedShift(),
        config=OnlineAdaptationConfig(window_size=5, label_delay=1),
        pca_components=2, control_limit_multiplier=1.0,
    )
    result = pipeline.run_replay(evaluation, np.arange(44), truth)
    successes = result.validations[result.validations["status"] == "confirmed"]
    if not successes.empty:
        first_available = int(successes["validation_time"].min())
        assert all(result.updates["time"] >= first_available)
        assert all(
            row["time"] > row["sample_times"][0]
            for row in result.updates.to_dict("records")
        )
    assert result.trials["time"].is_monotonic_increasing


def test_unlabelled_replay_never_uses_ground_truth():
    train, y, evaluation, truth = example_data()
    pipeline = OnlineAdaptiveBCI.from_calibration(
        train, y, policy=PeriodicRetrain(interval=1),
        config=OnlineAdaptationConfig(window_size=5, label_delay=None),
        pca_components=2,
    )
    result = pipeline.run_replay(evaluation, np.arange(44), truth)
    assert result.updates.empty
    assert result.trials["model_version"].eq(0).all()
    assert "correct" in result.trials  # post-run only


def test_prediction_before_update():
    train, y, evaluation, truth = example_data()
    pipe = OnlineAdaptiveBCI.from_calibration(
        train, y, policy=PeriodicRetrain(interval=1),
        config=OnlineAdaptationConfig(window_size=5, label_delay=0),
        pca_components=2,
    )
    before = pipe.classifier.predict(evaluation[0:1])[0]
    event = pipe.process_one(evaluation[0], 0)
    assert event["prediction"] == before
    pipe.receive_label(0, truth[0], available_at=0)
    assert event["model_version"] == 0
    assert pipe.updates[0]["model_version"] == 1
