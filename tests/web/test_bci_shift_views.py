import numpy as np

from src.web.bci_shift_views import (
    WINDOW_SIZE,
    detection_event_payload,
    feature_window_heatmap,
)


def test_heatmap_uses_training_reference_and_complete_windows_only():
    training = np.asarray([[0.0, 2.0], [1.0, 2.0], [2.0, 2.0],
                           [3.0, 2.0], [4.0, 2.0]])
    evaluation = np.column_stack((np.full(23, 5.0), np.full(23, 2.0)))
    payload = feature_window_heatmap(
        training, evaluation, ("varying_feature", "constant_feature"), window_size=10
    )
    assert len(payload["windows"]) == 2
    assert payload["windows"][0]["start_index"] == 0
    assert payload["windows"][1]["end_index"] == 19
    assert payload["windows"][0]["values"][0] > 0
    assert payload["windows"][0]["values"][1] is None
    assert payload["unscaled_feature_names"] == ["constant_feature"]


def test_detector_events_use_observation_indices_and_source_cue_times():
    rng = np.random.default_rng(24)
    train = rng.normal(0, 1, size=(80, 4))
    evaluation = rng.normal(0, 1, size=(70, 4))
    evaluation[35:] += 3.0
    cue_times = 2.0 + 7.0 * np.arange(70)
    payload = detection_event_payload(train, evaluation, cue_times)
    assert payload["config"]["window_size"] == WINDOW_SIZE
    assert len(payload["warnings"]) == len(evaluation)
    assert [row["index"] for row in payload["warnings"]] == list(range(70))
    for row in payload["warnings"]:
        assert np.isfinite(row["feature_value"])
        assert np.isfinite(row["prediction"])
        assert np.isfinite(row["lcl"])
        assert np.isfinite(row["ucl"])
        assert row["lcl"] < row["ucl"]
    for row in payload["validations"]:
        assert row["alarm_cue_onset_seconds"] == cue_times[row["alarm_index"]]
        if row["validation_index"] is not None:
            assert row["validation_index"] > row["alarm_index"]
            assert row["validation_cue_onset_seconds"] == cue_times[row["validation_index"]]
        assert row["status"] in {
            "confirmed", "rejected", "skipped_nearby_alarm",
            "insufficient_reference_window", "pending_current_window",
            "insufficient_degrees_of_freedom"
        }


def test_all_20_fbcsp_feature_labels_remain_distinct():
    rng = np.random.default_rng(2)
    training = rng.normal(size=(40, 20))
    evaluation = rng.normal(size=(36, 20))
    names = tuple(f"feature_{i}" for i in range(20))
    payload = feature_window_heatmap(training, evaluation, names)
    assert payload["feature_names"] == list(names)
    assert len(payload["windows"]) == 3
    assert all(len(row["values"]) == 20 for row in payload["windows"])


def test_kl_event_payload_uses_eligible_stage2_windows_and_calibration_bins():
    rng = np.random.default_rng(102)
    training = rng.normal(size=(90, 4))
    testing = rng.normal(size=(90, 4))
    testing[38:] += 4
    times = 1.2 + 3 * np.arange(90)
    payload = detection_event_payload(training, testing, times)
    kl = payload["kl_diagnostics"]
    assert kl["n_bins"] == 8
    assert kl["pseudocount"] == 0.5
    assert len(kl["events"]) == len(payload["validations"])
    assert kl["feature_names"] == ["PC1", "PC2"]
    for event, baseline in zip(kl["events"], payload["validations"]):
        assert event["alarm_index"] == baseline["alarm_index"]
        assert event["hotelling_status"] == baseline["status"]
        if event["kl_status"] == "diagnostic_only":
            assert event["forward"] >= 0
            assert event["reverse"] >= 0
            assert len(event["per_feature"]) == 2
            assert event["validation_index"] == baseline["validation_index"]
            assert event["reference_window"][1] == event["alarm_index"]
            assert event["current_window"][0] == event["alarm_index"] + 1
            assert len(event["histograms"]) == 2
            for histogram in event["histograms"]:
                assert len(histogram["reference"]) == 8
                assert len(histogram["current"]) == 8
                assert abs(sum(histogram["reference"]) - 1) < 1e-10
                assert abs(sum(histogram["current"]) - 1) < 1e-10
                assert histogram["edges"][0] == "-inf"
                assert histogram["edges"][-1] == "+inf"
        else:
            assert event["forward"] is None
            assert event["histograms"] == []
    assert kl["eligible_count"] == sum(
        e["kl_status"] == "diagnostic_only" for e in kl["events"]
    )
