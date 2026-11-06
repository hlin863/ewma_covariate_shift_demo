"""Stateful online EWMA must reproduce batch monitoring on identical features."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.detection.preprocessing import fit_cse_pca, transform_cse_features
from src.detection.stage1.sd_ewma import (
    OnlineSDEWMA,
    SD_EWMA_Config,
    SD_EWMA_State,
    fit_sd_ewma,
    run_sd_ewma,
)
from src.online.preprocessing import transform_cse_observation
from src.online.replay import OnlinePC1Monitor, replay_feature_rows


@pytest.mark.parametrize("mode", ["always", "frozen", "non_alarm"])
def test_scalar_stream_matches_existing_batch(mode):
    values = np.array([10., 10.2, 9.8, 14., 14.5, 13.8])
    times = np.arange(10, 16, dtype=int)
    config = SD_EWMA_Config(
        lambda_value=0.2,
        variance_smoothing=0.05,
        control_limit_multiplier=1.96,
        variance_update_mode=mode,
    )
    batch = run_sd_ewma(values, times, 10.0, 1.0, config)
    online = OnlineSDEWMA(SD_EWMA_State(10.0, 1.0), config)
    rows = [online.update_one(float(x), int(t)) for t, x in zip(times, values)]
    streamed = pd.DataFrame(rows)
    assert_frame_equal(batch, streamed[batch.columns], rtol=1e-12, atol=1e-12)
    assert online.state.n_observations == len(values)
    assert online.state.last_time == 15


def test_chunked_stream_and_restart_from_state():
    config = SD_EWMA_Config(lambda_value=0.2)
    detector = OnlineSDEWMA(SD_EWMA_State(10.0, 1.0), config)
    first = [detector.update_one(x, t) for t, x in [(0, 11.), (1, 12.)]]
    restored = OnlineSDEWMA(detector.state, config)
    second = [restored.update_one(x, t) for t, x in [(2, 15.), (3, 9.)]]
    batch = run_sd_ewma(
        np.array([11., 12., 15., 9.]), np.arange(4), 10., 1., config
    )
    assert_frame_equal(pd.DataFrame(first + second), batch)


def test_failed_update_does_not_change_state():
    detector = OnlineSDEWMA(
        SD_EWMA_State(10., 1.), SD_EWMA_Config(lambda_value=0.2)
    )
    detector.update_one(11., 0)
    saved = detector.state
    with pytest.raises(ValueError, match="strictly increasing"):
        detector.update_one(12., 0)
    assert detector.state == saved
    with pytest.raises(ValueError, match="finite"):
        detector.update_one(float("nan"), 1)
    assert detector.state == saved


def test_pca_one_row_transform_matches_batch():
    train = np.array([
        [1., 2., 3.], [2., 1., 4.], [3., 4., 2.],
        [4., 2., 1.], [5., 3., 6.], [6., 5., 4.],
    ])
    pca = fit_cse_pca(train, n_components=2)
    testing = np.array([[2., 3., 4.], [1., 5., 2.]])
    batch = transform_cse_features(testing, pca)
    online = np.vstack([transform_cse_observation(row, pca) for row in testing])
    np.testing.assert_allclose(online, batch)


@pytest.mark.parametrize("mode", ["always", "frozen", "non_alarm"])
def test_frozen_pca_feature_replay_matches_batch(mode):
    training = np.array([
        [1., 2., 3.], [2., 1., 4.], [3., 4., 2.],
        [4., 2., 1.], [5., 3., 6.], [6., 5., 4.],
    ])
    testing = np.array([
        [2., 3., 4.], [8., 4., 9.], [3., 2., 4.],
    ])
    times = np.array([10, 11, 12])
    monitor = OnlinePC1Monitor.from_training_features(
        training, pca_components=2, lambda_override=0.2,
        variance_update_mode=mode,
    )
    calibration = fit_sd_ewma(monitor.pca.first_component, lambda_override=0.2)
    projected = transform_cse_features(testing, monitor.pca)
    expected = run_sd_ewma(
        projected[:, 0], times, calibration.final_z,
        calibration.error_variance, monitor.detector.config,
    )
    output = list(monitor.replay(testing, times))
    actual = pd.DataFrame([item.stage1_trace for item in output])
    assert_frame_equal(actual[expected.columns], expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        np.vstack([item.projected_features for item in output]), projected
    )
    assert monitor.detector.state.n_observations == len(testing)


def test_replay_is_strict_and_preserves_vectors():
    features = np.array([[1., 2.], [3., 4.]])
    received = list(replay_feature_rows(features, np.array([5, 6])))
    received[0][1][0] = 99.
    assert features[0, 0] == 1.
    with pytest.raises(ValueError, match="strictly increasing"):
        list(replay_feature_rows(features, np.array([6, 5])))
    with pytest.raises(ValueError, match="integer"):
        list(replay_feature_rows(features, np.array([1.0, 2.0])))


def test_state_validations():
    with pytest.raises(ValueError, match="positive"):
        SD_EWMA_State(0., 0.)
    with pytest.raises(ValueError, match="integer"):
        SD_EWMA_State(0., 1., last_time=True)
    with pytest.raises(ValueError, match="non-negative"):
        SD_EWMA_State(0., 1., n_observations=-1)
