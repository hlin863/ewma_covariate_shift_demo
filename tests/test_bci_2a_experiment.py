from pathlib import Path

import numpy as np

from src.bci_2a_experiment import (
    DATASET_2A_CHANNELS,
    PUBLISHED_2A_RESULTS,
    build_dataset_2a_fbcsp_features,
    extract_dataset_2a_trials,
)
from src.bci_data import BCISessionData


def _session(session: str) -> BCISessionData:
    rng = np.random.default_rng(70 if session == "T" else 71)
    sfreq = 100.0
    signals = rng.normal(scale=0.25, size=(6000, len(DATASET_2A_CHANNELS)))
    annotations = []
    for index, onset in enumerate((2, 7, 12, 17, 22, 27, 32, 37)):
        code = "783" if session == "E" else ("769" if index % 2 == 0 else "770")
        start = int(onset * sfreq)
        stop = start + int(3 * sfreq)
        time = np.arange(stop - start) / sfreq
        channel = 0 if index % 2 == 0 else 5
        signals[start:stop, channel] += 1.5 * np.sin(2 * np.pi * 10.0 * time)
        annotations.append((float(onset), 0.0, code))
    return BCISessionData(
        subject=1,
        session=session,
        file_path=Path(f"A01{session}.gdf"),
        signals=signals,
        times=np.arange(signals.shape[0]) / sfreq,
        channel_names=DATASET_2A_CHANNELS,
        sampling_frequency=sfreq,
        annotations=tuple(annotations),
    )


def test_published_dataset_2a_targets_match_table1() -> None:
    assert set(PUBLISHED_2A_RESULTS) == {f"A{i:02d}" for i in range(1, 10)}
    assert PUBLISHED_2A_RESULTS["A01"] == (0.50, 12, 6)
    assert PUBLISHED_2A_RESULTS["A09"] == (0.70, 6, 4)


def test_extract_dataset_2a_training_trials_selects_ten_channels_and_left_right() -> None:
    result = extract_dataset_2a_trials(_session("T"))
    assert result.signals.shape == (8, 10, 300)
    assert result.channel_names == DATASET_2A_CHANNELS
    np.testing.assert_array_equal(result.labels, [0, 1, 0, 1, 0, 1, 0, 1])


def test_extract_dataset_2a_evaluation_trials_remains_unlabelled() -> None:
    result = extract_dataset_2a_trials(_session("E"))
    assert result.signals.shape == (8, 10, 300)
    np.testing.assert_array_equal(result.labels, np.full(8, -1))


def test_dataset_2a_fbcsp_produces_paper_filter_bank_features() -> None:
    training = extract_dataset_2a_trials(_session("T"))
    testing = extract_dataset_2a_trials(_session("E"))
    pipeline = build_dataset_2a_fbcsp_features(training, testing)
    assert pipeline.training.features.shape == (8, 20)
    assert pipeline.testing.features.shape == (8, 20)
    assert len(pipeline.training.feature_names) == 20
    assert np.isfinite(pipeline.training.features).all()
    assert np.isfinite(pipeline.testing.features).all()
