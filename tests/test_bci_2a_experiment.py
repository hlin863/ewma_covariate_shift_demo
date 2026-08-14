from pathlib import Path

import numpy as np

from src.bci_2a_experiment import (
    DATASET_2A_CHANNELS,
    DATASET_2A_EEG_MONTAGE,
    PUBLISHED_2A_RESULTS,
    build_dataset_2a_fbcsp_features,
    extract_dataset_2a_trials,
)
from src.bci_data import BCISessionData


def _session(
    session: str,
    *,
    channel_names: tuple[str, ...] = DATASET_2A_CHANNELS,
) -> BCISessionData:
    rng = np.random.default_rng(70 if session == "T" else 71)
    sfreq = 100.0
    signals = rng.normal(scale=0.25, size=(6000, len(channel_names)))
    annotations = []
    for index, onset in enumerate((2, 7, 12, 17, 22, 27, 32, 37)):
        code = "783" if session == "E" else ("769" if index % 2 == 0 else "770")
        start = int(onset * sfreq)
        stop = start + int(3 * sfreq)
        time = np.arange(stop - start) / sfreq
        channel = 0 if index % 2 == 0 else min(5, len(channel_names) - 1)
        signals[start:stop, channel] += 1.5 * np.sin(2 * np.pi * 10.0 * time)
        annotations.append((float(onset), 0.0, code))
    return BCISessionData(
        subject=1,
        session=session,
        file_path=Path(f"A01{session}.gdf"),
        signals=signals,
        times=np.arange(signals.shape[0]) / sfreq,
        channel_names=channel_names,
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


def test_extract_dataset_2a_resolves_generic_mne_gdf_channel_names_by_montage_order() -> None:
    # MNE can expose Dataset 2A GDF electrodes with one descriptive first
    # label followed by generic EEG-N labels.  The GDF acquisition order is
    # fixed, so the extractor must still recover the ten paper-selected sites.
    generic_names = tuple(
        "EEG-Fz" if index == 0 else f"EEG-{index - 1}"
        for index in range(len(DATASET_2A_EEG_MONTAGE))
    )
    session = _session("T", channel_names=generic_names)
    result = extract_dataset_2a_trials(session)

    assert result.signals.shape == (8, 10, 300)
    assert result.channel_names == DATASET_2A_CHANNELS

    expected_indices = [
        DATASET_2A_EEG_MONTAGE.index(name) for name in DATASET_2A_CHANNELS
    ]
    first_onset = int(2 * session.sampling_frequency)
    last_sample = first_onset + int(3 * session.sampling_frequency)
    expected_first_trial = session.signals[
        first_onset:last_sample, expected_indices
    ].T
    np.testing.assert_allclose(result.signals[0], expected_first_trial)


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
