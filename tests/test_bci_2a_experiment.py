from pathlib import Path

import numpy as np
import pytest

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
    n_trials: int = 8,
) -> BCISessionData:
    """Build a deterministic synthetic Dataset 2A-like session.

    Small training fixtures keep most unit tests fast. Evaluation tests can
    request the official Dataset 2A Session-II size of 288 cues so that the
    extractor is tested against the same cue/label-count contract as the
    released evaluation files.
    """

    rng = np.random.default_rng(70 if session == "T" else 71)
    sfreq = 100.0
    first_onset = 2.0
    spacing = 5.0
    last_onset = first_onset + (n_trials - 1) * spacing
    n_samples = int(np.ceil((last_onset + 4.0) * sfreq))

    signals = rng.normal(scale=0.25, size=(n_samples, len(channel_names)))
    annotations: list[tuple[float, float, str]] = []

    for index in range(n_trials):
        onset = first_onset + index * spacing
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


def _official_size_evaluation_labels() -> np.ndarray:
    """Return 288 labels with 72 trials from each official 2A class."""

    return np.tile(np.asarray([1, 2, 3, 4], dtype=int), 72)


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


def test_extract_dataset_2a_handles_real_mne_25_channel_layout_with_eog() -> None:
    channel_names = (
        "EEG-Fz", "EEG-0", "EEG-1", "EEG-2", "EEG-3", "EEG-4",
        "EEG-5", "EEG-C3", "EEG-6", "EEG-Cz", "EEG-7", "EEG-C4",
        "EEG-8", "EEG-9", "EEG-10", "EEG-11", "EEG-12", "EEG-13",
        "EEG-14", "EEG-Pz", "EEG-15", "EEG-16",
        "EOG-left", "EOG-central", "EOG-right",
    )
    session = _session("T", channel_names=channel_names)
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


def test_extract_dataset_2a_evaluation_requires_official_labels() -> None:
    with pytest.raises(ValueError, match="requires official evaluation labels"):
        extract_dataset_2a_trials(_session("E", n_trials=288))


def test_extract_dataset_2a_evaluation_filters_to_left_right() -> None:
    labels = _official_size_evaluation_labels()
    result = extract_dataset_2a_trials(
        _session("E", n_trials=288),
        evaluation_labels=labels,
    )

    assert result.signals.shape == (144, 10, 300)
    assert result.labels.shape == (144,)
    assert np.count_nonzero(result.labels == 0) == 72
    assert np.count_nonzero(result.labels == 1) == 72
    assert set(np.unique(result.labels)) == {0, 1}


def test_dataset_2a_fbcsp_produces_paper_filter_bank_features() -> None:
    training = extract_dataset_2a_trials(_session("T"))
    testing = extract_dataset_2a_trials(
        _session("E", n_trials=288),
        evaluation_labels=_official_size_evaluation_labels(),
    )
    pipeline = build_dataset_2a_fbcsp_features(training, testing)

    assert pipeline.training.features.shape == (8, 20)
    assert pipeline.testing.features.shape == (144, 20)
    assert len(pipeline.training.feature_names) == 20
    assert np.isfinite(pipeline.training.features).all()
    assert np.isfinite(pipeline.testing.features).all()
