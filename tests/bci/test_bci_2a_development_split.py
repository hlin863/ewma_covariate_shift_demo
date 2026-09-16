import numpy as np
import pytest

from src.bci_2a_development_split import Dataset2ADevelopmentSplit
from src.bci_2a_experiment import (
    DATASET_2A_CHANNELS,
    Dataset2ATrialSignalResult,
    split_dataset_2a_session1,
)


def _session1_trials() -> Dataset2ATrialSignalResult:
    n_trials = 20
    labels = np.asarray([0, 1] * 10, dtype=int)
    signals = np.arange(n_trials * 10 * 30, dtype=float).reshape(n_trials, 10, 30)
    return Dataset2ATrialSignalResult(
        signals=signals,
        labels=labels,
        times=np.arange(n_trials, dtype=int),
        cue_descriptions=np.asarray(
            ["769" if label == 0 else "770" for label in labels],
            dtype="U8",
        ),
        channel_names=DATASET_2A_CHANNELS,
        sampling_frequency=250.0,
        session_id="T",
    )


def test_session1_split_returns_explicit_development_split() -> None:
    split = split_dataset_2a_session1(_session1_trials())

    assert isinstance(split, Dataset2ADevelopmentSplit)
    assert split.training.signals.shape[0] == 14
    assert split.validation.signals.shape[0] == 6


def test_session1_split_is_stratified_and_preserves_all_trials_without_overlap() -> None:
    trials = _session1_trials()
    split = split_dataset_2a_session1(trials, validation_fraction=0.30, random_state=42)

    np.testing.assert_array_equal(np.bincount(split.training.labels), [7, 7])
    np.testing.assert_array_equal(np.bincount(split.validation.labels), [3, 3])

    training_times = set(split.training.times.tolist())
    validation_times = set(split.validation.times.tolist())
    assert training_times.isdisjoint(validation_times)
    assert training_times | validation_times == set(trials.times.tolist())


def test_session1_split_is_reproducible_for_fixed_seed() -> None:
    trials = _session1_trials()
    first = split_dataset_2a_session1(trials, random_state=7)
    second = split_dataset_2a_session1(trials, random_state=7)

    np.testing.assert_array_equal(first.training.times, second.training.times)
    np.testing.assert_array_equal(first.validation.times, second.validation.times)


def test_session1_split_rejects_evaluation_data() -> None:
    trials = _session1_trials()
    evaluation = Dataset2ATrialSignalResult(
        signals=trials.signals,
        labels=np.full(trials.labels.shape, -1, dtype=int),
        times=trials.times,
        cue_descriptions=np.full(trials.labels.shape, "783", dtype="U8"),
        channel_names=trials.channel_names,
        sampling_frequency=trials.sampling_frequency,
        session_id="E",
    )

    with pytest.raises(ValueError, match="Session-I/T"):
        split_dataset_2a_session1(evaluation)
