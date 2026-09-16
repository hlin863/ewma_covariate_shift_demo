from pathlib import Path

import numpy as np

from src.bci_2b_experiment import (
    PUBLISHED_2B_RESULTS,
    build_dataset_2b_fbcsp_features,
    concatenate_trial_signals,
    extract_dataset_2b_trials,
    run_dataset_2b_subject,
)
from src.bci_data import BCISessionData


def _session(
    session: str,
    evaluation: bool = False,
    channel_names: tuple[str, str, str] = ("C3", "Cz", "C4"),
) -> BCISessionData:
    rng = np.random.default_rng(7 + int(session[:2]))
    sampling_frequency = 100.0
    signals = rng.normal(scale=0.4, size=(5000, 3))
    annotations = []
    for index, onset in enumerate((2, 7, 12, 17, 22, 27, 32, 37)):
        code = "783" if evaluation else ("769" if index % 2 == 0 else "770")
        start = int(onset * sampling_frequency)
        stop = start + int(3 * sampling_frequency)
        time = np.arange(stop - start) / sampling_frequency
        channel = 0 if index % 2 == 0 else 2
        signals[start:stop, channel] += 2.0 * np.sin(2 * np.pi * 10.0 * time)
        annotations.append((float(onset), 0.0, code))
    return BCISessionData(
        subject=1,
        session=session,
        file_path=Path(f"B01{session}.gdf"),
        signals=signals,
        times=np.arange(signals.shape[0]) / sampling_frequency,
        channel_names=channel_names,
        sampling_frequency=sampling_frequency,
        annotations=tuple(annotations),
    )


def test_published_dataset_2b_targets_are_complete() -> None:
    assert set(PUBLISHED_2B_RESULTS) == {
        f"B{subject:02d}" for subject in range(1, 10)
    }
    assert PUBLISHED_2B_RESULTS["B01"] == (0.28, 14, 10)
    assert PUBLISHED_2B_RESULTS["B09"] == (0.45, 18, 7)


def test_extract_dataset_2b_trials_returns_raw_cue_tensors() -> None:
    result = extract_dataset_2b_trials(_session("01T"))
    assert result.signals.shape == (8, 3, 300)
    assert result.times.shape == (8,)
    assert result.channel_names == ("C3", "Cz", "C4")
    np.testing.assert_array_equal(result.labels, [0, 1, 0, 1, 0, 1, 0, 1])
    assert np.isfinite(result.signals).all()


def test_extract_dataset_2b_trials_accepts_mne_prefixed_channel_names() -> None:
    result = extract_dataset_2b_trials(
        _session("01T", channel_names=("EEG:C3", "EEG:Cz", "EEG:C4"))
    )
    assert result.signals.shape == (8, 3, 300)
    assert result.channel_names == ("C3", "Cz", "C4")


def test_evaluation_unknown_cues_do_not_become_training_classes() -> None:
    result = extract_dataset_2b_trials(_session("04E", evaluation=True))
    np.testing.assert_array_equal(result.labels, np.full(8, -1))


def test_concatenate_trial_signals_assigns_unique_times() -> None:
    first = extract_dataset_2b_trials(_session("01T"))
    second = extract_dataset_2b_trials(_session("02T"))
    result = concatenate_trial_signals([first, second])
    assert result.signals.shape == (16, 3, 300)
    np.testing.assert_array_equal(result.times, np.arange(16))
    assert np.unique(result.times).size == 16


def test_fbcsp_pipeline_produces_twenty_paper_features() -> None:
    training = concatenate_trial_signals(
        [
            extract_dataset_2b_trials(_session("01T")),
            extract_dataset_2b_trials(_session("02T")),
            extract_dataset_2b_trials(_session("03T")),
        ]
    )
    testing = concatenate_trial_signals(
        [
            extract_dataset_2b_trials(_session("04E", evaluation=True)),
            extract_dataset_2b_trials(_session("05E", evaluation=True)),
        ]
    )
    pipeline = build_dataset_2b_fbcsp_features(training, testing)
    assert pipeline.training.features.shape == (24, 20)
    assert pipeline.testing.features.shape == (16, 20)
    assert len(pipeline.training.feature_names) == 20
    assert np.isfinite(pipeline.training.features).all()
    assert np.isfinite(pipeline.testing.features).all()


def test_subject_experiment_uses_algorithm1_configuration() -> None:
    training_trials = concatenate_trial_signals(
        [
            extract_dataset_2b_trials(_session("01T")),
            extract_dataset_2b_trials(_session("02T")),
            extract_dataset_2b_trials(_session("03T")),
        ]
    )
    testing_trials = concatenate_trial_signals(
        [
            extract_dataset_2b_trials(_session("04E", evaluation=True)),
            extract_dataset_2b_trials(_session("05E", evaluation=True)),
        ]
    )
    pipeline = build_dataset_2b_fbcsp_features(training_trials, testing_trials)
    result = run_dataset_2b_subject(1, pipeline.training, pipeline.testing)

    assert result.subject == "B01"
    assert result.published_lambda == 0.28
    assert result.cse_result.effective_lambda == 0.28
    assert result.published_csw == 14
    assert result.published_csv == 10
    assert result.computed_csw == int(
        result.cse_result.warning_results["stage_1_alarm"].sum()
    )
    assert result.computed_csv == int(
        result.cse_result.validation_results["confirmed_shift"].sum()
    )
    assert set(result.cse_result.validation_results["status"]).issubset(
        {"confirmed", "rejected"}
    )
    assert "training_size" in result.cse_result.validation_results.columns
    assert "before_start_time" not in result.cse_result.validation_results.columns
