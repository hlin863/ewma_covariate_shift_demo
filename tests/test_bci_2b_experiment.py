from pathlib import Path

import numpy as np

from src.bci_2b_experiment import (
    PUBLISHED_2B_RESULTS,
    concatenate_trial_features,
    extract_dataset_2b_trials,
    run_dataset_2b_subject,
)
from src.bci_data import BCISessionData


def _session(session: str, shift: float = 0.0) -> BCISessionData:
    rng = np.random.default_rng(7 + int(session[:2]))
    sampling_frequency = 100.0
    signals = rng.normal(
        loc=shift,
        scale=1.0,
        size=(5000, 3),
    )
    annotations = tuple(
        (float(onset), 0.0, code)
        for onset, code in zip(
            (2, 7, 12, 17, 22, 27, 32, 37),
            ("769", "770", "769", "770", "769", "770", "769", "770"),
        )
    )
    return BCISessionData(
        subject=1,
        session=session,
        file_path=Path(f"B01{session}.gdf"),
        signals=signals,
        times=np.arange(signals.shape[0]) / sampling_frequency,
        channel_names=("C3", "Cz", "C4"),
        sampling_frequency=sampling_frequency,
        annotations=annotations,
    )


def test_published_dataset_2b_targets_are_complete() -> None:
    assert set(PUBLISHED_2B_RESULTS) == {
        f"B{subject:02d}" for subject in range(1, 10)
    }
    assert PUBLISHED_2B_RESULTS["B01"] == (0.28, 14, 10)
    assert PUBLISHED_2B_RESULTS["B09"] == (0.45, 18, 7)


def test_extract_dataset_2b_trials_uses_one_row_per_cue() -> None:
    result = extract_dataset_2b_trials(_session("01T"))

    assert result.features.shape == (8, 6)
    assert result.times.shape == (8,)
    assert result.feature_names == (
        "C3_mu_8_12",
        "C3_beta_14_30",
        "Cz_mu_8_12",
        "Cz_beta_14_30",
        "C4_mu_8_12",
        "C4_beta_14_30",
    )
    assert np.isfinite(result.features).all()


def test_concatenate_trial_features_assigns_unique_times() -> None:
    first = extract_dataset_2b_trials(_session("01T"))
    second = extract_dataset_2b_trials(_session("02T"))

    result = concatenate_trial_features([first, second])

    assert result.features.shape == (16, 6)
    np.testing.assert_array_equal(result.times, np.arange(16))
    assert np.unique(result.times).size == 16


def test_subject_experiment_uses_published_lambda_without_copying_counts() -> None:
    training = concatenate_trial_features(
        [
            extract_dataset_2b_trials(_session("01T")),
            extract_dataset_2b_trials(_session("02T")),
            extract_dataset_2b_trials(_session("03T")),
        ]
    )
    testing = concatenate_trial_features(
        [
            extract_dataset_2b_trials(_session("04E", shift=2.0)),
            extract_dataset_2b_trials(_session("05E", shift=2.0)),
        ]
    )

    result = run_dataset_2b_subject(
        1,
        training,
        testing,
        validation_before_size=3,
        validation_after_size=3,
    )

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
