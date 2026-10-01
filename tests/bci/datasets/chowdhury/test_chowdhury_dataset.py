from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.bci.datasets.chowdhury.data import (
    ChowdhuryEEGSession,
    load_patient_demographics,
    resolve_eeg_path,
)
from src.bci.datasets.chowdhury.experiment import (
    ChowdhuryExperimentConfig,
    prepare_subject_experiment,
)
from src.bci.datasets.chowdhury.metadata import (
    CHOWDHURY_PARTICIPANT_IDS,
    ChowdhuryParticipant,
    participant_index,
    validate_participant_id,
)


DEMOGRAPHIC_ROWS = [
    ["S01", 48, "Male", "Left", "Right", 8],
    ["S02", 71, "Male", "Left", "Right", 20],
]


def test_chowdhury_namespace_exposes_dataset_entry_points() -> None:
    from src.bci.datasets.chowdhury import (
        ChowdhuryParticipant as ExportedParticipant,
        load_patient_demographics as exported_loader,
        prepare_subject_experiment as exported_prepare,
    )

    assert ExportedParticipant is ChowdhuryParticipant
    assert exported_loader is load_patient_demographics
    assert exported_prepare is prepare_subject_experiment


def _write_demographics(directory: Path, rows: list[list[object]] | None = None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        DEMOGRAPHIC_ROWS if rows is None else rows,
        columns=[
            "participant_id",
            "age_years",
            "gender",
            "impaired_side",
            "dominant_side",
            "time_since_stroke_months",
        ],
    ).to_csv(directory / "patient_demographics.csv", index=False)


def test_participant_metadata_normalises_published_categories() -> None:
    participant = ChowdhuryParticipant("s03", 63, "male", "right", "RIGHT", 8)

    assert participant.participant_id == "S03"
    assert participant.gender == "Male"
    assert participant.impaired_hand == "Right"
    assert participant.is_dominant_side_impaired is True


@pytest.mark.parametrize("participant_id", ["S00", "S11", "1", "subject01"])
def test_validate_participant_id_rejects_out_of_cohort_values(
    participant_id: str,
) -> None:
    with pytest.raises(ValueError, match="S01 to S10"):
        validate_participant_id(participant_id)


def test_participant_index_rejects_duplicate_rows() -> None:
    participant = ChowdhuryParticipant("S01", 48, "Male", "Left", "Right", 8)
    with pytest.raises(ValueError, match="duplicate participant_id"):
        participant_index([participant, participant])


def test_load_patient_demographics_returns_typed_records(tmp_path: Path) -> None:
    _write_demographics(tmp_path)

    participants = load_patient_demographics(tmp_path)

    assert tuple(item.participant_id for item in participants) == ("S01", "S02")
    assert participants[1].time_since_stroke_months == 20


def test_repository_demographics_contains_published_ten_patient_cohort() -> None:
    data_directory = Path("data/raw/chowdhury_cse_uael")

    participants = load_patient_demographics(data_directory)

    assert tuple(item.participant_id for item in participants) == (
        CHOWDHURY_PARTICIPANT_IDS
    )
    assert sum(item.gender == "Female" for item in participants) == 4
    assert sum(item.is_dominant_side_impaired for item in participants) == 3


def test_load_patient_demographics_reports_missing_column(tmp_path: Path) -> None:
    pd.DataFrame({"participant_id": ["S01"]}).to_csv(
        tmp_path / "patient_demographics.csv", index=False
    )

    with pytest.raises(ValueError, match="missing columns"):
        load_patient_demographics(tmp_path)


def test_resolve_eeg_path_uses_explicit_filename_and_participant_folder(
    tmp_path: Path,
) -> None:
    expected = tmp_path / "eeg" / "S01" / "recording.mat"
    expected.parent.mkdir(parents=True)
    expected.write_bytes(b"placeholder")

    assert resolve_eeg_path(tmp_path, "s01", "recording.mat") == expected


def test_eeg_session_validates_samples_by_channels_shape() -> None:
    session = ChowdhuryEEGSession(
        participant_id="S01",
        session_id="online-1",
        file_path=Path("recording.mat"),
        signals=np.ones((10, 3)),
        times=np.arange(10) / 250.0,
        sampling_frequency=250.0,
        channel_names=("C3", "Cz", "C4"),
    )

    assert session.n_samples == 10
    assert session.n_channels == 3


def test_prepare_subject_experiment_joins_metadata_to_features(
    tmp_path: Path,
) -> None:
    _write_demographics(tmp_path)
    config = ChowdhuryExperimentConfig(participants=("S01", "S02"))

    experiment = prepare_subject_experiment(
        "s01",
        tmp_path,
        config,
        training_features=np.ones((4, 2)),
        testing_features=np.ones((3, 2)),
        testing_times=np.arange(3),
    )

    assert experiment.participant.participant_id == "S01"
    assert experiment.training_features.shape == (4, 2)


def test_prepare_subject_experiment_respects_configured_cohort(
    tmp_path: Path,
) -> None:
    _write_demographics(tmp_path)
    config = ChowdhuryExperimentConfig(participants=("S02",))

    with pytest.raises(ValueError, match="not enabled"):
        prepare_subject_experiment(
            "S01",
            tmp_path,
            config,
            training_features=np.ones((4, 2)),
            testing_features=np.ones((3, 2)),
            testing_times=np.arange(3),
        )
