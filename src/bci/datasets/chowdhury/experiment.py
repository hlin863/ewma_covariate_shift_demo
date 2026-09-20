"""Experiment boundary for applying the shared CSE pipeline to this cohort.

The module validates cohort selection and prepared feature matrices. It does
not claim a raw-EEG trial protocol that is absent from the demographic table.
Once the source recordings are available, their dataset-specific extraction
can feed the same ``prepare_subject_experiment`` function.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.bci.datasets.chowdhury.data import load_patient_demographics
from src.bci.datasets.chowdhury.metadata import (
    CHOWDHURY_PARTICIPANT_IDS,
    ChowdhuryParticipant,
    participant_index,
    validate_participant_id,
)
from src.detection import CSEConfig, CSEResult, run_cse


@dataclass(frozen=True)
class ChowdhuryExperimentConfig:
    """Verified cohort choices, separate from still-unknown EEG parameters."""

    participants: tuple[str, ...] = CHOWDHURY_PARTICIPANT_IDS
    use_impaired_hand: bool = True
    apply_cse: bool = True

    def __post_init__(self) -> None:
        participants = tuple(validate_participant_id(item) for item in self.participants)
        if not participants:
            raise ValueError("participants must not be empty.")
        if len(set(participants)) != len(participants):
            raise ValueError("participants must not contain duplicate IDs.")
        object.__setattr__(self, "participants", participants)


@dataclass(frozen=True)
class ChowdhurySubjectExperiment:
    """One participant's precomputed train/test features for shared CSE."""

    participant: ChowdhuryParticipant
    training_features: np.ndarray
    testing_features: np.ndarray
    testing_times: np.ndarray

    def __post_init__(self) -> None:
        training = np.asarray(self.training_features, dtype=float)
        testing = np.asarray(self.testing_features, dtype=float)
        times = np.asarray(self.testing_times)
        if training.ndim != 2 or training.shape[0] < 2 or training.shape[1] < 1:
            raise ValueError(
                "training_features must contain at least two observations and one feature."
            )
        if testing.ndim != 2 or testing.shape[0] < 1:
            raise ValueError("testing_features must be a non-empty two-dimensional matrix.")
        if training.shape[1] != testing.shape[1]:
            raise ValueError("training and testing feature counts must match.")
        if times.ndim != 1 or times.size != testing.shape[0]:
            raise ValueError("testing_times must contain one value per testing observation.")
        if np.unique(times).size != times.size:
            raise ValueError("testing_times must uniquely identify observations.")
        if not np.isfinite(training).all() or not np.isfinite(testing).all():
            raise ValueError("feature matrices must contain only finite values.")
        object.__setattr__(self, "training_features", training)
        object.__setattr__(self, "testing_features", testing)
        object.__setattr__(self, "testing_times", times)


@dataclass(frozen=True)
class ChowdhuryExperimentResult:
    participant_id: str
    warning_count: int
    confirmed_shift_count: int
    cse_result: CSEResult


def prepare_subject_experiment(
    participant_id: str,
    data_directory: str | Path,
    config: ChowdhuryExperimentConfig,
    *,
    training_features: np.ndarray,
    testing_features: np.ndarray,
    testing_times: np.ndarray,
) -> ChowdhurySubjectExperiment:
    """Join validated metadata to features produced by a dataset adapter."""

    canonical_id = validate_participant_id(participant_id)
    if canonical_id not in config.participants:
        raise ValueError(f"participant {canonical_id} is not enabled by the config.")
    participants = participant_index(load_patient_demographics(data_directory))
    if canonical_id not in participants:
        raise ValueError(f"participant {canonical_id} is missing from the demographics.")
    return ChowdhurySubjectExperiment(
        participant=participants[canonical_id],
        training_features=training_features,
        testing_features=testing_features,
        testing_times=testing_times,
    )


def run_chowdhury_subject(
    experiment: ChowdhurySubjectExperiment,
    *,
    cse_config: CSEConfig | None = None,
) -> ChowdhuryExperimentResult:
    """Run the repository's dataset-agnostic CSE detector on prepared features."""

    result = run_cse(
        training_features=experiment.training_features,
        testing_features=experiment.testing_features,
        testing_times=experiment.testing_times,
        config=CSEConfig() if cse_config is None else cse_config,
    )
    return ChowdhuryExperimentResult(
        participant_id=experiment.participant.participant_id,
        warning_count=int(result.warning_results["stage_1_alarm"].astype(bool).sum()),
        confirmed_shift_count=int(
            result.validation_results["confirmed_shift"].astype(bool).sum()
        ),
        cse_result=result,
    )
