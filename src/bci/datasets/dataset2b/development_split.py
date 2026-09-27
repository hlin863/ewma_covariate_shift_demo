"""The 2019 Dataset 2B session protocol and its 70/30 development split."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.bci.datasets.dataset2b.experiment import TrialSignalResult, concatenate_trial_signals
from src.bci.splitting import stratified_split_indices


@dataclass(frozen=True)
class Dataset2BDevelopmentSplit:
    training: TrialSignalResult
    validation: TrialSignalResult


@dataclass(frozen=True)
class Dataset2BCSEUAELProtocol:
    development: Dataset2BDevelopmentSplit
    evaluation: TrialSignalResult


def _subset_trials(trials: TrialSignalResult, indices: np.ndarray) -> TrialSignalResult:
    return TrialSignalResult(
        signals=trials.signals[indices],
        labels=trials.labels[indices],
        times=trials.times[indices],
        session_ids=trials.session_ids[indices],
        cue_descriptions=trials.cue_descriptions[indices],
        channel_names=trials.channel_names,
        sampling_frequency=trials.sampling_frequency,
    )


def split_dataset_2b_training_pool(
    trials: TrialSignalResult,
    *,
    validation_fraction: float = 0.30,
    random_state: int = 42,
) -> Dataset2BDevelopmentSplit:
    if trials.labels.shape != (trials.signals.shape[0],):
        raise ValueError("training pool must contain one label per trial.")
    if not np.isin(trials.session_ids, ["01T", "02T", "03T"]).all():
        raise ValueError("2019 training pool must contain only Sessions I–III.")
    indices = stratified_split_indices(
        trials.labels, validation_fraction=validation_fraction, random_state=random_state
    )
    return Dataset2BDevelopmentSplit(
        training=_subset_trials(trials, indices.training),
        validation=_subset_trials(trials, indices.validation),
    )


def prepare_cse_uael_2019_protocol(
    sessions: Sequence[TrialSignalResult],
    *,
    validation_fraction: float = 0.30,
    random_state: int = 42,
) -> Dataset2BCSEUAELProtocol:
    """I–III development pool, IV–V evaluation; partition the pool only once."""
    if len(sessions) != 5:
        raise ValueError("Dataset 2B requires five ordered sessions.")
    for number, session in enumerate(sessions, start=1):
        expected = f"{number:02d}{'T' if number <= 3 else 'E'}"
        if session.signals.shape[0] == 0 or not np.all(session.session_ids == expected):
            raise ValueError(f"session {number} must be a nonempty {expected} recording.")
    training_pool = concatenate_trial_signals(list(sessions[:3]))
    evaluation = concatenate_trial_signals(list(sessions[3:]))
    return Dataset2BCSEUAELProtocol(
        development=split_dataset_2b_training_pool(
            training_pool,
            validation_fraction=validation_fraction,
            random_state=random_state,
        ),
        evaluation=evaluation,
    )
