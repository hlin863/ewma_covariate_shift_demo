"""BCI Competition IV Dataset 2A reproduction helpers."""

import numpy as np

from src.bci.datasets.dataset2a.experiment import (
    DATASET_2A_CHANNELS,
    DATASET_2A_EEG_MONTAGE,
    PUBLISHED_2A_RESULTS,
    Dataset2AExperimentResult,
    Dataset2AFeaturePipelineResult,
    Dataset2ATrialFeatureResult,
    Dataset2ATrialSignalResult,
    _EVALUATION_CUE,
    _dataset_2a_channel_indices,
    build_dataset_2a_fbcsp_features,
    extract_dataset_2a_trials as _extract_dataset_2a_trials_impl,
    run_dataset_2a_subject,
    split_dataset_2a_session1,
)
from src.bci.datasets.dataset2a.development_pipeline import (
    Dataset2ADevelopmentFeaturePipelineResult,
    build_dataset_2a_development_fbcsp_features,
)
from src.bci.datasets.dataset2a.development_split import Dataset2ADevelopmentSplit
from src.bci.datasets.dataset2a.validation_calibration import (
    Dataset2AStage1ValidationCalibration,
    calibrate_stage1_control_limit_from_validation,
)


def _extract_unlabelled_dataset_2a_evaluation(
    session,
    *,
    seconds_after_cue: float,
) -> Dataset2ATrialSignalResult:
    """Extract all Session-II 783 cues when no released labels are available.

    This preserves the original GDF-only behaviour. The returned labels are -1,
    making it explicit that the 288 evaluation trials cannot be separated into
    left/right/feet/tongue from the evaluation GDF alone.
    """

    if seconds_after_cue <= 0.0:
        raise ValueError("seconds_after_cue must be positive.")

    sfreq = float(session.sampling_frequency)
    segment_size = int(round(seconds_after_cue * sfreq))
    channel_indices = _dataset_2a_channel_indices(session.channel_names)

    rows: list[np.ndarray] = []
    descriptions: list[str] = []
    for onset, _, description in session.annotations:
        code = str(description).strip()
        if code != _EVALUATION_CUE:
            continue

        start = int(round(float(onset) * sfreq))
        stop = start + segment_size
        if start < 0 or stop > session.signals.shape[0]:
            continue

        segment = session.signals[start:stop, channel_indices].T
        if not np.isfinite(segment).all():
            continue

        rows.append(segment)
        descriptions.append(code)

    if not rows:
        raise ValueError("no finite Dataset 2A Session-II evaluation trials were found.")

    signals = np.stack(rows)
    return Dataset2ATrialSignalResult(
        signals=signals,
        labels=np.full(signals.shape[0], -1, dtype=int),
        times=np.arange(signals.shape[0], dtype=int),
        cue_descriptions=np.asarray(descriptions, dtype="U8"),
        channel_names=DATASET_2A_CHANNELS,
        sampling_frequency=sfreq,
        session_id=str(session.session),
    )


def extract_dataset_2a_trials(
    session,
    *,
    evaluation_labels: np.ndarray | None = None,
    seconds_after_cue: float = 3.0,
) -> Dataset2ATrialSignalResult:
    """Extract Dataset 2A trials with an explicit GDF-only fallback.

    Session-I uses labelled 769/770 cues in the GDF. For Session-II, if official
    released labels are supplied, classes 1/2 are retained for the paper's
    left-vs-right experiment. If they are not supplied, all 783 evaluation cues
    are retained with label -1 so the existing GDF-only experiment can still run.
    """

    is_evaluation = str(session.session).upper() == "E"
    if not is_evaluation:
        return _extract_dataset_2a_trials_impl(
            session,
            seconds_after_cue=seconds_after_cue,
        )

    if evaluation_labels is None:
        return _extract_unlabelled_dataset_2a_evaluation(
            session,
            seconds_after_cue=seconds_after_cue,
        )

    labels = np.asarray(evaluation_labels, dtype=int).reshape(-1)
    if not np.isin(labels, [1, 2, 3, 4]).all():
        raise ValueError("Dataset 2A evaluation labels must be in {1, 2, 3, 4}.")

    return _extract_dataset_2a_trials_impl(
        session,
        evaluation_labels=labels,
        seconds_after_cue=seconds_after_cue,
    )


__all__ = [
    "DATASET_2A_CHANNELS",
    "DATASET_2A_EEG_MONTAGE",
    "PUBLISHED_2A_RESULTS",
    "Dataset2ADevelopmentFeaturePipelineResult",
    "Dataset2ADevelopmentSplit",
    "Dataset2AExperimentResult",
    "Dataset2AFeaturePipelineResult",
    "Dataset2AStage1ValidationCalibration",
    "Dataset2ATrialFeatureResult",
    "Dataset2ATrialSignalResult",
    "build_dataset_2a_development_fbcsp_features",
    "build_dataset_2a_fbcsp_features",
    "calibrate_stage1_control_limit_from_validation",
    "extract_dataset_2a_trials",
    "run_dataset_2a_subject",
    "split_dataset_2a_session1",
]
