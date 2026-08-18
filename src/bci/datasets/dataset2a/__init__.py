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
    build_dataset_2a_fbcsp_features,
    extract_dataset_2a_trials as _extract_dataset_2a_trials_unfiltered,
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


def extract_dataset_2a_trials(
    session,
    *,
    evaluation_labels: np.ndarray | None = None,
    seconds_after_cue: float = 3.0,
) -> Dataset2ATrialSignalResult:
    """Extract paper-compatible Dataset 2A trials.

    Session-I training data uses the labelled 769/770 left/right cues already
    present in the GDF. Session-II GDF files expose all evaluation cues as 783,
    so the official released class labels are required to reproduce the paper's
    binary left-vs-right experiment. Classes 3 (feet) and 4 (tongue) are removed
    only after each 783 cue has been paired with its corresponding released
    label, preserving cue/label alignment.
    """

    extracted = _extract_dataset_2a_trials_unfiltered(
        session,
        seconds_after_cue=seconds_after_cue,
    )

    is_evaluation = str(session.session).upper() == "E"
    if not is_evaluation:
        return extracted

    if evaluation_labels is None:
        raise ValueError(
            "Dataset 2A Session-II requires official evaluation labels "
            "for the paper reproduction."
        )

    labels = np.asarray(evaluation_labels, dtype=int).reshape(-1)
    if labels.size != extracted.signals.shape[0]:
        raise ValueError(
            "Dataset 2A evaluation cue/label mismatch: "
            f"found {extracted.signals.shape[0]} evaluation cues but "
            f"received {labels.size} labels."
        )
    if not np.isin(labels, [1, 2, 3, 4]).all():
        raise ValueError("Dataset 2A evaluation labels must be in {1, 2, 3, 4}.")

    keep = np.isin(labels, [1, 2])
    if not np.any(keep):
        raise ValueError("no Dataset 2A left/right evaluation trials were found.")

    selected_signals = np.asarray(extracted.signals[keep], dtype=float)
    selected_labels = np.where(labels[keep] == 1, 0, 1).astype(int)
    selected_descriptions = np.asarray(extracted.cue_descriptions[keep])

    return Dataset2ATrialSignalResult(
        signals=selected_signals,
        labels=selected_labels,
        times=np.arange(selected_signals.shape[0], dtype=int),
        cue_descriptions=selected_descriptions,
        channel_names=extracted.channel_names,
        sampling_frequency=float(extracted.sampling_frequency),
        session_id=extracted.session_id,
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
