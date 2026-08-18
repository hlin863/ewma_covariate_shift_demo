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


def extract_dataset_2a_trials(
    session,
    *,
    evaluation_labels: np.ndarray | None = None,
    seconds_after_cue: float = 3.0,
) -> Dataset2ATrialSignalResult:
    """Extract paper-compatible Dataset 2A left/right trials.

    Session-I uses the labelled 769/770 cues in the GDF. Session-II exposes
    evaluation cues as 783, so official released labels are required to retain
    only classes 1 (left) and 2 (right). The wrapper supports both the original
    extractor, which returns all Session-II cues unlabelled, and a directly
    integrated extractor that already accepts ``evaluation_labels``.
    """

    is_evaluation = str(session.session).upper() == "E"
    if is_evaluation and evaluation_labels is None:
        raise ValueError(
            "Dataset 2A Session-II requires official evaluation labels "
            "for the paper reproduction."
        )

    labels = None
    if is_evaluation:
        labels = np.asarray(evaluation_labels, dtype=int).reshape(-1)
        if not np.isin(labels, [1, 2, 3, 4]).all():
            raise ValueError("Dataset 2A evaluation labels must be in {1, 2, 3, 4}.")

    try:
        extracted = _extract_dataset_2a_trials_impl(
            session,
            evaluation_labels=labels,
            seconds_after_cue=seconds_after_cue,
        )
        implementation_accepts_labels = True
    except TypeError as error:
        if "evaluation_labels" not in str(error):
            raise
        extracted = _extract_dataset_2a_trials_impl(
            session,
            seconds_after_cue=seconds_after_cue,
        )
        implementation_accepts_labels = False

    if not is_evaluation:
        return extracted

    # A directly integrated implementation has already filtered the four-class
    # evaluation stream and mapped class IDs 1/2 to binary labels 0/1.
    if implementation_accepts_labels and np.all(np.asarray(extracted.labels) >= 0):
        if not np.isin(extracted.labels, [0, 1]).all():
            raise ValueError("filtered Dataset 2A evaluation labels must be binary 0/1.")
        return extracted

    # Compatibility path for the original extractor: one unlabelled row per
    # Session-II 783 cue is returned, so pair labels in cue order before filtering.
    if labels.size != extracted.signals.shape[0]:
        raise ValueError(
            "Dataset 2A evaluation cue/label mismatch: "
            f"found {extracted.signals.shape[0]} evaluation cues but "
            f"received {labels.size} labels."
        )

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
