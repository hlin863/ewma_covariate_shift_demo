"""Dataset 2A Session-I 70/30 development feature pipeline.

The 2019 CSE-UAEL paper states that the existing training dataset was further
partitioned into 70% training and 30% validation subsets. This module enforces
that boundary at the feature-learning stage: FBCSP is fitted only on the 70%
Session-I development-training subset, while the held-out 30% validation subset
and Session-II evaluation data are transformed with the already-fitted model.
"""

from dataclasses import dataclass

import numpy as np

from src.bci_2a_development_split import Dataset2ADevelopmentSplit
from src.bci_2a_experiment import (
    Dataset2ATrialFeatureResult,
    Dataset2ATrialSignalResult,
)
from src.fbcsp import FBCSPModel, fit_fbcsp, transform_fbcsp


@dataclass(frozen=True)
class Dataset2ADevelopmentFeaturePipelineResult:
    """FBCSP outputs for the paper-oriented Session-I development protocol."""

    model: FBCSPModel
    training: Dataset2ATrialFeatureResult
    validation: Dataset2ATrialFeatureResult
    testing: Dataset2ATrialFeatureResult


def _feature_names(model: FBCSPModel) -> tuple[str, ...]:
    names: list[str] = []
    for band_model in model.models:
        band = f"{band_model.low_hz:g}_{band_model.high_hz:g}Hz"
        for index in range(model.components_per_side):
            names.append(f"{band}_csp_high_{index + 1}")
        for index in range(model.components_per_side):
            names.append(f"{band}_csp_low_{index + 1}")
    return tuple(names)


def _as_feature_result(
    trials: Dataset2ATrialSignalResult,
    features: np.ndarray,
    names: tuple[str, ...],
) -> Dataset2ATrialFeatureResult:
    return Dataset2ATrialFeatureResult(
        features=np.asarray(features, dtype=float),
        times=np.asarray(trials.times),
        feature_names=names,
        cue_descriptions=np.asarray(trials.cue_descriptions),
        session_id=trials.session_id,
    )


def build_dataset_2a_development_fbcsp_features(
    split: Dataset2ADevelopmentSplit,
    testing: Dataset2ATrialSignalResult,
    *,
    components_per_side: int = 1,
) -> Dataset2ADevelopmentFeaturePipelineResult:
    """Fit FBCSP on the 70% subset and transform validation/test without refitting.

    The 30% validation subset is deliberately excluded from CSP fitting. This
    prevents validation information leaking into the spatial filters and keeps
    Session-II evaluation completely outside the Session-I model-fitting path.
    """

    training = split.training
    validation = split.validation

    if str(training.session_id).upper() != "T":
        raise ValueError("development training data must come from Dataset 2A Session-I/T.")
    if str(validation.session_id).upper() != "T":
        raise ValueError("development validation data must come from Dataset 2A Session-I/T.")
    if str(testing.session_id).upper() != "E":
        raise ValueError("testing data must come from Dataset 2A Session-II/E.")
    if training.channel_names != validation.channel_names or training.channel_names != testing.channel_names:
        raise ValueError("training, validation, and testing channel orders must match.")
    if not np.isclose(training.sampling_frequency, validation.sampling_frequency) or not np.isclose(
        training.sampling_frequency, testing.sampling_frequency
    ):
        raise ValueError("training, validation, and testing sampling frequencies must match.")
    if np.any(training.labels < 0):
        raise ValueError("development training trials must be labelled left/right trials.")
    if np.any(validation.labels < 0):
        raise ValueError("development validation trials must be labelled left/right trials.")

    model = fit_fbcsp(
        training.signals,
        training.labels,
        training.sampling_frequency,
        components_per_side=components_per_side,
    )
    training_features = transform_fbcsp(training.signals, model)
    validation_features = transform_fbcsp(validation.signals, model)
    testing_features = transform_fbcsp(testing.signals, model)
    names = _feature_names(model)

    return Dataset2ADevelopmentFeaturePipelineResult(
        model=model,
        training=_as_feature_result(training, training_features, names),
        validation=_as_feature_result(validation, validation_features, names),
        testing=_as_feature_result(testing, testing_features, names),
    )
