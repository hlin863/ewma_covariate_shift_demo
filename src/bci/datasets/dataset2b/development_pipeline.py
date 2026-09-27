"""Fit FBCSP on development training; transform held-out trials without fitting."""

from dataclasses import dataclass

import numpy as np

from src.bci.datasets.dataset2b.development_split import Dataset2BDevelopmentSplit
from src.bci.datasets.dataset2b.experiment import TrialFeatureResult, TrialSignalResult, _feature_names
from src.bci.fbcsp import FBCSPModel, fit_fbcsp, transform_fbcsp


@dataclass(frozen=True)
class Dataset2BDevelopmentFeaturePipelineResult:
    model: FBCSPModel
    training: TrialFeatureResult
    validation: TrialFeatureResult
    evaluation: TrialFeatureResult

    @property
    def testing(self) -> TrialFeatureResult:
        """Compatibility with existing detection/reporting callers."""
        return self.evaluation


def _as_features(trials: TrialSignalResult, model: FBCSPModel) -> TrialFeatureResult:
    return TrialFeatureResult(
        features=transform_fbcsp(trials.signals, model),
        times=trials.times,
        feature_names=_feature_names(model),
        session_ids=trials.session_ids,
        cue_descriptions=trials.cue_descriptions,
    )


def build_dataset_2b_development_fbcsp_features(
    split: Dataset2BDevelopmentSplit,
    evaluation: TrialSignalResult,
    *,
    components_per_side: int = 1,
) -> Dataset2BDevelopmentFeaturePipelineResult:
    training, validation = split.training, split.validation
    if any(item.channel_names != training.channel_names for item in (validation, evaluation)):
        raise ValueError("training, validation, and evaluation channel orders must match.")
    if any(not np.isclose(item.sampling_frequency, training.sampling_frequency)
           for item in (validation, evaluation)):
        raise ValueError("training, validation, and evaluation sampling frequencies must match.")
    if np.any(training.labels < 0) or np.any(validation.labels < 0):
        raise ValueError("development training and validation must be labelled.")
    if not np.isin(evaluation.session_ids, ["04E", "05E"]).all():
        raise ValueError("evaluation must contain only Sessions IV–V.")

    model = fit_fbcsp(
        training.signals, training.labels, training.sampling_frequency,
        components_per_side=components_per_side,
    )
    return Dataset2BDevelopmentFeaturePipelineResult(
        model=model,
        training=_as_features(training, model),
        validation=_as_features(validation, model),
        evaluation=_as_features(evaluation, model),
    )
