"""Adaptive-learning components for CSD-triggered BCI experiments.

The package sits downstream of ``src.detection``.  Drift detectors decide what
changed; adaptation policies decide whether that evidence should cause a model
update; classifier wrappers implement the update itself.
"""

from src.adaptation.classifier import Classifier, LinearSVMClassifier, RetrainableClassifier
from src.adaptation.ensemble import BaggingClassifier
from src.adaptation.policies import (
    AdaptationContext,
    AdaptationPolicy,
    NeverUpdate,
    PeriodicRetrain,
    RetrainOnValidatedShift,
    RetrainOnWarning,
)
from src.adaptation.supervised import (
    SupervisedAdaptationConfig,
    SupervisedAdaptationResult,
    run_supervised_adaptation,
)

__all__ = [
    "AdaptationContext",
    "AdaptationPolicy",
    "BaggingClassifier",
    "Classifier",
    "LinearSVMClassifier",
    "NeverUpdate",
    "PeriodicRetrain",
    "RetrainOnValidatedShift",
    "RetrainOnWarning",
    "RetrainableClassifier",
    "SupervisedAdaptationConfig",
    "SupervisedAdaptationResult",
    "run_supervised_adaptation",
]
