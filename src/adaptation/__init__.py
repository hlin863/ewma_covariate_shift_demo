"""Adaptive-learning components for CSD-triggered BCI experiments.

Module boundaries preserve the literature lineage: supervised adaptation for
Chowdhury et al. (2018), then PWKNN/transductive knowledge acquisition toward
Raza et al. (2019), with post-run evaluation kept separate from learner input.
"""

from src.adaptation.classifier import Classifier, LinearSVMClassifier, RetrainableClassifier
from src.adaptation.ensemble import BaggingClassifier
from src.adaptation.evaluation import (
    TransductiveEvaluationResult, UnsupervisedEvaluationResult,
    evaluate_transductive_adaptation, evaluate_unsupervised_adaptation,
)
from src.adaptation.policies import (
    AdaptationContext, AdaptationPolicy, NeverUpdate, PeriodicRetrain,
    RetrainOnValidatedShift, RetrainOnWarning,
)
from src.adaptation.pseudo_labelling import PseudoLabelBatch, PseudoLabeler, PWKNNPseudoLabeler
from src.adaptation.supervised import SupervisedAdaptationConfig, SupervisedAdaptationResult, run_supervised_adaptation
from src.adaptation.transductive import (
    TransductiveAdaptationConfig, TransductiveAdaptationResult,
    UnsupervisedAdaptationConfig, UnsupervisedAdaptationResult,
    run_transductive_adaptation, run_unsupervised_adaptation,
)

__all__ = [
    "AdaptationContext", "AdaptationPolicy", "BaggingClassifier", "Classifier",
    "LinearSVMClassifier", "NeverUpdate", "PeriodicRetrain", "PseudoLabelBatch",
    "PseudoLabeler", "PWKNNPseudoLabeler", "RetrainOnValidatedShift",
    "RetrainOnWarning", "RetrainableClassifier", "SupervisedAdaptationConfig",
    "SupervisedAdaptationResult", "TransductiveAdaptationConfig",
    "TransductiveAdaptationResult", "TransductiveEvaluationResult",
    "UnsupervisedAdaptationConfig", "UnsupervisedAdaptationResult",
    "UnsupervisedEvaluationResult", "evaluate_transductive_adaptation",
    "evaluate_unsupervised_adaptation", "run_supervised_adaptation",
    "run_transductive_adaptation", "run_unsupervised_adaptation",
]
