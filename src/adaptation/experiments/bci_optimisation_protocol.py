from dataclasses import dataclass
import numpy as np

from src.adaptation.twin_adapt.optimization import (
    DevelopmentData,
)


@dataclass(frozen=True)
class PreparedBCIExperiment:
    development: DevelopmentData
    x_evaluation: np.ndarray
    y_evaluation: np.ndarray
    dataset: str
    subject: int


def assemble_prepared_experiment(
    *,
    dataset,
    subject,
    features,
    y_train,
    y_validation,
    y_evaluation,
):
    """Adapt existing dataset-specific FBCSP outputs."""

    return PreparedBCIExperiment(
        development=DevelopmentData(
            x_train=np.asarray(features.training.features),
            y_train=np.asarray(y_train),
            x_validation=np.asarray(features.validation.features),
            y_validation=np.asarray(y_validation),
        ),
        x_evaluation=np.asarray(features.evaluation.features),
        y_evaluation=np.asarray(y_evaluation),
        dataset=dataset,
        subject=subject,
    )
