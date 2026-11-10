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



def prepare_bci_optimization_protocol(
    *, dataset: str, subject: int, data_root, seed: int = 42,
) -> PreparedBCIExperiment:
    """Reuse the existing 2A/2B loaders and development-fitted FBCSP protocol.

    The six-array return preserves the final evaluation partition; only the
    development training/validation arrays are supplied to configuration search.
    The validation subset is ordered by original trial index, though the
    existing stratified holdout need not be temporally contiguous.
    """
    from pathlib import Path
    from scripts.run_online_bci import prepare

    root = Path(data_root)
    x_train, y_train, x_validation, y_validation, x_eval, y_eval = prepare(
        dataset, subject,
        root / "bci_competition_iv_2a",
        root / "bci_competition_iv_2a_labels",
        root / "bci_competition_iv_2b",
        root / "bci_competition_iv_2b_labels",
        seed,
        include_validation=True,
    )
    development = DevelopmentData(
        x_train=np.asarray(x_train),
        y_train=np.asarray(y_train),
        x_validation=np.asarray(x_validation),
        y_validation=np.asarray(y_validation),
    )
    x_eval = np.asarray(x_eval)
    y_eval = np.asarray(y_eval)
    if x_eval.ndim != 2 or y_eval.ndim != 1:
        raise ValueError("Evaluation features/labels have invalid dimensions")
    if len(x_eval) != len(y_eval):
        raise ValueError("Evaluation feature/label counts differ")
    if x_eval.shape[1] != development.x_train.shape[1]:
        raise ValueError("Evaluation feature dimensions do not match development")
    if not np.isfinite(x_eval).all():
        raise ValueError("Evaluation features must be finite")
    return PreparedBCIExperiment(
        development=development,
        x_evaluation=x_eval,
        y_evaluation=y_eval,
        dataset=dataset,
        subject=subject,
    )
