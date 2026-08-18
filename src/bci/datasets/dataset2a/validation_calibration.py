"""Validation-error diagnostics for Dataset 2A Stage-I CSE calibration.

The 2019 CSE-UAEL paper states that Session-I is partitioned into training and
validation subsets and that validation samples are used for parameter
estimation.  It does not specify an exact rule for choosing the EWMA control
limit multiplier L from those validation samples.

This module therefore implements a transparent repo-specific calibration
extension: fit PCA and EWMA only on the development-training subset, transform
the held-out validation subset with the fitted PCA model, compute one-step-
ahead validation errors, standardise them by the pre-update EWMA error scale,
and choose L as the empirical (1 - target_false_alarm_rate) quantile.

The calibration never uses Session-II evaluation data or the published Table 1
CSW/CSV targets.
"""

from dataclasses import dataclass

import numpy as np

from src.detection.preprocessing import (
    extract_first_component,
    fit_cse_pca,
    transform_cse_features,
)
from src.detection.stage1 import fit_sd_ewma


@dataclass(frozen=True)
class Dataset2AStage1ValidationCalibration:
    """Diagnostics from held-out Session-I Stage-I calibration."""

    control_limit_multiplier: float
    target_false_alarm_rate: float
    observed_exceedance_rate: float
    validation_rmse: float
    n_validation_observations: int
    pc1_explained_variance_ratio: float


def calibrate_stage1_control_limit_from_validation(
    training_features: np.ndarray,
    validation_features: np.ndarray,
    *,
    lambda_value: float,
    pca_components: int | float | None = None,
    variance_smoothing: float = 0.05,
    variance_update_mode: str = "always",
    target_false_alarm_rate: float = 0.05,
) -> Dataset2AStage1ValidationCalibration:
    """Estimate Stage-I control multiplier L from held-out validation errors.

    This is a reproducible calibration extension rather than a claim about the
    exact unpublished MATLAB rule used in the paper.  The held-out Session-I
    validation subset is treated as development data; Session-II is untouched.

    ``variance_update_mode='non_alarm'`` is intentionally rejected because its
    variance path depends on the yet-unknown alarm threshold, creating a
    circular calibration problem.
    """

    training = np.asarray(training_features, dtype=float)
    validation = np.asarray(validation_features, dtype=float)
    if training.ndim != 2 or validation.ndim != 2:
        raise ValueError("training_features and validation_features must be 2D.")
    if training.shape[0] < 2 or validation.shape[0] < 2:
        raise ValueError("training and validation sets must each contain at least two rows.")
    if training.shape[1] != validation.shape[1]:
        raise ValueError("training and validation feature counts must match.")
    if not np.isfinite(training).all() or not np.isfinite(validation).all():
        raise ValueError("training and validation features must be finite.")
    if not 0.0 < lambda_value <= 1.0:
        raise ValueError("lambda_value must be in (0, 1].")
    if not 0.0 < variance_smoothing <= 1.0:
        raise ValueError("variance_smoothing must be in (0, 1].")
    if variance_update_mode not in {"always", "frozen", "non_alarm"}:
        raise ValueError("variance_update_mode must be 'always', 'frozen', or 'non_alarm'.")
    if variance_update_mode == "non_alarm":
        raise ValueError(
            "validation calibration does not support variance_update_mode='non_alarm' "
            "because the variance path depends on the unknown alarm threshold."
        )
    if not 0.0 < target_false_alarm_rate < 1.0:
        raise ValueError("target_false_alarm_rate must be in (0, 1).")

    pca_result = fit_cse_pca(training, n_components=pca_components)
    validation_transformed = transform_cse_features(validation, pca_result)
    training_pc1 = extract_first_component(pca_result.training_transformed)
    validation_pc1 = extract_first_component(validation_transformed)

    ewma_training = fit_sd_ewma(
        training_values=training_pc1,
        lambda_override=lambda_value,
    )

    previous_z = float(ewma_training.initial_z)
    previous_variance = float(ewma_training.error_variance)
    standardized_errors = np.empty(validation_pc1.size, dtype=float)
    raw_errors = np.empty(validation_pc1.size, dtype=float)

    for index, observation in enumerate(validation_pc1):
        previous_std = float(np.sqrt(previous_variance))
        error = float(observation - previous_z)
        raw_errors[index] = error
        standardized_errors[index] = abs(error) / previous_std

        previous_z = (
            float(lambda_value) * float(observation)
            + (1.0 - float(lambda_value)) * previous_z
        )
        if variance_update_mode == "always":
            previous_variance = float(
                variance_smoothing * error**2
                + (1.0 - variance_smoothing) * previous_variance
            )

    quantile = 1.0 - float(target_false_alarm_rate)
    control_limit_multiplier = float(
        np.quantile(standardized_errors, quantile, method="higher")
    )
    if control_limit_multiplier <= 0.0 or not np.isfinite(control_limit_multiplier):
        raise RuntimeError("validation errors did not produce a positive finite control limit.")

    observed_exceedance_rate = float(
        np.mean(standardized_errors > control_limit_multiplier)
    )
    validation_rmse = float(np.sqrt(np.mean(raw_errors**2)))

    return Dataset2AStage1ValidationCalibration(
        control_limit_multiplier=control_limit_multiplier,
        target_false_alarm_rate=float(target_false_alarm_rate),
        observed_exceedance_rate=observed_exceedance_rate,
        validation_rmse=validation_rmse,
        n_validation_observations=int(validation_pc1.size),
        pc1_explained_variance_ratio=float(pca_result.explained_variance_ratio[0]),
    )
