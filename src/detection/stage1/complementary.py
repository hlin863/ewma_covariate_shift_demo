"""Experimental Stage-I score and residual monitoring."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.detection.preprocessing import (
    fit_cse_pca,
    transform_cse_features,
)
from src.detection.stage1.sd_ewma import (
    SD_EWMA_Config,
    fit_sd_ewma,
    run_sd_ewma,
)


@dataclass
class ComplementaryModel:
    pca: object
    ewma: object
    q_scale: float
    score_limit: float
    residual_limit: float
    joint_limit: float


def _signals(features, pca, ewma, q_scale):
    scores = transform_cse_features(features, pca)

    reconstructed = pca.model.inverse_transform(scores)
    residuals = np.asarray(features) - reconstructed
    q = np.sum(residuals ** 2, axis=1)

    trace = run_sd_ewma(
        values=scores[:, 0],
        times=np.arange(len(scores)),
        initial_z=ewma.final_z,
        initial_error_variance=ewma.error_variance,
        config=SD_EWMA_Config(
            lambda_value=ewma.lambda_value,
            control_limit_multiplier=1.0,
            variance_update_mode="frozen",
        ),
    )

    score_stat = (
        np.abs(trace["error"].to_numpy())
        / ewma.error_standard_deviation
    )

    return scores[:, 0], q, score_stat, q / q_scale


def fit_complementary(
    training,
    calibration,
    alpha=0.01,
    lambda_value=0.2,
):
    """Fit on training; calibrate limits on separate no-change data."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one.")

    if np.asarray(training).shape[1] < 2:
        raise ValueError("At least two input features are required.")

    # Retain one direction, leaving a residual subspace.
    pca = fit_cse_pca(training, n_components=1)

    ewma = fit_sd_ewma(
        pca.first_component,
        lambda_override=lambda_value,
    )

    reconstructed = pca.model.inverse_transform(
        pca.training_transformed
    )
    q_train = np.sum(
        (np.asarray(training) - reconstructed) ** 2,
        axis=1,
    )
    q_scale = float(q_train.mean())

    if q_scale <= 0 or ewma.error_variance <= 0:
        raise ValueError(
            "Training must have nonzero score and residual variation."
        )

    _, _, score, residual = _signals(
        calibration, pca, ewma, q_scale
    )

    score_limit = float(
        np.quantile(score, 1 - alpha, method="higher")
    )
    residual_limit = float(
        np.quantile(residual, 1 - alpha, method="higher")
    )

    if min(score_limit, residual_limit) <= 0:
        raise ValueError("Calibration limits must be positive.")

    # Calibrate the maximum of the two normalised statistics.
    joint = np.maximum(
        score / score_limit,
        residual / residual_limit,
    )
    joint_limit = float(
        np.quantile(joint, 1 - alpha, method="higher")
    )

    return ComplementaryModel(
        pca=pca,
        ewma=ewma,
        q_scale=q_scale,
        score_limit=score_limit,
        residual_limit=residual_limit,
        joint_limit=joint_limit,
    )


def monitor(features, model):
    """Monitor one independent stream from the fitted initial state."""
    pc1, q, score, residual = _signals(
        features,
        model.pca,
        model.ewma,
        model.q_scale,
    )

    # Standalone methods.
    score_only = score > model.score_limit
    residual_only = residual > model.residual_limit

    # Routes within the combined method.
    score_route = (
        score > model.score_limit * model.joint_limit
    )
    residual_route = (
        residual > model.residual_limit * model.joint_limit
    )

    source = np.select(
        [
            score_route & residual_route,
            score_route,
            residual_route,
        ],
        ["both", "score", "residual"],
        default="none",
    )

    return pd.DataFrame({
        "time": np.arange(len(pc1)),
        "pc1_score": pc1,
        "residual_q": q,
        "score_stat": score,
        "residual_stat": residual,
        "score_only_alarm": score_only,
        "residual_only_alarm": residual_only,
        "score_alarm": score_route,
        "residual_alarm": residual_route,
        "stage_1_alarm": score_route | residual_route,
        "alarm_source": source,
    })