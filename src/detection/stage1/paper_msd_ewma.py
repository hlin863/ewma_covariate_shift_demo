"""The 2015 paper's multivariate EWMA (MSD-EWMA) control chart."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _matrix(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must have shape (n >= 2, d >= 1).")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


@dataclass(frozen=True)
class PaperMSDTrainingResult:
    """Stationary reference parameters used by Equation 7--9."""

    lambda_value: float
    reference_mean: np.ndarray
    covariance: np.ndarray
    inverse_covariance: np.ndarray
    n_features: int


def fit_paper_msd_ewma(
    training_values: np.ndarray,
    lambda_value: float = 0.10,
    regularization: float = 1e-9,
) -> PaperMSDTrainingResult:
    """Estimate the in-control mean and covariance from Phase-I data."""

    values = _matrix(training_values, "training_values")
    if not 0.0 < lambda_value <= 1.0:
        raise ValueError("lambda_value must be in (0, 1].")
    if regularization < 0.0 or not np.isfinite(regularization):
        raise ValueError("regularization must be finite and non-negative.")
    covariance = np.atleast_2d(np.cov(values, rowvar=False, ddof=1))
    covariance = covariance + regularization * np.eye(values.shape[1])
    return PaperMSDTrainingResult(
        lambda_value=float(lambda_value),
        reference_mean=values.mean(axis=0),
        covariance=covariance,
        inverse_covariance=np.linalg.pinv(covariance),
        n_features=int(values.shape[1]),
    )


def run_paper_msd_ewma(
    values: np.ndarray,
    times: np.ndarray,
    training: PaperMSDTrainingResult,
    control_limit: float,
) -> pd.DataFrame:
    """Apply the paper's time-varying MEWMA covariance statistic.

    For centered observations, Equation 7 updates ``z_i`` and Equation 9
    scales the stationary covariance by
    ``lambda/(2-lambda) * (1-(1-lambda)**(2*i))``.
    """

    observations = _matrix(values, "values")
    time_values = np.asarray(times)
    if time_values.ndim != 1 or time_values.size != observations.shape[0]:
        raise ValueError("times must be 1D and match values.")
    if observations.shape[1] != training.n_features:
        raise ValueError("values do not match the trained feature count.")
    if control_limit <= 0.0 or not np.isfinite(control_limit):
        raise ValueError("control_limit must be positive and finite.")

    lambda_value = training.lambda_value
    previous_z = np.zeros(training.n_features, dtype=float)
    records: list[dict[str, object]] = []
    for step, (time_value, observation) in enumerate(
        zip(time_values, observations, strict=True),
        start=1,
    ):
        centered = observation - training.reference_mean
        current_z = lambda_value * centered + (1.0 - lambda_value) * previous_z
        covariance_scale = (
            lambda_value
            / (2.0 - lambda_value)
            * (1.0 - (1.0 - lambda_value) ** (2 * step))
        )
        t_squared = float(
            current_z.T
            @ (training.inverse_covariance / covariance_scale)
            @ current_z
        )
        records.append(
            {
                "time": int(time_value),
                "t_squared": t_squared,
                "control_limit": float(control_limit),
                "stage_1_alarm": bool(t_squared > control_limit),
            }
        )
        previous_z = current_z
    return pd.DataFrame.from_records(records)
