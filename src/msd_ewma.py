"""Multivariate shift detection based on EWMA."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _validate_multivariate_values(
    values: np.ndarray,
) -> np.ndarray:
    """Validate a matrix of multivariate observations."""

    observations = np.asarray(values, dtype=float)

    if observations.ndim != 2:
        raise ValueError("values must have shape " "(n_samples, n_features).")

    if observations.shape[0] == 0:
        raise ValueError("values must contain at least one sample.")

    if observations.shape[1] == 0:
        raise ValueError("values must contain at least one feature.")

    if not np.isfinite(observations).all():
        raise ValueError("values must contain only finite observations.")

    return observations


def calculate_multivariate_ewma_path(
    values: np.ndarray,
    lambda_value: float,
    initial_z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate vector EWMA states and prediction errors."""

    observations = _validate_multivariate_values(values)

    if not 0.0 < lambda_value <= 1.0:
        raise ValueError("lambda_value must be in (0, 1].")

    initial_state = np.asarray(
        initial_z,
        dtype=float,
    )

    if initial_state.ndim != 1:
        raise ValueError("initial_z must be one-dimensional.")

    if initial_state.shape != (observations.shape[1],):
        raise ValueError("initial_z must have one value per feature.")

    if not np.isfinite(initial_state).all():
        raise ValueError("initial_z must contain only finite values.")

    ewma_states = np.empty_like(
        observations,
        dtype=float,
    )
    prediction_errors = np.empty_like(
        observations,
        dtype=float,
    )

    previous_z = initial_state.copy()

    for index, observation in enumerate(observations):
        prediction_errors[index] = observation - previous_z

        current_z = lambda_value * observation + (1.0 - lambda_value) * previous_z

        ewma_states[index] = current_z
        previous_z = current_z

    return ewma_states, prediction_errors


@dataclass(frozen=True)
class MSDTrainingResult:
    """Fitted parameters for multivariate SD-EWMA."""

    lambda_value: float
    initial_z: np.ndarray
    final_z: np.ndarray
    error_covariance: np.ndarray
    inverse_error_covariance: np.ndarray
    n_features: int


def fit_msd_ewma(
    training_values: np.ndarray,
    lambda_value: float,
) -> MSDTrainingResult:
    """Fit the multivariate EWMA detector."""

    observations = _validate_multivariate_values(training_values)

    initial_z = observations.mean(axis=0)

    ewma_states, prediction_errors = calculate_multivariate_ewma_path(
        values=observations,
        lambda_value=lambda_value,
        initial_z=initial_z,
    )

    error_covariance = np.atleast_2d(
        np.cov(
            prediction_errors,
            rowvar=False,
            ddof=0,
        )
    )

    inverse_error_covariance = np.linalg.pinv(error_covariance)

    return MSDTrainingResult(
        lambda_value=float(lambda_value),
        initial_z=initial_z.copy(),
        final_z=ewma_states[-1].copy(),
        error_covariance=error_covariance,
        inverse_error_covariance=(inverse_error_covariance),
        n_features=observations.shape[1],
    )


def run_msd_ewma(
    values: np.ndarray,
    times: np.ndarray,
    *,
    initial_z: np.ndarray,
    lambda_value: float,
    inverse_error_covariance: np.ndarray,
    control_limit: float,
) -> pd.DataFrame:
    """Run online multivariate EWMA shift detection."""

    observations = _validate_multivariate_values(values)
    time_values = np.asarray(times)

    if time_values.ndim != 1:
        raise ValueError("times must be one-dimensional.")

    if time_values.size != observations.shape[0]:
        raise ValueError("times and values must contain the same number of samples.")

    previous_z = np.asarray(initial_z, dtype=float).copy()
    inverse_covariance = np.asarray(
        inverse_error_covariance,
        dtype=float,
    )

    records: list[dict[str, object]] = []

    for time_value, observation in zip(
        time_values,
        observations,
        strict=True,
    ):
        prediction_error = observation - previous_z

        t_squared = float(prediction_error.T @ inverse_covariance @ prediction_error)

        alarm = t_squared > control_limit

        current_z = lambda_value * observation + (1.0 - lambda_value) * previous_z

        records.append(
            {
                "time": time_value,
                "t_squared": t_squared,
                "control_limit": control_limit,
                "stage_1_alarm": alarm,
            }
        )

        previous_z = current_z

    return pd.DataFrame.from_records(records)
