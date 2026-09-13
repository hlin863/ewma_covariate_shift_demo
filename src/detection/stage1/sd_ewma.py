from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EWMATrainingResult:
    lambda_value: float
    initial_z: float
    final_z: float
    error_variance: float
    error_standard_deviation: float


def _validate_univariate_values(
    values: np.ndarray,
    *,
    minimum_size: int = 1,
) -> np.ndarray:
    observations = np.asarray(values, dtype=float)
    if observations.ndim != 1:
        raise ValueError("values must be one-dimensional.")
    if observations.size < minimum_size:
        raise ValueError(f"values must contain at least {minimum_size} observations.")
    if not np.isfinite(observations).all():
        raise ValueError("values must contain only finite observations.")
    return observations


def calculate_ewma_training_path(
    values: np.ndarray,
    lambda_value: float,
    initial_z: float,
) -> tuple[np.ndarray, np.ndarray]:
    observations = np.asarray(values, dtype=float)
    if observations.ndim != 1:
        raise ValueError("values must be one-dimensional.")
    if not 0.0 < lambda_value <= 1.0:
        raise ValueError("lambda_value must be in (0, 1].")

    ewma_values = np.empty(observations.size)
    prediction_errors = np.empty(observations.size)
    previous_z = float(initial_z)
    for index, observation in enumerate(observations):
        prediction_errors[index] = observation - previous_z
        current_z = lambda_value * observation + (1.0 - lambda_value) * previous_z
        ewma_values[index] = current_z
        previous_z = current_z
    return ewma_values, prediction_errors


def estimate_lambda(
    values: np.ndarray,
    candidates: np.ndarray | None = None,
) -> tuple[float, pd.DataFrame]:
    observations = np.asarray(values, dtype=float)
    if candidates is None:
        candidates = np.arange(0.01, 1.01, 0.01)
    initial_z = float(observations.mean())
    records: list[dict[str, float]] = []
    for candidate in candidates:
        _, errors = calculate_ewma_training_path(
            values=observations,
            lambda_value=float(candidate),
            initial_z=initial_z,
        )
        records.append({"lambda": float(candidate), "sse": float(np.sum(errors**2))})
    search_results = pd.DataFrame(records)
    best_index = search_results["sse"].idxmin()
    best_lambda = float(search_results.loc[best_index, "lambda"])
    return best_lambda, search_results


def fit_sd_ewma(
    training_values: np.ndarray,
    *,
    lambda_override: float | None = None,
) -> EWMATrainingResult:
    observations = _validate_univariate_values(training_values, minimum_size=2)
    if lambda_override is not None:
        if not 0.0 < lambda_override <= 1.0:
            raise ValueError("lambda_override must be in the interval (0, 1].")
        effective_lambda = float(lambda_override)
    else:
        effective_lambda, _ = estimate_lambda(observations)

    initial_z = float(observations.mean())
    ewma_values, errors = calculate_ewma_training_path(
        values=observations,
        lambda_value=effective_lambda,
        initial_z=initial_z,
    )
    error_variance = float(np.mean(errors**2))
    return EWMATrainingResult(
        lambda_value=effective_lambda,
        initial_z=initial_z,
        final_z=float(ewma_values[-1]),
        error_variance=error_variance,
        error_standard_deviation=float(np.sqrt(error_variance)),
    )


@dataclass(frozen=True)
class SD_EWMA_Config:
    lambda_value: float
    variance_smoothing: float = 0.05
    control_limit_multiplier: float = 1.96
    variance_update_mode: str = "always"

    def __post_init__(self) -> None:
        if not 0.0 < self.lambda_value <= 1.0:
            raise ValueError("lambda_value must be in (0, 1].")
        if not 0.0 < self.variance_smoothing <= 1.0:
            raise ValueError("variance_smoothing must be in (0, 1].")
        if self.control_limit_multiplier <= 0.0:
            raise ValueError("control_limit_multiplier must be positive.")
        if self.variance_update_mode not in {"always", "frozen", "non_alarm"}:
            raise ValueError(
                "variance_update_mode must be 'always', 'frozen', or 'non_alarm'."
            )


def run_sd_ewma(
    values: np.ndarray,
    times: np.ndarray,
    initial_z: float,
    initial_error_variance: float,
    config: SD_EWMA_Config,
) -> pd.DataFrame:
    observations = _validate_univariate_values(values)
    time_values = np.asarray(times, dtype=float)

    if time_values.ndim != 1:
        raise ValueError("times must be one-dimensional.")
    if observations.size != time_values.size:
        raise ValueError("values and times must have equal length.")
    if not np.isfinite(time_values).all():
        raise ValueError("times must be finite.")
    if not np.equal(time_values, np.floor(time_values)).all():
        raise ValueError("times must contain integer observation indices.")
    if np.any(np.diff(time_values) <= 0):
        raise ValueError("times must be strictly increasing.")
    if initial_error_variance <= 0.0 or not np.isfinite(initial_error_variance):
        raise ValueError("initial_error_variance must be positive and finite.")
    if not np.isfinite(initial_z):
        raise ValueError("initial_z must be finite.")

    previous_z = float(initial_z)
    previous_variance = float(initial_error_variance)
    records: list[dict[str, float | int | str]] = []
    for time, observation in zip(time_values, observations):
        previous_std = float(np.sqrt(previous_variance))
        half_width = config.control_limit_multiplier * previous_std
        lcl = previous_z - half_width
        ucl = previous_z + half_width
        prediction_error = float(observation - previous_z)
        stage_1_alarm = bool(observation <= lcl or observation >= ucl)
        current_z = (
            config.lambda_value * observation + (1.0 - config.lambda_value) * previous_z
        )
        updated_variance = (
            config.variance_smoothing * prediction_error**2
            + (1.0 - config.variance_smoothing) * previous_variance
        )
        variance_updated = config.variance_update_mode == "always" or (
            config.variance_update_mode == "non_alarm" and not stage_1_alarm
        )
        current_variance = (
            float(updated_variance) if variance_updated else previous_variance
        )
        records.append(
            {
                "time": int(time),
                "x": float(observation),
                "prediction": float(previous_z),
                "ewma": float(current_z),
                "error": prediction_error,
                "error_variance_before": float(previous_variance),
                "error_std_before": previous_std,
                "control_limit_multiplier": float(config.control_limit_multiplier),
                "control_limit_half_width": float(half_width),
                "lcl": float(lcl),
                "ucl": float(ucl),
                "stage_1_alarm": int(stage_1_alarm),
                "error_variance_after": float(current_variance),
                "variance_update_mode": config.variance_update_mode,
                "lambda": float(config.lambda_value),
                "variance_smoothing": float(config.variance_smoothing),
                "variance_updated": int(variance_updated),
            }
        )
        previous_z = float(current_z)
        previous_variance = float(current_variance)
    return pd.DataFrame(records)
