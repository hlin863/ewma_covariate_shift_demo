"""Algorithm 1 Stage-II validation against the training reference distribution.

The 2019 CSE-UAEL paper states that each Stage-I warning is validated with a
Hotelling T-squared test comparing the current feature vector with the average
training feature vector.  With one trial-level feature vector at test time, the
well-defined statistical implementation is the one-sample Hotelling/Mahalanobis
statistic against the multivariate training distribution.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import f
from sklearn.covariance import LedoitWolf


@dataclass(frozen=True)
class TrainingReferenceHotellingConfig:
    """Configuration for Algorithm 1 training-reference validation."""

    alpha: float = 0.05
    covariance_method: str = "shrinkage"
    regularization: float = 1e-6

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0, 1).")
        if self.covariance_method not in {"empirical", "shrinkage"}:
            raise ValueError(
                "covariance_method must be 'empirical' or 'shrinkage'."
            )
        if self.regularization < 0.0:
            raise ValueError("regularization must not be negative.")


def _fit_training_reference(
    training_features: np.ndarray,
    *,
    covariance_method: str,
    regularization: float,
) -> tuple[np.ndarray, np.ndarray]:
    training = np.asarray(training_features, dtype=float)
    if training.ndim != 2:
        raise ValueError("training_features must be two-dimensional.")
    if training.shape[0] < 2 or training.shape[1] < 1:
        raise ValueError(
            "training_features must contain at least two rows and one column."
        )
    if not np.isfinite(training).all():
        raise ValueError("training_features must contain only finite values.")

    if covariance_method == "empirical":
        covariance = np.atleast_2d(np.cov(training, rowvar=False, ddof=1))
    else:
        covariance = LedoitWolf().fit(training).covariance_

    covariance = np.asarray(covariance, dtype=float)
    covariance += regularization * np.eye(training.shape[1])
    return training.mean(axis=0), covariance


def calculate_training_reference_hotelling(
    current_feature: np.ndarray,
    training_features: np.ndarray,
    *,
    covariance_method: str = "shrinkage",
    regularization: float = 1e-6,
) -> tuple[float, float, int, int, float]:
    """Calculate one-sample Hotelling T-squared against training features."""

    training = np.asarray(training_features, dtype=float)
    current = np.asarray(current_feature, dtype=float)
    if current.ndim != 1:
        raise ValueError("current_feature must be one-dimensional.")
    if training.ndim != 2 or training.shape[1] != current.size:
        raise ValueError(
            "current_feature and training_features must have equal feature counts."
        )
    if not np.isfinite(current).all():
        raise ValueError("current_feature must contain only finite values.")

    n_training, n_features = training.shape
    degrees_of_freedom_1 = int(n_features)
    degrees_of_freedom_2 = int(n_training - n_features)
    if degrees_of_freedom_2 <= 0:
        raise ValueError(
            "training sample is too small for the feature dimensionality."
        )

    training_mean, covariance = _fit_training_reference(
        training,
        covariance_method=covariance_method,
        regularization=regularization,
    )
    difference = current - training_mean
    t_squared = float(difference.T @ np.linalg.pinv(covariance) @ difference)

    # Predictive one-sample Hotelling conversion for a new observation against
    # a reference sample of size n.  It reduces to a scaled Mahalanobis test.
    scale = (
        n_training * degrees_of_freedom_2
        / (n_features * (n_training + 1) * (n_training - 1))
    )
    f_statistic = float(scale * t_squared)
    p_value = float(
        f.sf(f_statistic, degrees_of_freedom_1, degrees_of_freedom_2)
    )
    return (
        t_squared,
        f_statistic,
        degrees_of_freedom_1,
        degrees_of_freedom_2,
        p_value,
    )


def validate_algorithm1_alarms(
    training_features: np.ndarray,
    testing_features: np.ndarray,
    testing_times: np.ndarray,
    stage_1_results: pd.DataFrame,
    config: TrainingReferenceHotellingConfig | None = None,
) -> pd.DataFrame:
    """Validate every Stage-I warning against the training distribution."""

    validation_config = config or TrainingReferenceHotellingConfig()
    training = np.asarray(training_features, dtype=float)
    testing = np.asarray(testing_features, dtype=float)
    times = np.asarray(testing_times)

    if testing.ndim != 2 or times.ndim != 1:
        raise ValueError("testing_features must be 2D and testing_times must be 1D.")
    if testing.shape[0] != times.size:
        raise ValueError("testing_features and testing_times must have equal rows.")
    if training.ndim != 2 or training.shape[1] != testing.shape[1]:
        raise ValueError("training and testing feature counts must match.")
    if not np.isfinite(testing).all():
        raise ValueError("testing_features must contain only finite values.")

    required = {"time", "stage_1_alarm"}
    missing = required.difference(stage_1_results.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    columns = [
        "alarm_time",
        "validation_time",
        "training_size",
        "n_features",
        "hotelling_t_squared",
        "f_statistic",
        "degrees_of_freedom_1",
        "degrees_of_freedom_2",
        "p_value",
        "status",
        "confirmed_shift",
    ]
    records: list[dict[str, object]] = []

    alarm_times = stage_1_results.loc[
        stage_1_results["stage_1_alarm"].astype(bool), "time"
    ].drop_duplicates()

    for alarm_time in alarm_times:
        positions = np.flatnonzero(times == alarm_time)
        if positions.size != 1:
            raise ValueError(
                "Each alarm time must identify exactly one testing observation."
            )
        index = int(positions[0])
        (
            t_squared,
            f_statistic,
            df1,
            df2,
            p_value,
        ) = calculate_training_reference_hotelling(
            testing[index],
            training,
            covariance_method=validation_config.covariance_method,
            regularization=validation_config.regularization,
        )
        confirmed = bool(p_value < validation_config.alpha)
        records.append({
            "alarm_time": int(alarm_time),
            "validation_time": int(alarm_time),
            "training_size": int(training.shape[0]),
            "n_features": int(training.shape[1]),
            "hotelling_t_squared": t_squared,
            "f_statistic": f_statistic,
            "degrees_of_freedom_1": df1,
            "degrees_of_freedom_2": df2,
            "p_value": p_value,
            "status": "confirmed" if confirmed else "rejected",
            "confirmed_shift": confirmed,
        })

    return pd.DataFrame.from_records(records, columns=columns)
