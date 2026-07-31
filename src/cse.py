from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.cse_preprocessing import (
    CSEPCAResult,
    extract_first_component,
    fit_cse_pca,
    transform_cse_features,
)
from src.ewma import (
    EWMATrainingResult,
    SD_EWMA_Config,
    fit_sd_ewma,
    run_sd_ewma,
)
from src.multivariate_stage_2 import (
    HotellingConfig,
    validate_multivariate_alarms,
)


@dataclass(frozen=True)
class CSEConfig:
    """Configuration for the complete CSE algorithm."""

    pca_components: int | float | None = None
    variance_smoothing: float = 0.05
    control_limit_multiplier: float = 3.0
    validation_before_size: int = 50
    validation_after_size: int = 50
    validation_alpha: float = 0.05
    covariance_method: str = "shrinkage"
    covariance_regularization: float = 1e-6
    minimum_alarm_gap: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.variance_smoothing <= 1.0:
            raise ValueError(
                "variance_smoothing must be in (0, 1]."
            )

        if self.control_limit_multiplier <= 0.0:
            raise ValueError(
                "control_limit_multiplier must be positive."
            )

        if self.validation_before_size <= 0:
            raise ValueError(
                "validation_before_size must be positive."
            )

        if self.validation_after_size <= 0:
            raise ValueError(
                "validation_after_size must be positive."
            )

        if not 0.0 < self.validation_alpha < 1.0:
            raise ValueError(
                "validation_alpha must be in (0, 1)."
            )

        if self.covariance_regularization < 0.0:
            raise ValueError(
                "covariance_regularization must not be negative."
            )

        if (
            self.minimum_alarm_gap is not None
            and self.minimum_alarm_gap < 0
        ):
            raise ValueError(
                "minimum_alarm_gap must not be negative."
            )


@dataclass(frozen=True)
class CSEResult:
    """Outputs produced by the complete CSE algorithm."""

    pca_result: CSEPCAResult
    ewma_training_result: EWMATrainingResult
    training_signal: np.ndarray
    testing_transformed: np.ndarray
    testing_signal: np.ndarray
    warning_results: pd.DataFrame
    validation_results: pd.DataFrame


def _validate_cse_inputs(
    training_features: np.ndarray,
    testing_features: np.ndarray,
    testing_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate complete CSE inputs."""

    training = np.asarray(training_features, dtype=float)
    testing = np.asarray(testing_features, dtype=float)
    times = np.asarray(testing_times)

    if training.ndim != 2:
        raise ValueError(
            "training_features must be two-dimensional."
        )

    if testing.ndim != 2:
        raise ValueError(
            "testing_features must be two-dimensional."
        )

    if training.shape[0] < 2:
        raise ValueError(
            "training_features must contain at least two observations."
        )

    if testing.shape[0] < 1:
        raise ValueError(
            "testing_features must not be empty."
        )

    if training.shape[1] < 1:
        raise ValueError(
            "training_features must contain at least one feature."
        )

    if training.shape[1] != testing.shape[1]:
        raise ValueError(
            "training_features and testing_features must have "
            "the same number of feature columns."
        )

    if times.ndim != 1:
        raise ValueError(
            "testing_times must be one-dimensional."
        )

    if testing.shape[0] != times.size:
        raise ValueError(
            "testing_features and testing_times must contain "
            "the same number of observations."
        )

    if not np.isfinite(training).all():
        raise ValueError(
            "training_features must contain only finite values."
        )

    if not np.isfinite(testing).all():
        raise ValueError(
            "testing_features must contain only finite values."
        )

    if np.unique(times).size != times.size:
        raise ValueError(
            "testing_times must uniquely identify observations."
        )

    return training, testing, times


def _prepare_cse_features(
    training_features: np.ndarray,
    testing_features: np.ndarray,
    *,
    pca_components: int | float | None,
) -> tuple[
    CSEPCAResult,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Fit PCA and produce scalar training and testing signals."""

    pca_result = fit_cse_pca(
        training_features=training_features,
        n_components=pca_components,
    )

    testing_transformed = transform_cse_features(
        features=testing_features,
        fitted_result=pca_result,
    )

    training_signal = extract_first_component(
        pca_result.training_transformed,
    )

    testing_signal = extract_first_component(
        testing_transformed,
    )

    return (
        pca_result,
        training_signal,
        testing_transformed,
        testing_signal,
    )


def _run_cse_warning_stage(
    training_signal: np.ndarray,
    testing_signal: np.ndarray,
    testing_times: np.ndarray,
    config: CSEConfig,
) -> tuple[EWMATrainingResult, pd.DataFrame]:
    """Fit EWMA and produce covariate-shift warnings."""

    ewma_training_result = fit_sd_ewma(
        training_values=training_signal,
    )

    ewma_config = SD_EWMA_Config(
        lambda_value=ewma_training_result.lambda_value,
        variance_smoothing=config.variance_smoothing,
        control_limit_multiplier=(
            config.control_limit_multiplier
        ),
    )

    warning_results = run_sd_ewma(
        values=testing_signal,
        times=testing_times,
        initial_z=ewma_training_result.final_z,
        initial_error_variance=(
            ewma_training_result.error_variance
        ),
        config=ewma_config,
    )

    return ewma_training_result, warning_results


def _run_cse_validation_stage(
    transformed_features: np.ndarray,
    testing_times: np.ndarray,
    warning_results: pd.DataFrame,
    config: CSEConfig,
) -> pd.DataFrame:
    """Validate warnings using multivariate Hotelling testing."""

    hotelling_config = HotellingConfig(
        before_size=config.validation_before_size,
        after_size=config.validation_after_size,
        alpha=config.validation_alpha,
        covariance_method=config.covariance_method,
        regularization=config.covariance_regularization,
        minimum_alarm_gap=config.minimum_alarm_gap,
    )

    return validate_multivariate_alarms(
        features=transformed_features,
        times=testing_times,
        stage_1_results=warning_results,
        config=hotelling_config,
    )


def run_cse(
    training_features: np.ndarray,
    testing_features: np.ndarray,
    testing_times: np.ndarray,
    config: CSEConfig | None = None,
) -> CSEResult:
    """Run the complete Covariate Shift Estimation algorithm."""

    cse_config = config or CSEConfig()

    training, testing, times = _validate_cse_inputs(
        training_features=training_features,
        testing_features=testing_features,
        testing_times=testing_times,
    )

    (
        pca_result,
        training_signal,
        testing_transformed,
        testing_signal,
    ) = _prepare_cse_features(
        training_features=training,
        testing_features=testing,
        pca_components=cse_config.pca_components,
    )

    (
        ewma_training_result,
        warning_results,
    ) = _run_cse_warning_stage(
        training_signal=training_signal,
        testing_signal=testing_signal,
        testing_times=times,
        config=cse_config,
    )

    validation_results = _run_cse_validation_stage(
        transformed_features=testing_transformed,
        testing_times=times,
        warning_results=warning_results,
        config=cse_config,
    )

    return CSEResult(
        pca_result=pca_result,
        ewma_training_result=ewma_training_result,
        training_signal=training_signal,
        testing_transformed=testing_transformed,
        testing_signal=testing_signal,
        warning_results=warning_results,
        validation_results=validation_results,
    )