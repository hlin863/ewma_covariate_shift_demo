"""Filter-bank common spatial pattern features for the BCI reproduction path.

The signal-processing implementation is shared across the reproduction paths:
each band uses eighth-order zero-phase Butterworth filtering, a separate binary
CSP projection, and log-normalised variances from the extreme CSP components.
Two named filter-bank configurations preserve the methodological variants used
by the CSE-UAEL reproduction and the 2018 online adaptive BCI study.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh
from scipy.signal import butter, sosfiltfilt

CSE_UAEL_FILTER_BANK: tuple[tuple[float, float], ...] = tuple(
    (float(low), float(low + 4)) for low in range(8, 28, 2)
)

ONLINE_BCI_2018_FILTER_BANK: tuple[tuple[float, float], ...] = (
    (8.0, 12.0),
    (16.0, 24.0),
)

# Backward-compatible alias retained for existing Table 1 reproduction callers.
PAPER_FILTER_BANK = CSE_UAEL_FILTER_BANK


@dataclass(frozen=True)
class CSPModel:
    filters: np.ndarray
    low_hz: float
    high_hz: float


@dataclass(frozen=True)
class FBCSPModel:
    sampling_frequency: float
    models: tuple[CSPModel, ...]
    components_per_side: int

    @property
    def n_features(self) -> int:
        return len(self.models) * 2 * self.components_per_side


def _validate_trials(trials: np.ndarray) -> np.ndarray:
    values = np.asarray(trials, dtype=float)
    if values.ndim != 3:
        raise ValueError("trials must have shape (trials, channels, samples).")
    if values.shape[0] < 1 or values.shape[1] < 2 or values.shape[2] < 2:
        raise ValueError("trials must contain observations, channels, and samples.")
    if not np.isfinite(values).all():
        raise ValueError("trials must contain only finite values.")
    return values


def _validate_binary_labels(labels: np.ndarray, n_trials: int) -> np.ndarray:
    values = np.asarray(labels)
    if values.ndim != 1 or values.size != n_trials:
        raise ValueError("labels must be one-dimensional with one value per trial.")
    if np.unique(values).size != 2:
        raise ValueError("CSP requires exactly two training classes.")
    return values


def bandpass_trials(
    trials: np.ndarray,
    sampling_frequency: float,
    low_hz: float,
    high_hz: float,
) -> np.ndarray:
    """Apply the shared eighth-order zero-phase Butterworth band-pass filter."""
    values = _validate_trials(trials)
    if sampling_frequency <= 0.0:
        raise ValueError("sampling_frequency must be positive.")
    nyquist = sampling_frequency / 2.0
    if not 0.0 < low_hz < high_hz < nyquist:
        raise ValueError("band edges must lie strictly inside the Nyquist range.")
    sos = butter(
        8,
        [low_hz, high_hz],
        btype="bandpass",
        fs=sampling_frequency,
        output="sos",
    )
    return sosfiltfilt(sos, values, axis=-1)


def _normalised_trial_covariance(trial: np.ndarray) -> np.ndarray:
    covariance = trial @ trial.T
    trace = float(np.trace(covariance))
    if trace <= 0.0 or not np.isfinite(trace):
        raise ValueError("trial covariance must have a positive finite trace.")
    return covariance / trace


def fit_binary_csp(filtered_trials: np.ndarray, labels: np.ndarray) -> np.ndarray:
    trials = _validate_trials(filtered_trials)
    y = _validate_binary_labels(labels, trials.shape[0])
    classes = np.unique(y)
    class_covariances = []
    for label in classes:
        covariances = np.stack(
            [_normalised_trial_covariance(trial) for trial in trials[y == label]]
        )
        class_covariances.append(covariances.mean(axis=0))

    covariance_a, covariance_b = class_covariances
    composite = covariance_a + covariance_b
    regularisation = 1e-10 * np.eye(composite.shape[0])
    eigenvalues, eigenvectors = eigh(covariance_a, composite + regularisation)
    order = np.argsort(eigenvalues)[::-1]
    return np.asarray(eigenvectors[:, order].T, dtype=float)


def _select_extreme_components(
    projected_trials: np.ndarray,
    components_per_side: int,
) -> np.ndarray:
    n_components = projected_trials.shape[1]
    if components_per_side <= 0:
        raise ValueError("components_per_side must be positive.")
    if 2 * components_per_side > n_components:
        raise ValueError("twice components_per_side must not exceed the channel count.")
    indices = np.concatenate(
        [
            np.arange(components_per_side),
            np.arange(n_components - components_per_side, n_components),
        ]
    )
    return projected_trials[:, indices, :]


def log_normalised_variance(projected_trials: np.ndarray) -> np.ndarray:
    values = _validate_trials(projected_trials)
    variances = np.var(values, axis=-1, ddof=0)
    totals = variances.sum(axis=1, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError("projected trials must have positive total variance.")
    return np.log(np.maximum(variances / totals, 1e-20))


def fit_fbcsp(
    training_trials: np.ndarray,
    training_labels: np.ndarray,
    sampling_frequency: float,
    *,
    components_per_side: int = 1,
    filter_bank: tuple[tuple[float, float], ...] = CSE_UAEL_FILTER_BANK,
) -> FBCSPModel:
    trials = _validate_trials(training_trials)
    labels = _validate_binary_labels(training_labels, trials.shape[0])
    if not filter_bank:
        raise ValueError("filter_bank must not be empty.")
    if 2 * components_per_side > trials.shape[1]:
        raise ValueError("too many CSP components requested for the channels.")

    models = []
    for low_hz, high_hz in filter_bank:
        filtered = bandpass_trials(
            trials,
            sampling_frequency,
            low_hz,
            high_hz,
        )
        filters = fit_binary_csp(filtered, labels)
        models.append(CSPModel(filters, low_hz, high_hz))
    return FBCSPModel(
        sampling_frequency=float(sampling_frequency),
        models=tuple(models),
        components_per_side=components_per_side,
    )


def transform_fbcsp(trials: np.ndarray, model: FBCSPModel) -> np.ndarray:
    values = _validate_trials(trials)
    feature_blocks = []
    for band_model in model.models:
        filtered = bandpass_trials(
            values,
            model.sampling_frequency,
            band_model.low_hz,
            band_model.high_hz,
        )
        projected = np.einsum("kc,tcs->tks", band_model.filters, filtered)
        selected = _select_extreme_components(
            projected,
            model.components_per_side,
        )
        feature_blocks.append(log_normalised_variance(selected))
    return np.concatenate(feature_blocks, axis=1)


def fit_transform_fbcsp(
    training_trials: np.ndarray,
    training_labels: np.ndarray,
    testing_trials: np.ndarray,
    sampling_frequency: float,
    *,
    components_per_side: int = 1,
    filter_bank: tuple[tuple[float, float], ...] = CSE_UAEL_FILTER_BANK,
) -> tuple[FBCSPModel, np.ndarray, np.ndarray]:
    model = fit_fbcsp(
        training_trials,
        training_labels,
        sampling_frequency,
        components_per_side=components_per_side,
        filter_bank=filter_bank,
    )
    return (
        model,
        transform_fbcsp(training_trials, model),
        transform_fbcsp(testing_trials, model),
    )
