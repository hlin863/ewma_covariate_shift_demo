"""BCI Competition IV Dataset 2B experiment for the published CSE table.

This module keeps the published reference values separate from computed
results. Sessions 01T, 02T and 03T form the training stream; sessions 04E and
05E form the merged evaluation stream.  Each observation is a cue-aligned
3-second trial feature vector.

The current feature extractor remains a compact spectral baseline.  The CSE
execution itself now follows Algorithm 1 of the 2019 CSE-UAEL paper:

* PCA is fitted on training features and PC1 is monitored;
* testing EWMA starts from z0, the arithmetic mean of training PC1;
* the published subject lambda is used for the Table 1 comparison;
* every Stage-I CS warning is validated against the training PCA distribution.
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import welch

from src.bci_data import BCISessionData
from src.cse import CSEConfig, CSEResult, run_cse


PUBLISHED_2B_RESULTS: dict[str, tuple[float, int, int]] = {
    "B01": (0.28, 14, 10),
    "B02": (0.17, 18, 13),
    "B03": (0.60, 19, 12),
    "B04": (0.20, 11, 6),
    "B05": (0.10, 12, 8),
    "B06": (0.33, 22, 12),
    "B07": (0.30, 17, 11),
    "B08": (0.21, 27, 14),
    "B09": (0.45, 18, 7),
}

_CUE_DESCRIPTIONS = {"769", "770", "783"}


@dataclass(frozen=True)
class TrialFeatureResult:
    """Cue-aligned Dataset 2B feature observations."""

    features: np.ndarray
    times: np.ndarray
    feature_names: tuple[str, ...]
    session_ids: np.ndarray
    cue_descriptions: np.ndarray


@dataclass(frozen=True)
class Dataset2BExperimentResult:
    """Computed and published results for one Dataset 2B subject."""

    subject: str
    published_lambda: float
    published_csw: int
    published_csv: int
    computed_csw: int
    computed_csv: int
    training_trials: int
    testing_trials: int
    cse_result: CSEResult


def _trial_bandpower(
    segment: np.ndarray,
    sampling_frequency: float,
) -> np.ndarray:
    frequencies, power = welch(
        segment,
        fs=sampling_frequency,
        axis=0,
        nperseg=min(segment.shape[0], int(round(sampling_frequency))),
    )
    values: list[float] = []
    for channel_index in range(segment.shape[1]):
        for low, high in ((8.0, 12.0), (14.0, 30.0)):
            mask = (frequencies >= low) & (frequencies < high)
            band_power = np.trapezoid(
                power[mask, channel_index],
                frequencies[mask],
            )
            values.append(float(np.log10(max(band_power, 1e-20))))
    return np.asarray(values, dtype=float)


def extract_dataset_2b_trials(
    session: BCISessionData,
    *,
    seconds_after_cue: float = 3.0,
) -> TrialFeatureResult:
    """Extract one feature row per cue-aligned motor-imagery trial."""

    if seconds_after_cue <= 0.0:
        raise ValueError("seconds_after_cue must be positive.")
    sampling_frequency = float(session.sampling_frequency)
    segment_size = int(round(seconds_after_cue * sampling_frequency))
    if segment_size < 2:
        raise ValueError("trial segment is too short for the sampling rate.")

    rows: list[np.ndarray] = []
    times: list[float] = []
    descriptions: list[str] = []
    for onset, _, description in session.annotations:
        code = str(description).strip()
        if code not in _CUE_DESCRIPTIONS:
            continue
        start = int(round(float(onset) * sampling_frequency))
        stop = start + segment_size
        if start < 0 or stop > session.signals.shape[0]:
            continue
        segment = session.signals[start:stop]
        if not np.isfinite(segment).all():
            continue
        rows.append(_trial_bandpower(segment, sampling_frequency))
        times.append(float(onset))
        descriptions.append(code)

    if not rows:
        raise ValueError(
            "no finite Dataset 2B motor-imagery cue trials were found."
        )
    features = np.vstack(rows)
    feature_names = tuple(
        f"{channel}_{band}"
        for channel in session.channel_names
        for band in ("mu_8_12", "beta_14_30")
    )
    return TrialFeatureResult(
        features=features,
        times=np.asarray(times, dtype=float),
        feature_names=feature_names,
        session_ids=np.full(features.shape[0], session.session, dtype="U8"),
        cue_descriptions=np.asarray(descriptions, dtype="U8"),
    )


def concatenate_trial_features(
    results: list[TrialFeatureResult],
) -> TrialFeatureResult:
    """Join sessions and assign unique sequential CSE observation times."""

    if not results:
        raise ValueError("at least one trial feature result is required.")
    feature_names = results[0].feature_names
    if any(result.feature_names != feature_names for result in results[1:]):
        raise ValueError("all sessions must contain identical feature columns.")
    features = np.vstack([result.features for result in results])
    session_ids = np.concatenate([result.session_ids for result in results])
    descriptions = np.concatenate(
        [result.cue_descriptions for result in results]
    )
    return TrialFeatureResult(
        features=features,
        times=np.arange(features.shape[0], dtype=int),
        feature_names=feature_names,
        session_ids=session_ids,
        cue_descriptions=descriptions,
    )


def run_dataset_2b_subject(
    subject: int,
    training: TrialFeatureResult,
    testing: TrialFeatureResult,
    *,
    validation_alpha: float = 0.05,
    control_limit_multiplier: float = 3.0,
    variance_smoothing: float = 0.05,
) -> Dataset2BExperimentResult:
    """Run the paper's Algorithm 1 CSE path for one Dataset 2B subject."""

    subject_id = f"B{subject:02d}"
    if subject_id not in PUBLISHED_2B_RESULTS:
        raise ValueError("subject must be an integer from 1 to 9.")
    published_lambda, published_csw, published_csv = (
        PUBLISHED_2B_RESULTS[subject_id]
    )

    config = CSEConfig(
        pca_components=min(3, training.features.shape[1]),
        lambda_override=published_lambda,
        variance_smoothing=variance_smoothing,
        control_limit_multiplier=control_limit_multiplier,
        ewma_initialization="training_mean",
        validation_mode="algorithm1_training_reference",
        validation_alpha=validation_alpha,
        covariance_method="shrinkage",
    )
    cse_result = run_cse(
        training_features=training.features,
        testing_features=testing.features,
        testing_times=testing.times,
        config=config,
    )
    computed_csw = int(
        cse_result.warning_results["stage_1_alarm"].astype(bool).sum()
    )
    computed_csv = int(
        cse_result.validation_results["confirmed_shift"].astype(bool).sum()
    )
    return Dataset2BExperimentResult(
        subject=subject_id,
        published_lambda=published_lambda,
        published_csw=published_csw,
        published_csv=published_csv,
        computed_csw=computed_csw,
        computed_csv=computed_csv,
        training_trials=int(training.features.shape[0]),
        testing_trials=int(testing.features.shape[0]),
        cse_result=cse_result,
    )
