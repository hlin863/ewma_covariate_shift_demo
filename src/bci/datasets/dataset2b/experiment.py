"""BCI Competition IV Dataset 2B experiment for the published CSE table."""

from dataclasses import dataclass
import re

import numpy as np

from src.bci.data import BCISessionData
from src.bci.fbcsp import FBCSPModel, fit_transform_fbcsp
from src.detection import CSEConfig, CSEResult, run_cse

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
_TRAINING_LABELS = {"769": 0, "770": 1}


@dataclass(frozen=True)
class TrialSignalResult:
    signals: np.ndarray
    labels: np.ndarray
    times: np.ndarray
    session_ids: np.ndarray
    cue_descriptions: np.ndarray
    channel_names: tuple[str, ...]
    sampling_frequency: float


@dataclass(frozen=True)
class TrialFeatureResult:
    features: np.ndarray
    times: np.ndarray
    feature_names: tuple[str, ...]
    session_ids: np.ndarray
    cue_descriptions: np.ndarray


@dataclass(frozen=True)
class Dataset2BFeaturePipelineResult:
    model: FBCSPModel
    training: TrialFeatureResult
    testing: TrialFeatureResult


@dataclass(frozen=True)
class Dataset2BExperimentResult:
    subject: str
    published_lambda: float
    published_csw: int
    published_csv: int
    computed_csw: int
    computed_csv: int
    training_trials: int
    testing_trials: int
    cse_result: CSEResult


def _canonical_eeg_channel_name(name: str) -> str:
    compact = re.sub(r"[^A-Z0-9]", "", str(name).upper())
    for expected in ("C3", "CZ", "C4"):
        if compact.endswith(expected):
            return expected
    return compact


def _dataset_2b_channel_indices(
    channel_names: tuple[str, ...],
) -> tuple[int, int, int]:
    lookup: dict[str, int] = {}
    for index, name in enumerate(channel_names):
        canonical = _canonical_eeg_channel_name(name)
        if canonical in {"C3", "CZ", "C4"} and canonical not in lookup:
            lookup[canonical] = index

    missing = [name for name in ("C3", "CZ", "C4") if name not in lookup]
    if missing:
        available = ", ".join(str(name) for name in channel_names)
        raise ValueError(
            "Dataset 2B session must contain C3, Cz, and C4. "
            f"Missing: {', '.join(missing)}. Available channels: {available}"
        )
    return lookup["C3"], lookup["CZ"], lookup["C4"]


def extract_dataset_2b_trials(
    session: BCISessionData,
    *,
    seconds_after_cue: float = 3.0,
) -> TrialSignalResult:
    if seconds_after_cue <= 0.0:
        raise ValueError("seconds_after_cue must be positive.")
    sampling_frequency = float(session.sampling_frequency)
    segment_size = int(round(seconds_after_cue * sampling_frequency))
    if segment_size < 2:
        raise ValueError("trial segment is too short for the sampling rate.")

    required_channels = ("C3", "Cz", "C4")
    channel_indices = _dataset_2b_channel_indices(session.channel_names)
    rows: list[np.ndarray] = []
    times: list[float] = []
    labels: list[int] = []
    descriptions: list[str] = []
    for onset, _, description in session.annotations:
        code = str(description).strip()
        if code not in _CUE_DESCRIPTIONS:
            continue
        start = int(round(float(onset) * sampling_frequency))
        stop = start + segment_size
        if start < 0 or stop > session.signals.shape[0]:
            continue
        segment = session.signals[start:stop, channel_indices].T
        if not np.isfinite(segment).all():
            continue
        rows.append(segment)
        times.append(float(onset))
        labels.append(_TRAINING_LABELS.get(code, -1))
        descriptions.append(code)

    if not rows:
        raise ValueError("no finite Dataset 2B motor-imagery trials were found.")
    signals = np.stack(rows)
    return TrialSignalResult(
        signals=signals,
        labels=np.asarray(labels, dtype=int),
        times=np.asarray(times, dtype=float),
        session_ids=np.full(signals.shape[0], session.session, dtype="U8"),
        cue_descriptions=np.asarray(descriptions, dtype="U8"),
        channel_names=required_channels,
        sampling_frequency=sampling_frequency,
    )


def concatenate_trial_signals(results: list[TrialSignalResult]) -> TrialSignalResult:
    if not results:
        raise ValueError("at least one trial result is required.")
    first = results[0]
    if any(result.channel_names != first.channel_names for result in results[1:]):
        raise ValueError("all sessions must use identical channel ordering.")
    if any(
        not np.isclose(result.sampling_frequency, first.sampling_frequency)
        for result in results[1:]
    ):
        raise ValueError("all sessions must have the same sampling frequency.")
    if any(
        result.signals.shape[1:] != first.signals.shape[1:]
        for result in results[1:]
    ):
        raise ValueError("all sessions must have identical trial tensor shapes.")

    signals = np.concatenate([result.signals for result in results], axis=0)
    return TrialSignalResult(
        signals=signals,
        labels=np.concatenate([result.labels for result in results]),
        times=np.arange(signals.shape[0], dtype=int),
        session_ids=np.concatenate([result.session_ids for result in results]),
        cue_descriptions=np.concatenate([result.cue_descriptions for result in results]),
        channel_names=first.channel_names,
        sampling_frequency=first.sampling_frequency,
    )


def _feature_names(model: FBCSPModel) -> tuple[str, ...]:
    names: list[str] = []
    for band_model in model.models:
        band = f"{band_model.low_hz:g}_{band_model.high_hz:g}Hz"
        for index in range(model.components_per_side):
            names.append(f"{band}_csp_high_{index + 1}")
        for index in range(model.components_per_side):
            names.append(f"{band}_csp_low_{index + 1}")
    return tuple(names)


def build_dataset_2b_fbcsp_features(
    training: TrialSignalResult,
    testing: TrialSignalResult,
    *,
    components_per_side: int = 1,
) -> Dataset2BFeaturePipelineResult:
    if training.channel_names != testing.channel_names:
        raise ValueError("training and testing channel orders must match.")
    if not np.isclose(training.sampling_frequency, testing.sampling_frequency):
        raise ValueError("training and testing sampling frequencies must match.")
    if np.any(training.labels < 0):
        raise ValueError("all training trials must have left/right labels.")

    model, training_features, testing_features = fit_transform_fbcsp(
        training.signals,
        training.labels,
        testing.signals,
        training.sampling_frequency,
        components_per_side=components_per_side,
    )
    names = _feature_names(model)
    return Dataset2BFeaturePipelineResult(
        model=model,
        training=TrialFeatureResult(
            features=training_features,
            times=training.times,
            feature_names=names,
            session_ids=training.session_ids,
            cue_descriptions=training.cue_descriptions,
        ),
        testing=TrialFeatureResult(
            features=testing_features,
            times=testing.times,
            feature_names=names,
            session_ids=testing.session_ids,
            cue_descriptions=testing.cue_descriptions,
        ),
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
    subject_id = f"B{subject:02d}"
    if subject_id not in PUBLISHED_2B_RESULTS:
        raise ValueError("subject must be an integer from 1 to 9.")
    published_lambda, published_csw, published_csv = PUBLISHED_2B_RESULTS[subject_id]
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
    computed_csw = int(cse_result.warning_results["stage_1_alarm"].astype(bool).sum())
    computed_csv = int(cse_result.validation_results["confirmed_shift"].astype(bool).sum())
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
