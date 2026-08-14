"""BCI Competition IV Dataset 2A experiment for CSE Table 1 reproduction.

The 2019 paper uses subject session I (T) for training and session II (E) for
evaluation, selects ten motor-imagery EEG channels, keeps left/right hand
trials, extracts cue-aligned three-second trials, applies ten-band FBCSP, then
runs the CSE detector.
"""

from dataclasses import dataclass
import re

import numpy as np

from src.bci_data import BCISessionData
from src.cse import CSEConfig, CSEResult, run_cse
from src.fbcsp import FBCSPModel, fit_transform_fbcsp


PUBLISHED_2A_RESULTS: dict[str, tuple[float, int, int]] = {
    "A01": (0.50, 12, 6),
    "A02": (0.55, 15, 8),
    "A03": (0.60, 7, 6),
    "A04": (0.61, 10, 3),
    "A05": (0.72, 13, 8),
    "A06": (0.54, 12, 6),
    "A07": (0.57, 11, 4),
    "A08": (0.50, 11, 5),
    "A09": (0.70, 6, 4),
}

# The 22 EEG electrodes in the Dataset 2A GDF acquisition order. Some MNE
# versions expose several GDF channel labels as generic names such as EEG-0,
# EEG-1, ... rather than their 10-20 electrode labels. The acquisition order
# is fixed for Dataset 2A, so it provides a dataset-specific fallback.
DATASET_2A_EEG_MONTAGE = (
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
)

DATASET_2A_CHANNELS = (
    "C3", "FC3", "CP3", "C5", "C1",
    "C4", "FC4", "CP4", "C2", "C6",
)

_TRAINING_LABELS = {"769": 0, "770": 1}
_EVALUATION_CUE = "783"


@dataclass(frozen=True)
class Dataset2ATrialSignalResult:
    signals: np.ndarray
    labels: np.ndarray
    times: np.ndarray
    cue_descriptions: np.ndarray
    channel_names: tuple[str, ...]
    sampling_frequency: float
    session_id: str


@dataclass(frozen=True)
class Dataset2ATrialFeatureResult:
    features: np.ndarray
    times: np.ndarray
    feature_names: tuple[str, ...]
    cue_descriptions: np.ndarray
    session_id: str


@dataclass(frozen=True)
class Dataset2AFeaturePipelineResult:
    model: FBCSPModel
    training: Dataset2ATrialFeatureResult
    testing: Dataset2ATrialFeatureResult


@dataclass(frozen=True)
class Dataset2AExperimentResult:
    subject: str
    published_lambda: float
    published_csw: int
    published_csv: int
    computed_csw: int
    computed_csv: int
    training_trials: int
    testing_trials: int
    cse_result: CSEResult


def _canonical_channel_name(name: str) -> str:
    """Normalise common GDF/MNE channel-label punctuation and EEG prefixes."""

    return re.sub(r"[^A-Z0-9]", "", str(name).upper()).replace("EEG", "")


def _is_eog_channel(name: str) -> bool:
    """Return True for the three Dataset 2A electro-oculogram channels."""

    return "EOG" in re.sub(r"[^A-Z0-9]", "", str(name).upper())


def _dataset_2a_channel_indices(channel_names: tuple[str, ...]) -> tuple[int, ...]:
    """Resolve the ten paper-selected Dataset 2A electrodes.

    Prefer explicit electrode labels when the GDF reader exposes all of them.
    MNE commonly exposes Dataset 2A as 25 channels: 22 EEG channels followed
    by three EOG channels, while many of the EEG labels are generic ``EEG-N``
    names. In that case the 22 non-EOG channels are mapped through the fixed
    Dataset 2A acquisition montage. Explicit anchor labels such as C3/Cz/C4/Pz
    are checked against that order before the fallback is accepted.
    """

    expected = tuple(name.upper() for name in DATASET_2A_CHANNELS)
    lookup: dict[str, int] = {}

    for index, name in enumerate(channel_names):
        canonical = _canonical_channel_name(name)
        for target in expected:
            if canonical == target and target not in lookup:
                lookup[target] = index
                break

    if len(lookup) == len(expected):
        return tuple(lookup[name] for name in expected)

    # Do not use len(channel_names) == 22 here: MNE can retain the three EOG
    # channels even when the caller requested EEG data, yielding the observed
    # 25-channel layout. Strip EOG channels first and then validate the fixed
    # 22-electrode Dataset 2A acquisition order.
    eeg_indices = [
        index for index, name in enumerate(channel_names) if not _is_eog_channel(name)
    ]
    if len(eeg_indices) == len(DATASET_2A_EEG_MONTAGE):
        montage_lookup = {
            name.upper(): position
            for position, name in enumerate(DATASET_2A_EEG_MONTAGE)
        }

        # Validate every descriptive electrode label that MNE did preserve.
        # This prevents silently applying the montage fallback to an unrelated
        # 22-channel recording with a different order.
        descriptive_montage_names = set(montage_lookup)
        for absolute_index in eeg_indices:
            canonical = _canonical_channel_name(channel_names[absolute_index])
            if canonical not in descriptive_montage_names:
                continue
            expected_position = montage_lookup[canonical]
            if eeg_indices[expected_position] != absolute_index:
                raise ValueError(
                    "Dataset 2A channel order does not match the documented "
                    f"22-electrode montage at {channel_names[absolute_index]!r}."
                )

        return tuple(
            eeg_indices[montage_lookup[name]]
            for name in expected
        )

    missing = [name for name in expected if name not in lookup]
    available = ", ".join(str(name) for name in channel_names)
    raise ValueError(
        "Dataset 2A session is missing required channels: "
        + ", ".join(missing)
        + f". Available channels ({len(channel_names)}): {available}"
    )


def extract_dataset_2a_trials(
    session: BCISessionData,
    *,
    seconds_after_cue: float = 3.0,
) -> Dataset2ATrialSignalResult:
    """Extract the paper's ten-channel left/right motor-imagery trial tensors."""

    if seconds_after_cue <= 0.0:
        raise ValueError("seconds_after_cue must be positive.")
    sfreq = float(session.sampling_frequency)
    segment_size = int(round(seconds_after_cue * sfreq))
    channel_indices = _dataset_2a_channel_indices(session.channel_names)
    is_evaluation = str(session.session).upper() == "E"
    accepted = {_EVALUATION_CUE} if is_evaluation else set(_TRAINING_LABELS)

    rows: list[np.ndarray] = []
    labels: list[int] = []
    times: list[float] = []
    descriptions: list[str] = []
    for onset, _, description in session.annotations:
        code = str(description).strip()
        if code not in accepted:
            continue
        start = int(round(float(onset) * sfreq))
        stop = start + segment_size
        if start < 0 or stop > session.signals.shape[0]:
            continue
        segment = session.signals[start:stop, channel_indices].T
        if not np.isfinite(segment).all():
            continue
        rows.append(segment)
        labels.append(-1 if is_evaluation else _TRAINING_LABELS[code])
        times.append(float(onset))
        descriptions.append(code)

    if not rows:
        raise ValueError("no finite Dataset 2A left/right motor-imagery trials were found.")
    signals = np.stack(rows)
    return Dataset2ATrialSignalResult(
        signals=signals,
        labels=np.asarray(labels, dtype=int),
        times=np.arange(signals.shape[0], dtype=int),
        cue_descriptions=np.asarray(descriptions, dtype="U8"),
        channel_names=DATASET_2A_CHANNELS,
        sampling_frequency=sfreq,
        session_id=str(session.session),
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


def build_dataset_2a_fbcsp_features(
    training: Dataset2ATrialSignalResult,
    testing: Dataset2ATrialSignalResult,
    *,
    components_per_side: int = 1,
) -> Dataset2AFeaturePipelineResult:
    """Fit FBCSP on Session-I labels and transform Session-II without labels."""

    if training.channel_names != testing.channel_names:
        raise ValueError("training and testing channel orders must match.")
    if np.any(training.labels < 0):
        raise ValueError("Dataset 2A training trials must be labelled left/right trials.")
    model, train_features, test_features = fit_transform_fbcsp(
        training.signals,
        training.labels,
        testing.signals,
        training.sampling_frequency,
        components_per_side=components_per_side,
    )
    names = _feature_names(model)
    return Dataset2AFeaturePipelineResult(
        model=model,
        training=Dataset2ATrialFeatureResult(
            features=train_features,
            times=training.times,
            feature_names=names,
            cue_descriptions=training.cue_descriptions,
            session_id=training.session_id,
        ),
        testing=Dataset2ATrialFeatureResult(
            features=test_features,
            times=testing.times,
            feature_names=names,
            cue_descriptions=testing.cue_descriptions,
            session_id=testing.session_id,
        ),
    )


def run_dataset_2a_subject(
    subject: int,
    training: Dataset2ATrialFeatureResult,
    testing: Dataset2ATrialFeatureResult,
    *,
    validation_mode: str = "algorithm1_training_reference",
    validation_window_size: int = 10,
    validation_alpha: float = 0.05,
    control_limit_multiplier: float = 1.96,
    variance_smoothing: float = 0.05,
    variance_update_mode: str = "always",
) -> Dataset2AExperimentResult:
    """Run one subject and retain both computed and published Table 1 values."""

    subject_id = f"A{subject:02d}"
    if subject_id not in PUBLISHED_2A_RESULTS:
        raise ValueError("subject must be an integer from 1 to 9.")
    published_lambda, published_csw, published_csv = PUBLISHED_2A_RESULTS[subject_id]
    covariance_method = (
        "empirical" if validation_mode == "paper_two_sample" else "shrinkage"
    )
    result = run_cse(
        training_features=training.features,
        testing_features=testing.features,
        testing_times=testing.times,
        config=CSEConfig(
            pca_components=min(3, training.features.shape[1]),
            lambda_override=published_lambda,
            variance_smoothing=variance_smoothing,
            control_limit_multiplier=control_limit_multiplier,
            variance_update_mode=variance_update_mode,
            ewma_initialization="training_mean",
            validation_mode=validation_mode,
            validation_before_size=validation_window_size,
            validation_after_size=validation_window_size,
            validation_alpha=validation_alpha,
            covariance_method=covariance_method,
        ),
    )
    return Dataset2AExperimentResult(
        subject=subject_id,
        published_lambda=published_lambda,
        published_csw=published_csw,
        published_csv=published_csv,
        computed_csw=int(result.warning_results["stage_1_alarm"].astype(bool).sum()),
        computed_csv=int(result.validation_results["confirmed_shift"].astype(bool).sum()),
        training_trials=int(training.features.shape[0]),
        testing_trials=int(testing.features.shape[0]),
        cse_result=result,
    )
