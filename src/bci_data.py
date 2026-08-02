"""Data-loading helpers for BCI Competition IV Datasets 2A and 2B.

The CSE detector consumes feature matrices with shape
``(n_observations, n_features)``. Raw GDF loading is deliberately kept
separate from EEG preprocessing and feature extraction.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


_VALID_SESSIONS = {"T", "E"}
_DATASET_2B_SESSION_SUFFIX = {
    1: "T",
    2: "T",
    3: "T",
    4: "E",
    5: "E",
}


@dataclass(frozen=True)
class BCISessionData:
    """Raw EEG session loaded from one BCI Competition GDF file."""

    subject: int
    session: str
    file_path: Path
    signals: np.ndarray
    times: np.ndarray
    channel_names: tuple[str, ...]
    sampling_frequency: float
    annotations: tuple[tuple[float, float, str], ...]

    @property
    def n_samples(self) -> int:
        return int(self.signals.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.signals.shape[1])


def _validate_subject(subject: int) -> int:
    if isinstance(subject, bool) or not isinstance(subject, (int, np.integer)):
        raise ValueError("subject must be an integer from 1 to 9.")

    value = int(subject)
    if not 1 <= value <= 9:
        raise ValueError("subject must be an integer from 1 to 9.")
    return value


def _validate_session(session: str) -> str:
    value = str(session).upper()
    if value not in _VALID_SESSIONS:
        raise ValueError("session must be 'T' or 'E'.")
    return value


def _validate_dataset_2b_session(session: int) -> int:
    if isinstance(session, bool) or not isinstance(session, (int, np.integer)):
        raise ValueError("Dataset 2B session must be an integer from 1 to 5.")

    value = int(session)
    if value not in _DATASET_2B_SESSION_SUFFIX:
        raise ValueError("Dataset 2B session must be an integer from 1 to 5.")
    return value


def dataset_2a_filename(subject: int, session: str) -> str:
    """Return an official Dataset 2A GDF filename."""

    subject_value = _validate_subject(subject)
    session_value = _validate_session(session)
    return f"A{subject_value:02d}{session_value}.gdf"


def dataset_2b_filename(subject: int, session: int) -> str:
    """Return an official Dataset 2B GDF filename.

    Sessions 1--3 are training files and sessions 4--5 are evaluation files,
    for example ``B0101T.gdf`` and ``B0104E.gdf``.
    """

    subject_value = _validate_subject(subject)
    session_value = _validate_dataset_2b_session(session)
    suffix = _DATASET_2B_SESSION_SUFFIX[session_value]
    return f"B{subject_value:02d}{session_value:02d}{suffix}.gdf"


def resolve_dataset_2a_path(
    data_directory: str | Path,
    subject: int,
    session: str,
) -> Path:
    """Resolve and validate the expected Dataset 2A GDF path."""

    path = Path(data_directory) / dataset_2a_filename(subject, session)
    if not path.is_file():
        raise FileNotFoundError(
            "BCI Competition IV Dataset 2A file was not found: "
            f"{path}"
        )
    return path


def resolve_dataset_2b_path(
    data_directory: str | Path,
    subject: int,
    session: int,
) -> Path:
    """Resolve and validate the expected Dataset 2B GDF path."""

    path = Path(data_directory) / dataset_2b_filename(subject, session)
    if not path.is_file():
        raise FileNotFoundError(
            "BCI Competition IV Dataset 2B file was not found: "
            f"{path}"
        )
    return path


def _import_mne() -> Any:
    try:
        import mne
    except ImportError as exc:
        raise ImportError(
            "Loading GDF files requires MNE. Install it with: "
            "python -m pip install mne"
        ) from exc
    return mne


def _read_gdf(
    file_path: Path,
    *,
    subject: int,
    session: str,
    eeg_only: bool,
    preload: bool,
    allow_nonfinite: bool,
) -> BCISessionData:
    mne = _import_mne()
    raw = mne.io.read_raw_gdf(
        file_path,
        preload=preload,
        verbose="ERROR",
    )

    if eeg_only:
        raw = raw.copy().pick(picks="eeg")

    signals = np.asarray(raw.get_data(), dtype=float).T
    times = np.asarray(raw.times, dtype=float)

    if signals.ndim != 2:
        raise RuntimeError("Loaded EEG data must be two-dimensional.")
    if signals.shape[0] != times.size:
        raise RuntimeError(
            "Loaded EEG samples and time values must have equal length."
        )
    if not allow_nonfinite and not np.isfinite(signals).all():
        raise ValueError("Loaded EEG signals contain non-finite values.")

    annotations = tuple(
        (float(onset), float(duration), str(description))
        for onset, duration, description in zip(
            raw.annotations.onset,
            raw.annotations.duration,
            raw.annotations.description,
        )
    )

    return BCISessionData(
        subject=subject,
        session=session,
        file_path=file_path,
        signals=signals,
        times=times,
        channel_names=tuple(str(name) for name in raw.ch_names),
        sampling_frequency=float(raw.info["sfreq"]),
        annotations=annotations,
    )


def load_bci_competition_iv_2a_session(
    data_directory: str | Path,
    subject: int,
    session: str,
    *,
    eeg_only: bool = True,
    preload: bool = True,
) -> BCISessionData:
    """Load one Dataset 2A session using MNE."""

    subject_value = _validate_subject(subject)
    session_value = _validate_session(session)
    file_path = resolve_dataset_2a_path(
        data_directory=data_directory,
        subject=subject_value,
        session=session_value,
    )
    return _read_gdf(
        file_path,
        subject=subject_value,
        session=session_value,
        eeg_only=eeg_only,
        preload=preload,
        allow_nonfinite=False,
    )


def load_bci_competition_iv_2b_session(
    data_directory: str | Path,
    subject: int,
    session: int,
    *,
    eeg_only: bool = True,
    preload: bool = True,
) -> BCISessionData:
    """Load one Dataset 2B session using MNE.

    Dataset 2B recordings may contain non-finite separator samples between
    runs. They are preserved here so downstream window extraction can skip
    windows that cross a run boundary.
    """

    subject_value = _validate_subject(subject)
    session_value = _validate_dataset_2b_session(session)
    file_path = resolve_dataset_2b_path(
        data_directory=data_directory,
        subject=subject_value,
        session=session_value,
    )
    suffix = _DATASET_2B_SESSION_SUFFIX[session_value]
    return _read_gdf(
        file_path,
        subject=subject_value,
        session=f"{session_value:02d}{suffix}",
        eeg_only=eeg_only,
        preload=preload,
        allow_nonfinite=True,
    )


def load_cse_feature_file(
    file_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Load a processed CSE feature file stored in NumPy ``.npz`` format."""

    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSE feature file was not found: {path}")

    with np.load(path, allow_pickle=False) as contents:
        if "features" not in contents or "times" not in contents:
            raise ValueError(
                "feature file must contain 'features' and 'times' arrays."
            )

        features = np.asarray(contents["features"], dtype=float)
        times = np.asarray(contents["times"])
        labels = (
            np.asarray(contents["labels"])
            if "labels" in contents
            else None
        )

    if features.ndim != 2:
        raise ValueError("features must be a two-dimensional matrix.")
    if times.ndim != 1:
        raise ValueError("times must be one-dimensional.")
    if features.shape[0] != times.size:
        raise ValueError("features and times must have equal length.")

    if labels is not None:
        if labels.ndim != 1:
            raise ValueError("labels must be one-dimensional.")
        if labels.size != times.size:
            raise ValueError("labels and times must have equal length.")

    if not np.isfinite(features).all():
        raise ValueError("features must contain only finite values.")

    return features, times, labels
