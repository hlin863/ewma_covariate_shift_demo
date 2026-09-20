"""File loading and path helpers for the Chowdhury CSE-UAEL dataset.

Only the published demographic table is decoded here. The raw EEG parser is
deliberately left format-agnostic until the original recording files and their
session schema are available.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.bci.datasets.chowdhury.metadata import (
    ChowdhuryParticipant,
    participant_index,
    validate_participant_id,
)


DEMOGRAPHICS_FILENAME = "patient_demographics.csv"
DEMOGRAPHICS_COLUMNS = (
    "participant_id",
    "age_years",
    "gender",
    "impaired_side",
    "dominant_side",
    "time_since_stroke_months",
)


@dataclass(frozen=True)
class ChowdhuryEEGSession:
    """In-memory representation to be returned by a future raw EEG decoder."""

    participant_id: str
    session_id: str
    file_path: Path
    signals: np.ndarray
    times: np.ndarray
    sampling_frequency: float
    channel_names: tuple[str, ...]

    def __post_init__(self) -> None:
        participant_id = validate_participant_id(self.participant_id)
        session_id = str(self.session_id).strip()
        signals = np.asarray(self.signals, dtype=float)
        times = np.asarray(self.times, dtype=float)
        channel_names = tuple(str(name).strip() for name in self.channel_names)

        if not session_id:
            raise ValueError("session_id must not be empty.")
        if signals.ndim != 2:
            raise ValueError("signals must have shape (n_samples, n_channels).")
        if times.ndim != 1 or times.size != signals.shape[0]:
            raise ValueError("times must contain one value per signal sample.")
        if len(channel_names) != signals.shape[1] or not all(channel_names):
            raise ValueError("channel_names must contain one name per signal channel.")
        if not np.isfinite(signals).all() or not np.isfinite(times).all():
            raise ValueError("signals and times must contain only finite values.")
        if self.sampling_frequency <= 0.0:
            raise ValueError("sampling_frequency must be positive.")

        object.__setattr__(self, "participant_id", participant_id)
        object.__setattr__(self, "session_id", session_id)
        object.__setattr__(self, "file_path", Path(self.file_path))
        object.__setattr__(self, "signals", signals)
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "sampling_frequency", float(self.sampling_frequency))
        object.__setattr__(self, "channel_names", channel_names)

    @property
    def n_samples(self) -> int:
        return int(self.signals.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.signals.shape[1])


def resolve_demographics_path(data_directory: str | Path) -> Path:
    """Resolve the demographic CSV beneath a Chowdhury dataset root."""

    path = Path(data_directory) / DEMOGRAPHICS_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"Chowdhury demographic file was not found: {path}")
    return path


def resolve_participant_directory(
    data_directory: str | Path,
    participant_id: str,
) -> Path:
    """Resolve ``eeg/Sxx`` without assuming a recording file extension."""

    participant = validate_participant_id(participant_id)
    path = Path(data_directory) / "eeg" / participant
    if not path.is_dir():
        raise FileNotFoundError(
            f"Chowdhury participant EEG directory was not found: {path}"
        )
    return path


def resolve_eeg_path(
    data_directory: str | Path,
    participant_id: str,
    filename: str | Path,
) -> Path:
    """Resolve an explicitly named EEG file without inventing its format."""

    relative_name = Path(filename)
    if relative_name.is_absolute() or len(relative_name.parts) != 1:
        raise ValueError("filename must be a single relative file name.")
    path = resolve_participant_directory(data_directory, participant_id) / relative_name
    if not path.is_file():
        raise FileNotFoundError(f"Chowdhury EEG file was not found: {path}")
    return path


def _required_integer(value: object, *, field: str, row_number: int) -> int:
    if pd.isna(value):
        raise ValueError(f"row {row_number}: {field} must not be missing.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"row {row_number}: {field} must be an integer.") from exc
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"row {row_number}: {field} must be an integer.")
    return int(numeric)


def load_patient_demographics(
    data_directory: str | Path,
) -> tuple[ChowdhuryParticipant, ...]:
    """Load the published S01--S10 demographic table as typed records."""

    path = resolve_demographics_path(data_directory)
    frame = pd.read_csv(path)
    missing = [column for column in DEMOGRAPHICS_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"patient demographics missing columns: {missing}")
    if frame.empty:
        raise ValueError("patient demographics must contain at least one row.")

    participants: list[ChowdhuryParticipant] = []
    for offset, row in frame.loc[:, DEMOGRAPHICS_COLUMNS].iterrows():
        row_number = int(offset) + 2
        try:
            participants.append(
                ChowdhuryParticipant(
                    participant_id=row["participant_id"],
                    age_years=_required_integer(
                        row["age_years"], field="age_years", row_number=row_number
                    ),
                    gender=row["gender"],
                    impaired_side=row["impaired_side"],
                    dominant_side=row["dominant_side"],
                    time_since_stroke_months=_required_integer(
                        row["time_since_stroke_months"],
                        field="time_since_stroke_months",
                        row_number=row_number,
                    ),
                )
            )
        except ValueError as exc:
            if str(exc).startswith(f"row {row_number}:"):
                raise
            raise ValueError(f"row {row_number}: {exc}") from exc

    participant_index(participants)
    return tuple(participants)
