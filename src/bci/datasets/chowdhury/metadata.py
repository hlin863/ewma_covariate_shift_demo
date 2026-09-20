"""Participant metadata for the Chowdhury CSE-UAEL stroke cohort."""

from collections.abc import Iterable
from dataclasses import dataclass
import re


CHOWDHURY_PARTICIPANT_IDS = tuple(f"S{index:02d}" for index in range(1, 11))
VALID_GENDERS = frozenset({"Female", "Male"})
VALID_SIDES = frozenset({"Left", "Right"})


def validate_participant_id(value: str) -> str:
    """Return a canonical S01--S10 identifier or raise ``ValueError``."""

    if not isinstance(value, str):
        raise ValueError("participant_id must be a string from S01 to S10.")
    participant_id = value.strip().upper()
    if not re.fullmatch(r"S\d{2}", participant_id):
        raise ValueError("participant_id must use the S01 to S10 format.")
    if participant_id not in CHOWDHURY_PARTICIPANT_IDS:
        raise ValueError("participant_id must identify a participant from S01 to S10.")
    return participant_id


def _normalise_category(value: str, *, field: str, allowed: frozenset[str]) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be one of {sorted(allowed)}.")
    lookup = {item.casefold(): item for item in allowed}
    normalised = lookup.get(value.strip().casefold())
    if normalised is None:
        raise ValueError(f"{field} must be one of {sorted(allowed)}.")
    return normalised


def _validate_nonnegative_integer(value: int, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be a non-negative integer.")
    if value < 0:
        raise ValueError(f"{field} must be a non-negative integer.")
    return value


@dataclass(frozen=True)
class ChowdhuryParticipant:
    participant_id: str
    age_years: int
    gender: str
    impaired_side: str
    dominant_side: str
    time_since_stroke_months: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "participant_id",
            validate_participant_id(self.participant_id),
        )
        object.__setattr__(
            self,
            "gender",
            _normalise_category(
                self.gender,
                field="gender",
                allowed=VALID_GENDERS,
            ),
        )
        object.__setattr__(
            self,
            "impaired_side",
            _normalise_category(
                self.impaired_side,
                field="impaired_side",
                allowed=VALID_SIDES,
            ),
        )
        object.__setattr__(
            self,
            "dominant_side",
            _normalise_category(
                self.dominant_side,
                field="dominant_side",
                allowed=VALID_SIDES,
            ),
        )
        object.__setattr__(
            self,
            "age_years",
            _validate_nonnegative_integer(self.age_years, field="age_years"),
        )
        object.__setattr__(
            self,
            "time_since_stroke_months",
            _validate_nonnegative_integer(
                self.time_since_stroke_months,
                field="time_since_stroke_months",
            ),
        )

    @property
    def impaired_hand(self) -> str:
        return self.impaired_side

    @property
    def is_dominant_side_impaired(self) -> bool:
        return self.impaired_side == self.dominant_side


def participant_index(
    participants: Iterable[ChowdhuryParticipant],
) -> dict[str, ChowdhuryParticipant]:
    """Index participants by ID while rejecting ambiguous duplicate rows."""

    index: dict[str, ChowdhuryParticipant] = {}
    for participant in participants:
        if participant.participant_id in index:
            raise ValueError(
                f"duplicate participant_id: {participant.participant_id}"
            )
        index[participant.participant_id] = participant
    return index
