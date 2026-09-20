from dataclasses import dataclass


@dataclass(frozen=True)
class ChowdhuryParticipant:
    participant_id: str
    age_years: int
    gender: str
    impaired_side: str
    dominant_side: str
    time_since_stroke_months: int

    @property
    def impaired_hand(self) -> str:
        return self.impaired_side

    @property
    def is_dominant_side_impaired(self) -> bool:
        return self.impaired_side == self.dominant_side
