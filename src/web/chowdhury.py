"""Presentation helpers for the Chowdhury stroke-cohort metadata page."""

from collections import Counter
from statistics import mean, median

from src.bci.datasets.chowdhury.metadata import ChowdhuryParticipant


def build_chowdhury_cohort_view(
    participants: tuple[ChowdhuryParticipant, ...],
) -> dict[str, object]:
    """Build JSON/template-safe summaries without changing the source data."""

    if not participants:
        raise ValueError("at least one Chowdhury participant is required.")

    ordered = tuple(sorted(participants, key=lambda item: item.participant_id))
    ages = [item.age_years for item in ordered]
    stroke_months = [item.time_since_stroke_months for item in ordered]
    gender_counts = Counter(item.gender for item in ordered)
    impaired_side_counts = Counter(item.impaired_side for item in ordered)
    dominant_impaired = sum(item.is_dominant_side_impaired for item in ordered)
    max_age = max(ages) or 1
    max_stroke_months = max(stroke_months) or 1

    rows = []
    for participant in ordered:
        rows.append({
            "participant_id": participant.participant_id,
            "age_years": participant.age_years,
            "gender": participant.gender,
            "impaired_side": participant.impaired_side,
            "dominant_side": participant.dominant_side,
            "time_since_stroke_months": participant.time_since_stroke_months,
            "dominant_side_impaired": participant.is_dominant_side_impaired,
            "age_width": 100.0 * participant.age_years / max_age,
            "stroke_width": (
                100.0 * participant.time_since_stroke_months / max_stroke_months
            ),
        })

    total = len(ordered)
    return {
        "summary": {
            "participants": total,
            "mean_age": mean(ages),
            "minimum_age": min(ages),
            "maximum_age": max(ages),
            "median_stroke_months": median(stroke_months),
            "mean_stroke_months": mean(stroke_months),
            "minimum_stroke_months": min(stroke_months),
            "maximum_stroke_months": max(stroke_months),
            "dominant_side_impaired": dominant_impaired,
        },
        "gender_distribution": [
            {
                "label": label,
                "count": gender_counts.get(label, 0),
                "percent": 100.0 * gender_counts.get(label, 0) / total,
            }
            for label in ("Male", "Female")
        ],
        "impaired_side_distribution": [
            {
                "label": label,
                "count": impaired_side_counts.get(label, 0),
                "percent": 100.0 * impaired_side_counts.get(label, 0) / total,
            }
            for label in ("Left", "Right")
        ],
        "participants": rows,
    }
