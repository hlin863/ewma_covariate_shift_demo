from pathlib import Path

import pandas as pd

from app import app
from src.bci.datasets.chowdhury.metadata import ChowdhuryParticipant
from src.web.chowdhury import build_chowdhury_cohort_view


def _write_chowdhury_demographics(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            ["S01", 48, "Male", "Left", "Right", 8],
            ["S02", 71, "Male", "Left", "Right", 20],
            ["S03", 63, "Male", "Right", "Right", 8],
        ],
        columns=[
            "participant_id",
            "age_years",
            "gender",
            "impaired_side",
            "dominant_side",
            "time_since_stroke_months",
        ],
    ).to_csv(path / "patient_demographics.csv", index=False)



def test_chowdhury_cohort_view_computes_descriptive_statistics() -> None:
    participants = (
        ChowdhuryParticipant("S01", 48, "Male", "Left", "Right", 8),
        ChowdhuryParticipant("S02", 71, "Male", "Left", "Right", 20),
        ChowdhuryParticipant("S03", 63, "Male", "Right", "Right", 8),
    )

    cohort = build_chowdhury_cohort_view(participants)

    assert cohort["summary"]["participants"] == 3
    assert cohort["summary"]["mean_age"] == 182 / 3
    assert cohort["summary"]["median_stroke_months"] == 8
    assert cohort["summary"]["dominant_side_impaired"] == 1
    assert cohort["impaired_side_distribution"][0]["count"] == 2
    assert cohort["participants"][1]["age_width"] == 100.0
    assert cohort["participants"][1]["stroke_width"] == 100.0



def test_chowdhury_demographics_page_renders_visualisation(tmp_path: Path) -> None:
    _write_chowdhury_demographics(tmp_path)
    app.config.update(TESTING=True, CHOWDHURY_DATA_PATH=str(tmp_path))

    response = app.test_client().get("/chowdhury-demographics")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Chowdhury stroke cohort" in html
    assert "Cleveland dot plots" in html
    assert "Age by participant" in html
    assert "Time since stroke" in html
    assert html.count('class="cleveland-dot') == 6
    assert "Age and time since stroke" in html
    assert "S01" in html
    assert "Dominant side impaired" in html
    assert "Metadata is connected; raw EEG is not yet decoded" in html



def test_chowdhury_demographics_page_has_missing_data_state(
    tmp_path: Path,
) -> None:
    app.config.update(
        TESTING=True,
        CHOWDHURY_DATA_PATH=str(tmp_path / "missing"),
    )

    response = app.test_client().get("/chowdhury-demographics")

    assert response.status_code == 200
    assert "Cohort metadata could not be loaded" in response.get_data(as_text=True)
