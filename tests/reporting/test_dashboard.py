from pathlib import Path

import pandas as pd

from app import _dataset_summary, _load_results, app
from src.bci.datasets.chowdhury.metadata import ChowdhuryParticipant
from src.web.chowdhury import build_chowdhury_cohort_view
from src.web.dashboard import _build_results_catalog, _load_ks_validation_results


COLUMNS = [
    "dataset",
    "subject",
    "lambda",
    "published_csw",
    "computed_csw",
    "csw_difference",
    "published_csv",
    "computed_csv",
    "csv_difference",
]


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


def _write_results(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            ["2A", "A01", 0.50, 12, 16, 4, 6, 1, -5],
            ["2A", "A02", 0.55, 15, 16, 1, 8, 2, -6],
            ["2B", "B01", 0.28, 14, 17, 3, 10, 0, -10],
            ["2B", "B02", 0.17, 18, 13, -5, 13, 1, -12],
        ],
        columns=COLUMNS,
    ).to_csv(path, index=False)


def test_home_page_presents_research_scope_and_application_map() -> None:
    response = app.test_client().get("/")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "EWMA Covariate Shift Research Hub" in html
    assert "Current project progress" in html
    assert "CSE warning and validation are implemented and testable" in html
    assert "The full CSE-UAEL loop" in html
    assert "Table 1 dashboard" in html
    assert 'href="/table-1"' in html
    assert 'href="/results"' in html
    assert 'href="/figure-1"' in html
    assert 'href="/chowdhury-demographics"' in html
    assert 'href="/tests"' in html
    assert 'href="/papers/2015-raza-ewma-covariate-shift.pdf"' in html
    assert 'href="/papers/2018-chowdhury-online-adaptive-bci.pdf"' in html
    assert 'href="/papers/2019-raza-cse-uael.pdf"' in html
    assert html.count("Preview paper PDF") == 3


def test_paper_preview_serves_repository_pdf_inline() -> None:
    response = app.test_client().get(
        "/papers/2015-raza-ewma-covariate-shift.pdf"
    )

    assert response.status_code == 200
    assert response.mimetype == "application/pdf"
    assert response.data.startswith(b"%PDF-")
    assert response.headers["Content-Disposition"].startswith("inline;")


def test_paper_preview_rejects_non_pdf_files() -> None:
    response = app.test_client().get("/papers/README.md")

    assert response.status_code == 404


def test_dashboard_summary_uses_computed_and_published_values(tmp_path: Path) -> None:
    path = tmp_path / "comparison.csv"
    _write_results(path)
    frame = _load_results(path)
    summary = _dataset_summary(frame, "2A")

    assert summary["subjects"] == 2
    assert summary["published_csw_mean"] == 13.5
    assert summary["computed_csw_mean"] == 16.0
    assert summary["published_csv_mean"] == 7.0
    assert summary["computed_csv_mean"] == 1.5
    assert summary["csw_mae"] == 2.5
    assert summary["csv_mae"] == 5.5


def test_dashboard_renders_current_reproduction_results(tmp_path: Path) -> None:
    path = tmp_path / "comparison.csv"
    _write_results(path)
    app.config.update(TESTING=True, TABLE1_RESULTS_PATH=str(path))

    response = app.test_client().get("/table-1")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "BCI Competition IV" in html
    assert "A01" in html
    assert "B01" in html
    assert "Published CSV mean" in html
    assert "Computed CSV mean" in html
    assert "Research results" in html


def test_dashboard_shows_generation_command_when_results_missing(tmp_path: Path) -> None:
    app.config.update(
        TESTING=True,
        TABLE1_RESULTS_PATH=str(tmp_path / "missing.csv"),
    )

    response = app.test_client().get("/table-1")

    assert response.status_code == 200
    assert "run_bci_table1_reproduction.py" in response.get_data(as_text=True)


def test_results_catalog_detects_available_primary_result(tmp_path: Path) -> None:
    comparison_path = tmp_path / "metrics" / "bci_table1_comparison.csv"
    _write_results(comparison_path)
    figure_path = tmp_path / "figures" / "cse_lambda_stage1_warnings.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.write_bytes(b"png")

    catalog = _build_results_catalog(tmp_path)
    table1 = next(item for item in catalog if item["id"] == "bci-table1")
    lambda_sweep = next(item for item in catalog if item["id"] == "lambda-sweep")

    assert table1["status"] == "available"
    assert table1["available_count"] == 1
    assert table1["preview"]["rows"] == 4
    assert table1["preview"]["records"][0]["subject"] == "A01"
    assert any(artifact["is_figure"] for artifact in lambda_sweep["artifacts"])


def test_results_catalog_page_renders_domains_methods_and_theme_controls(
    tmp_path: Path,
) -> None:
    comparison_path = tmp_path / "metrics" / "bci_table1_comparison.csv"
    _write_results(comparison_path)
    figure_path = tmp_path / "figures" / "cse_lambda_stage1_warnings.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.write_bytes(b"png")
    app.config.update(TESTING=True, RESULTS_ROOT=str(tmp_path))

    response = app.test_client().get("/results")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Research results catalogue" in html
    assert "BCI Table 1 reproduction" in html
    assert "Raza 2015 D2 / Table III reproduction" in html
    assert "Datasets" in html
    assert "Methods" in html
    assert "Tracked fields" in html
    assert "computed_csw" in html
    assert "figures/cse_lambda_stage1_warnings.png" in html
    assert "Light" in html
    assert "Dark" in html
    assert "A01" in html
    assert "Not generated" in html


def test_results_catalog_serves_generated_output_files(tmp_path: Path) -> None:
    figure_path = tmp_path / "figures" / "cse_lambda_stage1_warnings.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.write_bytes(b"png")
    app.config.update(TESTING=True, RESULTS_ROOT=str(tmp_path))

    response = app.test_client().get("/outputs/figures/cse_lambda_stage1_warnings.png")

    assert response.status_code == 200
    assert response.data == b"png"


def _write_ks_validation_results(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            [50, 75, 25, 25, 0.20, 0.7102, "rejected", False],
            [100, 125, 25, 25, 0.60, 0.0001, "confirmed", True],
            [150, None, 25, 25, float("nan"), float("nan"), "pending_after_window", False],
        ],
        columns=[
            "alarm_time",
            "validation_time",
            "before_size",
            "after_size",
            "ks_statistic",
            "p_value",
            "status",
            "confirmed_shift",
        ],
    ).to_csv(path, index=False)


def test_ks_validation_view_computes_equation_6_evidence(tmp_path: Path) -> None:
    path = tmp_path / "metrics" / "paper2015" / "d2" / "stage2_validations.csv"
    _write_ks_validation_results(path)

    model = _load_ks_validation_results(tmp_path)

    assert model["available"] is True
    assert model["summary"]["total"] == 3
    assert model["summary"]["confirmed"] == 1
    assert model["summary"]["rejected"] == 1
    assert model["summary"]["pending"] == 1
    assert model["rows"][0]["scaled_statistic"] == 0.707107
    assert model["rows"][1]["scaled_statistic"] == 2.12132
    assert model["rows"][1]["margin"] == 0.76132
    assert model["rows"][1]["k_alpha"] == 1.36


def test_ks_validation_page_renders_under_results_hierarchy(tmp_path: Path) -> None:
    path = tmp_path / "metrics" / "paper2015" / "d2" / "stage2_validations.csv"
    _write_ks_validation_results(path)
    app.config.update(TESTING=True, RESULTS_ROOT=str(tmp_path))

    response = app.test_client().get("/results/paper2015-d2/ks-validation")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "K–S Stage-II validation" in html
    assert "Raza et al. (2015) · Equation (6)" in html
    assert "Scaled K–S statistic" in html
    assert "Critical value Kα" in html
    assert "confirmed" in html
    assert 'href="/results"' in html


def test_results_catalog_links_to_ks_validation_page(tmp_path: Path) -> None:
    app.config.update(TESTING=True, RESULTS_ROOT=str(tmp_path))

    response = app.test_client().get("/results")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "View K–S Stage-II validation" in html
    assert 'href="/results/paper2015-d2/ks-validation"' in html
