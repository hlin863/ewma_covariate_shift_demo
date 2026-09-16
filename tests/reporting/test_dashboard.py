from pathlib import Path

import pandas as pd

from app import _dataset_summary, _load_results, app
from src.web.dashboard import _build_results_catalog


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

    response = app.test_client().get("/")

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

    response = app.test_client().get("/")

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
