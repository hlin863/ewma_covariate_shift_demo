from pathlib import Path

import pandas as pd

from app import _dataset_summary, _load_results, app


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
    assert "A07 shift visualisation" in html


def test_dashboard_shows_generation_command_when_results_missing(tmp_path: Path) -> None:
    app.config.update(
        TESTING=True,
        TABLE1_RESULTS_PATH=str(tmp_path / "missing.csv"),
    )

    response = app.test_client().get("/")

    assert response.status_code == 200
    assert "run_bci_table1_reproduction.py" in response.get_data(as_text=True)


def test_covariate_shift_page_renders_mu_and_beta_panels(tmp_path: Path) -> None:
    app.config.update(
        TESTING=True,
        A07_COVARIATE_SHIFT_PATH=str(tmp_path / "missing.json"),
    )

    response = app.test_client().get("/covariate-shift")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Covariate shift across μ and β bands" in html
    assert "8–12 Hz" in html
    assert "14–30 Hz" in html
    assert "Paper-aligned schematic" in html
    assert 'data-band="mu"' in html
    assert 'data-band="beta"' in html
