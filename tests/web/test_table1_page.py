from pathlib import Path

from app import _dataset_summary, _load_results, app
from tests.web.helpers import write_table1_results as _write_results


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
