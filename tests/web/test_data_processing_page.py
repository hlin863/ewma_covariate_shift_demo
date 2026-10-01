"""The protocol page distinguishes source evidence from generated output."""

from pathlib import Path

import pandas as pd

from app import app
from src.web.data_processing import build_data_processing_view


def test_page_shows_split_and_paper_boundaries_without_generated_files(tmp_path: Path) -> None:
    app.config.update(TESTING=True, RESULTS_ROOT=str(tmp_path))
    response = app.test_client().get("/data-processing")
    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "280 fit · 70%" in html
    assert "120 validation · 30%" in html
    assert "320 held out" in html
    assert "User-provided PowerShell run transcript" in html
    assert "17" in html and "5" in html
    assert "§4.1.2 and §4.3" in html
    assert "Dataset 2A development" in html
    assert "10.1007/s00500-015-1937-5" in html
    assert "does not implement that published selection/evaluation sequence" in html
    assert 'href="/papers/2019-raza-cse-uael.pdf"' in html
    assert 'role="tablist"' in html


def test_page_uses_new_local_artifacts_when_present(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    pd.DataFrame([
        {"dataset": "2B", "subject": "B01", "method": "single_linear_svm", "accuracy": .6, "evaluation_scope": "Session III"},
        {"dataset": "2B", "subject": "B01", "method": "bagged_linear_svm", "accuracy": .7, "evaluation_scope": "Session III"},
    ]).to_csv(metrics / "bagging_bci_real.csv", index=False)
    pd.DataFrame([{
        "subject": "B01", "published_csw": 14, "computed_csw": 18,
        "published_csv": 10, "computed_csv": 6,
    }]).to_csv(metrics / "bci_2b_table1_results.csv", index=False)

    model = build_data_processing_view(tmp_path)
    assert model["accuracy_means"]["2B"]["bagged"] == .7
    assert model["accuracy_means"]["2B"]["subjects"] == 1
    assert model["accuracy_scopes"]["2B"] == ["Session III"]
    assert model["diagnostic"]["computed_csw"] == 18
    assert "Generated locally" in model["accuracy_source"]
    assert "Generated locally" in model["diagnostic_source"]


def test_bad_local_accuracy_file_falls_back_with_clear_provenance(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    (metrics / "bagging_bci_real.csv").write_text("bad,columns\n1,2\n", encoding="utf-8")
    model = build_data_processing_view(tmp_path)
    assert model["accuracy_error"] is not None
    assert len(model["accuracy_rows"]) == 18
    assert "User-provided" in model["accuracy_source"]
