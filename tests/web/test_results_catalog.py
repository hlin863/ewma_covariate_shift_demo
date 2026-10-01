from pathlib import Path

import pandas as pd

from app import app
from src.web.dashboard import _build_results_catalog
from tests.web.helpers import write_table1_results as _write_results


def _write_lambda_sse(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            [0.01, 10.0],
            [0.02, 7.5],
            [0.03, 8.0],
        ],
        columns=["lambda", "sse"],
    ).to_csv(path, index=False)



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
    sse_figure_path = tmp_path / "figures" / "cse_lambda_sse_curve.png"
    sse_figure_path.write_bytes(b"png")
    for filename in (
        "cse_lambda_confirmed_shifts.png",
        "cse_lambda_rci.png",
    ):
        (tmp_path / "figures" / filename).write_bytes(b"png")
    _write_lambda_sse(
        tmp_path / "metrics" / "cse_lambda_sse.csv"
    )
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
    assert "Synthetic CSE lambda analysis" in html
    assert "Prediction-error SSE" in html
    assert "Prediction-error SSE by lambda" in html
    assert "Lambda selection and downstream sensitivity" in html
    assert "Selected training parameter" in html
    assert "Best λ" in html
    assert "0.02" in html
    assert "Minimum SSE" in html
    assert "7.5000" in html
    assert "Downstream sensitivity" in html
    assert "figures/cse_lambda_sse_curve.png" in html
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



def test_results_catalog_links_to_ks_validation_page(tmp_path: Path) -> None:
    app.config.update(TESTING=True, RESULTS_ROOT=str(tmp_path))

    response = app.test_client().get("/results")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "View K–S Stage-II validation" in html
    assert 'href="/results/paper2015-d2/ks-validation"' in html
