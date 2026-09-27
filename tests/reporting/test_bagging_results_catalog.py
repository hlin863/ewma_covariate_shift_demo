from pathlib import Path

import pandas as pd

from app import app
from src.web.dashboard import _build_results_catalog


def _write_bagging_results(root: Path) -> None:
    metrics = root / "metrics" / "bagging_synthetic_bci.csv"
    metrics.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            ["2A", "single_linear_svm", 48, 144, 140, 0.9722, 0.01, 0.001, 1, 1, 1.0, 0.0],
            ["2A", "bagged_linear_svm", 48, 144, 141, 0.9792, 0.08, 0.006, 30, 30, 0.8, 0.04],
            ["2B", "single_linear_svm", 24, 16, 15, 0.9375, 0.01, 0.001, 1, 1, 1.0, 0.0],
            ["2B", "bagged_linear_svm", 24, 16, 16, 1.0, 0.06, 0.004, 30, 30, 0.8, 0.07],
        ],
        columns=[
            "dataset",
            "method",
            "training_trials",
            "testing_trials",
            "correct_predictions",
            "accuracy",
            "fit_seconds",
            "predict_seconds",
            "requested_estimators",
            "fitted_estimators",
            "sample_fraction",
            "mean_member_disagreement",
        ],
    ).to_csv(metrics, index=False)

    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    (figures / "bagging_synthetic_accuracy.png").write_bytes(b"png")
    (figures / "bagging_synthetic_disagreement.png").write_bytes(b"png")


def test_results_catalog_exposes_bagging_synthetic_artifacts(tmp_path: Path) -> None:
    _write_bagging_results(tmp_path)

    catalog = _build_results_catalog(tmp_path)
    bagging = next(item for item in catalog if item["id"] == "bagging-synthetic-bci")

    assert bagging["status"] == "available"
    assert bagging["available_count"] == 3
    assert bagging["has_figures"] is True
    assert bagging["preview"]["rows"] == 4
    assert bagging["preview"]["records"][0]["dataset"] == "2A"


def test_results_page_renders_bagging_visualisations(tmp_path: Path) -> None:
    _write_bagging_results(tmp_path)
    app.config.update(TESTING=True, RESULTS_ROOT=str(tmp_path))

    response = app.test_client().get("/results")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Bagging integration on synthetic 2A/2B" in html
    assert "Bootstrap aggregation" in html
    assert "mean_member_disagreement" in html
    assert "figures/bagging_synthetic_accuracy.png" in html
    assert "figures/bagging_synthetic_disagreement.png" in html
