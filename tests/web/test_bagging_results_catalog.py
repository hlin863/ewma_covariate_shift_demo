from pathlib import Path

import pandas as pd

from app import app
from src.web.dashboard import _build_results_catalog


def _write_synthetic_bagging_results(root: Path) -> None:
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


def _write_real_bagging_results(root: Path) -> None:
    metrics = root / "metrics" / "bagging_bci_real.csv"
    metrics.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            ["2A", "A01", "Session II", "single_linear_svm", 144, 0, 144, 100, 0.6944, 0.02, 0.001, 1, 1, 1.0, 0.0],
            ["2A", "A01", "Session II", "bagged_linear_svm", 144, 0, 144, 104, 0.7222, 0.08, 0.006, 30, 30, 0.8, 0.12],
            ["2B", "B01", "Sessions IV-V", "single_linear_svm", 240, 160, 320, 220, 0.6875, 0.02, 0.001, 1, 1, 1.0, 0.0],
            ["2B", "B01", "Sessions IV-V", "bagged_linear_svm", 240, 160, 320, 226, 0.7063, 0.08, 0.006, 30, 30, 0.8, 0.15],
        ],
        columns=[
            "dataset",
            "subject",
            "evaluation_scope",
            "method",
            "training_trials",
            "calibration_trials",
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
    (figures / "bagging_bci_real_accuracy.png").write_bytes(b"png")
    (figures / "bagging_bci_real_disagreement.png").write_bytes(b"png")


def test_results_catalog_exposes_synthetic_smoke_test(tmp_path: Path) -> None:
    _write_synthetic_bagging_results(tmp_path)

    catalog = _build_results_catalog(tmp_path)
    bagging = next(item for item in catalog if item["id"] == "bagging-synthetic-bci")

    assert bagging["status"] == "available"
    assert bagging["available_count"] == 3
    assert bagging["has_figures"] is True
    assert bagging["preview"]["rows"] == 4
    assert bagging["title"] == "Synthetic bagging smoke test"


def test_results_catalog_exposes_real_bci_bagging_artifacts(tmp_path: Path) -> None:
    _write_real_bagging_results(tmp_path)

    catalog = _build_results_catalog(tmp_path)
    bagging = next(item for item in catalog if item["id"] == "bagging-bci-real")

    assert bagging["status"] == "available"
    assert bagging["available_count"] == 3
    assert bagging["has_figures"] is True
    assert bagging["preview"]["rows"] == 4
    assert bagging["preview"]["records"][0]["subject"] == "A01"


def test_results_page_distinguishes_real_data_from_synthetic_smoke_test(
    tmp_path: Path,
) -> None:
    _write_real_bagging_results(tmp_path)
    _write_synthetic_bagging_results(tmp_path)
    app.config.update(TESTING=True, RESULTS_ROOT=str(tmp_path))

    response = app.test_client().get("/results")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Bagging evaluation on BCI Competition IV 2A/2B" in html
    assert "BCI Competition IV Dataset 2A" in html
    assert "BCI Competition IV Dataset 2B" in html
    assert "Synthetic bagging smoke test" in html
    assert "These generated fixtures are not BCI Competition IV research results" in html
    assert "figures/bagging_bci_real_accuracy.png" in html
    assert "figures/bagging_bci_real_disagreement.png" in html
