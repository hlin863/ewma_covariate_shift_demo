from pathlib import Path

import pandas as pd

from app import app
from src.web.dashboard import _load_ks_validation_results


def _write_ks_validation_results(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            [50, 75, 25, 25, 0.20, 0.7102, "rejected", False],
            [100, 125, 25, 25, 0.60, 0.0001, "confirmed", True],
            [150, None, 25, 25, float("nan"), float("nan"), "pending_after_window", False],
            [175, None, 25, 25, float("nan"), float("nan"), "skipped_nearby_alarm", False],
            [5, None, 25, 25, float("nan"), float("nan"), "insufficient_before_window", False],
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
    assert model["summary"]["total"] == 5
    assert model["summary"]["evaluated"] == 2
    assert model["summary"]["confirmed"] == 1
    assert model["summary"]["rejected"] == 1
    assert model["summary"]["pending"] == 1
    assert model["summary"]["skipped"] == 1
    assert model["summary"]["insufficient"] == 1
    assert model["summary"]["loss_total"] == 4
    assert model["loss_chart"]["rejected_end"] == 25.0
    assert model["loss_chart"]["pending_end"] == 50.0
    assert model["loss_chart"]["skipped_end"] == 75.0
    assert model["loss_chart"]["insufficient_end"] == 100.0
    assert len(model["evaluated_rows"]) == 2
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
    assert "Loss / non-confirmation cases" in html
    assert "Equation (6) evaluated warnings" in html
    assert "Full audit trail" in html
    assert "Rejected by K-S" in html
    assert "Pending future window" in html
    assert "Skipped nearby alarm" in html
    assert "Insufficient window" in html
    assert "confirmed" in html
    assert 'href="/results"' in html
