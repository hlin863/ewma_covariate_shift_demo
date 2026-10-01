from pathlib import Path

import pandas as pd

from app import app
from src.web.dashboard import _build_results_catalog


def _write_table3_comparison(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            ["SD-EWMA", 3.733602, 0.3, 0.0, 0.0, 9.666667, 0.0, 0.31, 0.189],
            ["TSSD-EWMA", 0.201816, 0.0, 55.555556, 11.1, 19.75, 10.0, 0.42, 0.224],
            ["ICI-CDT", 0.100908, 0.0, 0.0, 55.4, 27.888889, 40.0, 0.29, 0.196],
        ],
        columns=[
            "method",
            "computed_fp_percent",
            "published_fp_percent",
            "computed_fn_percent",
            "published_fn_percent",
            "computed_rci",
            "published_rci",
            "computed_ct_seconds",
            "published_ct_seconds",
        ],
    ).to_csv(path, index=False)



def test_table3_visualisation_builds_confusion_counts_and_tradeoff_data(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metrics" / "paper2015" / "d2" / "table3_comparison.csv"
    _write_table3_comparison(path)

    catalog = _build_results_catalog(tmp_path)
    table3 = next(item for item in catalog if item["id"] == "paper2015-d2")
    visual = table3["table3_visualisation"]

    assert visual is not None
    assert visual["true_shift_count"] == 9
    assert visual["non_shift_count"] == 991
    tssd = next(row for row in visual["methods"] if row["method"] == "TSSD-EWMA")
    assert tssd["fn"] == 5
    assert tssd["tp"] == 4
    assert tssd["fp"] == 2
    assert tssd["tn"] == 989
    assert tssd["computed_rci"] == 19.75
    assert tssd["published_rci"] == 10.0



def test_results_catalog_renders_table3_confusion_and_rci_ct_charts(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metrics" / "paper2015" / "d2" / "table3_comparison.csv"
    _write_table3_comparison(path)
    app.config.update(TESTING=True, RESULTS_ROOT=str(tmp_path))

    response = app.test_client().get("/results")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Table III visual analysis" in html
    assert "Confusion matrices" in html
    assert "RCI / CT trade-off" in html
    assert "TP" in html
    assert "FN" in html
    assert "FP" in html
    assert "TN" in html
    assert "991 non-shift observations" in html
