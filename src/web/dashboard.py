"""Flask dashboard for visualising BCI Table 1 reproduction results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from flask import Flask, render_template


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "outputs" / "metrics" / "bci_table1_comparison.csv"

app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / "templates"),
    static_folder=str(PROJECT_ROOT / "static"),
)
app.config.setdefault("TABLE1_RESULTS_PATH", str(DEFAULT_RESULTS_PATH))


def _load_results(path: str | Path) -> pd.DataFrame:
    results_path = Path(path)
    if not results_path.is_file():
        return pd.DataFrame()

    frame = pd.read_csv(results_path)
    required = {
        "dataset",
        "subject",
        "lambda",
        "published_csw",
        "computed_csw",
        "csw_difference",
        "published_csv",
        "computed_csv",
        "csv_difference",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "Table 1 comparison file is missing required columns: "
            + ", ".join(sorted(missing))
        )

    numeric = [
        "lambda",
        "published_csw",
        "computed_csw",
        "csw_difference",
        "published_csv",
        "computed_csv",
        "csv_difference",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame


def _dataset_summary(frame: pd.DataFrame, dataset: str) -> dict[str, float | int | str]:
    group = frame.loc[frame["dataset"] == dataset].copy()
    if group.empty:
        return {
            "dataset": dataset,
            "subjects": 0,
            "published_csw_mean": 0.0,
            "computed_csw_mean": 0.0,
            "published_csv_mean": 0.0,
            "computed_csv_mean": 0.0,
            "csw_mae": 0.0,
            "csv_mae": 0.0,
        }

    return {
        "dataset": dataset,
        "subjects": int(group.shape[0]),
        "published_csw_mean": float(group["published_csw"].mean()),
        "computed_csw_mean": float(group["computed_csw"].mean()),
        "published_csv_mean": float(group["published_csv"].mean()),
        "computed_csv_mean": float(group["computed_csv"].mean()),
        "csw_mae": float(group["csw_difference"].abs().mean()),
        "csv_mae": float(group["csv_difference"].abs().mean()),
    }


def _chart_rows(frame: pd.DataFrame, dataset: str) -> list[dict[str, object]]:
    group = frame.loc[frame["dataset"] == dataset].sort_values("subject")
    if group.empty:
        return []

    max_value = max(
        float(group["published_csw"].max()),
        float(group["computed_csw"].max()),
        float(group["published_csv"].max()),
        float(group["computed_csv"].max()),
        1.0,
    )

    rows: list[dict[str, object]] = []
    for _, row in group.iterrows():
        rows.append({
            "subject": str(row["subject"]),
            "lambda": float(row["lambda"]),
            "published_csw": int(row["published_csw"]),
            "computed_csw": int(row["computed_csw"]),
            "published_csv": int(row["published_csv"]),
            "computed_csv": int(row["computed_csv"]),
            "csw_difference": int(row["csw_difference"]),
            "csv_difference": int(row["csv_difference"]),
            "published_csw_width": 100.0 * float(row["published_csw"]) / max_value,
            "computed_csw_width": 100.0 * float(row["computed_csw"]) / max_value,
            "published_csv_width": 100.0 * float(row["published_csv"]) / max_value,
            "computed_csv_width": 100.0 * float(row["computed_csv"]) / max_value,
        })
    return rows


@app.get("/")
def dashboard():
    results_path = Path(app.config["TABLE1_RESULTS_PATH"])
    frame = _load_results(results_path)

    if frame.empty:
        return render_template(
            "dashboard.html",
            data_available=False,
            results_path=results_path,
            summaries=[],
            rows_2a=[],
            rows_2b=[],
        )

    summaries = [_dataset_summary(frame, "2A"), _dataset_summary(frame, "2B")]
    return render_template(
        "dashboard.html",
        data_available=True,
        results_path=results_path,
        summaries=summaries,
        rows_2a=_chart_rows(frame, "2A"),
        rows_2b=_chart_rows(frame, "2B"),
    )
