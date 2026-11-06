"""Additive online-learning experiment page; existing page routes are unchanged."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from flask import Blueprint, current_app, render_template, request

from src.online.experiments import run_online_comparison


online_results_bp = Blueprint("online_results", __name__)


def _saved_bci_summaries(root: Path) -> list[dict[str, object]]:
    """Read existing CLI outputs only. Never fabricate BCI experiment results."""
    records: list[dict[str, object]] = []
    for path in sorted((root / "online").glob("2*/subject_*/**/summary.csv"))[:60]:
        try:
            frame = pd.read_csv(path)
            required = {"dataset", "subject", "policy", "trials", "accuracy", "warnings", "updates"}
            if frame.empty or not required.issubset(frame.columns):
                continue
            row = frame.iloc[0]
            records.append({
                "dataset": str(row["dataset"]).upper(),
                "subject": str(row["subject"]),
                "policy": str(row["policy"]),
                "trials": int(row["trials"]),
                "accuracy": float(row["accuracy"]),
                "warnings": int(row["warnings"]),
                "updates": int(row["updates"]),
                "relative_path": str(path.relative_to(root)).replace("\\", "/"),
            })
        except (OSError, ValueError, TypeError, pd.errors.ParserError):
            continue
    return records


@online_results_bp.route("/results/online-learning", methods=["GET", "POST"])
def results():
    values = {
        "scenario": "feature_shift",
        "seed": 42,
        "trials": 100,
        "window_size": 10,
        "interval": 10,
        "label_delay": 0,
    }
    experiment, error = None, None
    if request.method == "POST":
        try:
            values["scenario"] = request.form.get("scenario", values["scenario"])
            if values["scenario"] not in {"none", "feature_shift", "relationship_shift"}:
                raise ValueError("Invalid scenario")
            for key in ("seed", "trials", "window_size", "interval", "label_delay"):
                values[key] = int(request.form.get(key, values[key]))
            experiment = run_online_comparison(
                scenario=values["scenario"],
                seed=values["seed"],
                trials=values["trials"],
                window_size=values["window_size"],
                interval=values["interval"],
                label_delay=None if values["label_delay"] == -1 else values["label_delay"],
            )
        except (ValueError, TypeError, OverflowError) as exc:
            error = str(exc)
    root = Path(current_app.config.get("RESULTS_ROOT", "outputs"))
    return render_template(
        "online_results.html", values=values, experiment=experiment, error=error,
        bci_summaries=_saved_bci_summaries(root),
    ), (400 if error else 200)
