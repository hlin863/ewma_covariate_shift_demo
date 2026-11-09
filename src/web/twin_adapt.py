"""Actual BCI TWIN-ADAPT execution and saved-evidence viewer."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from flask import Blueprint, current_app, render_template, request

from src.adaptation.experiments.twin_adapt_bci import run_bci_experiment
from src.adaptation.twin_adapt.config import TwinAdaptConfig

twin_adapt_bp = Blueprint("twin_adapt", __name__)


def _runs(root: Path) -> list[dict]:
    base = root / "experiments" / "twin_adapt_bci"
    entries = []
    for path in sorted(base.glob("2*/subject_*/*/summary.json"), reverse=True)[:80]:
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
            if not {"dataset", "subject", "accuracy", "updates", "run_id"}.issubset(summary):
                continue
            summary["relative_dir"] = path.parent.relative_to(root).as_posix()
            entries.append(summary)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return entries


def _trace(root: Path, entry: dict | None) -> dict:
    if entry is None:
        return {}
    directory = root / entry["relative_dir"]
    result = {}
    for key, name in (
        ("windows", "accuracy_windows.csv"),
        ("transitions", "state_transitions.csv"),
        ("updates", "updates.csv"),
        ("buffers", "adaptive_windows.csv"),
    ):
        file = directory / name
        if not file.is_file():
            result[key] = []
            continue
        try:
            result[key] = pd.read_csv(file).tail(100).where(
                lambda x: pd.notna(x), None
            ).to_dict(orient="records")
        except (OSError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError):
            result[key] = []
    return result


@twin_adapt_bp.route("/results/twin-adapt", methods=["GET", "POST"])
def results():
    root = Path(current_app.config.get("RESULTS_ROOT", "outputs"))
    defaults = TwinAdaptConfig()
    values = defaults.snapshot()
    values.update(dataset="2a", subject=1)
    error = None
    selected_id = request.args.get("run")
    if request.method == "POST":
        try:
            dataset = request.form.get("dataset", "2a")
            subject = int(request.form.get("subject", 1))
            kwargs = {key: int(request.form.get(key, getattr(defaults, key)))
                      for key in ("batch_size", "win_size", "win_size_max", "seed")}
            kwargs.update({key: float(request.form.get(key, getattr(defaults, key)))
                          for key in ("gamma", "delta")})
            delay = int(request.form.get("label_delay", 0))
            kwargs["label_delay"] = None if delay == -1 else delay
            config = TwinAdaptConfig(**kwargs)
            values.update(config.snapshot(), dataset=dataset, subject=subject)
            data_root = Path(current_app.config.get("BCI_RAW_DATA_ROOT", "data/raw"))
            # Run real local GDF/MAT data only; never manufacture experiment numbers.
            target, _ = run_bci_experiment(
                dataset=dataset, subject=subject, config=config,
                data_root=data_root, output_root=root / "experiments" / "twin_adapt_bci",
            )
            selected_id = target.name
        except (ValueError, TypeError, OSError, RuntimeError, ImportError) as exc:
            error = str(exc)
    runs = _runs(root)
    selected = next((row for row in runs if row["run_id"] == selected_id), None)
    if selected is None and runs:
        selected = runs[0]
    trace = _trace(root, selected)
    return render_template(
        "twin_adapt.html", values=values, runs=runs, selected=selected,
        trace=trace, error=error,
    ), (400 if error else 200)
