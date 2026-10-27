"""Join existing BCI Table 1 evidence with optional reconstruction-loss runs.

Always show previous subject results, including when reconstruction outputs do
not exist yet. This reader neither retrains models nor invents new loss values.
"""
from pathlib import Path
import json

import pandas as pd
from flask import Blueprint, current_app, render_template, request, send_from_directory, abort
from src.reporting.bci_comparison_figures import build_bci_comparison_figures

bci_reconstruction_bp = Blueprint("bci_reconstruction", __name__)


def load_comparison(root: Path, metrics_root: Path):
    baseline = {}
    table_path = metrics_root / "table1_pca3_h10.csv"
    if table_path.is_file():
        try:
            frame = pd.read_csv(table_path)
            for _, row in frame.iterrows():
                dataset = str(row.get("dataset", "")).strip().upper()
                subject = str(row.get("subject", "")).strip().upper()
                if dataset in {"2A", "2B"} and subject:
                    baseline[(dataset, subject)] = row.to_dict()
        except (OSError, ValueError, pd.errors.ParserError):
            pass

    manifest = {}
    entries = []
    issue = None
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            entries = manifest.get("subjects", [])
            if not isinstance(entries, list):
                raise ValueError("Invalid manifest subjects")
        except (OSError, ValueError, TypeError) as exc:
            issue = f"Cannot read reconstruction manifest: {exc}"
            manifest, entries = {}, []

    reconstruction = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        dataset = str(entry.get("dataset", "")).upper()
        try:
            number = int(entry.get("subject"))
        except (TypeError, ValueError):
            continue
        if dataset not in {"2A", "2B"} or not 1 <= number <= 9:
            continue
        subject = f"{'A' if dataset == '2A' else 'B'}{number:02d}"
        reconstruction[(dataset, subject)] = entry

    results = []
    for dataset, subject in sorted(set(baseline) | set(reconstruction)):
        previous = baseline.get((dataset, subject), {})
        entry = reconstruction.get((dataset, subject), {})
        number = int(subject[1:])
        trace = []
        trace_path = root / f"{dataset}_{number:02d}_loss_trace.csv"
        if entry and trace_path.is_file():
            try:
                frame = pd.read_csv(trace_path)
                required = ["trial_index", "reconstruction_mse", "pc1_score"]
                if set(required).issubset(frame.columns):
                    trace = frame[required].replace(
                        [float("inf"), -float("inf")], float("nan")
                    ).dropna().to_dict("records")
            except (OSError, ValueError, pd.errors.ParserError):
                pass
        results.append({
            "dataset": dataset, "subject": subject, "subject_number": number,
            "training_trials": entry.get("training_trials"),
            "evaluation_trials": entry.get("evaluation_trials"),
            "pca_components": entry.get("pca_components"),
            "mean_training_loss": entry.get("mean_training_loss"),
            "mean_evaluation_loss": entry.get("mean_evaluation_loss"),
            "batch_split": entry.get("batch_split"),
            "computed_csw": previous.get("computed_csw"),
            "computed_csv": previous.get("computed_csv"),
            "published_csw": previous.get("published_csw"),
            "published_csv": previous.get("published_csv"),
            "previous_pca_components": 3 if previous else None,
            "trace": trace,
            "has_reconstruction": bool(entry),
            "has_baseline": bool(previous),
        })
    return {
        "available": bool(results), "subjects": results,
        "warning": issue or (None if results else "No saved BCI subject results were found."),
        "provenance": manifest.get("provenance"),
        "limitations": manifest.get("warning"),
        "baseline_present": bool(baseline),
        "reconstruction_present": bool(reconstruction),
    }


@bci_reconstruction_bp.get("/results/abcd/bci")
def results():
    output_root = Path(current_app.config.get(
        "BCI_RECONSTRUCTION_PATH", "outputs/experiments/bci_reconstruction_loss"
    ))
    metrics_root = Path(current_app.config.get("BCI_METRICS_PATH", "outputs/metrics"))
    from src.web.dashboard import PROJECT_ROOT
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    if not metrics_root.is_absolute():
        metrics_root = PROJECT_ROOT / metrics_root
    figures_root = Path(current_app.config.get("BCI_FIGURES_PATH", "outputs/figures"))
    if not figures_root.is_absolute():
        figures_root = PROJECT_ROOT / figures_root
    try:
        build_bci_comparison_figures(
            metrics_path=metrics_root / "table1_pca3_h10.csv",
            reconstruction_path=output_root / "manifest.json",
            output_dir=figures_root,
        )
    except (OSError, ValueError, KeyError, TypeError):
        pass
    data = load_comparison(output_root, metrics_root)
    data["previous_figure"] = (figures_root / "bci_previous_method_comparison.png").is_file()
    data["reconstruction_figure"] = bool(data["reconstruction_present"] and (figures_root / "bci_reconstruction_comparison.png").is_file())
    selected = request.args.get("subject", "")
    all_rows = data["subjects"]
    chosen = next((r for r in all_rows if r["subject"] == selected),
                  all_rows[0] if all_rows else None)
    return render_template("bci_reconstruction.html", data=data, selected=chosen)

@bci_reconstruction_bp.get("/results/abcd/bci/figures/<path:filename>")
def figure(filename):
    allowed = {
        "bci_previous_method_comparison.png", "bci_previous_method_comparison.svg",
        "bci_reconstruction_comparison.png", "bci_reconstruction_comparison.svg",
    }
    if filename not in allowed:
        abort(404)
    from src.web.dashboard import PROJECT_ROOT
    folder = Path(current_app.config.get("BCI_FIGURES_PATH", "outputs/figures"))
    if not folder.is_absolute():
        folder = PROJECT_ROOT / folder
    return send_from_directory(folder, filename)
