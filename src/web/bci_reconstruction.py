"""Read saved BCI reconstruction outputs without running experiments on page load."""
from pathlib import Path
import json

import pandas as pd
from flask import Blueprint, current_app, render_template, request

bci_reconstruction_bp = Blueprint("bci_reconstruction", __name__)


def load_comparison(root: Path, metrics_root: Path):
    """Separate diagnostic metrics from reconstruction values; never invent missing runs."""
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return {"available": False, "subjects": [], "warning": "No BCI reconstruction manifest found. Run the CLI first."}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = manifest.get("subjects", [])
        if not isinstance(entries, list):
            raise ValueError("Invalid manifest subjects")
    except (OSError, ValueError, TypeError) as exc:
        return {"available": False, "subjects": [], "warning": f"Cannot read manifest: {exc}"}

    baseline = {}
    table_path = metrics_root / "table1_pca3_h10.csv"
    if table_path.is_file():
        try:
            for _, row in pd.read_csv(table_path).iterrows():
                key = (str(row.get("dataset", "")).upper().replace("DATASET ", "").replace("BCI ", "").strip(),
                       str(row.get("subject", "")).upper())
                baseline[key] = row.to_dict()
        except (OSError, ValueError, pd.errors.ParserError):
            pass

    results = []
    for entry in entries:
        dataset = str(entry.get("dataset", "")).upper()
        number = entry.get("subject")
        if dataset not in {"2A", "2B"}:
            continue
        try:
            number = int(number)
        except (TypeError, ValueError):
            continue
        if not 1 <= number <= 9:
            continue
        subject = f"{'A' if dataset == '2A' else 'B'}{number:02d}"
        trace_path = root / f"{dataset}_{number:02d}_loss_trace.csv"
        trace = []
        if trace_path.is_file():
            try:
                df = pd.read_csv(trace_path)
                required = {"trial_index", "reconstruction_mse", "pc1_score"}
                if required.issubset(df.columns):
                    trace = df[list(required)].replace([float("inf"), -float("inf")], float("nan")).dropna().to_dict("records")
            except (OSError, ValueError, pd.errors.ParserError):
                pass
        row = baseline.get((dataset, subject), {})
        results.append({
            "dataset": dataset, "subject": subject, "subject_number": number,
            "training_trials": entry.get("training_trials"),
            "evaluation_trials": entry.get("evaluation_trials"),
            "pca_components": entry.get("pca_components"),
            "mean_training_loss": entry.get("mean_training_loss"),
            "mean_evaluation_loss": entry.get("mean_evaluation_loss"),
            "batch_split": entry.get("batch_split"),
            "computed_csw": row.get("computed_csw"),
            "computed_csv": row.get("computed_csv"),
            "published_csw": row.get("published_csw"),
            "published_csv": row.get("published_csv"),
            "previous_pca_components": 3 if row else None,
            "trace": trace,
        })
    return {"available": bool(results), "subjects": results,
            "warning": None if results else "No valid subject runs found in the manifest.",
            "provenance": manifest.get("provenance"),
            "limitations": manifest.get("warning"),
            "baseline_present": bool(baseline)}


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
    data = load_comparison(output_root, metrics_root)
    selected = request.args.get("subject", "")
    all_rows = data["subjects"]
    chosen = next((r for r in all_rows if r["subject"] == selected), all_rows[0] if all_rows else None)
    return render_template("bci_reconstruction.html", data=data, selected=chosen)
