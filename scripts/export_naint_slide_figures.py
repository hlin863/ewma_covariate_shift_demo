"""Export source-grounded NAINT presentation figures from existing research payloads.

Real Dataset 2A figures require locally installed official GDF and evaluation-label
MAT files; D2 figures use committed experiment traces. Missing evidence is an
error, never an invitation to fabricate example measurements.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# Direct invocation (python scripts/export_naint_slide_figures.py) sets sys.path[0]
# to scripts/, not the repository root where the src package lives.
# Keep module invocation (python -m scripts.export_naint_slide_figures) working too.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_OUTPUT = ROOT / "outputs" / "presentations" / "naint_2026_11_20"


def _save(fig, destination: Path) -> list[str]:
    """Persist a vector and a slide-ready raster rendition."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in ("svg", "png"):
        output = destination.with_suffix("." + extension)
        fig.savefig(output, dpi=240, bbox_inches="tight", facecolor="white")
        paths.append(str(output))
    plt.close(fig)
    return paths


def _axes(title: str, ylabel: str, xlabel: str = "Evaluation trial index"):
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.set_title(title, fontsize=17, loc="left", fontweight="bold", pad=15)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.16)
    ax.spines[["top", "right"]].set_visible(False)
    return fig, ax


def export_2a_payload(payload: dict, destination: Path, subject: str) -> list[str]:
    """Export the exact FBCSP/CSE payload already used by the BCI EDA page."""
    if not payload or not payload.get("timelines"):
        raise ValueError("No real 2A FBCSP observations available; export cancelled")
    if not payload.get("cse_events") or not payload.get("heatmap"):
        raise ValueError("Missing detector or heatmap evidence; export cancelled")
    destination.mkdir(parents=True, exist_ok=True)
    paths = []

    # 1: chronological development/evaluation feature, one named feature only.
    selected = payload["timelines"][0]
    fig, ax = _axes(f"{subject} · FBCSP feature timeline", "Log-variance feature", "Trial index within session")
    for key, label, color in (
        ("development", "Development training", "#2C6E9F"),
        ("evaluation", "Held-out evaluation", "#D39146"),
    ):
        rows = selected[key]
        ax.plot([int(r["trial_index"]) for r in rows],
                [float(r["feature_value"]) for r in rows],
                label=label, color=color, lw=1.4, alpha=.86)
    ax.legend(frameon=False)
    fig.text(.12, .01, f"Feature: {selected['name']} · development-fitted FBCSP · seed 42",
             fontsize=9, color="#555555")
    paths += _save(fig, destination / "01_fbcsp_chronological_feature")

    # 2: PC1 monitored by the original EWMA pathway with control limits.
    rows = payload["cse_events"]["warnings"]
    if not rows:
        raise ValueError("Missing Stage-I observation-level trace")
    if any(r.get(k) is None for r in rows for k in ("feature_value", "prediction", "lcl", "ucl")):
        raise ValueError("Stage-I payload lacks recorded input/limits: update bci_shift_views.py")
    x = np.array([int(r["index"]) for r in rows])
    values = np.array([float(r["feature_value"]) for r in rows])
    lower = np.array([float(r["lcl"]) for r in rows])
    upper = np.array([float(r["ucl"]) for r in rows])
    mask = np.array([bool(r["warning"]) for r in rows])
    fig, ax = _axes(f"{subject} · Stage-I EWMA warnings", "PC1 score")
    ax.plot(x, values, color="#316D9C", lw=1.5, label="PC1 observation")
    ax.plot(x, lower, color="#C58B42", lw=1.1, ls="--", label="Control limits")
    ax.plot(x, upper, color="#C58B42", lw=1.1, ls="--")
    ax.scatter(x[mask], values[mask], color="#C4544C", marker="o", s=22,
               label="Stage-I warning", zorder=4)
    ax.legend(frameon=False, ncol=3)
    fig.text(.12, .01, "Exploratory EWMA run; warning ≠ validated shift", fontsize=9, color="#555555")
    paths += _save(fig, destination / "02_stage1_ewma_warning_trace")

    # 3: completion is a later observation, not the warning itself.
    records = payload["cse_events"]["validations"]
    fig, ax = _axes(f"{subject} · Stage-II validation timeline", "Event status")
    statuses = sorted({str(v["status"]) for v in records})
    if not statuses:
        statuses = ["No candidate records"]
    positions = {status: n for n, status in enumerate(statuses)}
    colors = {"confirmed": "#2C947F", "rejected": "#C58B42",
              "pending_current_window": "#74859A",
              "insufficient_degrees_of_freedom": "#BA6262",
              "insufficient_reference_window": "#74859A",
              "skipped_nearby_alarm": "#74859A"}
    for record in records:
        status = str(record["status"])
        start = int(record["alarm_index"])
        end = record.get("validation_index")
        y = positions[status]
        if end is not None:
            ax.plot([start, int(end)], [y, y], color=colors.get(status, "#718096"), alpha=.55)
            ax.scatter([int(end)], [y], s=35, color=colors.get(status, "#718096"), zorder=3)
        ax.scatter([start], [y], marker="|", s=80, color="#263B4E", zorder=4)
    ax.set_yticks(list(positions.values()), list(positions.keys()))
    ax.set_ylim(-.7, len(positions)-.3)
    fig.text(.12, .01, "Vertical ticks: warning; circles: completed test. Skipped/pending ≠ rejected.",
             fontsize=9, color="#555555")
    paths += _save(fig, destination / "03_stage2_validation_timeline")

    # 4: exploratory window-standardised feature changes, not confirmations.
    heatmap = payload["heatmap"]
    windows = heatmap["windows"]
    if not windows:
        raise ValueError("Too few complete windows for heatmap")
    matrix = np.array([[np.nan if v is None else float(v) for v in w["values"]]
                       for w in windows], dtype=float).T
    fig, ax = plt.subplots(figsize=(13, 6))
    finite = matrix[np.isfinite(matrix)]
    limit = max(1., float(np.percentile(np.abs(finite), 95))) if finite.size else 1.
    im = ax.imshow(matrix, aspect="auto", origin="lower", cmap="RdBu_r",
                   vmin=-limit, vmax=limit)
    ax.set_title(f"{subject} · FBCSP window-shift heatmap", loc="left", fontsize=17,
                 fontweight="bold", pad=15)
    ax.set_ylabel("FBCSP feature")
    ax.set_xlabel("Complete non-overlapping evaluation window")
    ax.set_yticks(np.arange(len(heatmap["feature_names"])), heatmap["feature_names"], fontsize=7)
    fig.colorbar(im, ax=ax, label="Mean difference / development SD")
    fig.text(.12, .01, "Exploratory standardised feature differences; not Stage-II confirmations.",
             fontsize=9, color="#555555")
    paths += _save(fig, destination / "04_fbcsp_window_shift_heatmap")

    manifest = {
        "source": "development-fitted BCI IV 2A GDF + official evaluation labels",
        "subject": subject, "experiment_type": "real_eeg_exploratory",
        "validation_protocol": payload["cse_events"]["config"],
        "development_trials": payload.get("development_count"),
        "evaluation_trials": payload.get("evaluation_count"),
        "seed": payload.get("seed"), "figures": paths,
        "caveat": "Not a numerical reproduction of the published CSW/CSV counts; no clinical online trial",
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return paths


def export_d2_saved(destination: Path, source: Path) -> list[str]:
    trace_path = source / "stage1_trace.csv"
    decisions_path = source / "stage2_validations.csv"
    if not trace_path.is_file() or not decisions_path.is_file():
        raise FileNotFoundError("D2 saved trace and validation CSVs are required")
    trace = pd.read_csv(trace_path)
    decisions = pd.read_csv(decisions_path)
    required_trace = {"time", "x", "lcl", "ucl", "stage_1_alarm"}
    required_decisions = {"alarm_time", "validation_time", "status"}
    if not required_trace.issubset(trace) or not required_decisions.issubset(decisions):
        raise ValueError("Saved D2 trace schema incomplete")
    if trace.empty:
        raise ValueError("Saved D2 trace empty")
    fig, ax = _axes("Synthetic D2 · EWMA and K–S validation",
                    "D2 observation", "D2 paper-indexed observation")
    ax.plot(trace["time"], trace["x"], color="#316D9C", lw=.9, label="Synthetic stream")
    ax.plot(trace["time"], trace["lcl"], color="#C58B42", ls="--", lw=1, label="EWMA limits")
    ax.plot(trace["time"], trace["ucl"], color="#C58B42", ls="--", lw=1)
    alerts = trace.loc[trace["stage_1_alarm"].astype(bool)]
    ax.scatter(alerts["time"], alerts["x"], s=8, color="#C4544C", label="Stage-I warning")
    complete = decisions.dropna(subset=["validation_time"])
    confirmed = complete.loc[complete["status"].eq("confirmed")]
    for _, record in confirmed.iterrows():
        ax.axvline(float(record["validation_time"]), color="#29927F", alpha=.42, lw=1.2)
    ax.legend(frameon=False, ncol=3)
    fig.text(.12, .01, "Synthetic D2, Stage-II K–S; green lines = confirmation completion, not BCI Hotelling.",
             fontsize=9, color="#555555")
    paths = _save(fig, destination / "05_synthetic_d2_ewma_ks_timeline")
    manifest = {
        "source": str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
        "experiment_type": "synthetic_d2_archived",
        "trace_rows": len(trace), "validation_records": len(decisions),
        "figures": paths,
        "caveat": "Independent D2 EWMA/K–S diagnostic, not the 2A EEG Hotelling experiment",
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("2a", "d2", "all"), default="all")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--data-2a", type=Path, default=ROOT / "data/raw/bci_competition_iv_2a")
    parser.add_argument("--labels-2a", type=Path, default=ROOT / "data/raw/bci_competition_iv_2a_labels")
    parser.add_argument("--d2-source", type=Path, default=ROOT / "outputs/metrics/paper2015/d2")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    exported = []
    if args.source in ("2a", "all"):
        if not 1 <= args.subject <= 9:
            parser.error("--subject must be 1..9")
        from flask import Flask
        from src.web.bci_eda import _load_dataset_2a_feature_timelines
        app = Flask(__name__)
        app.config.update(DATASET_2A_PATH=str(args.data_2a),
                          DATASET_2A_LABELS_PATH=str(args.labels_2a))
        with app.app_context():
            payload, message = _load_dataset_2a_feature_timelines(args.subject)
        if payload is None:
            parser.error(message)
        exported += export_2a_payload(payload, args.output_dir / "bci_2a" /
                                      f"A{args.subject:02d}", f"A{args.subject:02d}")
    if args.source in ("d2", "all"):
        exported += export_d2_saved(args.output_dir / "synthetic_d2", args.d2_source)
    print("\n".join(exported))


if __name__ == "__main__":
    main()
