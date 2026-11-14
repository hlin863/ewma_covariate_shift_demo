"""Rebuild publication-friendly figures from saved TWIN-ADAPT run evidence.

Plots live under <run>/figures; source CSV/JSON paths stay immutable.
No model or EEG processing is performed here.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _save_figure(fig, figures_dir: Path, stem: str) -> list[str]:
    """Write both reusable vector and slide-ready raster assets."""
    import matplotlib.pyplot as plt

    names = []
    for suffix in (".svg", ".png"):
        filename = stem + suffix
        fig.savefig(figures_dir / filename, dpi=200, bbox_inches="tight")
        names.append(filename)
    plt.close(fig)
    return names


def export_run_figures(run_dir: Path) -> dict:
    """Export figures from a saved fixed-config or optimisation run.

    All chart values come from the run's committed CSV files. This operation
    does not modify them. Repeat calls regenerate the same named assets.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    optimized = (run_dir / "candidate_history.csv").is_file()
    results_dir = run_dir / "final_evaluation" if optimized else run_dir
    windows_path = results_dir / "accuracy_windows.csv"
    updates_path = results_dir / "updates.csv"
    if not windows_path.is_file() or not updates_path.is_file():
        raise FileNotFoundError(
            "Expected accuracy_windows.csv and updates.csv in "
            f"{results_dir}"
        )
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict] = []

    def emit(fig, name: str, source: list[str], description: str) -> None:
        artifacts.append({
            "name": name,
            "files": _save_figure(fig, figures_dir, name),
            "source_files": source,
            "description": description,
        })

    windows = pd.read_csv(windows_path)
    if not windows.empty:
        required = {"batch", "prev_accuracy", "curr_accuracy",
                    "warning_threshold", "adaptation_threshold"}
        if not required.issubset(windows):
            raise ValueError(f"Missing accuracy-window columns: {required - set(windows)}")
        fig, ax = plt.subplots(figsize=(10, 4.8))
        x = windows["batch"].to_numpy()
        for key, label, style in (
            ("prev_accuracy", "Previous-window accuracy", "-"),
            ("curr_accuracy", "Current-window accuracy", "-"),
            ("warning_threshold", "Warning threshold", "--"),
            ("adaptation_threshold", "Adaptation threshold", "--"),
        ):
            ax.plot(x, windows[key].astype(float) * 100, style, marker="o",
                    markersize=3, label=label)
        ax.set(xlabel="Released batch", ylabel="Accuracy / threshold (%)",
               title="TWIN-ADAPT accuracy windows and decision thresholds")
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        emit(fig, "accuracy_windows", [str(windows_path.relative_to(run_dir))],
             "Measured within-batch accuracies and configured decision thresholds")

    updates = pd.read_csv(updates_path)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    if not updates.empty:
        if not {"batch", "version", "reason"}.issubset(updates):
            raise ValueError("Missing required update columns")
        ax.scatter(updates["batch"], updates["version"], s=70, zorder=3)
        for _, event in updates.iterrows():
            ax.annotate(str(event["reason"]), (event["batch"], event["version"]),
                        xytext=(4, 8), textcoords="offset points", fontsize=7,
                        rotation=18)
        ax.set_ylim(0, max(updates["version"]) + 1)
    else:
        ax.text(0.5, 0.5, "No model updates in this run",
                transform=ax.transAxes, ha="center", va="center")
    ax.set(xlabel="Released batch", ylabel="Model version",
           title="Recorded model-update events")
    ax.grid(alpha=0.2)
    emit(fig, "update_events", [str(updates_path.relative_to(run_dir))],
         "Actual model-update decisions; empty chart explicitly denotes zero updates")

    if optimized:
        history_path = run_dir / "candidate_history.csv"
        history = pd.read_csv(history_path)
        if not history.empty:
            required = {"stage", "budget", "score"}
            if not required.issubset(history):
                raise ValueError("Missing candidate-history score columns")
            fig, ax = plt.subplots(figsize=(9, 4.5))
            for stage, group in history.groupby("stage", sort=True):
                ax.scatter(group["budget"], group["score"], label=f"Stage {stage}",
                           s=45, alpha=0.65)
            ax.set(xlabel="Validation budget (trials)", ylabel="Objective score",
                   title="Candidate objective scores by validation budget")
            ax.grid(alpha=0.2)
            ax.legend()
            emit(fig, "candidate_scores", ["candidate_history.csv"],
                 "Every evaluated candidate score, including exact ties")

            if "prediction_diversity" in history:
                grouped = history.groupby(["stage", "budget"], sort=True)[
                    "prediction_diversity"].max().reset_index()
                fig, ax = plt.subplots(figsize=(9, 4.5))
                ax.bar([f"Stage {int(row.stage)} / {int(row.budget)}"
                        for row in grouped.itertuples()],
                       grouped["prediction_diversity"].astype(float) * 100)
                ax.set(xlabel="Search stage / validation trials",
                       ylabel="Maximum prediction disagreement (%)",
                       title="Within-stage candidate prediction diversity")
                ax.set_ylim(bottom=0)
                ax.grid(axis="y", alpha=0.2)
                emit(fig, "prediction_diversity", ["candidate_history.csv"],
                     "Stage-wide maximum pairwise prediction disagreement")

    manifest = {
        "run_dir": run_dir.name,
        "kind": "twin_adapt_optimization" if optimized else "twin_adapt_bci",
        "figure_count": len(artifacts),
        "artifacts": artifacts,
        "provenance": "Figures computed exclusively from CSV evidence in this run",
    }
    (figures_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    return manifest
