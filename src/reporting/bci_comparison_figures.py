"""Generate persistent BCI diagnostics and reconstruction-loss comparison figures.

Reads previously saved subject metrics; never reruns the EEG pipeline.
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_bci_comparison_figures(
    metrics_path=Path("outputs/metrics/table1_pca3_h10.csv"),
    reconstruction_path=Path("outputs/experiments/bci_reconstruction_loss/manifest.json"),
    output_dir=Path("outputs/figures"),
):
    metrics_path, reconstruction_path, output_dir = map(
        Path, (metrics_path, reconstruction_path, output_dir)
    )
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Existing BCI subject metrics unavailable: {metrics_path}")
    previous = pd.read_csv(metrics_path)
    required = {"dataset", "subject", "published_csw", "computed_csw",
                "published_csv", "computed_csv"}
    if not required.issubset(previous.columns):
        raise ValueError(f"Existing metrics missing: {sorted(required-set(previous.columns))}")
    previous = previous.sort_values(["dataset", "subject"]).reset_index(drop=True)
    if previous.empty:
        raise ValueError("No previous BCI subject results")
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = previous["subject"].astype(str).tolist()
    x = np.arange(len(previous))
    saved = []
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, constrained_layout=True)
    for ax, metric, heading in zip(
        axes, ("csw", "csv"),
        ("Stage-I EWMA warnings (CSW)", "Stage-II Hotelling confirmations (CSV)")
    ):
        ax.bar(x - .18, previous[f"published_{metric}"], width=.36, label="Published")
        ax.bar(x + .18, previous[f"computed_{metric}"], width=.36, label="Saved implementation")
        ax.set_title(heading)
        ax.set_ylabel("Event count")
        ax.grid(axis="y", alpha=.25)
        ax.legend()
    axes[-1].set_xticks(x, labels, rotation=45)
    axes[-1].set_xlabel("BCI Competition IV subjects · 2A and 2B")
    fig.suptitle("BCI Competition IV — published vs saved CSE results")
    for fmt in ("png", "svg"):
        dest = output_dir / f"bci_previous_method_comparison.{fmt}"
        fig.savefig(dest, dpi=180, bbox_inches="tight")
        saved.append(str(dest))
    plt.close(fig)

    reconstructed = []
    if reconstruction_path.is_file():
        document = json.loads(reconstruction_path.read_text(encoding="utf-8"))
        for row in document.get("subjects", []):
            if not isinstance(row, dict):
                continue
            dataset = str(row.get("dataset", "")).upper()
            if dataset not in {"2A", "2B"}:
                continue
            try:
                subject = f"{'A' if dataset == '2A' else 'B'}{int(row['subject']):02d}"
                before = float(row["mean_training_loss"])
                after = float(row["mean_evaluation_loss"])
            except (KeyError, ValueError, TypeError):
                continue
            if np.isfinite(before) and np.isfinite(after):
                reconstructed.append({"dataset": dataset, "subject": subject,
                                      "training_mse": before, "evaluation_mse": after,
                                      "delta_mse": after-before})
    if reconstructed:
        current = pd.DataFrame(reconstructed).sort_values(["dataset", "subject"])
        xx = np.arange(len(current))
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
        axes[0].bar(xx - .18, current["training_mse"], width=.36, label="Training")
        axes[0].bar(xx + .18, current["evaluation_mse"], width=.36, label="Evaluation")
        axes[0].set_ylabel("Mean reconstruction MSE")
        axes[0].set_title("Frozen PCA: reconstruction loss by BCI subject")
        axes[0].legend()
        axes[1].bar(xx, current["delta_mse"])
        axes[1].axhline(0, linewidth=1)
        axes[1].set_ylabel("Evaluation − training MSE")
        axes[1].set_title("Change in mean reconstruction loss")
        axes[1].set_xticks(xx, current["subject"], rotation=45)
        for ax in axes:
            ax.grid(axis="y", alpha=.25)
        fig.suptitle("BCI reconstruction-loss comparison (not validated drift events)")
        for fmt in ("png", "svg"):
            dest = output_dir / f"bci_reconstruction_comparison.{fmt}"
            fig.savefig(dest, dpi=180, bbox_inches="tight")
            saved.append(str(dest))
        plt.close(fig)
        current.to_csv(output_dir / "bci_reconstruction_comparison.csv", index=False)
        saved.append(str(output_dir / "bci_reconstruction_comparison.csv"))
    summary_path = output_dir / "bci_comparison_figures_manifest.json"
    summary_path.write_text(json.dumps({
        "previous_metrics": str(metrics_path),
        "reconstruction_manifest": str(reconstruction_path) if reconstructed else None,
        "existing_subject_count": len(previous),
        "reconstruction_subject_count": len(reconstructed),
        "artifacts": saved,
        "caveat": "Previous CSW/CSV event counts and reconstruction MSE are distinct measurements.",
    }, indent=2), encoding="utf-8")
    return saved


if __name__ == "__main__":
    build_bci_comparison_figures()
