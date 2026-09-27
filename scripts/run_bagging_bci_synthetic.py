"""Run the synthetic Dataset 2A/2B bagging integration experiment.

The runner writes a compact metrics table and two visual summaries consumed by
the Flask research-results catalogue.

Run from the repository root:

    python -m scripts.run_bagging_bci_synthetic
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.adaptation.experiments.bagging_synthetic_bci import (
    run_bagging_synthetic_bci_experiment,
)


def _plot_accuracy(frame: pd.DataFrame, path: Path) -> None:
    datasets = ["2A", "2B"]
    methods = ["single_linear_svm", "bagged_linear_svm"]
    labels = ["Single linear SVM", "Bagged linear SVM"]

    x = np.arange(len(datasets), dtype=float)
    width = 0.36

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for offset, (method, label) in enumerate(zip(methods, labels)):
        values = [
            float(
                frame.loc[
                    (frame["dataset"] == dataset) & (frame["method"] == method),
                    "accuracy",
                ].iloc[0]
            )
            for dataset in datasets
        ]
        positions = x + (offset - 0.5) * width
        bars = ax.bar(positions, values, width=width, label=label)
        ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3)

    ax.set_xticks(x, datasets)
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Classification accuracy")
    ax.set_title("Synthetic 2A/2B bagging integration")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_disagreement(frame: pd.DataFrame, path: Path) -> None:
    bagged = (
        frame.loc[frame["method"] == "bagged_linear_svm"]
        .set_index("dataset")
        .reindex(["2A", "2B"])
    )
    values = bagged["mean_member_disagreement"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(["2A", "2B"], values)
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3)
    ax.set_ylim(0.0, max(0.05, float(values.max()) * 1.2))
    ax.set_ylabel("Mean fraction of ensemble members disagreeing with final vote")
    ax.set_title("Bagging ensemble disagreement")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_and_write(
    output_root: str | Path = PROJECT_ROOT / "outputs",
    *,
    random_state: int = 42,
    n_estimators: int = 30,
    sample_fraction: float = 0.8,
) -> dict[str, Path]:
    """Run the integration experiment and write dashboard-ready artifacts."""

    root = Path(output_root)
    metrics_dir = root / "metrics"
    figures_dir = root / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    frame = run_bagging_synthetic_bci_experiment(
        random_state=random_state,
        n_estimators=n_estimators,
        sample_fraction=sample_fraction,
    )

    metrics_path = metrics_dir / "bagging_synthetic_bci.csv"
    accuracy_path = figures_dir / "bagging_synthetic_accuracy.png"
    disagreement_path = figures_dir / "bagging_synthetic_disagreement.png"

    frame.to_csv(metrics_path, index=False)
    _plot_accuracy(frame, accuracy_path)
    _plot_disagreement(frame, disagreement_path)

    return {
        "metrics": metrics_path,
        "accuracy_figure": accuracy_path,
        "disagreement_figure": disagreement_path,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test BaggingClassifier on deterministic synthetic 2A/2B BCI streams."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="Root directory for metrics/ and figures/ output.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--estimators", type=int, default=30)
    parser.add_argument("--sample-fraction", type=float, default=0.8)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    paths = run_and_write(
        args.output_root,
        random_state=args.seed,
        n_estimators=args.estimators,
        sample_fraction=args.sample_fraction,
    )

    frame = pd.read_csv(paths["metrics"])
    print(frame.to_string(index=False))
    print(f"Metrics: {paths['metrics']}")
    print(f"Accuracy figure: {paths['accuracy_figure']}")
    print(f"Disagreement figure: {paths['disagreement_figure']}")
    print("Dashboard: http://127.0.0.1:5000/results")


if __name__ == "__main__":
    main()
