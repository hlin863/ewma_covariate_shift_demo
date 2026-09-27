"""Run bagging on real BCI Competition IV Dataset 2A/2B files.

Run from the repository root after downloading the GDF recordings and official
evaluation-label MAT files:

    python -m scripts.run_bagging_bci

The synthetic runner remains available separately as a software smoke test:
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

from src.adaptation.experiments.bagging_bci import run_bagging_bci_experiment


def _plot_accuracy(frame: pd.DataFrame, path: Path) -> None:
    """Plot paired subject-level single-versus-bagged accuracy."""

    fig, ax = plt.subplots(figsize=(10, 5.5))
    positions: list[float] = []
    labels: list[str] = []
    cursor = 0.0

    for dataset in ("2A", "2B"):
        subset = frame.loc[frame["dataset"] == dataset]
        if subset.empty:
            continue
        subjects = sorted(subset["subject"].unique())
        for subject in subjects:
            subject_rows = subset.loc[subset["subject"] == subject]
            single = subject_rows.loc[
                subject_rows["method"] == "single_linear_svm", "accuracy"
            ]
            bagged = subject_rows.loc[
                subject_rows["method"] == "bagged_linear_svm", "accuracy"
            ]
            if single.empty or bagged.empty:
                continue
            x = cursor
            ax.plot(
                [x - 0.12, x + 0.12],
                [float(single.iloc[0]), float(bagged.iloc[0])],
                marker="o",
                linewidth=1.2,
            )
            positions.append(x)
            labels.append(str(subject))
            cursor += 1.0
        cursor += 0.75

    ax.set_xticks(positions, labels, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Classification accuracy")
    ax.set_title("Real BCI Competition IV: single versus bagged SVM")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_disagreement(frame: pd.DataFrame, path: Path) -> None:
    """Plot subject-level disagreement among bagged ensemble members."""

    bagged = frame.loc[frame["method"] == "bagged_linear_svm"].copy()
    if bagged.empty:
        raise ValueError("bagging result table contains no bagged_linear_svm rows.")

    labels = [str(value) for value in bagged["subject"]]
    values = bagged["mean_member_disagreement"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    bars = ax.bar(np.arange(len(values)), values)
    ax.set_xticks(np.arange(len(values)), labels, rotation=45, ha="right")
    ax.set_ylim(0.0, max(0.05, float(values.max()) * 1.15))
    ax.set_ylabel("Mean fraction of ensemble members disagreeing with final vote")
    ax.set_title("Real BCI Competition IV: bagging ensemble disagreement")
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=2, fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_and_write(
    output_root: str | Path = PROJECT_ROOT / "outputs",
    *,
    data_2a: str | Path = PROJECT_ROOT / "data" / "raw" / "bci_competition_iv_2a",
    labels_2a: str | Path = PROJECT_ROOT / "data" / "raw" / "bci_competition_iv_2a_labels",
    data_2b: str | Path = PROJECT_ROOT / "data" / "raw" / "bci_competition_iv_2b",
    labels_2b: str | Path | None = None,
    datasets: tuple[str, ...] = ("2a", "2b"),
    subjects: tuple[int, ...] = tuple(range(1, 10)),
    random_state: int = 42,
    n_estimators: int = 30,
    sample_fraction: float = 0.8,
) -> dict[str, Path]:
    """Run the real-data experiment and write dashboard-ready artifacts."""

    root = Path(output_root)
    metrics_dir = root / "metrics"
    figures_dir = root / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    labels_2b_path = Path(labels_2b) if labels_2b is not None else Path(data_2b)

    frame = run_bagging_bci_experiment(
        data_2a=data_2a,
        labels_2a=labels_2a,
        data_2b=data_2b,
        labels_2b=labels_2b_path,
        datasets=datasets,
        subjects=subjects,
        random_state=random_state,
        n_estimators=n_estimators,
        sample_fraction=sample_fraction,
    )

    if frame.empty:
        raise ValueError("No real-data bagging results were produced.")

    metrics_path = metrics_dir / "bagging_bci_real.csv"
    accuracy_path = figures_dir / "bagging_bci_real_accuracy.png"
    disagreement_path = figures_dir / "bagging_bci_real_disagreement.png"

    frame.to_csv(metrics_path, index=False)
    _plot_accuracy(frame, accuracy_path)
    _plot_disagreement(frame, disagreement_path)

    return {
        "metrics": metrics_path,
        "accuracy_figure": accuracy_path,
        "disagreement_figure": disagreement_path,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-2a",
        type=Path,
        default=Path("data/raw/bci_competition_iv_2a"),
    )
    parser.add_argument(
        "--labels-2a",
        type=Path,
        default=Path("data/raw/bci_competition_iv_2a_labels"),
    )
    parser.add_argument(
        "--data-2b",
        type=Path,
        default=Path("data/raw/bci_competition_iv_2b"),
    )
    parser.add_argument(
        "--labels-2b",
        type=Path,
        default=None,
        help=(
            "Directory containing Bxx04E.mat/Bxx05E.mat official labels. "
            "Defaults to --data-2b."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("2a", "2b"),
        default=("2a", "2b"),
    )
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        default=list(range(1, 10)),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--estimators", type=int, default=30)
    parser.add_argument("--sample-fraction", type=float, default=0.8)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    paths = run_and_write(
        args.output_root,
        data_2a=args.data_2a,
        labels_2a=args.labels_2a,
        data_2b=args.data_2b,
        labels_2b=args.labels_2b,
        datasets=tuple(args.datasets),
        subjects=tuple(args.subjects),
        random_state=args.seed,
        n_estimators=args.estimators,
        sample_fraction=args.sample_fraction,
    )

    frame = pd.read_csv(paths["metrics"])
    print(frame.to_string(index=False))
    print()
    print("Mean accuracy by dataset/method:")
    print(frame.groupby(["dataset", "method"])["accuracy"].agg(["mean", "std"]).to_string())
    print(f"\nMetrics: {paths['metrics']}")
    print(f"Accuracy figure: {paths['accuracy_figure']}")
    print(f"Disagreement figure: {paths['disagreement_figure']}")
    print("Dashboard: http://127.0.0.1:5000/results")


if __name__ == "__main__":
    main()
