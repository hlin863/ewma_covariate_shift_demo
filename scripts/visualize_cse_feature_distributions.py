"""Visualise processed BCI feature distributions before running CSE.

Example
-------
python scripts/visualize_cse_feature_distributions.py \
    data/processed/A01T_features.npz \
    data/processed/A01E_features.npz
"""

from argparse import ArgumentParser
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bci_data import load_cse_feature_file
from src.cse import CSEConfig, run_cse


def _parse_args():
    parser = ArgumentParser(
        description=(
            "Compare processed training and evaluation BCI feature "
            "distributions and visualise the CSE PCA signal."
        )
    )
    parser.add_argument("training_file", type=Path)
    parser.add_argument("evaluation_file", type=Path)
    parser.add_argument(
        "--split-index",
        type=int,
        default=None,
        help=(
            "Optional index dividing the evaluation session into before "
            "and after segments. Defaults to the midpoint."
        ),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("outputs/figures/bci_distributions"),
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=3,
    )
    return parser.parse_args()


def _summary_table(
    training: np.ndarray,
    evaluation_before: np.ndarray,
    evaluation_after: np.ndarray,
) -> pd.DataFrame:
    rows = []

    for feature_index in range(training.shape[1]):
        rows.append(
            {
                "feature": feature_index,
                "training_mean": training[:, feature_index].mean(),
                "training_std": training[:, feature_index].std(ddof=1),
                "evaluation_before_mean": (
                    evaluation_before[:, feature_index].mean()
                ),
                "evaluation_before_std": (
                    evaluation_before[:, feature_index].std(ddof=1)
                ),
                "evaluation_after_mean": (
                    evaluation_after[:, feature_index].mean()
                ),
                "evaluation_after_std": (
                    evaluation_after[:, feature_index].std(ddof=1)
                ),
            }
        )

    return pd.DataFrame(rows)


def _plot_feature_means(summary: pd.DataFrame, output_path: Path) -> None:
    x = np.arange(summary.shape[0])
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(
        x - width,
        summary["training_mean"],
        width,
        label="Training",
    )
    ax.bar(
        x,
        summary["evaluation_before_mean"],
        width,
        label="Evaluation before",
    )
    ax.bar(
        x + width,
        summary["evaluation_after_mean"],
        width,
        label="Evaluation after",
    )
    ax.set_title("Processed BCI Feature Means by Distribution Segment")
    ax.set_xlabel("Feature index")
    ax.set_ylabel("Mean")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["feature"].astype(str))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_pc1_distribution(
    training_signal: np.ndarray,
    testing_signal: np.ndarray,
    split_index: int,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(training_signal, bins=30, alpha=0.55, label="Training PC1")
    ax.hist(
        testing_signal[:split_index],
        bins=30,
        alpha=0.55,
        label="Evaluation before PC1",
    )
    ax.hist(
        testing_signal[split_index:],
        bins=30,
        alpha=0.55,
        label="Evaluation after PC1",
    )
    ax.set_title("CSE First Principal Component Distributions")
    ax.set_xlabel("PC1 value")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_pc1_stream(
    testing_times: np.ndarray,
    testing_signal: np.ndarray,
    split_index: int,
    warning_results: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(testing_times, testing_signal, label="PC1 signal")
    ax.axvline(
        testing_times[split_index],
        linestyle="--",
        label="Distribution split",
    )

    alarms = warning_results.loc[warning_results["stage_1_alarm"]]
    if not alarms.empty:
        ax.scatter(
            alarms["time"],
            alarms["x"],
            marker="x",
            label="Stage-I warnings",
        )

    ax.set_title("CSE PC1 Stream and EWMA Warnings")
    ax.set_xlabel("Observation time")
    ax.set_ylabel("PC1 value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = _parse_args()

    training, _, _ = load_cse_feature_file(args.training_file)
    evaluation, evaluation_times, _ = load_cse_feature_file(
        args.evaluation_file
    )

    if training.shape[1] != evaluation.shape[1]:
        raise ValueError(
            "Training and evaluation files must have equal feature counts."
        )

    split_index = args.split_index or (evaluation.shape[0] // 2)
    if not 1 <= split_index < evaluation.shape[0]:
        raise ValueError(
            "split-index must divide the evaluation observations into "
            "two non-empty segments."
        )

    config = CSEConfig(
        pca_components=min(args.pca_components, training.shape[1]),
        validation_before_size=20,
        validation_after_size=20,
        minimum_alarm_gap=20,
    )
    result = run_cse(
        training_features=training,
        testing_features=evaluation,
        testing_times=evaluation_times,
        config=config,
    )

    before = evaluation[:split_index]
    after = evaluation[split_index:]
    summary = _summary_table(training, before, after)

    output_directory = args.output_directory
    output_directory.mkdir(parents=True, exist_ok=True)

    summary_path = output_directory / "feature_distribution_summary.csv"
    summary.to_csv(summary_path, index=False)

    _plot_feature_means(
        summary,
        output_directory / "feature_means.png",
    )
    _plot_pc1_distribution(
        result.training_signal,
        result.testing_signal,
        split_index,
        output_directory / "pc1_distributions.png",
    )
    _plot_pc1_stream(
        evaluation_times,
        result.testing_signal,
        split_index,
        result.warning_results,
        output_directory / "pc1_stream_warnings.png",
    )

    print(summary.to_string(index=False))
    print(f"\nSaved outputs to: {output_directory.resolve()}")


if __name__ == "__main__":
    main()
