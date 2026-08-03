"""Archived visualisation utility for legacy processed BCI feature files.

This script expects the superseded `.npz` overlapping-window feature format.
It is retained for provenance and is not part of the active FBCSP replication.
"""

from argparse import ArgumentParser
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bci_data import load_cse_feature_file
from src.cse import CSEConfig, run_cse


def _parse_args():
    parser = ArgumentParser(description="ARCHIVED: visualise legacy processed BCI features.")
    parser.add_argument("training_file", type=Path)
    parser.add_argument("evaluation_file", type=Path)
    parser.add_argument("--split-index", type=int, default=None)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("archive/generated/bci_distributions"),
    )
    parser.add_argument("--pca-components", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    training, _, _ = load_cse_feature_file(args.training_file)
    evaluation, evaluation_times, _ = load_cse_feature_file(args.evaluation_file)
    if training.shape[1] != evaluation.shape[1]:
        raise ValueError("Training and evaluation files must have equal feature counts.")

    split_index = args.split_index or (evaluation.shape[0] // 2)
    config = CSEConfig(pca_components=min(args.pca_components, training.shape[1]))
    result = run_cse(training, evaluation, evaluation_times, config=config)

    output_directory = args.output_directory
    output_directory.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame({
        "feature": np.arange(training.shape[1]),
        "training_mean": training.mean(axis=0),
        "evaluation_before_mean": evaluation[:split_index].mean(axis=0),
        "evaluation_after_mean": evaluation[split_index:].mean(axis=0),
    })
    summary.to_csv(output_directory / "feature_distribution_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(evaluation_times, result.testing_signal, label="PC1 signal")
    alarms = result.warning_results.loc[result.warning_results["stage_1_alarm"].astype(bool)]
    if not alarms.empty:
        ax.scatter(alarms["time"], alarms["x"], marker="x", label="Stage-I warnings")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_directory / "pc1_stream_warnings.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
