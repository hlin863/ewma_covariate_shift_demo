"""Archived diagnostic Table 1 runner for legacy processed feature files.

This path consumes pre-generated overlapping-window band-power `.npz` files.
It is retained for provenance only. Use `scripts/run_bci_2b_experiment.py` for
the active trial-level FBCSP pipeline.
"""

from argparse import ArgumentParser
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bci_2b_reference import bci_2b_table1_reference
from src.bci_data import load_cse_feature_file
from src.cse import CSEConfig, run_cse


def _parse_args():
    parser = ArgumentParser(
        description="ARCHIVED: compare legacy Dataset 2B CSE output with Table 1."
    )
    parser.add_argument(
        "processed_directory",
        type=Path,
        nargs="?",
        default=Path("archive/generated/processed_bci_2b"),
    )
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    parser.add_argument("--pca-components", type=int, default=3)
    parser.add_argument("--before-size", type=int, default=20)
    parser.add_argument("--after-size", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("archive/generated/bci_2b_table1_comparison.csv"),
    )
    return parser.parse_args()


def _find_testing_file(directory: Path, prefix: str) -> Path | None:
    candidates = (
        directory / f"{prefix}_evaluation_features.npz",
        directory / f"{prefix}_session_04E_features.npz",
    )
    return next((path for path in candidates if path.is_file()), None)


def _evaluate(training_file: Path, testing_file: Path, config: CSEConfig):
    training, _, _ = load_cse_feature_file(training_file)
    testing, times, _ = load_cse_feature_file(testing_file)
    result = run_cse(training, testing, times, config=config)
    return {
        "observed_lambda": float(result.effective_lambda),
        "observed_csw": int(result.warning_results["stage_1_alarm"].sum()),
        "observed_csv": int(result.validation_results["confirmed_shift"].sum()),
        "observation_count": int(testing.shape[0]),
        "testing_file": str(testing_file),
    }


def main() -> None:
    args = _parse_args()
    reference = bci_2b_table1_reference()
    observed_rows = []
    for subject_number in args.subjects:
        prefix = f"B{subject_number:02d}"
        training_file = args.processed_directory / f"{prefix}_training_features.npz"
        testing_file = _find_testing_file(args.processed_directory, prefix)
        if not training_file.is_file() or testing_file is None:
            observed_rows.append({"subject": prefix, "status": "missing_processed_data"})
            continue
        training, _, _ = load_cse_feature_file(training_file)
        config = CSEConfig(
            pca_components=min(args.pca_components, training.shape[1]),
            validation_before_size=args.before_size,
            validation_after_size=args.after_size,
            validation_alpha=args.alpha,
            minimum_alarm_gap=args.after_size,
        )
        row = _evaluate(training_file, testing_file, config)
        row.update({"subject": prefix, "status": "computed"})
        observed_rows.append(row)

    observed = pd.DataFrame(observed_rows)
    comparison = reference.merge(observed, on="subject", how="left")
    comparison["lambda_error"] = (
        comparison["observed_lambda"] - comparison["paper_lambda"]
    ).abs()
    comparison["csw_error"] = (
        comparison["observed_csw"] - comparison["paper_csw"]
    ).abs()
    comparison["csv_error"] = (
        comparison["observed_csv"] - comparison["paper_csv"]
    ).abs()
    comparison["matches_paper"] = (
        comparison["lambda_error"].le(0.005)
        & comparison["csw_error"].eq(0)
        & comparison["csv_error"].eq(0)
    )
    comparison["experiment_unit"] = "overlapping EEG windows"
    comparison["replication_status"] = comparison["matches_paper"].map(
        {True: "matched", False: "not_matched"}
    )
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    main()
