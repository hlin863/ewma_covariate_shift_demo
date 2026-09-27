"""Run the Dataset 2B CSE experiment against published Table 1 values."""

from argparse import ArgumentParser
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bci_2b_diagnostics import run_dataset_2b_diagnostic
from src.bci_2b_experiment import (
    build_dataset_2b_fbcsp_features,
    concatenate_trial_signals,
    extract_dataset_2b_trials,
)
from src.bci_data import load_bci_competition_iv_2b_session


def _parse_args():
    parser = ArgumentParser(
        description=(
            "Run cue-aligned Dataset 2B CSE experiments using ten-band "
            "FBCSP features and published subject-specific lambdas."
        )
    )
    parser.add_argument(
        "data_directory",
        type=Path,
        nargs="?",
        default=Path("data/raw/bci_competition_iv_2b"),
    )
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        default=list(range(1, 10)),
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--control-limit-multiplier",
        type=float,
        default=1.96,
    )
    parser.add_argument("--variance-smoothing", type=float, default=0.05)
    parser.add_argument(
        "--variance-update-mode",
        choices=("always", "frozen", "non_alarm"),
        default="always",
    )
    parser.add_argument("--components-per-side", type=int, default=1)
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("outputs/metrics/bci_2b_table1_results.csv"),
    )
    parser.add_argument(
        "--warning-output-file",
        type=Path,
        default=Path("outputs/metrics/bci_2b_stage1_warnings.csv"),
    )
    return parser.parse_args()


def _load_features(
    data_directory: Path,
    subject: int,
    components_per_side: int,
):
    session_trials = {}
    for session_number in range(1, 6):
        session = load_bci_competition_iv_2b_session(
            data_directory=data_directory,
            subject=subject,
            session=session_number,
        )
        trials = extract_dataset_2b_trials(session)
        session_trials[session_number] = trials
        labelled = int((trials.labels >= 0).sum())
        print(
            f"B{subject:02d} session {session_number:02d}: "
            f"{trials.signals.shape[0]} accepted cue-aligned trials "
            f"({labelled} labelled)"
        )

    training_trials = concatenate_trial_signals(
        [session_trials[1], session_trials[2], session_trials[3]]
    )
    testing_trials = concatenate_trial_signals(
        [session_trials[4], session_trials[5]]
    )
    pipeline = build_dataset_2b_fbcsp_features(
        training_trials,
        testing_trials,
        components_per_side=components_per_side,
    )
    print(
        f"B{subject:02d}: FBCSP feature shape "
        f"train={pipeline.training.features.shape}, "
        f"test={pipeline.testing.features.shape}"
    )
    return pipeline.training, pipeline.testing


def main() -> None:
    args = _parse_args()
    rows: list[dict] = []
    warning_tables: list[pd.DataFrame] = []

    for subject in args.subjects:
        training, testing = _load_features(
            args.data_directory,
            subject,
            args.components_per_side,
        )
        result = run_dataset_2b_diagnostic(
            subject,
            training,
            testing,
            validation_alpha=args.alpha,
            control_limit_multiplier=args.control_limit_multiplier,
            variance_smoothing=args.variance_smoothing,
            variance_update_mode=args.variance_update_mode,
        )
        rows.append(result.summary_row())
        warning_tables.append(result.warning_rows(testing))

    table = pd.DataFrame(rows)
    numeric_mean_columns = [
        "lambda",
        "training_trials",
        "testing_trials",
        "published_csw",
        "computed_csw",
        "csw_difference",
        "published_csv",
        "computed_csv",
        "csv_difference",
        "control_limit_multiplier",
        "initial_half_width",
        "mean_half_width",
        "median_half_width",
        "final_half_width",
        "maximum_half_width",
    ]
    mean_row: dict = {
        "subject": "Mean",
        "variance_update_mode": args.variance_update_mode,
        "matches_published": bool(table["matches_published"].all()),
    }
    for column in numeric_mean_columns:
        mean_row[column] = table[column].mean()

    output = pd.concat([table, pd.DataFrame([mean_row])], ignore_index=True)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_file, index=False)

    nonempty_warnings = [frame for frame in warning_tables if not frame.empty]
    if nonempty_warnings:
        warning_output = pd.concat(nonempty_warnings, ignore_index=True)
    else:
        warning_output = pd.DataFrame(
            columns=[
                "subject",
                "lambda",
                "time",
                "session_id",
                "cue_description",
                "stage_1_alarm",
            ]
        )
    args.warning_output_file.parent.mkdir(parents=True, exist_ok=True)
    warning_output.to_csv(args.warning_output_file, index=False)

    print("\n" + output.to_string(index=False, float_format="%.4f"))
    print(f"\nSaved results to: {args.output_file.resolve()}")
    print(f"Saved Stage-I warnings to: {args.warning_output_file.resolve()}")
    print(
        "Published counts are reference targets; computed counts come from "
        "the loaded GDF recordings and are never replaced by those targets."
    )


if __name__ == "__main__":
    main()
