"""Run the Dataset 2B CSE experiment against the published Table 1 values."""

from argparse import ArgumentParser
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bci_2b_experiment import (
    concatenate_trial_features,
    extract_dataset_2b_trials,
    run_dataset_2b_subject,
)
from src.bci_data import load_bci_competition_iv_2b_session


def _parse_args():
    parser = ArgumentParser(
        description=(
            "Run cue-aligned Dataset 2B CSE experiments using the "
            "published subject-specific lambda values and the "
            "Algorithm 1 training-reference validation path."
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
        default=3.0,
    )
    parser.add_argument(
        "--variance-smoothing",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("outputs/metrics/bci_2b_table1_results.csv"),
    )
    return parser.parse_args()


def _load_features(data_directory: Path, subject: int):
    session_features = {}
    for session_number in range(1, 6):
        session = load_bci_competition_iv_2b_session(
            data_directory=data_directory,
            subject=subject,
            session=session_number,
        )
        trial_features = extract_dataset_2b_trials(session)
        session_features[session_number] = trial_features
        print(
            f"B{subject:02d} session {session_number:02d}: "
            f"{trial_features.features.shape[0]} cue-aligned trials"
        )

    training = concatenate_trial_features(
        [session_features[1], session_features[2], session_features[3]]
    )
    testing = concatenate_trial_features(
        [session_features[4], session_features[5]]
    )
    return training, testing


def main() -> None:
    args = _parse_args()
    rows = []

    for subject in args.subjects:
        training, testing = _load_features(args.data_directory, subject)
        result = run_dataset_2b_subject(
            subject,
            training,
            testing,
            validation_alpha=args.alpha,
            control_limit_multiplier=args.control_limit_multiplier,
            variance_smoothing=args.variance_smoothing,
        )

        rows.append({
            "subject": result.subject,
            "lambda": result.published_lambda,
            "training_trials": result.training_trials,
            "testing_trials": result.testing_trials,
            "published_csw": result.published_csw,
            "computed_csw": result.computed_csw,
            "csw_difference": result.computed_csw - result.published_csw,
            "published_csv": result.published_csv,
            "computed_csv": result.computed_csv,
            "csv_difference": result.computed_csv - result.published_csv,
            "matches_published": (
                result.computed_csw == result.published_csw
                and result.computed_csv == result.published_csv
            ),
        })

    table = pd.DataFrame(rows)
    mean_row = {
        "subject": "Mean",
        "lambda": table["lambda"].mean(),
        "training_trials": table["training_trials"].mean(),
        "testing_trials": table["testing_trials"].mean(),
        "published_csw": table["published_csw"].mean(),
        "computed_csw": table["computed_csw"].mean(),
        "csw_difference": table["csw_difference"].mean(),
        "published_csv": table["published_csv"].mean(),
        "computed_csv": table["computed_csv"].mean(),
        "csv_difference": table["csv_difference"].mean(),
        "matches_published": bool(table["matches_published"].all()),
    }
    output = pd.concat([table, pd.DataFrame([mean_row])], ignore_index=True)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_file, index=False)

    print("\n" + output.to_string(index=False, float_format="%.2f"))
    print(f"\nSaved results to: {args.output_file.resolve()}")
    print(
        "Published counts are reference targets; computed counts come from "
        "the loaded GDF recordings and are never replaced by those targets."
    )


if __name__ == "__main__":
    main()
