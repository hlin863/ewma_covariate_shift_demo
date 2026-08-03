"""Generate a paper-style Dataset 2B CSE summary table.

This script evaluates Session IV and Session V separately and reports:
subject, fitted/overridden lambda, CSW percentage, and CSV percentage.

The percentages are calculated over the processed observations in each file.
With the current sliding-window feature extractor, the denominator is EEG
windows, not the paper's 160 cue-aligned trials. Exact paper reproduction will
require trial-level feature extraction.
"""

from argparse import ArgumentParser
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bci_data import load_cse_feature_file
from src.cse import CSEConfig, run_cse


def _parse_args():
    parser = ArgumentParser(
        description="Generate Dataset 2B CSE Table 3-style results."
    )
    parser.add_argument(
        "processed_directory",
        type=Path,
        nargs="?",
        default=Path("data/processed/bci_2b"),
    )
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        default=list(range(1, 10)),
    )
    parser.add_argument("--pca-components", type=int, default=3)
    parser.add_argument("--before-size", type=int, default=20)
    parser.add_argument("--after-size", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("outputs/metrics/bci_2b_table3_results.csv"),
    )
    return parser.parse_args()


def _evaluate_session(
    training_file: Path,
    evaluation_file: Path,
    config: CSEConfig,
) -> dict[str, float | int]:
    training, _, _ = load_cse_feature_file(training_file)
    evaluation, times, _ = load_cse_feature_file(evaluation_file)

    result = run_cse(
        training_features=training,
        testing_features=evaluation,
        testing_times=times,
        config=config,
    )

    n_observations = int(result.warning_results.shape[0])
    csw_count = int(result.warning_results["stage_1_alarm"].sum())
    csv_count = int(result.validation_results["confirmed_shift"].sum())

    return {
        "lambda": float(result.effective_lambda),
        "n_observations": n_observations,
        "csw_count": csw_count,
        "csv_count": csv_count,
        "csw_percent": 100.0 * csw_count / n_observations,
        "csv_percent": 100.0 * csv_count / n_observations,
    }


def main() -> None:
    args = _parse_args()
    rows = []

    for subject in args.subjects:
        prefix = f"B{subject:02d}"
        training_file = (
            args.processed_directory / f"{prefix}_training_features.npz"
        )
        session_iv_file = (
            args.processed_directory / f"{prefix}_session_04E_features.npz"
        )
        session_v_file = (
            args.processed_directory / f"{prefix}_session_05E_features.npz"
        )

        missing = [
            path
            for path in (training_file, session_iv_file, session_v_file)
            if not path.is_file()
        ]
        if missing:
            print(
                f"Skipping {prefix}; missing: "
                + ", ".join(str(path) for path in missing)
            )
            continue

        training, _, _ = load_cse_feature_file(training_file)
        config = CSEConfig(
            pca_components=min(args.pca_components, training.shape[1]),
            validation_before_size=args.before_size,
            validation_after_size=args.after_size,
            validation_alpha=args.alpha,
            minimum_alarm_gap=args.after_size,
        )

        session_iv = _evaluate_session(
            training_file,
            session_iv_file,
            config,
        )
        session_v = _evaluate_session(
            training_file,
            session_v_file,
            config,
        )

        rows.append({
            "subject": prefix,
            "lambda": session_iv["lambda"],
            "session_iv_n": session_iv["n_observations"],
            "session_iv_csw_count": session_iv["csw_count"],
            "session_iv_csw_percent": session_iv["csw_percent"],
            "session_iv_csv_count": session_iv["csv_count"],
            "session_iv_csv_percent": session_iv["csv_percent"],
            "session_v_n": session_v["n_observations"],
            "session_v_csw_count": session_v["csw_count"],
            "session_v_csw_percent": session_v["csw_percent"],
            "session_v_csv_count": session_v["csv_count"],
            "session_v_csv_percent": session_v["csv_percent"],
        })

    if not rows:
        raise FileNotFoundError(
            "No complete subject feature sets were found. Run "
            "prepare_bci_2b_cse_features.py first."
        )

    table = pd.DataFrame(rows)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_file, index=False)

    display_columns = [
        "subject",
        "lambda",
        "session_iv_csw_percent",
        "session_iv_csv_percent",
        "session_v_csw_percent",
        "session_v_csv_percent",
    ]
    print(table[display_columns].to_string(index=False, float_format="%.2f"))
    print(f"\nSaved Table 3-style results to: {args.output_file.resolve()}")


if __name__ == "__main__":
    main()
