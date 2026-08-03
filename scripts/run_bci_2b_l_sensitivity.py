"""Sweep EWMA control-limit multipliers and variance update modes."""

from argparse import ArgumentParser
from pathlib import Path
import sys

import numpy as np
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


DEFAULT_L_VALUES = (1.0, 1.25, 1.5, 1.645, 1.96, 2.0, 2.25, 2.5, 3.0)
DEFAULT_VARIANCE_MODES = ("always", "frozen", "non_alarm")


def _parse_args():
    parser = ArgumentParser(
        description="Compare L values and EWMA variance update modes."
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
    parser.add_argument(
        "--l-values",
        type=float,
        nargs="+",
        default=list(DEFAULT_L_VALUES),
    )
    parser.add_argument(
        "--variance-update-modes",
        choices=DEFAULT_VARIANCE_MODES,
        nargs="+",
        default=list(DEFAULT_VARIANCE_MODES),
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--variance-smoothing", type=float, default=0.05)
    parser.add_argument("--components-per-side", type=int, default=1)
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("outputs/metrics/bci_2b_l_sensitivity.csv"),
    )
    return parser.parse_args()


def _load_features(data_directory: Path, subject: int, components_per_side: int):
    sessions = {}
    for session_number in range(1, 6):
        session = load_bci_competition_iv_2b_session(
            data_directory=data_directory,
            subject=subject,
            session=session_number,
        )
        sessions[session_number] = extract_dataset_2b_trials(session)

    training_trials = concatenate_trial_signals(
        [sessions[1], sessions[2], sessions[3]]
    )
    testing_trials = concatenate_trial_signals([sessions[4], sessions[5]])
    pipeline = build_dataset_2b_fbcsp_features(
        training_trials,
        testing_trials,
        components_per_side=components_per_side,
    )
    return pipeline.training, pipeline.testing


def main() -> None:
    args = _parse_args()
    if any(value <= 0.0 for value in args.l_values):
        raise ValueError("all L values must be positive.")

    rows: list[dict] = []
    for subject in args.subjects:
        training, testing = _load_features(
            args.data_directory,
            subject,
            args.components_per_side,
        )
        print(
            f"B{subject:02d}: loaded train={training.features.shape}, "
            f"test={testing.features.shape}"
        )
        for variance_mode in args.variance_update_modes:
            for multiplier in args.l_values:
                result = run_dataset_2b_diagnostic(
                    subject,
                    training,
                    testing,
                    validation_alpha=args.alpha,
                    control_limit_multiplier=float(multiplier),
                    variance_smoothing=args.variance_smoothing,
                    variance_update_mode=variance_mode,
                )
                row = result.summary_row()
                row["absolute_csw_error"] = abs(row["csw_difference"])
                row["absolute_csv_error"] = abs(row["csv_difference"])
                rows.append(row)
                print(
                    f"  mode={variance_mode:9s} L={multiplier:5.3f} "
                    f"CSW={result.computed_csw:3d} "
                    f"CSV={result.computed_csv:3d}"
                )

    output = pd.DataFrame(rows).sort_values(
        ["subject", "variance_update_mode", "control_limit_multiplier"]
    )
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_file, index=False)

    grouped = (
        output.groupby(
            ["variance_update_mode", "control_limit_multiplier"],
            as_index=False,
        )[
            [
                "computed_csw",
                "published_csw",
                "absolute_csw_error",
                "computed_csv",
                "published_csv",
                "absolute_csv_error",
                "mean_half_width",
            ]
        ]
        .mean()
        .sort_values("absolute_csw_error")
    )
    print("\nMean sensitivity summary:")
    print(grouped.to_string(index=False, float_format="%.4f"))
    print(f"\nSaved sensitivity results to: {args.output_file.resolve()}")


if __name__ == "__main__":
    main()
