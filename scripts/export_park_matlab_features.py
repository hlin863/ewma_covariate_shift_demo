"""Export the *existing* Park-page BCI trial features for MATLAB.

This script does not recalculate signal features or refit EEG transformations.
One subject and one session per CSV, using the Flask page's loader.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.web import app
from src.web.data_structures import _park_bci_features


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("2a", "2b"), required=True)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--session", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, help="Override the selected dataset's GDF directory")
    parser.add_argument("--labels-2a", type=Path, help="Optional official 2A evaluation-label directory")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not 1 <= args.subject <= 9:
        raise ValueError("--subject must be between 1 and 9.")
    if not 1 <= args.session <= (2 if args.dataset == "2a" else 5):
        raise ValueError("--session must be 1-2 for 2A or 1-5 for 2B.")

    with app.app_context():
        if args.data_path is not None:
            app.config["DATASET_2A_PATH" if args.dataset == "2a" else "DATASET_2B_PATH"] = str(args.data_path)
        if args.labels_2a is not None:
            app.config["DATASET_2A_LABELS_PATH"] = str(args.labels_2a)
        frame, description = _park_bci_features(args.dataset, args.subject, args.session)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, float_format="%.12g")
    print(f"Exported {len(frame)} ordered trial observations: {description}")
    print(f"CSV: {args.output} | columns: {', '.join(frame.columns)}")


if __name__ == "__main__":
    main()
