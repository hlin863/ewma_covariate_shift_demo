"""Prepare BCI Competition IV Dataset 2B features for CSE visualisation.

Example
-------
python scripts/prepare_bci_2b_cse_features.py \
    data/raw/bci_competition_iv_2b \
    --subject 1

This creates:
    data/processed/bci_2b/B01_training_features.npz
    data/processed/bci_2b/B01_evaluation_features.npz
"""

from argparse import ArgumentParser
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bci_data import load_bci_competition_iv_2b_session
from src.bci_features import (
    concatenate_feature_results,
    extract_bandpower_windows,
)


def _parse_args():
    parser = ArgumentParser(
        description=(
            "Extract sliding-window mu and beta log band-power features "
            "from BCI Competition IV Dataset 2B GDF sessions."
        )
    )
    parser.add_argument("data_directory", type=Path)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("data/processed/bci_2b"),
    )
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--step-seconds", type=float, default=0.5)
    return parser.parse_args()


def _save_result(path: Path, result) -> None:
    np.savez(
        path,
        features=result.features,
        times=result.times,
        feature_names=np.asarray(result.feature_names),
        session_ids=result.session_ids,
    )


def main() -> None:
    args = _parse_args()

    session_results = {}
    for session_number in range(1, 6):
        session = load_bci_competition_iv_2b_session(
            data_directory=args.data_directory,
            subject=args.subject,
            session=session_number,
        )
        result = extract_bandpower_windows(
            session,
            window_seconds=args.window_seconds,
            step_seconds=args.step_seconds,
        )
        session_results[session_number] = result
        print(
            f"Session {session.session}: "
            f"{result.features.shape[0]} windows x "
            f"{result.features.shape[1]} features"
        )

    training = concatenate_feature_results(
        [session_results[1], session_results[2]]
    )
    evaluation = concatenate_feature_results(
        [session_results[3], session_results[4], session_results[5]]
    )

    args.output_directory.mkdir(parents=True, exist_ok=True)
    prefix = f"B{args.subject:02d}"
    training_path = args.output_directory / f"{prefix}_training_features.npz"
    evaluation_path = args.output_directory / f"{prefix}_evaluation_features.npz"

    _save_result(training_path, training)
    _save_result(evaluation_path, evaluation)

    print(f"\nSaved training features: {training_path.resolve()}")
    print(f"Saved evaluation features: {evaluation_path.resolve()}")
    print(
        "Suggested visualisation split index: "
        f"{session_results[3].features.shape[0]} "
        "(boundary after session 03T)"
    )


if __name__ == "__main__":
    main()
