"""Archived legacy Dataset 2B feature preparation script.

This script belongs to the superseded overlapping-window mu/beta band-power
pipeline. It is retained only for provenance and must not be used for the
current Table 1 FBCSP replication.
"""

from argparse import ArgumentParser
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bci_data import load_bci_competition_iv_2b_session
from src.bci_features import concatenate_feature_results, extract_bandpower_windows


def _parse_args():
    parser = ArgumentParser(
        description=(
            "ARCHIVED: extract sliding-window mu and beta log band-power "
            "features from Dataset 2B GDF sessions."
        )
    )
    parser.add_argument("data_directory", type=Path)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("archive/generated/processed_bci_2b"),
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

    training = concatenate_feature_results(
        [session_results[1], session_results[2]]
    )
    args.output_directory.mkdir(parents=True, exist_ok=True)
    prefix = f"B{args.subject:02d}"
    outputs = {
        "training": (
            args.output_directory / f"{prefix}_training_features.npz",
            training,
        ),
        "session_iii": (
            args.output_directory / f"{prefix}_session_03T_features.npz",
            session_results[3],
        ),
        "session_iv": (
            args.output_directory / f"{prefix}_session_04E_features.npz",
            session_results[4],
        ),
        "session_v": (
            args.output_directory / f"{prefix}_session_05E_features.npz",
            session_results[5],
        ),
    }
    for _, (path, result) in outputs.items():
        _save_result(path, result)


if __name__ == "__main__":
    main()
