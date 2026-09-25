"""Export real 2A/2B Stage-I score and residual traces for the dashboard.

Run from the repository root: python -m scripts.run_complementary_bci --labels-2a data/raw/bci_competition_iv_2a_labels
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.bci.data import load_bci_competition_iv_2a_session, load_bci_competition_iv_2b_session
from src.bci.datasets.dataset2a import (
    build_dataset_2a_development_fbcsp_features,
    extract_dataset_2a_trials,
    split_dataset_2a_session1,
)
from src.bci.datasets.dataset2a.evaluation_labels import (
    load_dataset_2a_evaluation_labels,
    resolve_dataset_2a_evaluation_label_path,
)
from src.bci.datasets.dataset2b import (
    build_dataset_2b_fbcsp_features,
    concatenate_trial_signals,
    extract_dataset_2b_trials,
)
from src.detection.stage1.complementary import fit_complementary, monitor


def build_trace(training, calibration, testing, *, dataset, subject, session, alpha, lambda_value):
    """Fit on development features and preserve one row per evaluation trial."""
    model = fit_complementary(training, calibration, alpha=alpha, lambda_value=lambda_value)
    trace = monitor(testing, model)
    trace.insert(0, "session", str(session))
    trace.insert(0, "subject", str(subject))
    trace.insert(0, "dataset", dataset)
    return trace, {
        "dataset": dataset, "subject": subject, "session": str(session),
        "training_trials": len(training), "calibration_trials": len(calibration),
        "evaluation_trials": len(testing),
        "pc1_variance_ratio": float(model.pca.explained_variance_ratio[0]),
        "score_limit": model.score_limit, "residual_limit": model.residual_limit,
        "joint_limit": model.joint_limit,
    }


def run_2a(args, subject):
    training_session = load_bci_competition_iv_2a_session(args.data_2a, subject, "T")
    evaluation_session = load_bci_competition_iv_2a_session(args.data_2a, subject, "E")
    labels_path = resolve_dataset_2a_evaluation_label_path(
        subject, data_directory=args.data_2a, labels_directory=args.labels_2a
    )
    labels = load_dataset_2a_evaluation_labels(labels_path)
    split = split_dataset_2a_session1(
        extract_dataset_2a_trials(training_session), random_state=args.split_seed
    )
    pipeline = build_dataset_2a_development_fbcsp_features(
        split, extract_dataset_2a_trials(evaluation_session, evaluation_labels=labels)
    )
    if len(pipeline.testing.features) != 144:
        raise ValueError(f"A{subject:02d}: expected 144 left/right evaluation trials")
    return build_trace(
        pipeline.training.features, pipeline.validation.features, pipeline.testing.features,
        dataset="2A", subject=f"A{subject:02d}", session="E",
        alpha=args.alpha, lambda_value=args.lambda_value,
    )


def run_2b(args, subject):
    trials = [extract_dataset_2b_trials(load_bci_competition_iv_2b_session(
        args.data_2b, subject, session
    )) for session in range(1, 6)]
    # Sessions I-II fit FBCSP/PCA/EWMA; III is an independent threshold calibration set.
    # IV and V are evaluated in chronological order with the same initial monitoring state.
    training = concatenate_trial_signals(trials[:2])
    evaluation = concatenate_trial_signals(trials[2:])
    pipeline = build_dataset_2b_fbcsp_features(training, evaluation)
    n_cal = len(trials[2].signals)
    features = pipeline.testing.features
    calibration, test = features[:n_cal], features[n_cal:]
    trace, info = build_trace(
        pipeline.training.features, calibration, test,
        dataset="2B", subject=f"B{subject:02d}", session="IV-V",
        alpha=args.alpha, lambda_value=args.lambda_value,
    )
    session_ids = np.asarray(pipeline.testing.session_ids)[n_cal:]
    trace["session"] = [str(value) for value in session_ids]
    return trace, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-2a", type=Path, default=Path("data/raw/bci_competition_iv_2a"))
    parser.add_argument("--labels-2a", type=Path, default=Path("data/raw/bci_competition_iv_2a_labels"))
    parser.add_argument("--data-2b", type=Path, default=Path("data/raw/bci_competition_iv_2b"))
    parser.add_argument("--datasets", nargs="+", choices=("2a", "2b"), default=("2a", "2b"))
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--lambda-value", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/complementary_bci"))
    args = parser.parse_args()
    if any(not 1 <= subject <= 9 for subject in args.subjects):
        parser.error("subjects must be numbered 1 through 9")
    for dataset, directory in (("2a", args.data_2a), ("2b", args.data_2b)):
        if dataset in args.datasets and not directory.is_dir():
            parser.error(f"Missing {dataset.upper()} recordings directory: {directory}")
    if "2a" in args.datasets and not args.labels_2a.is_dir():
        parser.error(f"Missing 2A evaluation-label directory: {args.labels_2a}")

    traces, metadata = [], []
    for subject in args.subjects:
        for dataset, runner in (("2a", run_2a), ("2b", run_2b)):
            if dataset not in args.datasets:
                continue
            trace, info = runner(args, subject)
            traces.append(trace)
            metadata.append(info)
            print(f"{info['subject']}: {len(trace)} trials, {int(trace.stage_1_alarm.sum())} combined warnings")
    args.output.mkdir(parents=True, exist_ok=True)
    pd.concat(traces, ignore_index=True).to_csv(args.output / "traces.csv", index=False)
    pd.DataFrame(metadata).to_csv(args.output / "calibration.csv", index=False)
    (args.output / "config.json").write_text(json.dumps({
        "datasets": args.datasets, "subjects": args.subjects, "alpha": args.alpha,
        "lambda_value": args.lambda_value, "split_seed": args.split_seed,
        "protocol_2a": "Session I stratified 70/30 fit/calibration; Session II left/right evaluation",
        "protocol_2b": "Sessions I-II fit; III calibration; IV-V evaluation",
    }, indent=2), encoding="utf-8")
    print(f"Dashboard: http://127.0.0.1:5000/results/complementary-bci (after starting Flask)")


if __name__ == "__main__":
    main()
