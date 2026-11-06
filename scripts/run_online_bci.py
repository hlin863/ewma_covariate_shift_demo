"""Run the connected online-adaptive feature pipeline on existing BCI datasets.

This is trial-level feature replay (offline zero-phase FBCSP preprocessing), not
causal EEG acquisition. Sessions/splits match existing repository builders.
"""
from argparse import ArgumentParser
from pathlib import Path
import sys
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.adaptation.policies import (
    NeverUpdate, PeriodicRetrain, RetrainOnWarning, RetrainOnValidatedShift,
)
from src.online.pipeline import OnlineAdaptiveBCI, OnlineAdaptationConfig


def prepare(dataset, subject, data_2a, labels_2a, data_2b, seed):
    if dataset == "2a":
        from src.bci.data import load_bci_competition_iv_2a_session
        from src.bci.datasets.dataset2a import (
            extract_dataset_2a_trials, split_dataset_2a_session1,
            build_dataset_2a_development_fbcsp_features,
        )
        from src.bci.datasets.dataset2a.evaluation_labels import (
            load_dataset_2a_evaluation_labels, resolve_dataset_2a_evaluation_label_path,
        )
        if labels_2a is None:
            raise ValueError("2A requires --labels-2a for supervised replay")
        first = extract_dataset_2a_trials(load_bci_competition_iv_2a_session(data_2a, subject, "T"))
        path = resolve_dataset_2a_evaluation_label_path(
            subject, data_directory=data_2a, labels_directory=labels_2a
        )
        last = extract_dataset_2a_trials(
            load_bci_competition_iv_2a_session(data_2a, subject, "E"),
            evaluation_labels=load_dataset_2a_evaluation_labels(path),
        )
        split = split_dataset_2a_session1(first, random_state=seed)
        features = build_dataset_2a_development_fbcsp_features(split, last)
        return features.training.features, split.training.labels, features.testing.features, last.labels
    from src.bci.data import load_bci_competition_iv_2b_session
    from src.bci.datasets.dataset2b import (
        extract_dataset_2b_trials, prepare_cse_uael_2019_protocol,
        build_dataset_2b_development_fbcsp_features,
    )
    sessions = [
        extract_dataset_2b_trials(load_bci_competition_iv_2b_session(data_2b, subject, i))
        for i in range(1, 6)
    ]
    protocol = prepare_cse_uael_2019_protocol(sessions, random_state=seed)
    features = build_dataset_2b_development_fbcsp_features(
        protocol.development, protocol.evaluation
    )
    return (features.training.features, protocol.development.training.labels,
            features.evaluation.features, protocol.evaluation.labels)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["2a", "2b"], required=True)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--data-2a", type=Path, default=Path("data/raw/bci_competition_iv_2a"))
    parser.add_argument("--labels-2a", type=Path)
    parser.add_argument("--data-2b", type=Path, default=Path("data/raw/bci_competition_iv_2b"))
    parser.add_argument("--policy", choices=["never", "periodic", "warning", "validated"], default="validated")
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--label-delay", type=int, default=0,
                        help="Delay in trials; -1 means no labels released to the learner")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pca-components", type=int, default=2)
    parser.add_argument("--lambda-value", type=float, default=0.2)
    parser.add_argument("--output", type=Path, default=Path("outputs/online"))
    args = parser.parse_args()
    x_train, y_train, x_test, y_test = prepare(
        args.dataset, args.subject, args.data_2a, args.labels_2a, args.data_2b, args.seed
    )
    policies = dict(
        never=NeverUpdate(),
        periodic=PeriodicRetrain(args.interval),
        warning=RetrainOnWarning(),
        validated=RetrainOnValidatedShift(),
    )
    config = OnlineAdaptationConfig(
        window_size=args.window_size,
        label_delay=None if args.label_delay < 0 else args.label_delay,
    )
    pipeline = OnlineAdaptiveBCI.from_calibration(
        x_train, y_train, policy=policies[args.policy], config=config,
        pca_components=args.pca_components,
        lambda_override=args.lambda_value,
    )
    result = pipeline.run_replay(x_test, np.arange(len(x_test)), y_test)
    target = args.output / args.dataset / f"subject_{args.subject:02d}" / args.policy
    target.mkdir(parents=True, exist_ok=True)
    result.trials.to_csv(target / "trials.csv", index=False)
    result.validations.to_csv(target / "stage2.csv", index=False)
    result.updates.to_csv(target / "updates.csv", index=False)
    summary = dict(dataset=args.dataset, subject=args.subject, policy=args.policy,
                   trials=len(result.trials), accuracy=float(result.trials["correct"].mean()),
                   warnings=int(result.trials["stage1_warning"].sum()),
                   validations=int(result.validations["status"].eq("confirmed").sum())
                   if not result.validations.empty else 0,
                   updates=len(result.updates), label_delay=args.label_delay)
    pd.DataFrame([summary]).to_csv(target / "summary.csv", index=False)
    print(pd.Series(summary).to_string())
    print(f"Saved outputs to {target}")


if __name__ == "__main__":
    main()
