"""BCI Competition IV 2A/2B -> existing FBCSP -> frozen PCA -> loss trace.

Run from repository root, e.g.
python -m scripts.run_bci_reconstruction_loss --dataset 2b --subjects 1
python -m scripts.run_bci_reconstruction_loss --dataset 2a --subjects 1 --labels-2a data/raw/bci_competition_iv_2a_labels
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from src.bci.data import (
    load_bci_competition_iv_2a_session,
    load_bci_competition_iv_2b_session,
)
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
    build_dataset_2b_development_fbcsp_features,
    extract_dataset_2b_trials,
    prepare_cse_uael_2019_protocol,
)
from src.bci.pipelines.reconstruction_loss import evaluate_bci_reconstruction_loss


def feature_pipeline(args, subject):
    if args.dataset == "2a":
        if args.labels_2a is None:
            raise ValueError("Dataset 2A requires official Session-II labels via --labels-2a")
        first = load_bci_competition_iv_2a_session(args.data_2a, subject, "T")
        second = load_bci_competition_iv_2a_session(args.data_2a, subject, "E")
        labels_path = resolve_dataset_2a_evaluation_label_path(
            subject, data_directory=args.data_2a, labels_directory=args.labels_2a
        )
        labels = load_dataset_2a_evaluation_labels(labels_path)
        development = split_dataset_2a_session1(
            extract_dataset_2a_trials(first),
            validation_fraction=args.validation_fraction,
            random_state=args.seed,
        )
        evaluation = extract_dataset_2a_trials(second, evaluation_labels=labels)
        return build_dataset_2a_development_fbcsp_features(
            development, evaluation, components_per_side=args.components_per_side
        )

    sessions = [
        extract_dataset_2b_trials(
            load_bci_competition_iv_2b_session(args.data_2b, subject, number)
        )
        for number in range(1, 6)
    ]
    protocol = prepare_cse_uael_2019_protocol(
        sessions, validation_fraction=args.validation_fraction, random_state=args.seed
    )
    return build_dataset_2b_development_fbcsp_features(
        protocol.development, protocol.evaluation,
        components_per_side=args.components_per_side,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["2a", "2b"], required=True)
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    parser.add_argument("--data-2a", type=Path, default=Path("data/raw/bci_competition_iv_2a"))
    parser.add_argument("--labels-2a", type=Path)
    parser.add_argument("--data-2b", type=Path, default=Path("data/raw/bci_competition_iv_2b"))
    parser.add_argument("--output", type=Path, default=Path("outputs/experiments/bci_reconstruction_loss"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.30)
    parser.add_argument("--components-per-side", type=int, default=1)
    parser.add_argument("--pca-components", type=int, default=1)
    parser.add_argument("--min-side", type=int, default=10)
    parser.add_argument("--bernstein-M", type=float, default=None,
                        help="Optional exploratory batch score; not a calibrated p-value")
    args = parser.parse_args()
    if any(not 1 <= s <= 9 for s in args.subjects):
        parser.error("subjects must be integers from 1 to 9")
    if not 0 < args.validation_fraction < 1:
        parser.error("validation-fraction must lie in (0,1)")
    args.output.mkdir(parents=True, exist_ok=True)
    summary = []
    for subject in args.subjects:
        pipeline = feature_pipeline(args, subject)
        train, evaluation = pipeline.training, pipeline.testing
        result = evaluate_bci_reconstruction_loss(
            train.features, evaluation.features,
            times=evaluation.times,
            feature_names=train.feature_names,
            n_components=args.pca_components,
            candidate_min_side=args.min_side,
            bernstein_M=args.bernstein_M,
        )
        key = f"{args.dataset.upper()}_{subject:02d}"
        result.evaluation_trace.to_csv(args.output / f"{key}_loss_trace.csv", index=False)
        pd.DataFrame({
            "feature": train.feature_names,
            "training_mean_residual_sq": result.feature_mse_before,
            "evaluation_mean_residual_sq": result.feature_mse_after,
        }).to_csv(args.output / f"{key}_feature_residuals.csv", index=False)
        record = {
            "dataset": args.dataset.upper(), "subject": subject,
            "training_trials": len(train.features),
            "evaluation_trials": len(evaluation.features),
            "pca_components": result.pca_components,
            "mean_training_loss": float(result.training_loss.mean()),
            "mean_evaluation_loss": float(result.evaluation_loss.mean()),
            "batch_split": result.batch_split,
        }
        summary.append(record)
        print(json.dumps(record, default=str))
    manifest_path = args.output / "manifest.json"
    previous = []
    if manifest_path.is_file():
        try:
            prior = json.loads(manifest_path.read_text(encoding="utf-8"))
            previous = prior.get("subjects", [])
            if not isinstance(previous, list):
                raise ValueError("Malformed previous manifest")
        except (ValueError, OSError) as exc:
            raise RuntimeError(f"Cannot safely update existing manifest: {exc}") from exc
    updated_keys = {(row["dataset"], int(row["subject"])) for row in summary}
    retained = [
        row for row in previous
        if isinstance(row, dict)
        and (str(row.get("dataset", "")).upper(), int(row.get("subject", -1))) not in updated_keys
    ]
    combined = sorted(retained + summary, key=lambda row: (row["dataset"], int(row["subject"])))
    (args.output / "manifest.json").write_text(json.dumps({
        "provenance": "BCI Competition IV; repo development FBCSP protocol; Heyden Eq(1) teaching extension",
        "warning": "Session transitions are not annotated change points; batch Bernstein is exploratory",
        "config": {k: str(v) for k, v in vars(args).items()},
        "subjects": combined,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
