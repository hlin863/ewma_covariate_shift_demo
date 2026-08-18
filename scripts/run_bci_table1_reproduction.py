"""Reproduce the 2019 paper's CSE Table 1 layout for Datasets 2A and 2B."""

from argparse import ArgumentParser, ArgumentTypeError
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bci.data import (
    load_bci_competition_iv_2a_session,
    load_bci_competition_iv_2b_session,
)
from src.bci.datasets.dataset2a import (
    build_dataset_2a_development_fbcsp_features,
    extract_dataset_2a_trials,
    run_dataset_2a_subject,
    split_dataset_2a_session1,
)
from src.bci.datasets.dataset2a.evaluation_labels import (
    load_dataset_2a_evaluation_labels,
    resolve_dataset_2a_evaluation_label_path,
)
from src.bci.datasets.dataset2b import (
    PUBLISHED_2B_RESULTS,
    build_dataset_2b_fbcsp_features,
    concatenate_trial_signals,
    extract_dataset_2b_trials,
)
from src.detection import CSEConfig, run_cse
from src.reporting.table1 import (
    Table1Row,
    comparison_dataframe,
    paper_style_dataframe,
    paper_style_markdown,
)


def _parse_pca_components(value: str) -> int | float | None:
    """Parse PCA retention from CLI."""

    text = str(value).strip().lower()
    if text in {"all", "none"}:
        return None

    try:
        if any(marker in text for marker in (".", "e")):
            parsed_float = float(text)
            if 0.0 < parsed_float < 1.0:
                return parsed_float
            raise ArgumentTypeError(
                "--pca-components as a float must be in (0, 1), for example 0.95."
            )
        parsed_int = int(text)
    except ValueError as error:
        raise ArgumentTypeError(
            "--pca-components must be 'all', a positive integer, or a float in (0, 1)."
        ) from error

    if parsed_int < 1:
        raise ArgumentTypeError("--pca-components as an integer must be at least 1.")
    return parsed_int


def _fraction(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed < 1.0:
        raise ArgumentTypeError("fraction must be in (0, 1).")
    return parsed


def _parse_args():
    parser = ArgumentParser(
        description=(
            "Run the CSE experiment for BCI Competition IV Datasets 2A and 2B "
            "and save computed results in the side-by-side Table 1 structure."
        )
    )
    parser.add_argument(
        "--data-2a", type=Path, default=Path("data/raw/bci_competition_iv_2a")
    )
    parser.add_argument(
        "--labels-2a",
        type=Path,
        default=Path("data/raw/bci_competition_iv_2a_labels"),
        help=(
            "Directory containing the separately released Dataset 2A Session-II "
            "true-label MAT files (A01E.mat ... A09E.mat). The original Dataset 2A "
            "signal download contains GDF files only."
        ),
    )
    parser.add_argument(
        "--data-2b", type=Path, default=Path("data/raw/bci_competition_iv_2b")
    )
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    parser.add_argument(
        "--validation-mode",
        choices=(
            "paper_two_sample",
            "algorithm1_training_reference",
            "retrospective_windows",
        ),
        default="paper_two_sample",
    )
    parser.add_argument("--validation-window-size", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--control-limit-multiplier", type=float, default=1.96)
    parser.add_argument("--variance-smoothing", type=float, default=0.05)
    parser.add_argument(
        "--variance-update-mode",
        choices=("always", "frozen", "non_alarm"),
        default="always",
    )
    parser.add_argument("--components-per-side", type=int, default=1)
    parser.add_argument(
        "--pca-components",
        type=_parse_pca_components,
        default=None,
        metavar="N|FRACTION|all",
        help=(
            "PCA components retained before CSE. Use a positive integer (e.g. 1 or 3), "
            "a variance fraction such as 0.95, or 'all'. Stage I still monitors PC1 only."
        ),
    )
    parser.add_argument(
        "--session1-validation-fraction",
        type=_fraction,
        default=0.30,
        help=(
            "Dataset 2A Session-I fraction held out for validation before FBCSP/PCA/CSE "
            "model fitting. The paper specifies 30%%; default: 0.30."
        ),
    )
    parser.add_argument(
        "--session1-split-seed",
        type=int,
        default=42,
        help=(
            "Deterministic seed for the stratified Dataset 2A Session-I 70/30 split. "
            "The paper does not report its exact random partition."
        ),
    )
    parser.add_argument(
        "--calibrate-control-limit-from-validation",
        action="store_true",
        help=(
            "Dataset 2A only: estimate the Stage-I control-limit multiplier L from "
            "held-out Session-I validation prediction errors. This is a transparent "
            "repo calibration extension; the paper does not report its exact L-selection rule."
        ),
    )
    parser.add_argument(
        "--validation-false-alarm-rate",
        type=_fraction,
        default=0.05,
        help=(
            "Target upper-tail rate used when validation-error calibration of L is enabled. "
            "Default: 0.05."
        ),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("outputs/metrics/bci_table1_reproduction.md"),
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=Path("outputs/metrics/bci_table1_comparison.csv"),
    )
    return parser.parse_args()


def _run_2a(args, subject: int) -> Table1Row:
    session1 = load_bci_competition_iv_2a_session(args.data_2a, subject, "T")
    session2 = load_bci_competition_iv_2a_session(args.data_2a, subject, "E")

    label_path = resolve_dataset_2a_evaluation_label_path(
        subject,
        data_directory=args.data_2a,
        labels_directory=args.labels_2a,
    )
    evaluation_labels = load_dataset_2a_evaluation_labels(label_path)

    session1_trials = extract_dataset_2a_trials(session1)
    testing_trials = extract_dataset_2a_trials(
        session2,
        evaluation_labels=evaluation_labels,
    )

    if testing_trials.signals.shape[0] != 144:
        raise RuntimeError(
            "Dataset 2A paper reproduction expects 144 left/right Session-II trials; "
            f"{testing_trials.signals.shape[0]} were extracted for A{subject:02d}."
        )

    development_split = split_dataset_2a_session1(
        session1_trials,
        validation_fraction=args.session1_validation_fraction,
        random_state=args.session1_split_seed,
    )
    pipeline = build_dataset_2a_development_fbcsp_features(
        development_split,
        testing_trials,
        components_per_side=args.components_per_side,
    )
    result = run_dataset_2a_subject(
        subject,
        pipeline.training,
        pipeline.testing,
        validation=pipeline.validation,
        pca_components=args.pca_components,
        validation_mode=args.validation_mode,
        validation_window_size=args.validation_window_size,
        validation_alpha=args.alpha,
        control_limit_multiplier=args.control_limit_multiplier,
        variance_smoothing=args.variance_smoothing,
        variance_update_mode=args.variance_update_mode,
        calibrate_control_limit_from_validation=(
            args.calibrate_control_limit_from_validation
        ),
        validation_false_alarm_rate=args.validation_false_alarm_rate,
    )
    calibration_text = ""
    if result.validation_calibration is not None:
        calibration = result.validation_calibration
        calibration_text = (
            f", Lval={calibration.control_limit_multiplier:.3f}, "
            f"valRMSE={calibration.validation_rmse:.3f}, "
            f"valTail={calibration.observed_exceedance_rate:.3f}"
        )
    print(
        f"{result.subject}: session1={session1_trials.signals.shape[0]}, "
        f"dev_train={pipeline.training.features.shape[0]}, "
        f"validation={pipeline.validation.features.shape[0]}, "
        f"test={result.testing_trials}, "
        f"PCA={result.cse_result.pca_result.n_components}, "
        f"PC1var={result.cse_result.pca_result.explained_variance_ratio[0]:.3f}, "
        f"L={result.selected_control_limit_multiplier:.3f}, "
        f"CSW={result.computed_csw}/{result.published_csw}, "
        f"CSV={result.computed_csv}/{result.published_csv}"
        f"{calibration_text}"
    )
    return Table1Row(
        dataset="2A",
        subject=result.subject,
        lambda_value=result.published_lambda,
        published_csw=result.published_csw,
        published_csv=result.published_csv,
        computed_csw=result.computed_csw,
        computed_csv=result.computed_csv,
    )


def _run_2b(args, subject: int) -> Table1Row:
    session_trials = []
    for session_number in range(1, 6):
        session = load_bci_competition_iv_2b_session(
            args.data_2b, subject, session_number
        )
        session_trials.append(extract_dataset_2b_trials(session))
    training_trials = concatenate_trial_signals(session_trials[:3])
    testing_trials = concatenate_trial_signals(session_trials[3:])
    pipeline = build_dataset_2b_fbcsp_features(
        training_trials,
        testing_trials,
        components_per_side=args.components_per_side,
    )

    subject_id = f"B{subject:02d}"
    published_lambda, published_csw, published_csv = PUBLISHED_2B_RESULTS[subject_id]
    covariance_method = (
        "empirical" if args.validation_mode == "paper_two_sample" else "shrinkage"
    )
    cse_result = run_cse(
        training_features=pipeline.training.features,
        testing_features=pipeline.testing.features,
        testing_times=pipeline.testing.times,
        config=CSEConfig(
            pca_components=args.pca_components,
            lambda_override=published_lambda,
            variance_smoothing=args.variance_smoothing,
            control_limit_multiplier=args.control_limit_multiplier,
            variance_update_mode=args.variance_update_mode,
            ewma_initialization="training_mean",
            validation_mode=args.validation_mode,
            validation_before_size=args.validation_window_size,
            validation_after_size=args.validation_window_size,
            validation_alpha=args.alpha,
            covariance_method=covariance_method,
        ),
    )
    computed_csw = int(cse_result.warning_results["stage_1_alarm"].astype(bool).sum())
    computed_csv = int(
        cse_result.validation_results["confirmed_shift"].astype(bool).sum()
    )
    print(
        f"{subject_id}: train={pipeline.training.features.shape[0]}, "
        f"test={pipeline.testing.features.shape[0]}, "
        f"PCA={cse_result.pca_result.n_components}, "
        f"PC1var={cse_result.pca_result.explained_variance_ratio[0]:.3f}, "
        f"CSW={computed_csw}/{published_csw}, CSV={computed_csv}/{published_csv}"
    )
    return Table1Row(
        dataset="2B",
        subject=subject_id,
        lambda_value=published_lambda,
        published_csw=published_csw,
        published_csv=published_csv,
        computed_csw=computed_csw,
        computed_csv=computed_csv,
    )


def main() -> None:
    args = _parse_args()
    rows: list[Table1Row] = []
    for subject in args.subjects:
        rows.append(_run_2a(args, subject))
        rows.append(_run_2b(args, subject))

    markdown = paper_style_markdown(rows)
    paper_table = paper_style_dataframe(rows)
    comparison = comparison_dataframe(rows)

    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(markdown, encoding="utf-8")
    args.comparison_output.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.comparison_output, index=False)

    print("\nComputed Table 1 reproduction:\n")
    print(paper_table.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print(f"\nSaved paper-style table to: {args.markdown_output.resolve()}")
    print(f"Saved published/computed comparison to: {args.comparison_output.resolve()}")
    print(
        "Dataset 2A uses a stratified Session-I development split before FBCSP fitting: "
        f"validation_fraction={args.session1_validation_fraction:.2f}, "
        f"seed={args.session1_split_seed}."
    )
    if args.calibrate_control_limit_from_validation:
        print(
            "Dataset 2A Stage-I L was calibrated from held-out Session-I validation "
            "prediction errors only; Session-II and Table 1 targets were not used."
        )
    print(
        "Published values are retained only as reference targets. "
        "The formatted table always contains computed CSW/CSV values."
    )


if __name__ == "__main__":
    main()
