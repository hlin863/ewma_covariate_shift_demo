"""Reproduce the 2019 paper's CSE Table 1 layout for Datasets 2A and 2B."""

from argparse import ArgumentParser
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bci_2a_experiment import (
    build_dataset_2a_fbcsp_features,
    extract_dataset_2a_trials,
    run_dataset_2a_subject,
)
from src.bci_2b_experiment import (
    PUBLISHED_2B_RESULTS,
    build_dataset_2b_fbcsp_features,
    concatenate_trial_signals,
    extract_dataset_2b_trials,
)
from src.bci_data import (
    load_bci_competition_iv_2a_session,
    load_bci_competition_iv_2b_session,
)
from src.cse import CSEConfig, run_cse
from src.table1_reproduction import (
    Table1Row,
    comparison_dataframe,
    paper_style_dataframe,
    paper_style_markdown,
)


def _parse_args():
    parser = ArgumentParser(
        description=(
            "Run the CSE experiment for BCI Competition IV Datasets 2A and 2B "
            "and save computed results in the side-by-side Table 1 structure."
        )
    )
    parser.add_argument(
        "--data-2a",
        type=Path,
        default=Path("data/raw/bci_competition_iv_2a"),
    )
    parser.add_argument(
        "--data-2b",
        type=Path,
        default=Path("data/raw/bci_competition_iv_2b"),
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
    training_session = load_bci_competition_iv_2a_session(args.data_2a, subject, "T")
    testing_session = load_bci_competition_iv_2a_session(args.data_2a, subject, "E")
    training_trials = extract_dataset_2a_trials(training_session)
    testing_trials = extract_dataset_2a_trials(testing_session)
    pipeline = build_dataset_2a_fbcsp_features(
        training_trials,
        testing_trials,
        components_per_side=args.components_per_side,
    )
    result = run_dataset_2a_subject(
        subject,
        pipeline.training,
        pipeline.testing,
        validation_mode=args.validation_mode,
        validation_window_size=args.validation_window_size,
        validation_alpha=args.alpha,
        control_limit_multiplier=args.control_limit_multiplier,
        variance_smoothing=args.variance_smoothing,
        variance_update_mode=args.variance_update_mode,
    )
    print(
        f"{result.subject}: train={result.training_trials}, test={result.testing_trials}, "
        f"CSW={result.computed_csw}/{result.published_csw}, "
        f"CSV={result.computed_csv}/{result.published_csv}"
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
        session = load_bci_competition_iv_2b_session(args.data_2b, subject, session_number)
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
    covariance_method = "empirical" if args.validation_mode == "paper_two_sample" else "shrinkage"
    cse_result = run_cse(
        training_features=pipeline.training.features,
        testing_features=pipeline.testing.features,
        testing_times=pipeline.testing.times,
        config=CSEConfig(
            pca_components=min(3, pipeline.training.features.shape[1]),
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
    computed_csv = int(cse_result.validation_results["confirmed_shift"].astype(bool).sum())
    print(
        f"{subject_id}: train={pipeline.training.features.shape[0]}, "
        f"test={pipeline.testing.features.shape[0]}, "
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
        "Published values are retained only as reference targets. "
        "The formatted table always contains computed CSW/CSV values."
    )


if __name__ == "__main__":
    main()
