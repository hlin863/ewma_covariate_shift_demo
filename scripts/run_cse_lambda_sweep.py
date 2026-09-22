"""Run reproducible CSE lambda estimation and sensitivity experiments.

The script first reproduces the Stage-I paper criterion for selecting the
EWMA smoothing parameter by minimising one-step-ahead prediction-error SSE on
the PCA-derived training signal. It then compares fixed lambda values while
keeping the PCA, control-limit, and Hotelling validation settings constant.

Outputs include the lambda-SSE search table and curve plus the downstream
lambda-sensitivity metrics and charts under ``outputs``.
"""

from pathlib import Path
from time import perf_counter
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.cse import CSEConfig, run_cse
from src.detection.preprocessing import extract_first_component, fit_cse_pca
from src.detection.stage1.sd_ewma import estimate_lambda


METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

LAMBDA_VALUES = (0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00)
TRUE_SHIFT_TIME = 120


def make_experiment_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create the deterministic multivariate mean-shift test scenario."""

    rng = np.random.default_rng(42)
    covariance = np.array([
        [1.0, 0.35, 0.20],
        [0.35, 1.0, 0.25],
        [0.20, 0.25, 1.0],
    ])

    training_features = rng.multivariate_normal(
        mean=np.zeros(3),
        cov=covariance,
        size=300,
    )
    before_shift = rng.multivariate_normal(
        mean=np.zeros(3),
        cov=covariance,
        size=TRUE_SHIFT_TIME,
    )
    after_shift = rng.multivariate_normal(
        mean=np.full(3, 5.0),
        cov=covariance,
        size=120,
    )

    testing_features = np.vstack((before_shift, after_shift))
    testing_times = np.arange(testing_features.shape[0])

    return training_features, testing_features, testing_times


def run_lambda_sse_analysis() -> tuple[pd.DataFrame, float]:
    """Estimate lambda from Stage-I training data using prediction-error SSE."""

    training, _, _ = make_experiment_data()
    pca_result = fit_cse_pca(
        training_features=training,
        n_components=2,
    )
    training_signal = extract_first_component(
        pca_result.training_transformed
    )

    best_lambda, search_results = estimate_lambda(
        values=training_signal,
    )

    return search_results, best_lambda


def run_lambda_sweep() -> pd.DataFrame:
    """Run CSE once for every fixed lambda and collect summary metrics."""

    training, testing, times = make_experiment_data()
    records: list[dict[str, float | int]] = []

    for lambda_value in LAMBDA_VALUES:
        config = CSEConfig(
            pca_components=2,
            lambda_override=lambda_value,
            variance_smoothing=0.05,
            control_limit_multiplier=3.0,
            validation_before_size=20,
            validation_after_size=20,
            validation_alpha=0.05,
            covariance_method="shrinkage",
            covariance_regularization=1e-6,
            minimum_alarm_gap=20,
        )

        start = perf_counter()
        result = run_cse(
            training_features=training,
            testing_features=testing,
            testing_times=times,
            config=config,
        )
        elapsed = perf_counter() - start

        warnings = result.warning_results.loc[
            result.warning_results["stage_1_alarm"] == 1
        ]
        validation = result.validation_results
        confirmed = validation.loc[validation["confirmed_shift"]]

        first_warning_time = (
            int(warnings["time"].min())
            if not warnings.empty
            else np.nan
        )
        first_confirmation_time = (
            int(confirmed["validation_time"].min())
            if not confirmed.empty
            else np.nan
        )
        cse_rci = (
            first_confirmation_time - TRUE_SHIFT_TIME
            if not np.isnan(first_confirmation_time)
            else np.nan
        )

        records.append({
            "lambda": result.effective_lambda,
            "warning_count": int(warnings.shape[0]),
            "confirmed_count": int(confirmed.shape[0]),
            "rejected_count": int(
                (validation["status"] == "rejected").sum()
            ),
            "pending_count": int(
                (validation["status"] == "pending_after_window").sum()
            ),
            "first_warning_time": first_warning_time,
            "first_confirmation_time": first_confirmation_time,
            "cse_rci": cse_rci,
            "computation_time_seconds": elapsed,
        })

    return pd.DataFrame.from_records(records)


def save_lambda_sse_curve(
    results: pd.DataFrame,
    best_lambda: float,
    filename: str,
) -> None:
    """Save the paper-grounded lambda-versus-SSE optimisation curve."""

    best_row = results.loc[
        results["lambda"] == best_lambda
    ].iloc[0]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        results["lambda"],
        results["sse"],
        marker="o",
        markersize=3,
        linewidth=1.5,
    )
    ax.scatter(
        [best_lambda],
        [best_row["sse"]],
        marker="*",
        s=180,
        zorder=3,
    )
    ax.annotate(
        (
            f"minimum SSE\n"
            f"lambda = {best_lambda:.2f}"
        ),
        xy=(best_lambda, best_row["sse"]),
        xytext=(14, 18),
        textcoords="offset points",
    )
    ax.set_title(
        "EWMA Lambda Selection by Prediction-Error SSE"
    )
    ax.set_xlabel("EWMA smoothing parameter (lambda)")
    ax.set_ylabel("Sum of squared prediction errors")
    ax.grid(axis="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=200)
    plt.close(fig)


def save_bar_chart(
    results: pd.DataFrame,
    column: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    """Save one lambda comparison as an individual bar chart."""

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = results["lambda"].map(lambda value: f"{value:.2f}")
    ax.bar(labels, results[column])
    ax.set_title(title)
    ax.set_xlabel("EWMA smoothing parameter (lambda)")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=200)
    plt.close(fig)


def main() -> None:
    """Run the experiment and save all tabular and visual outputs."""

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    sse_results, best_lambda = run_lambda_sse_analysis()
    sse_results.to_csv(
        METRICS_DIR / "cse_lambda_sse.csv",
        index=False,
    )
    save_lambda_sse_curve(
        sse_results,
        best_lambda=best_lambda,
        filename="cse_lambda_sse_curve.png",
    )

    results = run_lambda_sweep()
    results.to_csv(
        METRICS_DIR / "cse_lambda_sensitivity.csv",
        index=False,
    )

    save_bar_chart(
        results,
        column="warning_count",
        ylabel="Stage-I warning count",
        title="Effect of Lambda on CSE Stage-I Warnings",
        filename="cse_lambda_stage1_warnings.png",
    )
    save_bar_chart(
        results,
        column="confirmed_count",
        ylabel="Confirmed shift count",
        title="Effect of Lambda on Confirmed CSE Shifts",
        filename="cse_lambda_confirmed_shifts.png",
    )
    save_bar_chart(
        results,
        column="cse_rci",
        ylabel="Recognition Capability Index",
        title="Effect of Lambda on CSE Recognition Delay",
        filename="cse_lambda_rci.png",
    )

    minimum_sse = float(
        sse_results.loc[
            sse_results["lambda"] == best_lambda,
            "sse",
        ].iloc[0]
    )
    print(
        f"Minimum-SSE lambda: {best_lambda:.2f} "
        f"(SSE={minimum_sse:.6f})"
    )
    print("\nLambda sensitivity:")
    print(results.to_string(index=False))
    print(f"\nSaved metrics to: {METRICS_DIR}")
    print(f"Saved figures to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
