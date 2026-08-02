"""Run a reproducible CSE lambda-sensitivity experiment.

The script compares fixed EWMA smoothing values while keeping the PCA,
control-limit, and Hotelling validation settings constant. It writes a CSV
summary and three bar charts under ``outputs``.
"""

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.cse import CSEConfig, run_cse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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

    print(results.to_string(index=False))
    print(f"\nSaved metrics to: {METRICS_DIR}")
    print(f"Saved figures to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
