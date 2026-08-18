"""Diagnostic CSE execution for BCI Competition IV Dataset 2B."""

from dataclasses import dataclass

import pandas as pd

from src.bci.datasets.dataset2b.experiment import PUBLISHED_2B_RESULTS, TrialFeatureResult
from src.detection import CSEConfig, CSEResult, run_cse


@dataclass(frozen=True)
class Dataset2BDiagnosticResult:
    subject: str
    published_lambda: float
    published_csw: int
    published_csv: int
    computed_csw: int
    computed_csv: int
    training_trials: int
    testing_trials: int
    cse_result: CSEResult

    def summary_row(self) -> dict[str, float | int | str | bool]:
        warnings = self.cse_result.warning_results
        half_widths = warnings["control_limit_half_width"]
        return {
            "subject": self.subject,
            "lambda": self.published_lambda,
            "training_trials": self.training_trials,
            "testing_trials": self.testing_trials,
            "published_csw": self.published_csw,
            "computed_csw": self.computed_csw,
            "csw_difference": self.computed_csw - self.published_csw,
            "published_csv": self.published_csv,
            "computed_csv": self.computed_csv,
            "csv_difference": self.computed_csv - self.published_csv,
            "control_limit_multiplier": float(warnings.iloc[0]["control_limit_multiplier"]),
            "variance_update_mode": str(warnings.iloc[0]["variance_update_mode"]),
            "initial_half_width": float(half_widths.iloc[0]),
            "mean_half_width": float(half_widths.mean()),
            "median_half_width": float(half_widths.median()),
            "final_half_width": float(half_widths.iloc[-1]),
            "maximum_half_width": float(half_widths.max()),
            "matches_published": (
                self.computed_csw == self.published_csw
                and self.computed_csv == self.published_csv
            ),
        }

    def warning_rows(self, testing: TrialFeatureResult) -> pd.DataFrame:
        warnings = self.cse_result.warning_results.copy()
        warnings.insert(0, "subject", self.subject)
        warnings.insert(1, "lambda", self.published_lambda)
        warnings["session_id"] = testing.session_ids
        warnings["cue_description"] = testing.cue_descriptions
        return warnings.loc[warnings["stage_1_alarm"].astype(bool)].reset_index(drop=True)


def run_dataset_2b_diagnostic(
    subject: int,
    training: TrialFeatureResult,
    testing: TrialFeatureResult,
    *,
    validation_alpha: float = 0.05,
    control_limit_multiplier: float = 1.96,
    variance_smoothing: float = 0.05,
    variance_update_mode: str = "always",
) -> Dataset2BDiagnosticResult:
    subject_id = f"B{subject:02d}"
    if subject_id not in PUBLISHED_2B_RESULTS:
        raise ValueError("subject must be an integer from 1 to 9.")
    published_lambda, published_csw, published_csv = PUBLISHED_2B_RESULTS[subject_id]

    result = run_cse(
        training_features=training.features,
        testing_features=testing.features,
        testing_times=testing.times,
        config=CSEConfig(
            pca_components=min(3, training.features.shape[1]),
            lambda_override=published_lambda,
            variance_smoothing=variance_smoothing,
            control_limit_multiplier=control_limit_multiplier,
            variance_update_mode=variance_update_mode,
            ewma_initialization="training_mean",
            validation_mode="algorithm1_training_reference",
            validation_alpha=validation_alpha,
            covariance_method="shrinkage",
        ),
    )
    computed_csw = int(result.warning_results["stage_1_alarm"].sum())
    computed_csv = int(result.validation_results["confirmed_shift"].astype(bool).sum())
    return Dataset2BDiagnosticResult(
        subject=subject_id,
        published_lambda=published_lambda,
        published_csw=published_csw,
        published_csv=published_csv,
        computed_csw=computed_csw,
        computed_csv=computed_csv,
        training_trials=int(training.features.shape[0]),
        testing_trials=int(testing.features.shape[0]),
        cse_result=result,
    )
