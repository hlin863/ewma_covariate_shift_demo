"""Read-only evidence model for the data-processing and protocol page."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REFERENCE_PATH = Path(__file__).resolve().parents[2] / "data" / "reference" / "bci_processing_2026-09-27.json"


def _accuracy_rows(reference: dict) -> list[dict[str, object]]:
    rows = []
    for dataset, values in reference["bagging_accuracy"].items():
        for index, (single, bagged) in enumerate(
            zip(values["single"], values["bagged"], strict=True), start=1
        ):
            rows.append({
                "dataset": dataset,
                "subject": f"{'A' if dataset == '2A' else 'B'}{index:02d}",
                "single": float(single),
                "bagged": float(bagged),
                "scope": "Session II" if dataset == "2A" else "Session III (GDF-only holdout)",
            })
    return rows


def _live_accuracy(path: Path) -> list[dict[str, object]]:
    frame = pd.read_csv(path)
    required = {"dataset", "subject", "method", "accuracy", "evaluation_scope"}
    if not required.issubset(frame.columns):
        raise ValueError("The bagging CSV does not contain the expected result fields.")
    frame = frame.loc[
        frame["dataset"].isin(["2A", "2B"])
        & frame["method"].isin(["single_linear_svm", "bagged_linear_svm"])
    ].copy()
    frame["accuracy"] = pd.to_numeric(frame["accuracy"], errors="raise")
    if frame.empty or frame["accuracy"].isna().any() or not frame["accuracy"].between(0, 1).all():
        raise ValueError("The bagging CSV has no valid accuracy observations.")
    if frame.duplicated(["dataset", "subject", "method"]).any():
        raise ValueError("The bagging CSV has duplicate dataset/subject/method rows.")
    scopes = frame.groupby(["dataset", "subject"])["evaluation_scope"].nunique()
    if (scopes != 1).any():
        raise ValueError("Paired classifiers must share an evaluation scope.")
    scope_lookup = frame.drop_duplicates(["dataset", "subject"]).set_index(
        ["dataset", "subject"]
    )["evaluation_scope"]
    paired = frame.pivot(index=["dataset", "subject"], columns="method", values="accuracy")
    if paired.isna().any().any() or set(paired.columns) != {"single_linear_svm", "bagged_linear_svm"}:
        raise ValueError("Every subject needs both single and bagged classifier results.")
    return [
        {
            "dataset": str(dataset), "subject": str(subject),
            "single": float(values["single_linear_svm"]),
            "bagged": float(values["bagged_linear_svm"]),
            "scope": str(scope_lookup.loc[(dataset, subject)]),
        }
        for (dataset, subject), values in paired.sort_index().iterrows()
    ]


def build_data_processing_view(outputs_root: Path) -> dict[str, object]:
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    rows = _accuracy_rows(reference)
    accuracy_source = reference["provenance"]
    accuracy_error = None
    bagging_path = outputs_root / "metrics" / "bagging_bci_real.csv"
    if bagging_path.is_file():
        try:
            rows = _live_accuracy(bagging_path)
            accuracy_source = "Generated locally: outputs/metrics/bagging_bci_real.csv"
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            accuracy_error = f"Local bagging CSV could not be used ({exc}); showing the supplied run transcript."

    diagnostic = reference["diagnostic_b01"].copy()
    diagnostic_source = reference["provenance"]
    diagnostic_path = outputs_root / "metrics" / "bci_2b_table1_results.csv"
    if diagnostic_path.is_file():
        try:
            frame = pd.read_csv(diagnostic_path)
            row = frame.loc[frame["subject"] == "B01"]
            if not row.empty:
                values = row.iloc[0]
                diagnostic = {
                    key: int(values[key]) for key in
                    ("published_csw", "computed_csw", "published_csv", "computed_csv")
                }
                diagnostic["mode"] = "algorithm1_training_reference"
                diagnostic_source = "Generated locally: outputs/metrics/bci_2b_table1_results.csv"
        except (OSError, ValueError, KeyError, pd.errors.ParserError):
            pass

    means = {}
    accuracy_scopes = {}
    for dataset in ("2A", "2B"):
        subset = [row for row in rows if row["dataset"] == dataset]
        if subset:
            means[dataset] = {
                "single": sum(float(row["single"]) for row in subset) / len(subset),
                "bagged": sum(float(row["bagged"]) for row in subset) / len(subset),
                "subjects": len(subset),
            }
            accuracy_scopes[dataset] = sorted({str(row["scope"]) for row in subset})
    return {
        "split": reference["split_b01"],
        "split_source": reference["provenance"],
        "diagnostic": diagnostic,
        "diagnostic_source": diagnostic_source,
        "accuracy_rows": rows,
        "accuracy_source": accuracy_source,
        "accuracy_error": accuracy_error,
        "accuracy_means": means,
        "accuracy_scopes": accuracy_scopes,
    }
