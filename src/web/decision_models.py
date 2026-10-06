"""Read-only explorer for saved BCI decision-model evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from flask import Blueprint, current_app, render_template, request


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS = PROJECT_ROOT / "outputs" / "metrics" / "bci_decision_models.csv"
DEFAULT_DETAILS = PROJECT_ROOT / "outputs" / "decision_models"

decision_models_bp = Blueprint("decision_models", __name__)


def _load_metrics(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    required = {
        "dataset", "subject", "evaluation_scope", "method", "role",
        "training_trials", "testing_trials", "feature_count", "accuracy",
        "correct_predictions", "fit_seconds", "predict_seconds",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "Decision-model metrics missing columns: " + ", ".join(sorted(missing))
        )
    return frame


@decision_models_bp.get("/results/decision-models")
def explorer():
    metrics_path = Path(
        current_app.config.get("BCI_DECISION_METRICS_PATH", DEFAULT_METRICS)
    )
    details_root = Path(
        current_app.config.get("BCI_DECISION_DETAILS_PATH", DEFAULT_DETAILS)
    )
    error = None
    try:
        frame = _load_metrics(metrics_path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        frame = pd.DataFrame()
        error = str(exc)

    options: list[tuple[str, str]] = []
    if not frame.empty:
        options = sorted({
            (str(dataset).upper(), str(subject))
            for dataset, subject in frame[["dataset", "subject"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        })

    selected = (
        str(request.args.get("dataset", "")).upper(),
        str(request.args.get("subject", "")),
    )
    if selected not in options:
        selected = options[0] if options else None

    detail = None
    selected_metrics: list[dict[str, object]] = []
    example = None
    examples: list[dict[str, object]] = []

    if selected:
        dataset, subject = selected
        selected_metrics = (
            frame.loc[
                (frame["dataset"].astype(str).str.upper() == dataset)
                & (frame["subject"].astype(str) == subject)
            ]
            .sort_values("method")
            .to_dict("records")
        )
        detail_path = details_root / f"{dataset.lower()}_{subject}.json"
        if detail_path.is_file():
            try:
                detail = json.loads(detail_path.read_text(encoding="utf-8"))
                examples = list(detail.get("examples", []))
                requested_trial = request.args.get("trial")
                wanted = None
                if requested_trial is not None:
                    try:
                        wanted = int(requested_trial)
                    except ValueError:
                        wanted = None
                if wanted is not None:
                    example = next(
                        (
                            item for item in examples
                            if int(item.get("trial_index", -1)) == wanted
                        ),
                        None,
                    )
                if example is None and examples:
                    example = examples[0]
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                error = str(exc)
        else:
            error = f"Detailed evidence file not found: {detail_path}"

    return render_template(
        "decision_models.html",
        metrics_path=metrics_path,
        details_root=details_root,
        data_available=bool(options),
        options=options,
        selected=selected,
        metrics=selected_metrics,
        detail=detail,
        examples=examples,
        example=example,
        error=error,
    )
