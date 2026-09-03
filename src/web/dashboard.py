"""Flask dashboard for visualising reproduction and experiment results."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd
from flask import Flask, abort, render_template, send_from_directory

from src.web.figure1 import build_dataset_2a_figure1, serialise_figure1


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "outputs" / "metrics" / "bci_table1_comparison.csv"
DEFAULT_OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_DATASET_2A_PATH = PROJECT_ROOT / "data" / "raw" / "bci_competition_iv_2a"
DEFAULT_DATASET_2A_LABELS_PATH = PROJECT_ROOT / "data" / "raw" / "bci_competition_iv_2a_labels"


RESULT_DEFINITIONS = (
    {
        "id": "bci-table1",
        "title": "BCI Table 1 reproduction",
        "domain": "bci",
        "domain_label": "BCI reproduction",
        "datasets": ["BCI Competition IV 2A", "BCI Competition IV 2B"],
        "methods": ["FBCSP", "PCA", "EWMA Stage I", "Hotelling T² Stage II"],
        "purpose": "Published-versus-computed CSE warning and validation counts for the paper-style Table 1 experiment.",
        "command": "python scripts/run_bci_table1_reproduction.py --validation-mode paper_two_sample",
        "primary_artifact": "metrics/bci_table1_comparison.csv",
        "preview_columns": [
            "dataset",
            "subject",
            "lambda",
            "published_csw",
            "computed_csw",
            "csw_difference",
            "published_csv",
            "computed_csv",
            "csv_difference",
        ],
        "artifacts": [
            ("metrics/bci_table1_reproduction.md", "Paper-style Markdown table"),
            ("metrics/bci_table1_comparison.csv", "Published/computed comparison"),
        ],
    },
    {
        "id": "bci-2b-diagnostics",
        "title": "Dataset 2B diagnostics",
        "domain": "bci",
        "domain_label": "BCI diagnostics",
        "datasets": ["BCI Competition IV 2B"],
        "methods": ["FBCSP", "PCA", "EWMA Stage I", "Stage-II validation"],
        "purpose": "Subject-level control-limit diagnostics plus the exact locations of Stage-I warnings in the 2B stream.",
        "command": "python scripts/run_bci_2b_experiment.py",
        "primary_artifact": "metrics/bci_2b_table1_results.csv",
        "preview_columns": [
            "subject",
            "lambda",
            "published_csw",
            "computed_csw",
            "published_csv",
            "computed_csv",
            "control_limit_multiplier",
            "mean_half_width",
        ],
        "artifacts": [
            ("metrics/bci_2b_table1_results.csv", "Subject diagnostic results"),
            ("metrics/bci_2b_stage1_warnings.csv", "Stage-I warning events"),
        ],
    },
    {
        "id": "bci-2b-l-sensitivity",
        "title": "Dataset 2B control-limit sensitivity",
        "domain": "bci",
        "domain_label": "BCI sensitivity",
        "datasets": ["BCI Competition IV 2B"],
        "methods": ["EWMA", "L sweep", "Variance update modes"],
        "purpose": "Tests how control-limit multiplier L and EWMA variance-update assumptions affect CSW/CSV reproduction error.",
        "command": "python scripts/run_bci_2b_l_sensitivity.py",
        "primary_artifact": "metrics/bci_2b_l_sensitivity.csv",
        "preview_columns": [
            "subject",
            "variance_update_mode",
            "control_limit_multiplier",
            "computed_csw",
            "absolute_csw_error",
            "computed_csv",
            "absolute_csv_error",
        ],
        "artifacts": [
            ("metrics/bci_2b_l_sensitivity.csv", "L and variance-mode sensitivity table"),
        ],
    },
    {
        "id": "lambda-sweep",
        "title": "Synthetic CSE lambda sweep",
        "domain": "synthetic",
        "domain_label": "Synthetic sensitivity",
        "datasets": ["Deterministic 3D Gaussian mean shift"],
        "methods": ["PCA", "EWMA Stage I", "Hotelling Stage II", "Lambda sweep"],
        "purpose": "Measures how the EWMA smoothing parameter changes warning count, validation count, recognition delay and computation time.",
        "command": "python scripts/run_cse_lambda_sweep.py",
        "primary_artifact": "metrics/cse_lambda_sensitivity.csv",
        "preview_columns": [
            "lambda",
            "warning_count",
            "confirmed_count",
            "rejected_count",
            "first_warning_time",
            "first_confirmation_time",
            "cse_rci",
            "computation_time_seconds",
        ],
        "artifacts": [
            ("metrics/cse_lambda_sensitivity.csv", "Lambda sensitivity metrics"),
            ("figures/cse_lambda_stage1_warnings.png", "Stage-I warning-count chart"),
            ("figures/cse_lambda_confirmed_shifts.png", "Confirmed-shift chart"),
            ("figures/cse_lambda_rci.png", "Recognition-delay chart"),
        ],
    },
    {
        "id": "paper2015-d2",
        "title": "Raza 2015 D2 / Table III reproduction",
        "domain": "synthetic",
        "domain_label": "Paper 2015 reproduction",
        "datasets": ["D2 AR jumping-mean stream"],
        "methods": ["SD-EWMA", "TSSD-EWMA", "ICI-CDT", "K-S Stage II"],
        "purpose": "Auditable Table III reproduction with detector traces, event-level scoring and published-versus-computed FP, FN, RCI and CT.",
        "command": "python scripts/run_2015_synthetic_reproduction.py --dataset d2",
        "primary_artifact": "metrics/paper2015/d2/table3_comparison.csv",
        "preview_columns": [
            "method",
            "computed_fp_percent",
            "published_fp_percent",
            "computed_fn_percent",
            "published_fn_percent",
            "computed_rci",
            "published_rci",
            "computed_ct_seconds",
            "published_ct_seconds",
        ],
        "artifacts": [
            ("metrics/paper2015/d2/summary.json", "Run summary"),
            ("metrics/paper2015/d2/stage1_trace.csv", "Full Stage-I detector trace"),
            ("metrics/paper2015/d2/stage2_validations.csv", "Stage-II validation trace"),
            ("metrics/paper2015/d2/stage1_events.csv", "Stage-I event scoring"),
            ("metrics/paper2015/d2/stage2_events.csv", "Stage-II event scoring"),
            ("metrics/paper2015/d2/table3_computed.csv", "Computed Table III"),
            ("metrics/paper2015/d2/table3_comparison.csv", "Published/computed Table III comparison"),
            ("metrics/paper2015/d2/table3_computed.md", "Human-readable Table III"),
            ("metrics/paper2015/d2/ici_cdt_trace.csv", "ICI-CDT trace"),
            ("metrics/paper2015/d2/table3_sd_ewma_events.csv", "SD-EWMA Table III events"),
            ("metrics/paper2015/d2/table3_tssd_ewma_events.csv", "TSSD-EWMA Table III events"),
            ("metrics/paper2015/d2/table3_ici_cdt_events.csv", "ICI-CDT Table III events"),
        ],
    },
)


app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / "templates"),
    static_folder=str(PROJECT_ROOT / "static"),
)
app.config.setdefault("TABLE1_RESULTS_PATH", str(DEFAULT_RESULTS_PATH))
app.config.setdefault("RESULTS_ROOT", str(DEFAULT_OUTPUTS_ROOT))
app.config.setdefault("DATASET_2A_PATH", str(DEFAULT_DATASET_2A_PATH))
app.config.setdefault("DATASET_2A_LABELS_PATH", str(DEFAULT_DATASET_2A_LABELS_PATH))


def _load_results(path: str | Path) -> pd.DataFrame:
    results_path = Path(path)
    if not results_path.is_file():
        return pd.DataFrame()

    frame = pd.read_csv(results_path)
    required = {
        "dataset",
        "subject",
        "lambda",
        "published_csw",
        "computed_csw",
        "csw_difference",
        "published_csv",
        "computed_csv",
        "csv_difference",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "Table 1 comparison file is missing required columns: "
            + ", ".join(sorted(missing))
        )

    numeric = [
        "lambda",
        "published_csw",
        "computed_csw",
        "csw_difference",
        "published_csv",
        "computed_csv",
        "csv_difference",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame


def _dataset_summary(frame: pd.DataFrame, dataset: str) -> dict[str, float | int | str]:
    group = frame.loc[frame["dataset"] == dataset].copy()
    if group.empty:
        return {
            "dataset": dataset,
            "subjects": 0,
            "published_csw_mean": 0.0,
            "computed_csw_mean": 0.0,
            "published_csv_mean": 0.0,
            "computed_csv_mean": 0.0,
            "csw_mae": 0.0,
            "csv_mae": 0.0,
        }

    return {
        "dataset": dataset,
        "subjects": int(group.shape[0]),
        "published_csw_mean": float(group["published_csw"].mean()),
        "computed_csw_mean": float(group["computed_csw"].mean()),
        "published_csv_mean": float(group["published_csv"].mean()),
        "computed_csv_mean": float(group["computed_csv"].mean()),
        "csw_mae": float(group["csw_difference"].abs().mean()),
        "csv_mae": float(group["csv_difference"].abs().mean()),
    }


def _chart_rows(frame: pd.DataFrame, dataset: str) -> list[dict[str, object]]:
    group = frame.loc[frame["dataset"] == dataset].sort_values("subject")
    if group.empty:
        return []

    max_value = max(
        float(group["published_csw"].max()),
        float(group["computed_csw"].max()),
        float(group["published_csv"].max()),
        float(group["computed_csv"].max()),
        1.0,
    )

    rows: list[dict[str, object]] = []
    for _, row in group.iterrows():
        rows.append({
            "subject": str(row["subject"]),
            "lambda": float(row["lambda"]),
            "published_csw": int(row["published_csw"]),
            "computed_csw": int(row["computed_csw"]),
            "published_csv": int(row["published_csv"]),
            "computed_csv": int(row["computed_csv"]),
            "csw_difference": int(row["csw_difference"]),
            "csv_difference": int(row["csv_difference"]),
            "published_csw_width": 100.0 * float(row["published_csw"]) / max_value,
            "computed_csw_width": 100.0 * float(row["computed_csw"]) / max_value,
            "published_csv_width": 100.0 * float(row["published_csv"]) / max_value,
            "computed_csv_width": 100.0 * float(row["computed_csv"]) / max_value,
        })
    return rows


def _normalise_preview_value(value: object) -> object:
    if pd.isna(value):
        return "n/a"
    if isinstance(value, float):
        return round(value, 6)
    if hasattr(value, "item"):
        return value.item()
    return value


def _csv_preview(path: Path, preferred_columns: list[str]) -> dict[str, object] | None:
    if not path.is_file() or path.suffix.lower() != ".csv":
        return None

    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        return {
            "error": str(error),
            "rows": 0,
            "columns": [],
            "missing_columns": preferred_columns,
            "records": [],
        }

    columns = [column for column in preferred_columns if column in frame.columns]
    if not columns:
        columns = list(frame.columns[:8])

    preview = frame.loc[:, columns].head(8)
    records = [
        {column: _normalise_preview_value(value) for column, value in row.items()}
        for row in preview.to_dict(orient="records")
    ]
    return {
        "error": None,
        "rows": int(frame.shape[0]),
        "column_count": int(frame.shape[1]),
        "missing_columns": [
            column for column in preferred_columns if column not in frame.columns
        ],
        "columns": columns,
        "records": records,
    }


def _build_results_catalog(outputs_root: str | Path) -> list[dict[str, object]]:
    root = Path(outputs_root)
    catalog: list[dict[str, object]] = []

    for definition in RESULT_DEFINITIONS:
        item = deepcopy(definition)
        artifacts = []
        for relative_path, label in definition["artifacts"]:
            absolute_path = root / relative_path
            artifacts.append({
                "relative_path": relative_path,
                "label": label,
                "available": absolute_path.is_file(),
                "is_figure": (
                    absolute_path.is_file()
                    and absolute_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
                ),
                "size_kb": (
                    round(absolute_path.stat().st_size / 1024.0, 1)
                    if absolute_path.is_file()
                    else None
                ),
            })

        primary_path = root / str(definition["primary_artifact"])
        item["artifacts"] = artifacts
        item["available_count"] = sum(artifact["available"] for artifact in artifacts)
        item["artifact_count"] = len(artifacts)
        item["has_figures"] = any(artifact["is_figure"] for artifact in artifacts)
        item["status"] = "available" if primary_path.is_file() else "not-generated"
        item["preview"] = _csv_preview(
            primary_path,
            list(definition["preview_columns"]),
        )
        catalog.append(item)

    return catalog


@app.get("/outputs/<path:filename>")
def outputs_file(filename: str):
    outputs_root = Path(app.config["RESULTS_ROOT"]).resolve()
    requested = (outputs_root / filename).resolve()
    if requested != outputs_root and outputs_root not in requested.parents:
        abort(404)
    if not requested.is_file():
        abort(404)
    return send_from_directory(outputs_root, filename)


@app.get("/")
def dashboard():
    results_path = Path(app.config["TABLE1_RESULTS_PATH"])
    frame = _load_results(results_path)

    if frame.empty:
        return render_template(
            "dashboard.html",
            data_available=False,
            results_path=results_path,
            summaries=[],
            rows_2a=[],
            rows_2b=[],
        )

    summaries = [_dataset_summary(frame, "2A"), _dataset_summary(frame, "2B")]
    return render_template(
        "dashboard.html",
        data_available=True,
        results_path=results_path,
        summaries=summaries,
        rows_2a=_chart_rows(frame, "2A"),
        rows_2b=_chart_rows(frame, "2B"),
    )


@app.get("/results")
def results_catalog():
    outputs_root = Path(app.config["RESULTS_ROOT"])
    result_groups = _build_results_catalog(outputs_root)
    available_experiments = sum(
        group["status"] == "available" for group in result_groups
    )
    available_artifacts = sum(
        int(group["available_count"]) for group in result_groups
    )
    total_artifacts = sum(int(group["artifact_count"]) for group in result_groups)

    return render_template(
        "results.html",
        results_root=outputs_root,
        result_groups=result_groups,
        available_experiments=available_experiments,
        experiment_count=len(result_groups),
        available_artifacts=available_artifacts,
        total_artifacts=total_artifacts,
    )


@app.get("/figure-1")
def figure_1():
    data_directory = Path(app.config["DATASET_2A_PATH"])
    labels_directory = Path(app.config["DATASET_2A_LABELS_PATH"])
    try:
        figure = build_dataset_2a_figure1(
            data_directory,
            subject=7,
            labels_directory=labels_directory,
        )
    except (FileNotFoundError, ImportError, ValueError, RuntimeError) as error:
        return render_template(
            "figure1.html",
            data_available=False,
            error_message=str(error),
            data_directory=data_directory,
            figure=None,
        )

    return render_template(
        "figure1.html",
        data_available=True,
        error_message=None,
        data_directory=data_directory,
        figure=serialise_figure1(figure),
    )
