"""Flask dashboard for visualising reproduction and experiment results."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd
from flask import Flask, abort, render_template, send_from_directory

from src.bci.datasets.chowdhury.data import load_patient_demographics
from src.web.chowdhury import build_chowdhury_cohort_view
from src.web.figure1 import build_dataset_2a_figure1, serialise_figure1
from src.web.home import build_home_page_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "outputs" / "metrics" / "bci_table1_comparison.csv"
DEFAULT_OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
PAPERS_ROOT = PROJECT_ROOT / "papers"
DEFAULT_DATASET_2A_PATH = PROJECT_ROOT / "data" / "raw" / "bci_competition_iv_2a"
DEFAULT_DATASET_2A_LABELS_PATH = PROJECT_ROOT / "data" / "raw" / "bci_competition_iv_2a_labels"
DEFAULT_CHOWDHURY_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "chowdhury_cse_uael"


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
        "title": "Synthetic CSE lambda analysis",
        "domain": "synthetic",
        "domain_label": "Synthetic sensitivity",
        "datasets": ["Deterministic 3D Gaussian mean shift"],
        "methods": [
            "PCA",
            "Prediction-error SSE",
            "EWMA Stage I",
            "Hotelling Stage II",
            "Lambda sweep",
        ],
        "purpose": (
            "Shows the paper-grounded SSE criterion used to select lambda, then "
            "measures how fixed lambda values change warning count, validation "
            "count, recognition delay and computation time."
        ),
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
            ("metrics/cse_lambda_sse.csv", "Lambda–SSE optimisation table"),
            ("figures/cse_lambda_sse_curve.png", "Prediction-error SSE by lambda"),
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
        "detail_endpoint": "ks_validation_results",
        "detail_label": "View K–S Stage-II validation",
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
app.config.setdefault("CHOWDHURY_DATA_PATH", str(DEFAULT_CHOWDHURY_DATA_PATH))


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


def _lambda_analysis_visualisation(outputs_root: str | Path) -> dict[str, object] | None:
    """Build the focused lambda-estimation and sensitivity presentation model."""

    root = Path(outputs_root)
    sse_path = root / "metrics" / "cse_lambda_sse.csv"
    if not sse_path.is_file():
        return None

    try:
        frame = pd.read_csv(sse_path)
    except (OSError, ValueError, pd.errors.ParserError):
        return None

    required = {"lambda", "sse"}
    if required.difference(frame.columns) or frame.empty:
        return None

    lambda_values = pd.to_numeric(frame["lambda"], errors="coerce")
    sse_values = pd.to_numeric(frame["sse"], errors="coerce")
    valid = lambda_values.notna() & sse_values.notna()
    if not valid.any():
        return None

    clean = pd.DataFrame({
        "lambda": lambda_values.loc[valid],
        "sse": sse_values.loc[valid],
    })
    best_row = clean.loc[clean["sse"].idxmin()]

    figures = (
        {
            "relative_path": "figures/cse_lambda_stage1_warnings.png",
            "label": "Stage-I warning count",
            "detail": "How fixed lambda changes the number of online EWMA warnings.",
        },
        {
            "relative_path": "figures/cse_lambda_confirmed_shifts.png",
            "label": "Confirmed shifts",
            "detail": "Whether Stage-II validation changes as lambda varies.",
        },
        {
            "relative_path": "figures/cse_lambda_rci.png",
            "label": "Recognition delay",
            "detail": "Detection delay measured by the CSE recognition capability index.",
        },
    )

    return {
        "best_lambda": float(best_row["lambda"]),
        "minimum_sse": float(best_row["sse"]),
        "candidate_count": int(clean.shape[0]),
        "sse_figure": {
            "relative_path": "figures/cse_lambda_sse_curve.png",
            "label": "Prediction-error SSE by lambda",
            "available": (root / "figures" / "cse_lambda_sse_curve.png").is_file(),
        },
        "sensitivity_figures": [
            {
                **figure,
                "available": (root / figure["relative_path"]).is_file(),
            }
            for figure in figures
        ],
    }


def _table3_visualisation(outputs_root: str | Path) -> dict[str, object] | None:
    """Build confusion-matrix and RCI/CT views for the D2 Table III result."""

    path = Path(outputs_root) / "metrics" / "paper2015" / "d2" / "table3_comparison.csv"
    if not path.is_file():
        return None

    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError):
        return None

    required = {
        "method",
        "computed_fp_percent",
        "computed_fn_percent",
        "computed_rci",
        "published_rci",
        "computed_ct_seconds",
        "published_ct_seconds",
    }
    if required.difference(frame.columns):
        return None

    true_shift_count = 9
    scope_observation_count = 1000
    non_shift_count = scope_observation_count - true_shift_count

    methods: list[dict[str, object]] = []
    rci_values: list[float] = []
    ct_values: list[float] = []

    for _, source in frame.iterrows():
        fp_percent = float(source["computed_fp_percent"])
        fn_percent = float(source["computed_fn_percent"])
        false_negative = int(round((fn_percent / 100.0) * true_shift_count))
        true_positive = max(true_shift_count - false_negative, 0)
        false_positive = int(round((fp_percent / 100.0) * non_shift_count))
        true_negative = max(non_shift_count - false_positive, 0)

        computed_rci = float(source["computed_rci"])
        published_rci = float(source["published_rci"])
        computed_ct = float(source["computed_ct_seconds"])
        published_ct = float(source["published_ct_seconds"])
        rci_values.extend([computed_rci, published_rci])
        ct_values.extend([computed_ct, published_ct])

        methods.append(
            {
                "method": str(source["method"]),
                "tp": true_positive,
                "fn": false_negative,
                "fp": false_positive,
                "tn": true_negative,
                "computed_fp_percent": round(fp_percent, 4),
                "computed_fn_percent": round(fn_percent, 4),
                "computed_rci": round(computed_rci, 4),
                "published_rci": round(published_rci, 4),
                "computed_ct_seconds": round(computed_ct, 4),
                "published_ct_seconds": round(published_ct, 4),
            }
        )

    rci_max = max(rci_values + [1.0])
    ct_max = max(ct_values + [0.001])
    for method in methods:
        method["computed_rci_width"] = 100.0 * float(method["computed_rci"]) / rci_max
        method["published_rci_width"] = 100.0 * float(method["published_rci"]) / rci_max
        method["computed_ct_width"] = 100.0 * float(method["computed_ct_seconds"]) / ct_max
        method["published_ct_width"] = 100.0 * float(method["published_ct_seconds"]) / ct_max

    return {
        "methods": methods,
        "true_shift_count": true_shift_count,
        "non_shift_count": non_shift_count,
        "scope_observation_count": scope_observation_count,
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
        item["table3_visualisation"] = (
            _table3_visualisation(root)
            if definition["id"] == "paper2015-d2"
            else None
        )
        item["lambda_analysis"] = (
            _lambda_analysis_visualisation(root)
            if definition["id"] == "lambda-sweep"
            else None
        )
        catalog.append(item)

    return catalog


KS_VALIDATION_RELATIVE_PATH = Path("metrics/paper2015/d2/stage2_validations.csv")


def _load_ks_validation_results(outputs_root: str | Path) -> dict[str, object]:
    """Build a paper-facing Stage-II K-S validation view from generated output."""

    results_path = Path(outputs_root) / KS_VALIDATION_RELATIVE_PATH
    empty_summary = {
        "total": 0,
        "evaluated": 0,
        "confirmed": 0,
        "rejected": 0,
        "pending": 0,
        "skipped": 0,
        "insufficient": 0,
        "other": 0,
        "loss_total": 0,
    }
    empty_chart = {
        "rejected_end": 0.0,
        "pending_end": 0.0,
        "skipped_end": 0.0,
        "insufficient_end": 0.0,
        "other_end": 0.0,
        "segments": [],
    }
    if not results_path.is_file():
        return {
            "available": False,
            "results_path": results_path,
            "rows": [],
            "evaluated_rows": [],
            "loss_rows": [],
            "summary": empty_summary,
            "loss_chart": empty_chart,
            "alpha": 0.05,
            "k_alpha": 1.36,
        }

    frame = pd.read_csv(results_path)
    required = {
        "alarm_time",
        "before_size",
        "after_size",
        "ks_statistic",
        "p_value",
        "status",
        "confirmed_shift",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "Stage-II validation file is missing required columns: "
            + ", ".join(sorted(missing))
        )

    alpha = 0.05
    k_alpha = 1.36
    rows: list[dict[str, object]] = []
    for _, source in frame.iterrows():
        n1 = float(source["before_size"])
        n2 = float(source["after_size"])
        ks_statistic = (
            float(source["ks_statistic"])
            if pd.notna(source["ks_statistic"])
            else float("nan")
        )
        scaled = (
            float(source["scaled_statistic"])
            if "scaled_statistic" in frame.columns
            and pd.notna(source["scaled_statistic"])
            else (
                float((n1 * n2 / (n1 + n2)) ** 0.5 * ks_statistic)
                if n1 > 0 and n2 > 0 and pd.notna(ks_statistic)
                else float("nan")
            )
        )
        row_k_alpha = (
            float(source["k_alpha"])
            if "k_alpha" in frame.columns and pd.notna(source["k_alpha"])
            else k_alpha
        )
        margin = (
            scaled - row_k_alpha
            if pd.notna(scaled) and pd.notna(row_k_alpha)
            else float("nan")
        )
        status = str(source["status"])
        rows.append(
            {
                "alarm_time": _normalise_preview_value(source["alarm_time"]),
                "validation_time": _normalise_preview_value(source.get("validation_time")),
                "before_size": int(n1),
                "after_size": int(n2),
                "ks_statistic": _normalise_preview_value(ks_statistic),
                "scaled_statistic": _normalise_preview_value(scaled),
                "k_alpha": _normalise_preview_value(row_k_alpha),
                "margin": _normalise_preview_value(margin),
                "p_value": _normalise_preview_value(source["p_value"]),
                "status": status,
                "confirmed_shift": bool(source["confirmed_shift"]),
            }
        )

    statuses = frame["status"].fillna("").astype(str)
    confirmed = int((statuses == "confirmed").sum())
    rejected = int((statuses == "rejected").sum())
    pending = int(statuses.str.startswith("pending").sum())
    skipped = int(statuses.str.startswith("skipped").sum())
    insufficient = int(statuses.str.startswith("insufficient").sum())
    recognised = (
        (statuses == "confirmed")
        | (statuses == "rejected")
        | statuses.str.startswith("pending")
        | statuses.str.startswith("skipped")
        | statuses.str.startswith("insufficient")
    )
    other = int((~recognised).sum())
    evaluated = confirmed + rejected
    loss_total = rejected + pending + skipped + insufficient + other

    summary = {
        "total": int(frame.shape[0]),
        "evaluated": evaluated,
        "confirmed": confirmed,
        "rejected": rejected,
        "pending": pending,
        "skipped": skipped,
        "insufficient": insufficient,
        "other": other,
        "loss_total": loss_total,
    }

    loss_counts = [
        ("Rejected by K-S", rejected, "rejected"),
        ("Pending future window", pending, "pending"),
        ("Skipped nearby alarm", skipped, "skipped"),
        ("Insufficient window", insufficient, "insufficient"),
        ("Other", other, "other"),
    ]
    cumulative = 0.0
    segments = []
    chart_bounds: dict[str, float] = {}
    for label, count, key in loss_counts:
        percent = 100.0 * count / loss_total if loss_total else 0.0
        cumulative += percent
        chart_bounds[f"{key}_end"] = round(cumulative, 4)
        segments.append(
            {
                "label": label,
                "key": key,
                "count": count,
                "percent": round(percent, 1),
            }
        )

    loss_chart = {
        **chart_bounds,
        "segments": segments,
    }

    evaluated_rows = [
        row for row in rows if row["status"] in {"confirmed", "rejected"}
    ]
    loss_rows = [
        row for row in rows if row["status"] != "confirmed"
    ]

    return {
        "available": True,
        "results_path": results_path,
        "rows": rows,
        "evaluated_rows": evaluated_rows,
        "loss_rows": loss_rows,
        "summary": summary,
        "loss_chart": loss_chart,
        "alpha": alpha,
        "k_alpha": k_alpha,
    }

@app.get("/outputs/<path:filename>")
def outputs_file(filename: str):
    outputs_root = Path(app.config["RESULTS_ROOT"]).resolve()
    requested = (outputs_root / filename).resolve()
    if requested != outputs_root and outputs_root not in requested.parents:
        abort(404)
    if not requested.is_file():
        abort(404)
    return send_from_directory(outputs_root, filename)


@app.get("/papers/<path:filename>")
def paper_file(filename: str):
    papers_root = PAPERS_ROOT.resolve()
    requested = (papers_root / filename).resolve()
    if (
        requested.suffix.lower() != ".pdf"
        or requested.parent != papers_root
        or not requested.is_file()
    ):
        abort(404)
    return send_from_directory(
        papers_root,
        filename,
        mimetype="application/pdf",
        as_attachment=False,
    )


@app.get("/")
def home():
    return render_template("home.html", **build_home_page_model())


@app.get("/table-1")
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


@app.get("/results/paper2015-d2/ks-validation")
def ks_validation_results():
    """Show per-warning Stage-II K-S evidence for the Raza 2015 D2 reproduction."""

    outputs_root = Path(app.config["RESULTS_ROOT"])
    model = _load_ks_validation_results(outputs_root)
    return render_template(
        "ks_validation.html",
        **model,
    )


@app.get("/chowdhury-demographics")
def chowdhury_demographics():
    data_directory = Path(app.config["CHOWDHURY_DATA_PATH"])
    try:
        participants = load_patient_demographics(data_directory)
        cohort = build_chowdhury_cohort_view(participants)
    except (FileNotFoundError, OSError, ValueError) as error:
        return render_template(
            "chowdhury_demographics.html",
            data_available=False,
            data_directory=data_directory,
            error_message=str(error),
            cohort=None,
        )

    return render_template(
        "chowdhury_demographics.html",
        data_available=True,
        data_directory=data_directory,
        error_message=None,
        cohort=cohort,
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
