"""Dataset navigation and read-only inspection of source data structures."""

from pathlib import Path

import numpy as np
import pandas as pd
from flask import Blueprint, abort, current_app, render_template, request, url_for
from scipy.io import loadmat, whosmat

from src.pdm.datasets.turbofan import load_turbofan_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
data_structures_bp = Blueprint("data_structures", __name__)

DATASETS = (
    {"key": "2a", "name": "BCI Competition IV 2A", "kind": "EEG recordings",
     "format": "GDF + MAT evaluation labels", "structure": "Cue-aligned EEG trial tensors",
     "description": "Inspect trial amplitude, channel distributions and μ/β power by subject and session.",
     "endpoint": "bci_eda.data_distributions", "args": {"dataset": "2a"}},
    {"key": "2b", "name": "BCI Competition IV 2B", "kind": "EEG recordings",
     "format": "GDF", "structure": "Five sessions per subject, converted to EEG trials",
     "description": "Browse the training and evaluation sessions for each subject.",
     "endpoint": "bci_eda.data_distributions", "args": {"dataset": "2b"}},
    {"key": "turbofan", "name": "NASA Turbofan / C-MAPSS", "kind": "Engine sensor time series",
     "format": "Whitespace-delimited TXT", "structure": "26 columns, grouped by engine and cycle",
     "description": "Explore FD001–FD004 source tables, engine counts, sensor values and remaining useful life.",
     "endpoint": "data_structures.inspect_dataset", "args": {"dataset": "turbofan"}},
    {"key": "chowdhury", "name": "Chowdhury stroke cohort", "kind": "Clinical participant metadata",
     "format": "CSV", "structure": "One row per participant",
     "description": "View participant ages, stroke history and cohort distributions.",
     "endpoint": "chowdhury_demographics", "args": {}},
    {"key": "algae", "name": "NASA Algae Raceway", "kind": "Biological growth measurements",
     "format": "MATLAB MAT", "structure": "Nested structures for three raceways",
     "description": "Inspect measurement fields, sampling counts and time ranges without aligning the signals.",
     "endpoint": "data_structures.inspect_dataset", "args": {"dataset": "algae"}},
    {"key": "synthetic", "name": "Synthetic Gaussian mean shift", "kind": "Controlled detector input",
     "format": "CSV", "structure": "Time, scalar observation and known regime metadata",
     "description": "Inspect the generated input stream and its simulation ground truth.",
     "endpoint": "data_structures.inspect_dataset", "args": {"dataset": "synthetic"}},
)


def _catalogue():
    return [dict(item, url=url_for(item["endpoint"], **item["args"]),
                 active=request.endpoint == item["endpoint"] and all(
                     request.view_args.get(key, request.args.get(key, "2a")) == value
                     for key, value in item["args"].items()))
            for item in DATASETS if item["endpoint"] in current_app.view_functions]


@data_structures_bp.app_context_processor
def dataset_navigation():
    return {"dataset_navigation": _catalogue()}


@data_structures_bp.get("/data-structures")
def overview():
    return render_template("data_structures.html", datasets=_catalogue(), selected=None)


def _preview(frame, title):
    preview = frame.head(6).round(4).astype(object)
    return {"title": title, "columns": frame.columns.tolist(),
            "rows": preview.where(pd.notna(preview), "Missing").values.tolist()}


def _schema(frame):
    return [{"field": str(column), "type": str(frame[column].dtype),
             "missing": int(frame[column].isna().sum())} for column in frame.columns]


def _turbofan_view():
    subset = request.args.get("subset", "FD001").strip().upper()
    if subset not in {"FD001", "FD002", "FD003", "FD004"}:
        abort(400, description="Choose FD001, FD002, FD003 or FD004.")
    directory = current_app.config.get(
        "TURBOFAN_DATA_PATH", PROJECT_ROOT / "data/raw/turbofan_engine_degradation"
    )
    data = load_turbofan_dataset(directory, subset)
    train = data.training_with_rul()
    test = data.testing_with_rul()
    return {
        "subset": subset,
        "metrics": [("Training rows", len(train)), ("Test rows", len(test)),
                    ("Training engines", train.unit.nunique()),
                    ("Test engines", test.unit.nunique())],
        "schema": _schema(data.training),
        "previews": [_preview(train, "Training observations with derived RUL"),
                     _preview(test, "Test observations with offline RUL")],
        "summaries": [{"field": column,
                       "train_min": float(data.training[column].min()),
                       "train_median": float(data.training[column].median()),
                       "train_max": float(data.training[column].max()),
                       "test_median": float(data.testing[column].median())}
                      for column in data.training.columns[2:]],
        "notes": ["Training engines run to failure. Test trajectories stop before failure; their RUL includes NASA's endpoint offset.",
                  "Train and test unit numbers refer to separate fleets. RUL is a target, not a sensor feature.",
                  "These tables show source values before scaling, sensor selection or active-learning queries."],
    }


def _algae_view():
    path = Path(current_app.config.get(
        "ALGAE_DATA_PATH", PROJECT_ROOT / "data/raw/Algae Raceway/algae.mat"
    ))
    variables = [{"field": name, "shape": " × ".join(map(str, shape)), "type": kind}
                 for name, shape, kind in whosmat(path)]
    raw = loadmat(path, variable_names=["algae"], squeeze_me=True,
                  struct_as_record=False)
    if "algae" not in raw:
        raise ValueError("The MAT file does not contain the raw algae structure.")
    raceways = np.atleast_1d(raw["algae"])
    try:
        raceway = int(request.args.get("raceway", "1"))
    except ValueError:
        abort(400, description="Choose a valid raceway number.")
    if not 1 <= raceway <= len(raceways):
        abort(400, description="Choose a valid raceway number.")
    selected = raceways[raceway - 1]
    signals = []
    for field in getattr(selected, "_fieldnames", []):
        value = getattr(selected, field)
        if hasattr(value, "data") and hasattr(value, "time_num"):
            measurements = np.asarray(value.data, dtype=float)
            times = np.asarray(value.time_num, dtype=float).reshape(-1)
            finite_times = times[np.isfinite(times)]
            signals.append({"field": field, "values": measurements.size,
                            "shape": " × ".join(map(str, measurements.shape)) or "scalar",
                            "missing": int((~np.isfinite(measurements)).sum()),
                            "start": float(finite_times.min()) if finite_times.size else None,
                            "end": float(finite_times.max()) if finite_times.size else None})
    return {"raceway": raceway, "raceways": range(1, len(raceways) + 1),
            "metrics": [("Raceways", len(raceways)), ("Top-level variables", len(variables)),
                        ("Measurement fields", len(signals))],
            "variables": variables, "signals": signals,
            "notes": ["Raw algae measurements have separate data and time_num arrays for each signal.",
                      "Sampling frequencies differ. Counts below are numeric values, and time_num is measured in days from the experiment start.",
                      "algae_ts and algae_kg are separate source representations. This view inspects raw algae without resampling or constructing labels."],
            }


def _synthetic_view():
    path = current_app.config.get(
        "SYNTHETIC_DATA_PATH", PROJECT_ROOT / "data/raw/gaussian_mean_shift.csv"
    )
    frame = pd.read_csv(path)
    return {"metrics": [("Observations", len(frame)), ("Columns", len(frame.columns))],
            "schema": _schema(frame), "previews": [_preview(frame, "Source observations")],
            "notes": ["This is a generated scalar input stream. true_regime and true_shift are simulation metadata for offline scoring."]}


@data_structures_bp.get("/data-structures/<dataset>")
def inspect_dataset(dataset):
    readers = {"turbofan": _turbofan_view, "algae": _algae_view, "synthetic": _synthetic_view}
    if dataset not in readers:
        abort(404)
    selected = next(item for item in DATASETS if item["key"] == dataset)
    try:
        view = readers[dataset]()
        error = None
    except (FileNotFoundError, OSError, ValueError) as exc:
        view, error = {}, str(exc)
    return render_template("data_structures.html", datasets=_catalogue(),
                           selected=selected, view=view, error=error)
