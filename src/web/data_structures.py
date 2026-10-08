"""Dataset navigation and read-only inspection of source data structures."""

from pathlib import Path

import numpy as np
import pandas as pd
from flask import Blueprint, abort, current_app, render_template, request, url_for
from scipy.io import loadmat, whosmat

from src.pdm.datasets.turbofan import load_turbofan_dataset
from src.detection.stage1.park_mean_cusum import mean_cusum_curves
from src.detection.multivariate_inspection import (
    demonstration_stream, describe_windows, load_multivariate_csv,
)


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
    {"key": "park", "name": "Park (2023) · Multivariate structure", "kind": "WP2 structural-change inspection",
     "format": "Multivariate CSV or generated demonstration", "structure": "Ordered sensor channels, mean/variance/correlation windows",
     "description": "Explore changes in cross-channel dependence without claiming an implemented wavelet/CUSUM detector.",
     "endpoint": "data_structures.inspect_dataset", "args": {"dataset": "park"}},
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


def _navigation_groups(items):
    """Research-source groups used by the shared dropdown; keep URLs unchanged."""
    specifications = (
        ("BCI Competition IV", "EEG benchmark datasets", ("2a", "2b")),
        ("Research papers", "Paper-linked sources", ("chowdhury", "park")),
        ("Industrial and biological data", "Applied monitoring datasets", ("turbofan", "algae")),
        ("Synthetic benchmarks", "Controlled distributions", ("synthetic",)),
    )
    indexed = {item["key"]: item for item in items}
    return [
        {"title": title, "subtitle": subtitle,
         "items": [indexed[key] for key in keys if key in indexed],
         "active": any(indexed[key]["active"] for key in keys if key in indexed)}
        for title, subtitle, keys in specifications
        if any(key in indexed for key in keys)
    ]


@data_structures_bp.app_context_processor
def dataset_navigation():
    items = _catalogue()
    return {"dataset_navigation": items, "dataset_navigation_groups": _navigation_groups(items)}


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



def _park_bci_features(dataset: str, subject: int, session_index: int) -> tuple[pd.DataFrame, str]:
    """Build positive trial-level inputs using existing BCI EDA loaders.

    Never combine separate sessions into one sequence. No labels are used.
    """
    from src.web.bci_eda import _load_dataset_2a, _load_dataset_2b
    sessions, errors = (_load_dataset_2a(subject) if dataset == "2a"
                        else _load_dataset_2b(subject))
    name = ("Session I" if session_index == 1 else "Session II") if dataset == "2a" else f"Session {session_index}"
    selected = next((s for s in sessions if str(s["label"]).startswith(name)), None)
    if selected is None:
        raise ValueError(f"Unable to load {dataset.upper()} subject {subject:02d} {name}: " + "; ".join(errors))
    metrics = selected["_metrics"]
    # µ and β are logged in dB relative to µV²; convert back to strictly
    # positive band-power values before Eq (2.2) mean normalisation.
    frame = pd.DataFrame({
        "time": np.arange(len(metrics["rms_uv"]), dtype=float),
        "mu_power_uv2": np.power(10., np.asarray(metrics["mu_db"]) / 10.),
        "beta_power_uv2": np.power(10., np.asarray(metrics["beta_db"]) / 10.),
        "rms_uv": np.asarray(metrics["rms_uv"], dtype=float),
    })
    return frame, f"BCI IV {dataset.upper()} · Subject {subject:02d} · {name}"


def _park_view():
    """Separate Park-inspired window summaries and Section 2.1 mean-CUSUM."""
    source = request.args.get("source", "demo").strip().lower()
    if source not in {"demo", "2a", "2b", "csv"}:
        abort(400, description="Choose demo, 2a, 2b, or csv.")
    try:
        subject = int(request.args.get("subject", "1"))
        session_index = int(request.args.get("session", "1"))
        window = int(request.args.get("window", "40"))
        min_segment = int(request.args.get("min_segment", "8"))
    except (TypeError, ValueError):
        abort(400, description="Subject, session and window must be integers.")
    if not 1 <= subject <= 9:
        abort(400, description="Subject must range from 1 to 9.")
    if source in {"2a", "2b"} and not 1 <= session_index <= (2 if source == "2a" else 5):
        abort(400, description="Session is outside the dataset's available range.")
    if source == "demo":
        frame = demonstration_stream()
        # Demo sensor values are signed and near mean zero. Squared positive
        # amplitudes are descriptive inputs, not the paper's spectral method.
        for channel in ("sensor_1", "sensor_2", "sensor_3"):
            frame[channel] = frame[channel].to_numpy() ** 2 + 1.0
        source_label = "Synthetic demonstration · positive squared amplitudes"
    elif source == "csv":
        path = current_app.config.get("PARK_MULTIVARIATE_DATA_PATH")
        if not path:
            raise ValueError("Configure PARK_MULTIVARIATE_DATA_PATH before selecting CSV.")
        frame = load_multivariate_csv(path)
        source_label = "Configured multivariate CSV"
    else:
        frame, source_label = _park_bci_features(source, subject, session_index)
    if window < 4 or window > len(frame):
        abort(400, description="Window must range from 4 to the sample count.")
    rows = describe_windows(frame, window_size=window)
    features = frame.drop(columns=["time"]).to_numpy(dtype=float)
    cusum = mean_cusum_curves(features, min_segment=min_segment)
    cusum["variables"] = [str(col) for col in frame.columns if col != "time"]
    return {
        "metrics": [("Observations", len(frame)), ("Variables", len(frame.columns) - 1),
                    ("Complete windows", len(rows))],
        "window": window, "windows": rows,
        "cusum": cusum, "source": source, "subject": subject,
        "session": session_index, "min_segment": min_segment,
        "source_label": source_label,
        "schema": _schema(frame),
        "previews": [_preview(frame, "Ordered multivariate observations")],
        "notes": [
            "Source: " + source_label + ". One selected BCI session is analysed at a time.",
            "For BCI IV, each observation is a cue-aligned trial, not a continuous raw EEG sample. Features are positive µ/β power (µV²) and RMS amplitude (µV).",
            "Section 2.1 candidate peaks are retrospective positions in this selected sequence; unimodality/peak-agreement rules and hypothesis-test calibration are NOT implemented.",
            "A candidate mean change is not confirmed covariate shift, adaptation utility or a published Park reproduction.",
            "These descriptive results do not retrain FBCSP, PCA or SVM models. Trial boundaries are not concatenated.",
        ],
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
    readers = {"turbofan": _turbofan_view, "algae": _algae_view, "synthetic": _synthetic_view, "park": _park_view}
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
