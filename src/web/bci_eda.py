"""Exploratory distribution views for the local BCI Competition IV recordings.

The page deliberately consumes the same GDF loaders and three-second cue-aligned
trial extractors used by the reproduction code. It summarizes raw trial
distributions for visual inspection; it does not turn descriptive differences
into covariate-shift confirmations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from flask import Blueprint, current_app, render_template, request

from src.bci.data import (
    dataset_2a_filename,
    dataset_2b_filename,
    load_bci_competition_iv_2a_session,
    load_bci_competition_iv_2b_session,
)
from src.bci.datasets.dataset2a import extract_dataset_2a_trials
from src.bci.datasets.dataset2a.evaluation_labels import (
    load_dataset_2a_evaluation_labels,
)
from src.bci.datasets.dataset2b import extract_dataset_2b_trials


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_2A_PATH = PROJECT_ROOT / "data" / "raw" / "bci_competition_iv_2a"
DEFAULT_DATASET_2A_LABELS_PATH = (
    PROJECT_ROOT / "data" / "raw" / "bci_competition_iv_2a_labels"
)
DEFAULT_DATASET_2B_PATH = PROJECT_ROOT / "data" / "raw" / "bci_competition_iv_2b"

bci_eda_bp = Blueprint("bci_eda", __name__)


def _safe_subject(value: str | None) -> int:
    try:
        subject = int(value or 1)
    except (TypeError, ValueError):
        return 1
    return subject if 1 <= subject <= 9 else 1


def _band_power_db_uv2(
    signals: np.ndarray,
    sampling_frequency: float,
    low_hz: float,
    high_hz: float,
) -> np.ndarray:
    """Return one approximate log-band-power value per trial.

    The estimate is intentionally descriptive: each channel is de-meaned,
    Hann-windowed, transformed with an rFFT, integrated over the requested
    frequency band, converted from V² to µV², then averaged across channels.
    """

    values = np.asarray(signals, dtype=float)
    if values.ndim != 3:
        raise ValueError("trial signals must have shape (trials, channels, samples).")
    n_samples = int(values.shape[-1])
    if n_samples < 4:
        raise ValueError("at least four samples per trial are required for spectral EDA.")

    sfreq = float(sampling_frequency)
    frequencies = np.fft.rfftfreq(n_samples, d=1.0 / sfreq)
    mask = (frequencies >= float(low_hz)) & (frequencies <= float(high_hz))
    if int(mask.sum()) < 2:
        raise ValueError(
            f"sampling rate {sfreq:g} Hz does not resolve {low_hz:g}-{high_hz:g} Hz."
        )

    centred = values - values.mean(axis=-1, keepdims=True)
    window = np.hanning(n_samples)
    window_energy = float(np.square(window).sum())
    spectrum = np.fft.rfft(centred * window, axis=-1)
    psd = np.square(np.abs(spectrum)) / max(sfreq * window_energy, 1e-20)
    frequency_step = float(frequencies[1] - frequencies[0])
    band_power_v2 = psd[..., mask].sum(axis=-1) * frequency_step
    trial_power_uv2 = np.maximum(band_power_v2.mean(axis=1) * 1e12, 1e-20)
    return 10.0 * np.log10(trial_power_uv2)


def _class_counts(labels: np.ndarray) -> dict[str, int]:
    y = np.asarray(labels, dtype=int).reshape(-1)
    return {
        "left": int(np.sum(y == 0)),
        "right": int(np.sum(y == 1)),
        "unlabelled": int(np.sum(y < 0)),
        "other": int(np.sum(y > 1)),
    }


def _summarise_trials(trials, *, label: str, role: str) -> dict[str, object]:
    signals = np.asarray(trials.signals, dtype=float)
    if signals.ndim != 3 or signals.shape[0] == 0:
        raise ValueError("EDA requires at least one trial tensor.")
    if not np.isfinite(signals).all():
        raise ValueError("EDA trial tensors must contain only finite values.")

    scaled_uv = signals * 1e6
    trial_rms_uv = np.sqrt(np.mean(np.square(scaled_uv), axis=(1, 2)))
    trial_p2p_uv = np.ptp(scaled_uv, axis=-1).mean(axis=1)
    mu_db = _band_power_db_uv2(
        signals, float(trials.sampling_frequency), 8.0, 12.0
    )
    beta_db = _band_power_db_uv2(
        signals, float(trials.sampling_frequency), 16.0, 24.0
    )

    channel_summaries = []
    for index, channel_name in enumerate(trials.channel_names):
        flattened = scaled_uv[:, index, :].reshape(-1)
        q05, q25, median, q75, q95 = np.percentile(
            flattened, [5.0, 25.0, 50.0, 75.0, 95.0]
        )
        channel_summaries.append(
            {
                "channel": str(channel_name),
                "q05": round(float(q05), 4),
                "q25": round(float(q25), 4),
                "median": round(float(median), 4),
                "q75": round(float(q75), 4),
                "q95": round(float(q95), 4),
            }
        )

    return {
        "label": label,
        "role": role,
        "session_id": str(getattr(trials, "session_id", "")),
        "trials": int(signals.shape[0]),
        "channels": int(signals.shape[1]),
        "samples_per_trial": int(signals.shape[2]),
        "sampling_frequency": round(float(trials.sampling_frequency), 3),
        "duration_seconds": round(
            float(signals.shape[2]) / float(trials.sampling_frequency), 3
        ),
        "class_counts": _class_counts(trials.labels),
        "median_rms_uv": round(float(np.median(trial_rms_uv)), 4),
        "median_p2p_uv": round(float(np.median(trial_p2p_uv)), 4),
        "median_mu_db": round(float(np.median(mu_db)), 4),
        "median_beta_db": round(float(np.median(beta_db)), 4),
        "channel_summaries": channel_summaries,
        "_metrics": {
            "rms_uv": trial_rms_uv,
            "mu_db": mu_db,
            "beta_db": beta_db,
        },
    }


def _histogram_payload(
    sessions: list[dict[str, object]],
    metric: str,
    *,
    bins: int = 28,
) -> dict[str, object]:
    arrays = [
        np.asarray(session["_metrics"][metric], dtype=float)
        for session in sessions
        if metric in session.get("_metrics", {})
    ]
    if not arrays:
        return {"centres": [], "series": []}

    combined = np.concatenate(arrays)
    lower = float(np.min(combined))
    upper = float(np.max(combined))
    if np.isclose(lower, upper):
        padding = max(abs(lower) * 0.05, 0.5)
        lower -= padding
        upper += padding
    edges = np.linspace(lower, upper, int(bins) + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])

    series = []
    for session in sessions:
        values = np.asarray(session["_metrics"][metric], dtype=float)
        counts, _ = np.histogram(values, bins=edges)
        total = max(int(counts.sum()), 1)
        series.append(
            {
                "label": str(session["label"]),
                "values": (counts.astype(float) * 100.0 / total).round(4).tolist(),
            }
        )
    return {
        "centres": centres.round(6).tolist(),
        "series": series,
    }


def _serialise_sessions(sessions: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {key: value for key, value in session.items() if key != "_metrics"}
        for session in sessions
    ]


def _inventory() -> list[dict[str, object]]:
    data_2a = Path(
        current_app.config.get("DATASET_2A_PATH", DEFAULT_DATASET_2A_PATH)
    )
    labels_2a = Path(
        current_app.config.get(
            "DATASET_2A_LABELS_PATH", DEFAULT_DATASET_2A_LABELS_PATH
        )
    )
    data_2b = Path(
        current_app.config.get("DATASET_2B_PATH", DEFAULT_DATASET_2B_PATH)
    )

    expected_2a = [
        data_2a / dataset_2a_filename(subject, session)
        for subject in range(1, 10)
        for session in ("T", "E")
    ]
    expected_labels = [
        labels_2a / f"A{subject:02d}E.mat" for subject in range(1, 10)
    ]
    expected_2b = [
        data_2b / dataset_2b_filename(subject, session)
        for subject in range(1, 10)
        for session in range(1, 6)
    ]

    def item(name: str, path: Path, expected: list[Path]) -> dict[str, object]:
        present = sum(file_path.is_file() for file_path in expected)
        missing = [file_path.name for file_path in expected if not file_path.is_file()]
        return {
            "name": name,
            "path": str(path),
            "present": int(present),
            "expected": len(expected),
            "missing_preview": missing[:5],
        }

    return [
        item("Dataset 2A GDF", data_2a, expected_2a),
        item("Dataset 2A labels", labels_2a, expected_labels),
        item("Dataset 2B GDF", data_2b, expected_2b),
    ]


def _load_dataset_2a(subject: int) -> tuple[list[dict[str, object]], list[str]]:
    data_directory = Path(
        current_app.config.get("DATASET_2A_PATH", DEFAULT_DATASET_2A_PATH)
    )
    labels_directory = Path(
        current_app.config.get(
            "DATASET_2A_LABELS_PATH", DEFAULT_DATASET_2A_LABELS_PATH
        )
    )
    sessions: list[dict[str, object]] = []
    errors: list[str] = []

    for session_name, label, role in (
        ("T", "Session I", "training / calibration"),
        ("E", "Session II", "evaluation"),
    ):
        try:
            loaded = load_bci_competition_iv_2a_session(
                data_directory, subject, session_name
            )
            if session_name == "E":
                label_path = labels_directory / f"A{subject:02d}E.mat"
                if label_path.is_file():
                    official_labels = load_dataset_2a_evaluation_labels(label_path)
                    trials = extract_dataset_2a_trials(
                        loaded, evaluation_labels=official_labels
                    )
                    role = "evaluation · released left/right labels"
                else:
                    trials = extract_dataset_2a_trials(loaded)
                    role = "evaluation · GDF-only marginal"
            else:
                trials = extract_dataset_2a_trials(loaded)
            sessions.append(_summarise_trials(trials, label=label, role=role))
        except (FileNotFoundError, ImportError, OSError, ValueError) as exc:
            errors.append(f"{label}: {exc}")

    return sessions, errors


def _load_dataset_2b(subject: int) -> tuple[list[dict[str, object]], list[str]]:
    data_directory = Path(
        current_app.config.get("DATASET_2B_PATH", DEFAULT_DATASET_2B_PATH)
    )
    sessions: list[dict[str, object]] = []
    errors: list[str] = []

    for session_number in range(1, 6):
        suffix = "T" if session_number <= 3 else "E"
        label = f"Session {session_number} ({suffix})"
        role = (
            "training / calibration cue stream"
            if session_number <= 3
            else "evaluation cue stream"
        )
        try:
            loaded = load_bci_competition_iv_2b_session(
                data_directory, subject, session_number
            )
            trials = extract_dataset_2b_trials(loaded)
            sessions.append(_summarise_trials(trials, label=label, role=role))
        except (FileNotFoundError, ImportError, OSError, ValueError) as exc:
            errors.append(f"{label}: {exc}")

    return sessions, errors


@bci_eda_bp.get("/data-distributions")
def data_distributions():
    """Render interactive descriptive EDA for local Dataset 2A/2B recordings."""

    dataset = str(request.args.get("dataset", "2a")).strip().lower()
    if dataset not in {"2a", "2b"}:
        dataset = "2a"
    subject = _safe_subject(request.args.get("subject"))

    if dataset == "2a":
        sessions, errors = _load_dataset_2a(subject)
        dataset_label = "BCI Competition IV Dataset 2A"
        subject_label = f"A{subject:02d}"
    else:
        sessions, errors = _load_dataset_2b(subject)
        dataset_label = "BCI Competition IV Dataset 2B"
        subject_label = f"B{subject:02d}"

    histograms = {
        "rms_uv": _histogram_payload(sessions, "rms_uv"),
        "mu_db": _histogram_payload(sessions, "mu_db"),
        "beta_db": _histogram_payload(sessions, "beta_db"),
    }
    public_sessions = _serialise_sessions(sessions)
    chart_payload = {
        "sessions": public_sessions,
        "histograms": histograms,
    }

    return render_template(
        "bci_eda.html",
        dataset=dataset,
        dataset_label=dataset_label,
        subject=subject,
        subject_label=subject_label,
        subjects=range(1, 10),
        sessions=public_sessions,
        chart_payload=chart_payload,
        inventory=_inventory(),
        errors=errors,
    )
