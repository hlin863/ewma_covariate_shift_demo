"""Exploratory, provenance-aware feature-shift and CSE event chart payloads.

Uses the same development-fitted FBCSP rows for both chart layers. No
evaluation labels are used for fitting or detector monitoring; charts
are diagnostics, not paper-reproduction claims or ground-truth drift.
"""
from __future__ import annotations

import numpy as np

from src.detection.core import CSEConfig, run_cse

WINDOW_SIZE = 10
PCA_COMPONENTS = 2


def feature_window_heatmap(training_features, evaluation_features, feature_names,
                           window_size: int = WINDOW_SIZE) -> dict:
    """Nonoverlapping, complete evaluation windows versus development reference."""
    train = np.asarray(training_features, dtype=float)
    evaluation = np.asarray(evaluation_features, dtype=float)
    if train.ndim != 2 or evaluation.ndim != 2 or train.shape[1] != evaluation.shape[1]:
        raise ValueError("Feature-space dimensions must agree")
    if len(feature_names) != train.shape[1] or window_size < 2:
        raise ValueError("Invalid feature names or heatmap window size")
    if not np.isfinite(train).all() or not np.isfinite(evaluation).all():
        raise ValueError("Features must be finite")
    reference_mean = train.mean(axis=0)
    reference_std = train.std(axis=0, ddof=1)
    valid = reference_std > 1e-10
    windows = []
    for start in range(0, len(evaluation) - window_size + 1, window_size):
        stop = start + window_size
        mean = evaluation[start:stop].mean(axis=0)
        differences = np.full(train.shape[1], np.nan)
        differences[valid] = (mean[valid] - reference_mean[valid]) / reference_std[valid]
        windows.append({
            "start_index": int(start),
            "end_index": int(stop - 1),
            "values": [round(float(v), 5) if np.isfinite(v) else None for v in differences],
        })
    return {
        "feature_names": list(feature_names),
        "windows": windows,
        "window_size": window_size,
        "unscaled_feature_names": [str(feature_names[i]) for i in np.flatnonzero(~valid)],
        "statistic": "window mean minus development mean, divided by development sample SD",
    }


def _optional_number(value):
    if value is None:
        return None
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def detection_event_payload(training_features, evaluation_features,
                            evaluation_cue_onsets) -> dict:
    """Run a single clearly specified exploratory Stage-I/Stage-II configuration.

    Warnings refer to the evaluation feature row, and Stage-II decisions
    retain separate warning and validation times (the latter may be later).
    """
    evaluation = np.asarray(evaluation_features, dtype=float)
    onsets = np.asarray(evaluation_cue_onsets, dtype=float)
    if evaluation.ndim != 2 or len(evaluation) != len(onsets):
        raise ValueError("Evaluation features and source cue times must align")
    if not np.isfinite(onsets).all():
        raise ValueError("Cue onset times must be finite")
    config = CSEConfig(
        pca_components=PCA_COMPONENTS,
        lambda_override=0.2,
        ewma_initialization="training_final",
        validation_mode="paper_two_sample",
        validation_before_size=WINDOW_SIZE,
        validation_after_size=WINDOW_SIZE,
        covariance_method="shrinkage",
        validation_alpha=0.05,
    )
    result = run_cse(
        training_features=np.asarray(training_features, dtype=float),
        testing_features=evaluation,
        testing_times=np.arange(len(evaluation), dtype=int),
        config=config,
    )
    warnings = []
    for record in result.warning_results.to_dict("records"):
        idx = int(record["time"])
        warnings.append({
            "index": idx,
            "cue_onset_seconds": float(onsets[idx]),
            "ewma": _optional_number(record.get("ewma")),
            "warning": bool(record["stage_1_alarm"]),
        })
    validations = []
    for record in result.validation_results.to_dict("records"):
        alarm_index = int(record["alarm_time"])
        finish_index = _optional_number(record.get("validation_time"))
        finish_index = int(finish_index) if finish_index is not None else None
        validations.append({
            "alarm_index": alarm_index,
            "alarm_cue_onset_seconds": float(onsets[alarm_index]),
            "validation_index": finish_index,
            "validation_cue_onset_seconds": (
                float(onsets[finish_index]) if finish_index is not None else None
            ),
            "status": str(record["status"]),
            "confirmed": bool(record["confirmed_shift"]),
            "p_value": _optional_number(record.get("p_value")),
            "reference_start_index": _optional_number(record.get("reference_start_time")),
            "current_end_index": _optional_number(record.get("current_end_time")),
        })
    return {
        "warnings": warnings,
        "validations": validations,
        "config": {
            "pca_components": PCA_COMPONENTS,
            "lambda": 0.2,
            "initialization": "training_final",
            "validation_mode": "paper_two_sample",
            "window_size": WINDOW_SIZE,
            "covariance_method": "shrinkage",
            "alpha": 0.05,
            "status": "exploratory recomputation, not a published-count reproduction",
        },
        "warning_count": sum(int(w["warning"]) for w in warnings),
        "confirmed_count": sum(int(v["confirmed"]) for v in validations),
    }
