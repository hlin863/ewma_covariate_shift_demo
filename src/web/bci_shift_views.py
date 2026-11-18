"""Exploratory, provenance-aware feature-shift and CSE event chart payloads.

Uses the same development-fitted FBCSP rows for both chart layers. No
evaluation labels are used for fitting or detector monitoring; charts
are diagnostics, not paper-reproduction claims or ground-truth drift.
"""
from __future__ import annotations

import numpy as np

from src.detection.core import CSEConfig, run_cse
from src.detection.stage2.kl_diagnostics import (
    KLDiagnosticConfig, attach_kl_diagnostics, fit_kl_histogram_edges,
)

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
                            evaluation_cue_onsets=None) -> dict:
    """Run a single clearly specified exploratory Stage-I/Stage-II configuration.

    Warnings refer to the evaluation feature row, and Stage-II decisions
    retain separate warning and validation times (the latter may be later).
    """
    evaluation = np.asarray(evaluation_features, dtype=float)
    onsets = (None if evaluation_cue_onsets is None
              else np.asarray(evaluation_cue_onsets, dtype=float))
    if evaluation.ndim != 2 or (onsets is not None and len(evaluation) != len(onsets)):
        raise ValueError("Evaluation features and source cue times must align")
    if onsets is not None and not np.isfinite(onsets).all():
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
    # Opt-in descriptive diagnostics on the exact retained PCA representation
    # and eligible windows already selected by the unchanged Hotelling runner.
    kl_config = KLDiagnosticConfig(n_bins=8, pseudocount=0.5)
    kl_records = attach_kl_diagnostics(
        validation_results=result.validation_results,
        testing_features=result.testing_transformed,
        testing_times=np.arange(len(evaluation), dtype=int),
        calibration_features=result.pca_result.training_transformed,
        config=kl_config,
    )
    kl_edges = fit_kl_histogram_edges(
        result.pca_result.training_transformed, kl_config
    )
    kl_events = []
    for record in kl_records.to_dict("records"):
        alarm_idx = int(record["alarm_time"])
        status = str(record["kl_status"])
        event = {
            "alarm_index": alarm_idx,
            "validation_index": (
                int(record["validation_time"])
                if record["validation_time"] is not None
                and np.isfinite(record["validation_time"])
                else None
            ),
            "hotelling_status": str(record["status"]),
            "kl_status": status,
            "forward": _optional_number(record["kl_reference_to_current"]),
            "reverse": _optional_number(record["kl_current_to_reference"]),
            "per_feature": (
                [float(x) for x in record["kl_per_feature"]]
                if record["kl_per_feature"] is not None else []
            ),
            "histograms": [],
        }
        if status == "diagnostic_only":
            h = int(record["sample_size"])
            before = result.testing_transformed[alarm_idx-h+1:alarm_idx+1]
            after = result.testing_transformed[alarm_idx+1:alarm_idx+h+1]
            event["reference_window"] = [alarm_idx-h+1, alarm_idx]
            event["current_window"] = [alarm_idx+1, alarm_idx+h]
            for column, boundaries in enumerate(kl_edges):
                a = np.histogram(before[:, column], bins=boundaries)[0].astype(float)
                b = np.histogram(after[:, column], bins=boundaries)[0].astype(float)
                a += kl_config.pseudocount
                b += kl_config.pseudocount
                a /= a.sum()
                b /= b.sum()
                event["histograms"].append({
                    "reference": a.tolist(),
                    "current": b.tolist(),
                    "edges": [
                        ("-inf" if np.isneginf(edge) else
                         "+inf" if np.isposinf(edge) else float(edge))
                        for edge in boundaries
                    ],
                })
        kl_events.append(event)
    warnings = []
    for record in result.warning_results.to_dict("records"):
        idx = int(record["time"])
        warnings.append({
            "index": idx,
            "cue_onset_seconds": float(onsets[idx]) if onsets is not None else None,
            "ewma": _optional_number(record.get("ewma")),
            "feature_value": _optional_number(record.get("x")),
            "prediction": _optional_number(record.get("prediction")),
            "lcl": _optional_number(record.get("lcl")),
            "ucl": _optional_number(record.get("ucl")),
            "warning": bool(record["stage_1_alarm"]),
        })
    validations = []
    for record in result.validation_results.to_dict("records"):
        alarm_index = int(record["alarm_time"])
        finish_index = _optional_number(record.get("validation_time"))
        finish_index = int(finish_index) if finish_index is not None else None
        validations.append({
            "alarm_index": alarm_index,
            "alarm_cue_onset_seconds": (float(onsets[alarm_index]) if onsets is not None else None),
            "validation_index": finish_index,
            "validation_cue_onset_seconds": (
                float(onsets[finish_index]) if onsets is not None and finish_index is not None else None
            ),
            "status": str(record["status"]),
            "confirmed": bool(record["confirmed_shift"]),
            "p_value": _optional_number(record.get("p_value")),
            "reference_start_index": _optional_number(record.get("reference_start_time")),
            "current_end_index": _optional_number(record.get("current_end_time")),
        })
    return {
        "kl_diagnostics": {
            "events": kl_events,
            "feature_names": [
                f"PC{index + 1}" for index in range(result.testing_transformed.shape[1])
            ],
            "n_bins": kl_config.n_bins,
            "pseudocount": kl_config.pseudocount,
            "method": "marginal histogram KL; descriptive, not joint KL or a significance test",
            "eligible_count": sum(e["kl_status"] == "diagnostic_only" for e in kl_events),
        },
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
