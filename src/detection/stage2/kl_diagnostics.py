"""Optional histogram KL diagnostics for Stage-II CSE warning windows.

This module does not implement a new confirmation test. It preserves existing
Hotelling status/confirmed_shift decisions, with no model update side effects.
Calibration-fitted histogram bins prevent evaluation-dependent bin selection.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import rel_entr


@dataclass(frozen=True)
class KLDiagnosticConfig:
    n_bins: int = 8
    pseudocount: float = 0.5

    def __post_init__(self):
        if not isinstance(self.n_bins, int) or self.n_bins < 2:
            raise ValueError("n_bins must be an integer >= 2")
        if not np.isfinite(self.pseudocount) or self.pseudocount <= 0:
            raise ValueError("pseudocount must be finite and positive")


def _matrix(values, name):
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError(f"{name} must be a finite, nonempty (observations, features) array")
    return x


def fit_kl_histogram_edges(calibration_features, config=None):
    """Fit fixed, equal-width histogram boundaries on calibration features only.

    Uses open-ended first and last bins so out-of-range evaluation values are
    included, not silently discarded. Constant training features use a
    symmetric unit-width calibration range; all columns keep n_bins bins.
    """
    config = config or KLDiagnosticConfig()
    x = _matrix(calibration_features, "calibration_features")
    edges = []
    for k in range(x.shape[1]):
        low, high = float(x[:, k].min()), float(x[:, k].max())
        if low == high:
            low -= 0.5
            high += 0.5
        cuts = np.linspace(low, high, config.n_bins + 1)
        cuts[0], cuts[-1] = -np.inf, np.inf
        edges.append(cuts)
    return np.asarray(edges)


def histogram_kl_scores(reference, current, edges, *, pseudocount=0.5):
    """Per-feature discrete KL(P_before || Q_after), plus its reverse.

    All probability bins share calibration-fitted boundaries. The mean is a
    descriptive marginal diagnostic, not joint KL or a calibrated p-value.
    """
    before = _matrix(reference, "reference")
    after = _matrix(current, "current")
    cuts = np.asarray(edges, dtype=float)
    if before.shape[1] != after.shape[1]:
        raise ValueError("Reference/current feature counts differ")
    if cuts.ndim != 2 or cuts.shape[0] != before.shape[1] or cuts.shape[1] < 3:
        raise ValueError("edges must be (n_features, n_bins + 1)")
    if not np.all(np.diff(cuts, axis=1) > 0):
        raise ValueError("Histogram edges must be strictly increasing")
    if not np.isfinite(pseudocount) or pseudocount <= 0:
        raise ValueError("pseudocount must be positive")
    forward, reverse = [], []
    for k in range(before.shape[1]):
        a = np.histogram(before[:, k], bins=cuts[k])[0].astype(float) + pseudocount
        b = np.histogram(after[:, k], bins=cuts[k])[0].astype(float) + pseudocount
        a /= a.sum()
        b /= b.sum()
        forward.append(float(rel_entr(a, b).sum()))
        reverse.append(float(rel_entr(b, a).sum()))
    return np.asarray(forward), np.asarray(reverse)


def attach_kl_diagnostics(
    validation_results,
    testing_features,
    testing_times,
    calibration_features,
    *,
    config=None,
):
    """Attach descriptive KL scores to existing paper-two-sample validation rows.

    Uses each row's recorded window size and alarm time. Evaluates only rows
    with status confirmed/rejected and a complete before/after window.
    Does not alter existing Hotelling status, p_value, or confirmed_shift.
    Neither testing labels nor classifier predictions are used.
    """
    config = config or KLDiagnosticConfig()
    x = _matrix(testing_features, "testing_features")
    train = _matrix(calibration_features, "calibration_features")
    times = np.asarray(testing_times)
    if x.shape[1] != train.shape[1]:
        raise ValueError("Calibration/testing feature counts differ")
    if times.ndim != 1 or len(times) != len(x) or len(np.unique(times)) != len(times):
        raise ValueError("testing_times must uniquely identify testing observations")
    required = {"alarm_time", "sample_size", "status"}
    if not required.issubset(validation_results.columns):
        raise ValueError(f"Missing required validation columns: {sorted(required - set(validation_results.columns))}")
    result = validation_results.copy()
    result["kl_reference_to_current"] = np.nan
    result["kl_current_to_reference"] = np.nan
    result["kl_per_feature"] = None
    result["kl_status"] = "not_evaluated"
    result["kl_method"] = "marginal_histogram_pseudocount"
    result["kl_n_bins"] = config.n_bins
    edges = fit_kl_histogram_edges(train, config)
    index_by_time = {value: i for i, value in enumerate(times.tolist())}
    for row_idx, row in result.iterrows():
        if row["status"] not in {"confirmed", "rejected"}:
            result.at[row_idx, "kl_status"] = "ineligible_" + str(row["status"])
            continue
        i = index_by_time.get(row["alarm_time"])
        if i is None:
            raise ValueError("Validation alarm_time absent from testing_times")
        h = int(row["sample_size"])
        if h < 2 or i - h + 1 < 0 or i + h >= len(x):
            result.at[row_idx, "kl_status"] = "incomplete_window"
            continue
        before = x[i-h+1:i+1]
        after = x[i+1:i+h+1]
        a, b = histogram_kl_scores(before, after, edges, pseudocount=config.pseudocount)
        result.at[row_idx, "kl_reference_to_current"] = float(a.mean())
        result.at[row_idx, "kl_current_to_reference"] = float(b.mean())
        result.at[row_idx, "kl_per_feature"] = a.tolist()
        result.at[row_idx, "kl_status"] = "diagnostic_only"
    return result
