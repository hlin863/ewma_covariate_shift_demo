"""Paired, synthetic benchmark: Park (2023) binary mean rule versus proposed CPSS.

This is an exploratory *offline mean-change* study, not a reproduction of
Park's Section 3 simulation models or evidence about online ARL/BCI accuracy.
"""
from dataclasses import asdict, dataclass
import csv
import json
from math import ceil
from pathlib import Path

import numpy as np

from src.detection.stage1.park_cpss import calculate_cpss_score
from src.detection.stage1.park_mean_cusum import (
    evaluate_mean_change,
    mean_cusum_curves,
)

SCENARIOS = ("stationary", "mean_shift", "variance_shift")


@dataclass(frozen=True)
class BenchmarkConfig:
    observations: int = 160
    channels: int = 4
    calibration_seeds: int = 12
    evaluation_seeds: int = 24
    perturbations: int = 8
    jitter_sd: float = 0.12
    gamma: float = 1.0
    target_fpr: float = 0.05
    mean_shift_size: float = 2.0
    variance_factor: float = 1.8
    seed: int = 2026
    min_segment: int = 8

    def __post_init__(self):
        if self.observations < 100 or self.channels < 2:
            raise ValueError("Use at least 100 observations and 2 channels.")
        if min(self.calibration_seeds, self.evaluation_seeds, self.perturbations) < 1:
            raise ValueError("All run counts must be positive.")
        if self.min_segment < 2 or self.min_segment * 2 >= self.observations:
            raise ValueError("Invalid minimum segment.")
        if not (0 < self.target_fpr < 1):
            raise ValueError("target_fpr must lie in (0, 1).")
        if (not np.isfinite(self.jitter_sd) or self.jitter_sd < 0 or
                not np.isfinite(self.gamma) or self.gamma < 0 or
                not np.isfinite(self.mean_shift_size) or self.mean_shift_size <= 0 or
                not np.isfinite(self.variance_factor) or self.variance_factor <= 1):
            raise ValueError("Invalid perturbation, penalty or shift configuration.")


def make_stream(scenario: str, config: BenchmarkConfig, seed: int) -> np.ndarray:
    """Positive-mean, temporally dependent synthetic features; paired noise seed.

    Not Park's signed AR/MA simulation. The variance-only condition changes
    dispersion without intentionally changing the population mean.
    """
    if scenario not in SCENARIOS:
        raise ValueError("Unknown benchmark scenario.")
    rng = np.random.default_rng(seed)
    n, p = config.observations, config.channels
    shared = rng.standard_normal((n, 1))
    innovation = 0.85 * rng.standard_normal((n, p)) + 0.25 * shared
    state = np.zeros_like(innovation)
    for t in range(n):
        state[t] = innovation[t] + (0.45 * state[t - 1] if t else 0)
    centre = 8.0 + 0.15 * np.arange(p)
    values = centre + 0.65 * state
    breakpoint = n // 2
    if scenario == "mean_shift":
        values[breakpoint:, :ceil(p / 2)] += config.mean_shift_size
    elif scenario == "variance_shift":
        values[breakpoint:] = centre + config.variance_factor * (
            values[breakpoint:] - centre)
    if np.any(values <= 0):
        raise ValueError("Synthetic positive-feature assumption violated.")
    return values


def evaluate_trial(values: np.ndarray, config: BenchmarkConfig, jitter_seed: int) -> dict:
    """Use the exact same Park pipeline for the binary rule and every CPSS run.

    Invalid/unsampleable perturbations contribute zero, rather than being
    silently dropped from the CPSS denominator.
    """
    T = len(values)
    baseline = evaluate_mean_change(
        mean_cusum_curves(values, min_segment=config.min_segment))
    rng = np.random.default_rng(jitter_seed)
    amax, aavg, bmax, bavg = [], [], [], []
    contributions, confirmed_positions, gaps = [], [], []
    ineligible = 0
    for _ in range(config.perturbations):
        jittered = values + rng.normal(0, config.jitter_sd, size=values.shape)
        if np.any(jittered <= 0):
            raise ValueError("Jitter produced a non-positive feature value.")
        result = evaluate_mean_change(
            mean_cusum_curves(jittered, min_segment=config.min_segment))
        if result["status"] == "ineligible":
            # A_max=A_avg=1 yields a failed unimodality product.
            a1, a2, b1, b2 = 1, 1, 1, 1
            ineligible += 1
        else:
            a1 = result["maximum_check"]["A"]
            a2 = result["average_check"]["A"]
            b1 = result["b_max_smoothed"]
            b2 = result["b_avg_smoothed"]
            gaps.append(abs(b1 - b2))
            if result["mean_change"]:
                confirmed_positions.append(result["estimated_position"])
        amax.append(a1)
        aavg.append(a2)
        bmax.append(b1)
        bavg.append(b2)
        passed = a1 * a2 == 0 and abs(b1 - b2) < np.sqrt(T) / 2
        contributions.append(float(passed) * np.exp(
            -config.gamma * abs(b1 - b2) / (T - 1)))
    score = calculate_cpss_score(
        amax, aavg, bmax, bavg, T=T, gamma=config.gamma)
    assert np.isclose(score, np.mean(contributions))
    return {
        "binary_decision": int(baseline["mean_change"] is True),
        "binary_eligible": int(baseline["status"] != "ineligible"),
        "binary_position": baseline["estimated_position"],
        "cpss_score": float(score),
        "cpss_position": (float(np.median(confirmed_positions))
                          if confirmed_positions else None),
        "perturbation_pass_rate": len(confirmed_positions) / config.perturbations,
        "contribution_std": _std(contributions),
        "contribution_iqr": _iqr(contributions),
        "peak_iqr": _iqr(confirmed_positions),
        "gap_iqr": _iqr(gaps),
        "ineligible_perturbations": ineligible,
    }


def _std(values):
    return float(np.std(values, ddof=1)) if len(values) > 1 else None


def _iqr(values):
    return (float(np.percentile(values, 75) - np.percentile(values, 25))
            if values else None)


def calibrate_threshold(null_scores, target_fpr: float) -> float:
    """Conservative empirical upper-tail threshold using disjoint null runs.

    Calls above threshold are strictly greater; ties never count as alarms.
    Small null calibration sets may imply an achieved FPR of zero.
    """
    scores = np.sort(np.asarray(null_scores, dtype=float))
    if scores.ndim != 1 or len(scores) == 0 or not np.isfinite(scores).all():
        raise ValueError("Nonempty finite calibration scores required.")
    rank = min(len(scores), ceil((len(scores) + 1) * (1 - target_fpr)))
    return float(scores[rank - 1])


def _ratio(a, b):
    return float(a / b) if b else None


def binary_metrics(labels, predictions):
    labels, predictions = np.asarray(labels, dtype=int), np.asarray(predictions, dtype=int)
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    sensitivity = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": _ratio(tp + tn, len(labels)),
        "balanced_accuracy": (float((sensitivity + specificity) / 2)
                              if sensitivity is not None and specificity is not None else None),
        "sensitivity": sensitivity, "specificity": specificity,
        "precision": _ratio(tp, tp + fp), "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "false_positive_rate": _ratio(fp, fp + tn),
    }


def roc_auc(scores, labels):
    """Mann-Whitney AUC with 0.5 credit for ties; no extra dependencies."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pos, neg = scores[labels == 1], np.sort(scores[labels == 0])
    if not len(pos) or not len(neg):
        return None
    left = np.searchsorted(neg, pos, side="left")
    right = np.searchsorted(neg, pos, side="right")
    return float(np.sum(left + 0.5 * (right - left)) / (len(pos) * len(neg)))


def _spread(values):
    values = list(values)
    if not values:
        return {"mean": None, "std": None, "median": None, "iqr": None,
                "p10": None, "p90": None, "count": 0}
    return {
        "mean": float(np.mean(values)), "std": _std(values),
        "median": float(np.median(values)), "iqr": _iqr(values),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)), "count": len(values),
    }


def run_benchmark(config: BenchmarkConfig) -> dict:
    # Calibration and evaluation use disjoint generator seeds.
    # Both detectors see matched underlying streams at each evaluation seed.
    calibration_scores = []
    for i in range(config.calibration_seeds):
        seed = config.seed + 100000 + i
        calibration_scores.append(evaluate_trial(
            make_stream("stationary", config, seed), config, seed + 300000
        )["cpss_score"])
    threshold = calibrate_threshold(calibration_scores, config.target_fpr)
    rows = []
    for i in range(config.evaluation_seeds):
        seed = config.seed + 200000 + i
        for scenario in SCENARIOS:
            obs = make_stream(scenario, config, seed)
            row = evaluate_trial(obs, config, seed + 400000)
            row.update({
                "scenario": scenario, "seed": seed, "change_at": config.observations // 2
                if scenario != "stationary" else None,
                "mean_change_truth": int(scenario == "mean_shift"),
                "cpss_decision": int(row["cpss_score"] > threshold),
            })
            for name in ("binary", "cpss"):
                position = row[name + "_position"]
                row[name + "_absolute_position_error"] = (
                    abs(position - config.observations // 2)
                    if scenario == "mean_shift" and row[name + "_decision"] and position is not None
                    else None)
            rows.append(row)

    # Mean-change classification metrics exclude variance_shift, which is a
    # separately reported second-order nuisance case.
    primary = [row for row in rows if row["scenario"] != "variance_shift"]
    truth = [row["mean_change_truth"] for row in primary]
    results = {}
    for name in ("binary", "cpss"):
        predictions = [row[name + "_decision"] for row in primary]
        method = binary_metrics(truth, predictions)
        method["roc_auc"] = roc_auc(
            [row["cpss_score"] if name == "cpss" else row["binary_decision"]
             for row in primary], truth)
        method["position_absolute_error"] = _spread([
            row[name + "_absolute_position_error"] for row in rows
            if row[name + "_absolute_position_error"] is not None])
        method["stationary_positive_rate"] = float(np.mean([
            row[name + "_decision"] for row in rows if row["scenario"] == "stationary"]))
        method["variance_only_positive_rate"] = float(np.mean([
            row[name + "_decision"] for row in rows if row["scenario"] == "variance_shift"]))
        results[name] = method

    scenario_spread = {
        scenario: {
            "cpss_score": _spread([r["cpss_score"] for r in rows if r["scenario"] == scenario]),
            "binary_decision": _spread([r["binary_decision"] for r in rows if r["scenario"] == scenario]),
            "within_stream_cpss_contribution_std": _spread([
                r["contribution_std"] for r in rows
                if r["scenario"] == scenario and r["contribution_std"] is not None]),
            "within_stream_peak_iqr": _spread([
                r["peak_iqr"] for r in rows
                if r["scenario"] == scenario and r["peak_iqr"] is not None]),
            "ineligible_binary": sum(1 for r in rows if r["scenario"] == scenario
                                     and not r["binary_eligible"]),
            "ineligible_perturbations": sum(r["ineligible_perturbations"] for r in rows
                                            if r["scenario"] == scenario),
        } for scenario in SCENARIOS
    }
    return {
        "source": "Exploratory positive-feature AR(1) synthetic benchmark; not Park Figure 3 reproduction",
        "config": asdict(config),
        "calibration": {
            "target_fpr": config.target_fpr, "threshold": threshold,
            "seed_range": [config.seed + 100000,
                           config.seed + 100000 + config.calibration_seeds - 1],
            "scores": calibration_scores, "rule": "score > threshold; ties rejected",
        },
        "evaluation_seed_range": [config.seed + 200000,
                                   config.seed + 200000 + config.evaluation_seeds - 1],
        "classification": results,
        "spread": scenario_spread,
        "rows": rows,
        "limitations": [
            "Park binary rule is an offline first-order candidate, not an online detector.",
            "Synthetic feature distributions are not Park's signed AR/MA Section 3 models.",
            "CPSS jitter is a sensitivity perturbation, not calibrated measurement uncertainty.",
            "Threshold is null-calibrated on a finite disjoint sample; evaluation FPR can differ.",
            "Localization error is evaluated only for detected mean changes.",
            "Variance-only changes are a second-order stress case, not a mean-change positive.",
        ],
    }


def export_benchmark(result: dict, output_dir: Path) -> dict:
    """Export machine-readable seed-level records and an auditable summary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    rows = result["rows"]
    with (output_dir / "trials.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    compact = {key: value for key, value in result.items() if key != "rows"}
    (output_dir / "summary.json").write_text(
        json.dumps(compact, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"summary": str(output_dir / "summary.json"),
            "trials": str(output_dir / "trials.csv")}
