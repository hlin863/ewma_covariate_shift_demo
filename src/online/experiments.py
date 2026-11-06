"""Controlled online adaptive learning comparison using the existing synthetic generator.

One policy receives one feature vector at a time. All policies use the same
calibration and evaluation data, and true labels are released according to the
configured trial-delay. Data and results are generated locally, not published
BCI Competition IV reproduction evidence.
"""
from __future__ import annotations

import numpy as np

from src.adaptation.experiments.diethe import DietheConfig, synthetic_features
from src.adaptation.policies import NeverUpdate, PeriodicRetrain, RetrainOnValidatedShift
from src.online.pipeline import OnlineAdaptationConfig, OnlineAdaptiveBCI


def run_online_comparison(
    *,
    scenario: str = "feature_shift",
    seed: int = 42,
    trials: int = 100,
    label_delay: int | None = 0,
    window_size: int = 10,
    interval: int = 10,
) -> dict[str, object]:
    """Return measured run results, not illustrative or pre-filled metrics."""
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an integer in [0, 2**32 - 1]")
    if isinstance(trials, bool) or not isinstance(trials, int) or not 80 <= trials <= 500:
        raise ValueError("trials must be between 80 and 500")
    if isinstance(window_size, bool) or not isinstance(window_size, int) or not 4 <= window_size <= 50:
        raise ValueError("window_size must be between 4 and 50")
    if isinstance(interval, bool) or not isinstance(interval, int) or not 1 <= interval <= 100:
        raise ValueError("interval must be between 1 and 100")
    if label_delay is not None and (
        isinstance(label_delay, bool) or not isinstance(label_delay, int)
        or label_delay < 0 or label_delay > 500
    ):
        raise ValueError("label_delay must be between 0 and 500 or None")

    generated, change_index = synthetic_features(
        DietheConfig(seed=seed, scenario=scenario, trials=trials)
    )
    policies = (
        ("never", NeverUpdate()),
        ("periodic", PeriodicRetrain(interval)),
        ("validated", RetrainOnValidatedShift()),
    )
    summary: list[dict[str, object]] = []
    traces: dict[str, list[dict[str, object]]] = {}
    for name, policy in policies:
        pipeline = OnlineAdaptiveBCI.from_calibration(
            generated["training_features"], generated["training_labels"],
            policy=policy,
            config=OnlineAdaptationConfig(
                window_size=window_size, label_delay=label_delay,
                covariance_method="shrinkage",
            ),
            pca_components=3, lambda_override=0.2,
            control_limit_multiplier=2.5,
        )
        result = pipeline.run_replay(
            generated["evaluation_features"],
            generated["evaluation_times"],
            generated["evaluation_labels"],
        )
        trials_table = result.trials
        completed = result.validations
        validated = completed.loc[completed["status"] == "confirmed"] if not completed.empty else completed
        updates = result.updates
        post_change_accuracy = (
            float(trials_table.iloc[change_index:]["correct"].mean())
            if change_index is not None else None
        )
        summary.append({
            "policy": name,
            "accuracy": float(trials_table["correct"].mean()),
            "post_change_accuracy": post_change_accuracy,
            "warnings": int(trials_table["stage1_warning"].sum()),
            "confirmed": int(validated["alarm_time"].nunique()) if not validated.empty else 0,
            "updates": int(len(updates)),
            "training_size": int(result.final_classifier.training_size),
            "retrain_seconds": float(updates["retrain_seconds"].sum()) if not updates.empty else 0.0,
            "pending": int(result.remaining_pending_warnings),
        })
        traces[name] = [{
            "time": int(row.time),
            "ewma": float(row.ewma),
            "warning": bool(row.stage1_warning),
            "correct": bool(row.correct),
            "version": int(row.model_version),
        } for row in trials_table.itertuples(index=False)]

    baseline_accuracy = summary[0]["accuracy"]
    for row in summary:
        row["gain_vs_never"] = float(row["accuracy"] - baseline_accuracy)
    return {
        "summary": summary,
        "traces": traces,
        "scenario": scenario,
        "seed": seed,
        "trials": trials,
        "change_index": change_index,
        "label_delay": label_delay,
        "window_size": window_size,
        "interval": interval,
        "provenance": "synthetic online feature-level replay; not BCI 2A/2B experimental evidence",
    }
