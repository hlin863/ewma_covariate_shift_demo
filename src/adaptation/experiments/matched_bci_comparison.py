# src/adaptation/experiments/matched_bci_comparison.py

"""Matched frozen versus accuracy-triggered SVM comparison."""

from dataclasses import replace
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4
import hashlib
import json
import subprocess

import numpy as np
import pandas as pd

from scripts.run_online_bci import prepare
from src.adaptation.classifier import build_fresh_model
from src.adaptation.twin_adapt.config import TwinAdaptConfig
from src.adaptation.twin_adapt.controller import adapt_model
from src.online.batch_stream import make_bci_dataflow


def _fingerprint(*arrays):
    """Hash the effective arrays used by both comparison arms."""
    digest = hashlib.sha256()

    for array in arrays:
        value = np.ascontiguousarray(np.asarray(array))
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())

    return digest.hexdigest()


def run_matched_bci_comparison(
    *,
    dataset: str,
    subject: int,
    config: TwinAdaptConfig,
    data_root: Path = Path("data/raw"),
    output_root: Path = Path("outputs/experiments/matched_bci_comparison"),
):
    """Compare two independent SVMs on identical EEG features."""

    if dataset not in ("2a", "2b") or not 1 <= subject <= 9:
        raise ValueError("Invalid BCI dataset or subject")

    root = Path(data_root)

    # STEP 1: Reuse existing BCI feature preparation.
    prepared = prepare(
        dataset,
        subject,
        root / "bci_competition_iv_2a",
        root / "bci_competition_iv_2a_labels",
        root / "bci_competition_iv_2b",
        root / "bci_competition_iv_2b_labels",
        config.seed,
    )

    if len(prepared) == 6:
        x_train, y_train = prepared[0], prepared[1]
        x_eval, y_eval = prepared[4], prepared[5]
    elif len(prepared) == 4:
        x_train, y_train, x_eval, y_eval = (
            prepared[0],
            prepared[1],
            prepared[2],
            prepared[3],
        )
    else:
        raise ValueError("prepare() returned an unexpected number of arrays")

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_eval = np.asarray(x_eval)
    y_eval = np.asarray(y_eval)

    data_hash = _fingerprint(x_train, y_train, x_eval, y_eval)

    # STEP 2: Fit two independent SVMs from the same data.
    frozen_model = build_fresh_model(x_train, y_train, seed=config.seed)

    adaptive_model = build_fresh_model(x_train, y_train, seed=config.seed)

    # STEP 3: Withhold feedback only for the frozen arm.
    frozen_config = replace(config, label_delay=None)

    frozen_result = adapt_model(
        make_bci_dataflow(x_eval, batch_size=config.batch_size),
        frozen_model,
        config=frozen_config,
        evaluation_labels=y_eval,
    )

    # STEP 4: Execute the existing adaptation controller.
    adaptive_result = adapt_model(
        make_bci_dataflow(x_eval, batch_size=config.batch_size),
        adaptive_model,
        config=config,
        evaluation_labels=y_eval,
    )

    assert frozen_result.update_count == 0

    # STEP 5: Compare predictions trial by trial.
    frozen_trials = frozen_result.trials
    adaptive_trials = adaptive_result.trials

    if not np.array_equal(
        frozen_trials["trial_index"].to_numpy(),
        adaptive_trials["trial_index"].to_numpy(),
    ):
        raise ValueError("Comparison trial indices differ")

    paired = pd.DataFrame(
        {
            "trial_index": frozen_trials["trial_index"],
            "true_label": frozen_trials["true_label"],
            "frozen_prediction": frozen_trials["prediction"],
            "adaptive_prediction": adaptive_trials["prediction"],
            "frozen_correct": frozen_trials["correct"],
            "adaptive_correct": adaptive_trials["correct"],
            "adaptive_model_version": adaptive_trials["model_version"],
        }
    )

    frozen_accuracy = float(frozen_result.accuracy)
    adaptive_accuracy = float(adaptive_result.accuracy)

    updates = adaptive_result.updates

    retrain_seconds = (
        float(updates["retrain_seconds"].sum()) if not updates.empty else 0.0
    )

    # STEP 6: Record configuration and outcomes.
    summary = {
        "dataset": dataset,
        "subject": subject,
        "seed": config.seed,
        "data_fingerprint": data_hash,
        "training_trials": len(x_train),
        "evaluation_trials": len(x_eval),
        "frozen_accuracy": frozen_accuracy,
        "adaptive_accuracy": adaptive_accuracy,
        "accuracy_difference_pp": 100 * (adaptive_accuracy - frozen_accuracy),
        "frozen_updates": frozen_result.update_count,
        "adaptive_updates": adaptive_result.update_count,
        "adaptive_retrain_seconds": retrain_seconds,
        "adaptive_config": config.snapshot(),
        "frozen_config": frozen_config.snapshot(),
        "comparison": "matched_frozen_vs_accuracy_triggered",
        "evaluation_type": "offline_feature_level_replay",
    }

    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unavailable"

    summary["commit"] = revision

    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid4().hex[:8]
    )

    destination = Path(output_root) / dataset / f"subject_{subject:02d}" / run_id

    destination.mkdir(parents=True, exist_ok=False)

    # STEP 7: Preserve the per-arm evidence.
    for name, result in (
        ("frozen", frozen_result),
        ("adaptive", adaptive_result),
    ):
        arm_dir = destination / name
        arm_dir.mkdir()

        result.trials.to_csv(arm_dir / "trials.csv", index=False)
        result.accuracy_windows.to_csv(arm_dir / "accuracy_windows.csv", index=False)
        result.transitions.to_csv(arm_dir / "state_transitions.csv", index=False)
        result.adaptive_windows.to_csv(arm_dir / "adaptive_windows.csv", index=False)
        result.updates.to_csv(arm_dir / "updates.csv", index=False)

    paired.to_csv(
        destination / "paired_predictions.csv",
        index=False,
    )

    (destination / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    return destination, summary
