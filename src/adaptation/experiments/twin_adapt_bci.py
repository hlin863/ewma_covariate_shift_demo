"""BCI IV 2A/2B TWIN-ADAPT experiment orchestration and evidence exports.

All feature preparation is delegated to the existing online BCI runner.
Labels are used by the controller only at configured batch-release times.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from uuid import uuid4

import numpy as np
import pandas as pd

from src.adaptation.classifier import LinearSVMClassifier
from src.adaptation.twin_adapt.config import TwinAdaptConfig
from src.adaptation.twin_adapt.controller import adapt_model
from src.online.batch_stream import make_bci_dataflow


DEFAULT_DATA = Path("data/raw")
DEFAULT_OUTPUT = Path("outputs/experiments/twin_adapt_bci")


def run_bci_experiment(
    *, dataset: str, subject: int, config: TwinAdaptConfig,
    data_root: Path = DEFAULT_DATA, output_root: Path = DEFAULT_OUTPUT,
) -> tuple[Path, dict]:
    """Execute actual GDF/MAT feature replay and save an isolated run."""
    if dataset not in ("2a", "2b") or not 1 <= subject <= 9:
        raise ValueError("dataset must be 2a/2b and subject must be 1..9")
    # Keep the same session split and fitted FBCSP used by run_online_bci.
    from scripts.run_online_bci import prepare
    root = Path(data_root)
    x_train, y_train, x_eval, y_eval = prepare(
        dataset, subject,
        root / "bci_competition_iv_2a",
        root / "bci_competition_iv_2a_labels",
        root / "bci_competition_iv_2b",
        root / "bci_competition_iv_2b_labels",
        config.seed,
    )
    x_train, x_eval = np.asarray(x_train), np.asarray(x_eval)
    y_train, y_eval = np.asarray(y_train), np.asarray(y_eval)
    if x_train.ndim != 2 or x_eval.ndim != 2 or x_train.shape[1] != x_eval.shape[1]:
        raise ValueError("Development/evaluation feature dimensions differ")
    if len(x_eval) != len(y_eval) or len(x_train) != len(y_train):
        raise ValueError("Features and official labels are not aligned")
    model = LinearSVMClassifier(random_state=config.seed).fit(x_train, y_train)
    stream = make_bci_dataflow(x_eval, batch_size=config.batch_size)
    result = adapt_model(stream, model, config=config, evaluation_labels=y_eval)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid4().hex[:8]
    destination = Path(output_root) / dataset / f"subject_{subject:02d}" / run_id
    destination.mkdir(parents=True, exist_ok=False)
    tables = {
        "trials.csv": result.trials,
        "accuracy_windows.csv": result.accuracy_windows,
        "state_transitions.csv": result.transitions,
        "adaptive_windows.csv": result.adaptive_windows,
        "updates.csv": result.updates,
    }
    for filename, frame in tables.items():
        frame.to_csv(destination / filename, index=False)
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unavailable"
    summary = {
        "dataset": dataset, "subject": subject, "run_id": run_id,
        "classifier": "linear_svm", "training_trials": int(len(x_train)),
        "evaluation_trials": int(len(x_eval)), "feature_count": int(x_eval.shape[1]),
        "accuracy": float(result.accuracy), "updates": result.update_count,
        "warning_events": int(result.accuracy_windows["action"].eq("warning_started").sum())
            if not result.accuracy_windows.empty else 0,
        "state_transitions": int(len(result.transitions)),
        "pending_feedback_batches": result.pending_feedback_batches,
        "retrain_seconds": float(result.updates["retrain_seconds"].sum())
            if not result.updates.empty else 0.0,
        "commit": revision, "interpretation": "operational TWIN-ADAPT; feature-level replay",
        "label_protocol": "official evaluation truth released to controller after full prediction batch",
        "config": config.snapshot(),
        "dataset_protocol": "2A: session I development/session II evaluation; 2B: sessions I-III development/IV-V evaluation",
    }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame([{k: v for k, v in summary.items() if k != "config"}]).to_csv(
        destination / "summary.csv", index=False
    )
    return destination, summary
