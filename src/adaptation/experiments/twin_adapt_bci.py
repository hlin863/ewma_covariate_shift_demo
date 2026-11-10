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
        "warning_events": int((
            result.accuracy_windows["curr_accuracy"]
            < result.accuracy_windows["warning_threshold"]
        ).sum()) if not result.accuracy_windows.empty else 0,
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
    # Stable catalogue entry; individual run folders remain immutable.
    pd.DataFrame([{k: v for k, v in summary.items() if k != "config"}]).to_csv(
        Path(output_root) / "latest_summary.csv", index=False
    )
    return destination, summary



def run_optimized_bci_experiment(
    *, dataset: str, subject: int, search_space, budgets,
    objective=None, initial_candidates: int = 12,
    promotion_fraction: float = 1 / 3,
    data_root: Path = DEFAULT_DATA,
    output_root: Path = Path("outputs/experiments/twin_adapt_optimization"),
) -> tuple[Path, dict]:
    """Select settings on development data, then perform one held-out BCI replay.

    This is a separate experiment path: the original fixed-config runner and
    historical outputs are intentionally unchanged.
    """
    from time import perf_counter
    from src.adaptation.classifier import build_fresh_model
    from src.adaptation.experiments.bci_optimisation_protocol import (
        prepare_bci_optimization_protocol,
    )
    from src.adaptation.twin_adapt.objective import CostObjectiveConfig
    from src.adaptation.twin_adapt.optimization import optimize_adaptation

    if dataset not in ("2a", "2b") or not 1 <= subject <= 9:
        raise ValueError("dataset must be 2a/2b and subject must be 1..9")
    if objective is None:
        objective = CostObjectiveConfig()

    prepared = prepare_bci_optimization_protocol(
        dataset=dataset, subject=subject, data_root=data_root, seed=search_space.seed,
    )
    started = perf_counter()
    best_config, history = optimize_adaptation(
        development_data=prepared.development,
        config_space=search_space,
        budgets=budgets,
        initial_candidates=initial_candidates,
        promotion_fraction=promotion_fraction,
        objective=objective,
    )
    search_seconds = perf_counter() - started

    # No reuse of a candidate's mutated fitted model; no final-evaluation data
    # was accessed during configuration search.
    model = build_fresh_model(
        prepared.development.x_train, prepared.development.y_train,
        seed=best_config.seed,
    )
    stream = make_bci_dataflow(
        prepared.x_evaluation, batch_size=best_config.batch_size,
    )
    final_result = adapt_model(
        stream, model, config=best_config,
        evaluation_labels=prepared.y_evaluation,
    )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid4().hex[:8]
    destination = Path(output_root) / dataset / f"subject_{subject:02d}" / run_id
    destination.mkdir(parents=True, exist_ok=False)

    def save_json(name, value):
        (destination / name).write_text(
            json.dumps(value, indent=2, allow_nan=False), encoding="utf-8",
        )

    pd.DataFrame(history).assign(
        candidate=lambda df: df["candidate"].map(json.dumps)
    ).to_csv(destination / "candidate_history.csv", index=False)
    save_json("search_space.json", {
        "gamma_values": search_space.gamma_values,
        "delta_values": search_space.delta_values,
        "win_size_values": search_space.win_size_values,
        "win_size_max_values": search_space.win_size_max_values,
        "batch_size": search_space.batch_size,
        "label_delay": search_space.label_delay,
        "seed": search_space.seed,
    })
    save_json("objective_config.json", asdict(objective))
    save_json("best_config.json", best_config.snapshot())
    save_json("development_split.json", {
        "dataset": dataset,
        "subject": subject,
        "training_trials": len(prepared.development.y_train),
        "validation_trials": len(prepared.development.y_validation),
        "evaluation_trials": len(prepared.y_evaluation),
        "validation_protocol": "seeded stratified development holdout, ordered by original trial indices",
        "fbcsp_fit": "development training only",
        "evaluation_used_in_search": False,
        "selected_budgets": list(budgets),
    })

    final_dir = destination / "final_evaluation"
    final_dir.mkdir()
    for name, table in (
        ("trials.csv", final_result.trials),
        ("accuracy_windows.csv", final_result.accuracy_windows),
        ("state_transitions.csv", final_result.transitions),
        ("adaptive_windows.csv", final_result.adaptive_windows),
        ("updates.csv", final_result.updates),
    ):
        table.to_csv(final_dir / name, index=False)

    summary = {
        "run_id": run_id,
        "dataset": dataset,
        "subject": subject,
        "accuracy": float(final_result.accuracy),
        "updates": final_result.update_count,
        "retrain_seconds": float(final_result.updates["retrain_seconds"].sum())
            if not final_result.updates.empty else 0.0,
        "search_seconds": float(search_seconds),
        "candidate_evaluations": len(history),
        "best_config": best_config.snapshot(),
        "objective_config": asdict(objective),
        "training_trials": len(prepared.development.y_train),
        "validation_trials": len(prepared.development.y_validation),
        "evaluation_trials": len(prepared.y_evaluation),
        "interpretation": "operational TWIN-ADAPT; development-only configuration search",
    }
    save_json("optimization_summary.json", summary)
    (final_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    return destination, summary
