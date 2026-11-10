"""Actual BCI TWIN-ADAPT execution and saved-evidence viewer."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from flask import Blueprint, current_app, render_template, request

from src.adaptation.experiments.twin_adapt_bci import (
    run_bci_experiment, run_optimized_bci_experiment,
)
from src.adaptation.twin_adapt.objective import CostObjectiveConfig
from src.adaptation.twin_adapt.optimization import TwinAdaptSearchSpace
from src.adaptation.twin_adapt.config import TwinAdaptConfig

twin_adapt_bp = Blueprint("twin_adapt", __name__)


def _runs(root: Path) -> list[dict]:
    base = root / "experiments" / "twin_adapt_bci"
    entries = []
    for path in sorted(base.glob("2*/subject_*/*/summary.json"), reverse=True)[:80]:
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
            if not {"dataset", "subject", "accuracy", "updates", "run_id"}.issubset(summary):
                continue
            summary["relative_dir"] = path.parent.relative_to(root).as_posix()
            entries.append(summary)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return entries


def _trace(root: Path, entry: dict | None) -> dict:
    if entry is None:
        return {}
    directory = root / entry["relative_dir"]
    result = {}
    for key, name in (
        ("windows", "accuracy_windows.csv"),
        ("transitions", "state_transitions.csv"),
        ("updates", "updates.csv"),
        ("buffers", "adaptive_windows.csv"),
    ):
        file = directory / name
        if not file.is_file():
            result[key] = []
            continue
        try:
            result[key] = pd.read_csv(file).tail(100).where(
                lambda x: pd.notna(x), None
            ).to_dict(orient="records")
        except (OSError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError):
            result[key] = []
    return result


def _local_dataset_readiness(raw_root: Path) -> dict:
    """Report local data presence only; never infer GDF files from archived CSVs."""
    checks = {}
    for dataset in ("2a", "2b"):
        recordings = raw_root / f"bci_competition_iv_{dataset}"
        labels = raw_root / f"bci_competition_iv_{dataset}_labels"
        gdf_count = sum(1 for _ in recordings.rglob("*.gdf")) if recordings.is_dir() else 0
        mat_count = sum(1 for _ in labels.rglob("*.mat")) if labels.is_dir() else 0
        checks[dataset] = {
            "recordings": str(recordings),
            "labels": str(labels),
            "gdf_count": gdf_count,
            "label_count": mat_count,
            "ready": gdf_count > 0 and mat_count > 0,
        }
    return checks


def _optimized_runs(root: Path) -> list[dict]:
    """Read only genuine saved optimisation evidence, never synthetic placeholders."""
    base = root / "experiments" / "twin_adapt_optimization"
    entries = []
    for path in sorted(base.glob("2*/subject_*/*/optimization_summary.json"), reverse=True)[:80]:
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            if not {"dataset", "subject", "accuracy", "updates", "run_id"}.issubset(row):
                continue
            row["relative_dir"] = path.parent.relative_to(root).as_posix()
            entries.append(row)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return entries


def _paired_real_bci_results(
    fixed_runs: list[dict], optimized_runs: list[dict],
) -> list[dict]:
    """Descriptive baseline-vs-search comparison with explicit protocol checks."""
    seen = set()
    rows = []
    for optimized in optimized_runs:
        key = (optimized["dataset"], optimized["subject"])
        if key in seen:
            continue
        seen.add(key)
        fixed = next(
            (row for row in fixed_runs
             if (row["dataset"], row["subject"]) == key), None
        )
        if fixed is None:
            continue
        config = optimized.get("best_config") or {}
        original = fixed.get("config") or {}
        aligned = all((
            optimized.get("training_trials") == fixed.get("training_trials"),
            optimized.get("evaluation_trials") == fixed.get("evaluation_trials"),
            config.get("batch_size") == original.get("batch_size"),
            config.get("label_delay") == original.get("label_delay"),
            config.get("seed") == original.get("seed"),
        ))
        rows.append({
            "dataset": optimized["dataset"],
            "subject": optimized["subject"],
            "fixed_accuracy": fixed["accuracy"],
            "optimized_accuracy": optimized["accuracy"],
            "fixed_updates": fixed["updates"],
            "optimized_updates": optimized["updates"],
            "fixed_retrain": fixed.get("retrain_seconds", 0.0),
            "optimized_retrain": optimized.get("retrain_seconds", 0.0),
            "difference_pp": 100 * (optimized["accuracy"] - fixed["accuracy"]),
            "protocol_aligned": aligned,
            "fixed_run": fixed["run_id"],
            "optimized_run": optimized["run_id"],
        })
    return rows


def _optimization_history(root: Path, entry: dict | None) -> list[dict]:
    if entry is None:
        return []
    file = root / entry["relative_dir"] / "candidate_history.csv"
    if not file.is_file():
        return []
    try:
        frame = pd.read_csv(file)
        frame = frame.tail(120)
        return frame.where(pd.notna(frame), None).to_dict(orient="records")
    except (OSError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return []


def _accuracy_plot(windows: list[dict]) -> dict:
    """Plot only stored measured accuracy windows and thresholds."""
    valid = []
    for row in windows:
        try:
            current = float(row["curr_accuracy"])
            warning = float(row["warning_threshold"])
            batch = int(row["batch"])
        except (KeyError, ValueError, TypeError):
            continue
        if not 0 <= current <= 1 or not 0 <= warning <= 1:
            continue
        valid.append((batch, current, warning))
    if not valid:
        return {"current": "", "warning": "", "count": 0}
    step = 540 / max(len(valid) - 1, 1)
    def points(pos):
        return " ".join(
            f"{30 + i * step:.1f},{180 - 155 * row[pos]:.1f}"
            for i, row in enumerate(valid)
        )
    return {"current": points(1), "warning": points(2), "count": len(valid)}


@twin_adapt_bp.route("/results/twin-adapt", methods=["GET", "POST"])
def results():
    root = Path(current_app.config.get("RESULTS_ROOT", "outputs"))
    defaults = TwinAdaptConfig()
    values = defaults.snapshot()
    values.update(dataset="2a", subject=1)
    optimized_values = {
        "dataset": "2a", "subject": 1, "batch_size": 10,
        "budgets": "20 40", "candidates": 6, "seed": 42,
        "alpha": 0.2, "beta": 0.02, "reference_seconds": 1.0,
        "promotion_fraction": 1 / 3, "label_delay": 0,
    }
    error = None
    selected_id = request.args.get("run")
    selected_opt_id = request.args.get("optimized_run")
    if request.method == "POST":
        try:
            data_root = Path(current_app.config.get("BCI_RAW_DATA_ROOT", "data/raw"))
            if request.form.get("mode") == "optimized":
                dataset = request.form.get("dataset", "2a")
                subject = int(request.form.get("subject", 1))
                batch_size = int(request.form.get("opt_batch_size", 10))
                budgets = [
                    int(item) for item in request.form.get("budgets", "20 40")
                    .replace(",", " ").split()
                ]
                candidates = int(request.form.get("candidates", 6))
                seed = int(request.form.get("seed", 42))
                alpha = float(request.form.get("alpha", 0.2))
                beta = float(request.form.get("beta", 0.02))
                reference = float(request.form.get("reference_seconds", 1.0))
                promotion = float(request.form.get("promotion_fraction", 1 / 3))
                delay = int(request.form.get("label_delay", 0))
                optimized_values.update({
                    "dataset": dataset, "subject": subject,
                    "batch_size": batch_size, "budgets": " ".join(map(str, budgets)),
                    "candidates": candidates, "seed": seed, "alpha": alpha,
                    "beta": beta, "reference_seconds": reference,
                    "promotion_fraction": promotion, "label_delay": delay,
                })
                if not 1 <= candidates <= 81:
                    raise ValueError("Candidate count must be between 1 and 81")
                if not budgets or len(budgets) > 8:
                    raise ValueError("Specify 1 to 8 validation budgets")
                if len(budgets) > 1 and not 0 < promotion <= 1:
                    raise ValueError("Invalid promotion fraction")
                space = TwinAdaptSearchSpace(
                    batch_size=batch_size,
                    label_delay=None if delay == -1 else delay,
                    seed=seed,
                )
                objective = CostObjectiveConfig(
                    alpha=alpha, beta=beta, reference_seconds=reference,
                )
                target, _ = run_optimized_bci_experiment(
                    dataset=dataset, subject=subject, search_space=space,
                    budgets=budgets, objective=objective,
                    initial_candidates=candidates,
                    promotion_fraction=promotion, data_root=data_root,
                    output_root=root / "experiments" / "twin_adapt_optimization",
                )
                selected_opt_id = target.name
            else:
                dataset = request.form.get("dataset", "2a")
                subject = int(request.form.get("subject", 1))
                kwargs = {key: int(request.form.get(key, getattr(defaults, key)))
                          for key in ("batch_size", "win_size", "win_size_max", "seed")}
                kwargs.update({key: float(request.form.get(key, getattr(defaults, key)))
                              for key in ("gamma", "delta")})
                delay = int(request.form.get("label_delay", 0))
                kwargs["label_delay"] = None if delay == -1 else delay
                config = TwinAdaptConfig(**kwargs)
                values.update(config.snapshot(), dataset=dataset, subject=subject)
                target, _ = run_bci_experiment(
                    dataset=dataset, subject=subject, config=config,
                    data_root=data_root,
                    output_root=root / "experiments" / "twin_adapt_bci",
                )
                selected_id = target.name
        except (ValueError, TypeError, OSError, RuntimeError, ImportError) as exc:
            error = str(exc)
    runs = _runs(root)
    selected = next((row for row in runs if row["run_id"] == selected_id), None)
    if selected is None and runs:
        selected = runs[0]
    trace = _trace(root, selected)
    raw_root = Path(current_app.config.get("BCI_RAW_DATA_ROOT", "data/raw"))
    data_readiness = _local_dataset_readiness(raw_root)
    optimized_runs = _optimized_runs(root)
    selected_opt = next(
        (row for row in optimized_runs if row["run_id"] == selected_opt_id), None
    )
    if selected_opt is None and optimized_runs:
        selected_opt = optimized_runs[0]
    fixed_reference = next(
        (row for row in runs if selected_opt is not None
         and row["dataset"] == selected_opt["dataset"]
         and row["subject"] == selected_opt["subject"]), None
    )
    return render_template(
        "twin_adapt.html", values=values, runs=runs, selected=selected,
        trace=trace, error=error, accuracy_plot=_accuracy_plot(trace.get("windows", [])),
        optimized_runs=optimized_runs, selected_opt=selected_opt,
        optimized_values=optimized_values,
        optimization_history=_optimization_history(root, selected_opt),
        data_readiness=data_readiness, fixed_reference=fixed_reference,
        paired_results=_paired_real_bci_results(runs, optimized_runs),
    ), (400 if error else 200)
