"""Regression tests for development-only cost-aware TWIN-ADAPT optimisation."""
import json

import numpy as np
import pytest

from src.adaptation.twin_adapt.objective import CostObjectiveConfig
from src.adaptation.twin_adapt.optimization import (
    DevelopmentData,
    TwinAdaptSearchSpace,
    calculate_objective_score,
    optimize_adaptation,
    sample_valid_configs,
)
from src.online.batch_stream import make_validation_stream


def fixture_development():
    rng = np.random.default_rng(41)
    train = rng.normal(size=(60, 4))
    validation = rng.normal(size=(80, 4))
    return DevelopmentData(
        x_train=train,
        y_train=(train[:, 0] > 0).astype(int),
        x_validation=validation,
        y_validation=(validation[:, 0] > 0).astype(int),
    )


def fixture_space():
    return TwinAdaptSearchSpace(
        gamma_values=(0.93, 0.95),
        delta_values=(0.80, 0.85),
        win_size_values=(5,),
        win_size_max_values=(40, 80),
        batch_size=20,
        seed=41,
    )


def test_candidate_sampling_is_unique_and_reproducible():
    one = sample_valid_configs(fixture_space(), n=6)
    two = sample_valid_configs(fixture_space(), n=6)
    assert one == two
    assert len(set(one)) == 6
    assert all(0 < c.delta < c.gamma < 1 for c in one)


def test_objective_penalizes_retraining_and_updates():
    objective = CostObjectiveConfig(alpha=0.2, beta=0.02, reference_seconds=1.0)
    low = calculate_objective_score(
        0.8, {"update_rate": 0.01, "retrain_seconds": 0.1}, objective,
    )
    high = calculate_objective_score(
        0.8, {"update_rate": 0.10, "retrain_seconds": 0.9}, objective,
    )
    assert low > high
    assert calculate_objective_score(
        0.8, {"update_rate": 1., "retrain_seconds": 100.},
        CostObjectiveConfig(alpha=0, beta=0),
    ) == 0.8


def test_development_training_labels_must_match():
    data = fixture_development()
    with pytest.raises(ValueError, match="same length"):
        DevelopmentData(
            x_train=data.x_train, y_train=data.y_train[:-1],
            x_validation=data.x_validation, y_validation=data.y_validation,
        )


def test_validation_budget_rejected_when_too_large():
    data = fixture_development()
    with pytest.raises(ValueError, match="budget"):
        make_validation_stream(
            data.x_validation, data.y_validation,
            batch_size=20, budget=81,
        )


def test_multifidelity_search_promotion_and_history():
    best, history = optimize_adaptation(
        fixture_development(),
        fixture_space(),
        budgets=[40, 80],
        initial_candidates=6,
        objective=CostObjectiveConfig(alpha=0, beta=0),
    )
    assert best in sample_valid_configs(fixture_space(), n=6)
    assert len(history) == 8  # Six initial candidates, two promoted finalists.
    assert [sum(row["stage"] == stage for row in history) for stage in (0, 1)] == [6, 2]
    assert sum(row["promoted"] and row["stage"] == 1 for row in history) == 1
    assert all(row["budget"] in (40, 80) for row in history)
    assert all(np.isfinite(row["score"]) for row in history)


def test_invalid_budget_sequence_rejected():
    with pytest.raises(ValueError, match="strictly increasing"):
        optimize_adaptation(
            fixture_development(), fixture_space(),
            budgets=[80, 40], initial_candidates=2,
        )


def test_optimized_bci_writes_independent_evaluation_and_search_evidence(monkeypatch, tmp_path):
    from scripts import run_online_bci
    from src.adaptation.experiments.twin_adapt_bci import run_optimized_bci_experiment

    rng = np.random.default_rng(13)
    x_train = rng.normal(size=(60, 4))
    x_val = rng.normal(size=(80, 4))
    x_eval = rng.normal(size=(80, 4))
    calls = []

    def prepared_data(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs["include_validation"] is True
        return (
            x_train, (x_train[:, 0] > 0).astype(int),
            x_val, (x_val[:, 0] > 0).astype(int),
            x_eval, (x_eval[:, 0] > 0).astype(int),
        )

    monkeypatch.setattr(run_online_bci, "prepare", prepared_data)
    dest, summary = run_optimized_bci_experiment(
        dataset="2a", subject=1,
        search_space=fixture_space(),
        budgets=[40, 80],
        initial_candidates=6,
        objective=CostObjectiveConfig(alpha=0, beta=0),
        data_root=tmp_path / "raw", output_root=tmp_path / "out",
    )
    assert calls and len(calls) == 1
    assert summary["candidate_evaluations"] == 8
    assert summary["evaluation_trials"] == 80
    assert summary["validation_trials"] == 80
    assert (dest / "candidate_history.csv").is_file()
    assert (dest / "best_config.json").is_file()
    assert (dest / "final_evaluation" / "trials.csv").is_file()
    assert json.loads((dest / "development_split.json").read_text())[
        "evaluation_used_in_search"
    ] is False
    assert not (tmp_path / "out" / "latest_summary.csv").exists()


def test_bci_results_page_shows_committed_a01_b01_evidence(monkeypatch):
    """The page must visualise saved genuine BCI data, not fabricate search results."""
    from src.web import app
    monkeypatch.setitem(app.config, "TESTING", True)
    monkeypatch.setitem(app.config, "RESULTS_ROOT", "outputs")
    with app.test_client() as client:
        response = client.get("/results/twin-adapt")
    assert response.status_code == 200
    assert b"88.19%" in response.data
    assert b"66.25%" in response.data
    assert b"Measured accuracy trajectory" in response.data
    assert b"Saved real-BCI optimisation runs" in response.data


def test_optimisation_form_calls_real_runner_and_shows_saved_evidence(monkeypatch, tmp_path):
    """Mock only the expensive data-processing boundary, not the view's result loader."""
    from src.web import app
    from src.web import twin_adapt as twin_adapt_view

    root = tmp_path / "outputs"
    called = []
    def fake_optimised_runner(**kwargs):
        called.append(kwargs)
        dest = (
            root / "experiments" / "twin_adapt_optimization"
            / "2a" / "subject_01" / "fixture_run"
        )
        dest.mkdir(parents=True)
        summary = {
            "run_id": "fixture_run", "dataset": "2a", "subject": 1,
            "accuracy": 0.75, "updates": 1,
            "search_seconds": 1.25, "candidate_evaluations": 2,
            "best_config": {"gamma": 0.95, "delta": 0.85},
            "objective_config": {"alpha": 0.2, "beta": 0.02},
        }
        (dest / "optimization_summary.json").write_text(json.dumps(summary))
        import pandas as pd
        pd.DataFrame([{
            "stage": 0, "budget": 20, "accuracy": 0.75,
            "updates": 1, "retrain_seconds": 0.05, "replay_seconds": 0.2,
            "score": 0.74, "promoted": True, "candidate": "{}",
        }]).to_csv(dest / "candidate_history.csv", index=False)
        return dest, summary

    monkeypatch.setattr(twin_adapt_view, "run_optimized_bci_experiment", fake_optimised_runner)
    monkeypatch.setitem(app.config, "TESTING", True)
    monkeypatch.setitem(app.config, "RESULTS_ROOT", str(root))
    with app.test_client() as client:
        response = client.post("/results/twin-adapt", data={
            "mode": "optimized", "dataset": "2a", "subject": "1",
            "opt_batch_size": "10", "budgets": "20 40",
            "candidates": "2", "seed": "42",
            "alpha": "0.2", "beta": "0.02",
            "reference_seconds": "1", "promotion_fraction": "0.5",
            "label_delay": "0",
        })
    assert response.status_code == 200
    assert len(called) == 1
    assert called[0]["budgets"] == [20, 40]
    assert b"fixture_run" in response.data
    assert b"Selected configuration and candidate history" in response.data
