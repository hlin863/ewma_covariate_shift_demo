"""Output-layout regression tests: figures should accompany their source run."""
import json

import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from src.adaptation.experiments.run_figures import export_run_figures


def test_optimized_run_figures_and_manifest(tmp_path):
    run = tmp_path / "optimization_run"
    final = run / "final_evaluation"
    final.mkdir(parents=True)
    pd.DataFrame([
        {"stage": 0, "budget": 40, "score": 0.725,
         "prediction_diversity": 0.0},
        {"stage": 0, "budget": 40, "score": 0.725,
         "prediction_diversity": 0.0},
    ]).to_csv(run / "candidate_history.csv", index=False)
    pd.DataFrame([
        {"batch": 0, "prev_accuracy": 0.7, "curr_accuracy": 0.5,
         "warning_threshold": 0.665, "adaptation_threshold": 0.63},
        {"batch": 1, "prev_accuracy": 0.8, "curr_accuracy": 0.6,
         "warning_threshold": 0.76, "adaptation_threshold": 0.72},
    ]).to_csv(final / "accuracy_windows.csv", index=False)
    pd.DataFrame([
        {"batch": 0, "version": 1, "reason": "initial_severe_accuracy_drop"}
    ]).to_csv(final / "updates.csv", index=False)

    source_before = (run / "candidate_history.csv").read_bytes()
    manifest = export_run_figures(run)

    assert manifest["kind"] == "twin_adapt_optimization"
    assert manifest["figure_count"] == 4
    assert {a["name"] for a in manifest["artifacts"]} == {
        "accuracy_windows", "update_events",
        "candidate_scores", "prediction_diversity",
    }
    for item in manifest["artifacts"]:
        for filename in item["files"]:
            assert (run / "figures" / filename).stat().st_size > 0
        for source in item["source_files"]:
            assert (run / source).exists()
    assert (run / "candidate_history.csv").read_bytes() == source_before
    assert json.loads((run / "figures" / "manifest.json").read_text()) == manifest

    # Rebuilding never moves or changes the underlying experiment evidence.
    assert export_run_figures(run) == manifest


def test_fixed_run_without_updates_has_explicit_empty_chart(tmp_path):
    run = tmp_path / "fixed_run"
    run.mkdir()
    pd.DataFrame([
        {"batch": 0, "prev_accuracy": 0.75, "curr_accuracy": 0.75,
         "warning_threshold": 0.71, "adaptation_threshold": 0.67},
    ]).to_csv(run / "accuracy_windows.csv", index=False)
    (run / "updates.csv").write_text("", encoding="utf-8")

    manifest = export_run_figures(run)

    assert manifest["kind"] == "twin_adapt_bci"
    assert manifest["figure_count"] == 2
    assert (run / "figures" / "update_events.svg").exists()
    assert (run / "figures" / "accuracy_windows.png").exists()


def test_incomplete_run_is_rejected_without_inventing_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="accuracy_windows.csv"):
        export_run_figures(tmp_path)
