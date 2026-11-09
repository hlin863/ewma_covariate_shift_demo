"""Flask TWIN-ADAPT page and real BCI orchestration boundary tests."""
import json

import numpy as np
import pandas as pd

from src.web import app


def test_page_no_saved_evidence(tmp_path):
    app.config.update(TESTING=True, RESULTS_ROOT=str(tmp_path))
    with app.test_client() as client:
        response = client.get("/results/twin-adapt")
    assert response.status_code == 200
    assert b"No BCI TWIN-ADAPT runs saved yet" in response.data


def test_page_reads_saved_measured_evidence(tmp_path):
    root = tmp_path / "experiments" / "twin_adapt_bci" / "2a" / "subject_01" / "run123"
    root.mkdir(parents=True)
    (root / "summary.json").write_text(json.dumps({
        "dataset": "2a", "subject": 1, "run_id": "run123",
        "accuracy": 0.75, "updates": 1, "retrain_seconds": 0.3,
        "classifier": "linear_svm", "feature_count": 3, "training_trials": 50,
        "evaluation_trials": 40, "commit": "test",
        "interpretation": "operational", "pending_feedback_batches": 0,
    }))
    pd.DataFrame([{
        "batch": 0, "released_at_batch": 0, "prev_accuracy": 0.8,
        "curr_accuracy": 0.6, "warning_threshold": 0.76,
        "adaptation_threshold": 0.68, "action": "initial_update",
    }]).to_csv(root / "accuracy_windows.csv", index=False)
    app.config.update(TESTING=True, RESULTS_ROOT=str(tmp_path))
    with app.test_client() as client:
        response = client.get("/results/twin-adapt?run=run123")
    assert response.status_code == 200
    assert b"75.00%" in response.data
    assert b"initial_update" in response.data


def test_form_invalid_thresholds_do_not_run(tmp_path):
    app.config.update(TESTING=True, RESULTS_ROOT=str(tmp_path))
    with app.test_client() as client:
        response = client.post("/results/twin-adapt", data={
            "dataset": "2a", "subject": "1", "gamma": "0.7", "delta": "0.9"
        })
    assert response.status_code == 400
    assert b"Require 0 &lt; delta &lt; gamma &lt; 1" in response.data
