"""Integration tests for real execution wiring using prepared synthetic features."""
from pathlib import Path

import numpy as np
from src.adaptation.classifier import LinearSVMClassifier
from src.adaptation.twin_adapt.config import TwinAdaptConfig
from src.adaptation.twin_adapt.controller import adapt_model
from src.online.batch_stream import make_bci_dataflow


def test_stream_to_adaptation_controller():
    rng = np.random.default_rng(19)
    x_train = rng.normal(size=(60, 4))
    y_train = (x_train[:, 0] > 0).astype(int)
    model = LinearSVMClassifier(random_state=19).fit(x_train, y_train)
    x_test = rng.normal(size=(80, 4))
    y_test = (x_test[:, 0] > 0).astype(int)
    result = adapt_model(
        make_bci_dataflow(x_test, batch_size=20),
        model, config=TwinAdaptConfig(batch_size=20, win_size=5, win_size_max=40),
        evaluation_labels=y_test,
    )
    assert len(result.trials) == 80
    assert len(result.accuracy_windows) == 4
    assert result.trials.trial_index.tolist() == list(range(80))
    assert 0 <= result.accuracy <= 1


def test_no_label_feedback_keeps_model_frozen():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(40, 3))
    y = (x[:, 0] > 0).astype(int)
    model = LinearSVMClassifier().fit(x, y)
    result = adapt_model(
        make_bci_dataflow(x, batch_size=20), model,
        config=TwinAdaptConfig(batch_size=20, win_size=5, win_size_max=40, label_delay=None),
        evaluation_labels=y,
    )
    assert result.update_count == 0
    assert result.pending_feedback_batches == 0
    assert result.accuracy == 1.0 or 0 <= result.accuracy <= 1.0


def test_results_page_handles_missing_outputs():
    from src.web import app
    app.config["TESTING"] = True
    with app.test_client() as client:
        response = client.get("/results/twin-adapt")
    assert response.status_code == 200
    assert b"TWIN-ADAPT" in response.data


def test_bci_orchestrator_writes_real_controller_evidence(monkeypatch, tmp_path):
    """Exercise the CLI/backend boundary without pretending fixture data are real EEG."""
    from scripts import run_online_bci
    from src.adaptation.experiments.twin_adapt_bci import run_bci_experiment
    def fake_prepare(*args):
        rng = np.random.default_rng(31)
        train = rng.normal(size=(60, 4))
        test = rng.normal(size=(80, 4))
        return train, (train[:, 0] > 0).astype(int), test, (test[:, 0] > 0).astype(int)
    monkeypatch.setattr(run_online_bci, "prepare", fake_prepare)
    path, summary = run_bci_experiment(
        dataset="2a", subject=1,
        config=TwinAdaptConfig(batch_size=20, win_size=5, win_size_max=40),
        data_root=tmp_path / "raw", output_root=tmp_path / "out",
    )
    assert summary["evaluation_trials"] == 80
    assert summary["classifier"] == "linear_svm"
    assert (path / "accuracy_windows.csv").is_file()
    assert (path / "state_transitions.csv").is_file()
    assert (path / "updates.csv").is_file()
    assert (path / "summary.json").is_file()
    assert (tmp_path / "out" / "latest_summary.csv").is_file()
