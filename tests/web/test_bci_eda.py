from pathlib import Path
from types import SimpleNamespace

import numpy as np
from flask import Flask

from src.web.bci_eda import _summarise_trials, bci_eda_bp


def _synthetic_trials() -> SimpleNamespace:
    sampling_frequency = 128.0
    samples = 128
    time = np.arange(samples, dtype=float) / sampling_frequency
    trials = []
    labels = []
    for index in range(6):
        left = 8e-6 * np.sin(2.0 * np.pi * 10.0 * time + index * 0.1)
        centre = 5e-6 * np.sin(2.0 * np.pi * 20.0 * time + index * 0.05)
        right = 6e-6 * np.sin(2.0 * np.pi * 10.0 * time + 0.4)
        trials.append(np.stack((left, centre, right), axis=0))
        labels.append(index % 2)
    return SimpleNamespace(
        signals=np.stack(trials),
        labels=np.asarray(labels, dtype=int),
        channel_names=("C3", "Cz", "C4"),
        sampling_frequency=sampling_frequency,
        session_id="01T",
    )


def test_summarise_trials_builds_distribution_metrics() -> None:
    summary = _summarise_trials(
        _synthetic_trials(), label="Session 1 (T)", role="training"
    )

    assert summary["trials"] == 6
    assert summary["channels"] == 3
    assert summary["class_counts"] == {
        "left": 3,
        "right": 3,
        "unlabelled": 0,
        "other": 0,
    }
    assert len(summary["channel_summaries"]) == 3
    assert np.isfinite(summary["_metrics"]["rms_uv"]).all()
    assert np.isfinite(summary["_metrics"]["mu_db"]).all()
    assert np.isfinite(summary["_metrics"]["beta_db"]).all()


def test_data_distributions_page_handles_missing_local_raw_data(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    app = Flask(
        __name__,
        template_folder=str(project_root / "templates"),
        static_folder=str(project_root / "static"),
    )
    app.register_blueprint(bci_eda_bp)
    app.config.update(
        TESTING=True,
        DATASET_2A_PATH=str(tmp_path / "2a"),
        DATASET_2A_LABELS_PATH=str(tmp_path / "2a_labels"),
        DATASET_2B_PATH=str(tmp_path / "2b"),
    )

    response = app.test_client().get("/data-distributions?dataset=2a&subject=1")

    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert "EEG data distributions before shift detection" in page
    assert "No analyzable local sessions were found for A01" in page
    assert "0 / 18" in page
    assert "0 / 9" in page
    assert "0 / 45" in page
