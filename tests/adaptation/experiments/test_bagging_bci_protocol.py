from pathlib import Path

import numpy as np

import src.adaptation.experiments.bagging_bci as bagging_bci
from src.bci.data import BCISessionData


def _session(session_number: int) -> BCISessionData:
    session = f"{session_number:02d}T"
    rng = np.random.default_rng(100 + session_number)
    sampling_frequency = 100.0
    signals = rng.normal(scale=0.4, size=(5000, 3))
    annotations = []
    for index, onset in enumerate((2, 7, 12, 17, 22, 27, 32, 37)):
        code = "769" if index % 2 == 0 else "770"
        start = int(onset * sampling_frequency)
        stop = start + int(3 * sampling_frequency)
        time = np.arange(stop - start) / sampling_frequency
        channel = 0 if index % 2 == 0 else 2
        signals[start:stop, channel] += 2.0 * np.sin(2 * np.pi * 10.0 * time)
        annotations.append((float(onset), 0.0, code))
    return BCISessionData(
        subject=1,
        session=session,
        file_path=Path(f"B01{session}.gdf"),
        signals=signals,
        times=np.arange(signals.shape[0]) / sampling_frequency,
        channel_names=("C3", "Cz", "C4"),
        sampling_frequency=sampling_frequency,
        annotations=tuple(annotations),
    )


def test_dataset_2b_bagging_defaults_to_gdf_only_session3_holdout(monkeypatch) -> None:
    monkeypatch.setattr(
        bagging_bci,
        "load_bci_competition_iv_2b_session",
        lambda data_directory, subject, session: _session(session),
    )

    rows = bagging_bci.run_dataset_2b_bagging(
        data_directory="unused",
        labels_directory=None,
        subject=1,
        n_estimators=5,
    )

    assert len(rows) == 2
    assert {row["method"] for row in rows} == {
        "single_linear_svm",
        "bagged_linear_svm",
    }
    assert {row["evaluation_scope"] for row in rows} == {
        "Session III (GDF-only holdout)"
    }
    assert {row["training_trials"] for row in rows} == {16}
    assert {row["testing_trials"] for row in rows} == {8}
    assert {row["calibration_trials"] for row in rows} == {0}
    assert all(0.0 <= row["accuracy"] <= 1.0 for row in rows)
