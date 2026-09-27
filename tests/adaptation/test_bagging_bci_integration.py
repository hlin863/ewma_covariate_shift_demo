from pathlib import Path

import pandas as pd

from scripts.run_bagging_bci_synthetic import run_and_write


def test_bagging_integrates_with_synthetic_2a_and_2b(tmp_path: Path) -> None:
    paths = run_and_write(tmp_path)

    frame = pd.read_csv(paths["metrics"])

    assert set(frame["dataset"]) == {"2A", "2B"}
    assert set(frame["method"]) == {
        "single_linear_svm",
        "bagged_linear_svm",
    }
    assert frame.shape[0] == 4
    assert frame["accuracy"].between(0.0, 1.0).all()
    assert frame["fit_seconds"].ge(0.0).all()
    assert frame["predict_seconds"].ge(0.0).all()

    two_a = frame.loc[frame["dataset"] == "2A"]
    two_b = frame.loc[frame["dataset"] == "2B"]
    assert set(two_a["training_trials"]) == {48}
    assert set(two_a["testing_trials"]) == {144}
    assert set(two_b["training_trials"]) == {24}
    assert set(two_b["testing_trials"]) == {16}

    bagged = frame.loc[frame["method"] == "bagged_linear_svm"]
    assert (bagged["requested_estimators"] == 30).all()
    assert bagged["fitted_estimators"].between(1, 30).all()
    assert (bagged["sample_fraction"] == 0.8).all()
    assert bagged["mean_member_disagreement"].between(0.0, 1.0).all()

    for key in ("accuracy_figure", "disagreement_figure"):
        assert paths[key].is_file()
        assert paths[key].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
