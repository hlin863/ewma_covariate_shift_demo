from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from src.bci.datasets.dataset2b.evaluation_labels import (
    dataset_2b_evaluation_label_filename,
    load_dataset_2b_evaluation_labels,
    resolve_dataset_2b_evaluation_label_path,
)


def test_dataset_2b_label_filename_uses_evaluation_session_number() -> None:
    assert dataset_2b_evaluation_label_filename(1, 4) == "B0104E.mat"
    assert dataset_2b_evaluation_label_filename(9, 5) == "B0905E.mat"


def test_dataset_2b_label_loader_converts_official_1_2_encoding(tmp_path: Path) -> None:
    path = tmp_path / "B0104E.mat"
    labels = np.tile(np.asarray([1, 2], dtype=int), 80)
    savemat(path, {"classlabel": labels.reshape(-1, 1)})

    loaded = load_dataset_2b_evaluation_labels(path)

    assert loaded.shape == (160,)
    assert set(np.unique(loaded)) == {0, 1}
    np.testing.assert_array_equal(loaded[:4], np.asarray([0, 1, 0, 1]))


def test_dataset_2b_label_resolver_checks_explicit_directory_first(tmp_path: Path) -> None:
    data = tmp_path / "signals"
    labels = tmp_path / "labels"
    data.mkdir()
    labels.mkdir()
    expected = labels / "B0205E.mat"
    savemat(expected, {"classlabel": np.ones((160, 1), dtype=int)})

    resolved = resolve_dataset_2b_evaluation_label_path(
        2,
        5,
        data_directory=data,
        labels_directory=labels,
    )

    assert resolved == expected


def test_dataset_2b_label_loader_rejects_wrong_length(tmp_path: Path) -> None:
    path = tmp_path / "B0104E.mat"
    savemat(path, {"classlabel": np.ones((159, 1), dtype=int)})

    with pytest.raises(ValueError, match="160 labels"):
        load_dataset_2b_evaluation_labels(path)
