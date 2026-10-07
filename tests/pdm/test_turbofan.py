from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.pdm.datasets.turbofan import COLUMN_NAMES, TurbofanDataset, load_turbofan_dataset


def _row(unit, cycle):
    return [unit, cycle, *np.arange(24, dtype=float)]


@pytest.fixture
def data_directory(tmp_path):
    # Unequal lengths and shuffled rows expose alignment and sorting mistakes.
    np.savetxt(tmp_path / "train_FD001.txt", [_row(2, 1), _row(1, 2), _row(1, 1)])
    np.savetxt(tmp_path / "test_FD001.txt", [_row(2, 2), _row(1, 1), _row(2, 1)])
    (tmp_path / "RUL_FD001.txt").write_text("7\n11\n")
    return tmp_path


def test_load_and_derive_rul_without_changing_raw_tables(data_directory):
    data = TurbofanDataset.load(data_directory, "fd001")
    assert data.dataset == "FD001"
    assert data.training.columns.tolist() == COLUMN_NAMES
    assert data.training[["unit", "cycle"]].values.tolist() == [[1, 1], [1, 2], [2, 1]]
    assert data.training["unit"].dtype == np.dtype("int64")
    assert data.test_rul.tolist() == [7, 11]
    assert data.training_with_rul()["rul"].tolist() == [1, 0, 0]
    assert data.testing_with_rul()["rul"].tolist() == [7, 12, 11]
    assert "rul" not in data.training
    assert "rul" not in data.testing
    modified = data.training_with_rul()
    modified.loc[0, "sensor_1"] = -999
    assert data.training.loc[0, "sensor_1"] != -999


def test_whitespace_and_single_observation(tmp_path):
    row = "\t  ".join(str(value) for value in _row(1, 1)) + "  \n"
    for split in ("train", "test"):
        (tmp_path / f"{split}_FD001.txt").write_text(row)
    (tmp_path / "RUL_FD001.txt").write_text("0\n")
    data = load_turbofan_dataset(tmp_path)
    assert data.training.shape == (1, 26)
    assert data.test_rul.shape == (1,)
    assert data.testing_with_rul()["rul"].tolist() == [0]


def test_missing_file_reports_path(data_directory):
    path = data_directory / "RUL_FD001.txt"
    path.unlink()
    with pytest.raises(FileNotFoundError, match="RUL_FD001.txt"):
        load_turbofan_dataset(data_directory)


@pytest.mark.parametrize("subset", ["FD000", "FD005", "FD01", "", None])
def test_invalid_subset(data_directory, subset):
    with pytest.raises(ValueError, match="dataset must be"):
        load_turbofan_dataset(data_directory, subset)


@pytest.mark.parametrize("rows, message", [
    ([_row(1, 1)[:-1]], "expected 26 columns"),
    ([_row(1, 1) + [0]], "expected 26 columns"),
    ([_row(1, 1), _row(1, 2)[:-1]], "rectangular numeric table"),
    ([[1, 1, *([np.nan] * 24)]], "finite"),
    ([[1, 1, *([np.inf] * 24)]], "finite"),
    ([_row(0, 1)], "positive integer"),
    ([_row(1.5, 1)], "positive integer"),
    ([_row(1, 1.5)], "positive integer"),
    ([_row(1, 1), _row(3, 1)], "unit IDs"),
    ([_row(1, 1), _row(1, 1)], "unique consecutive cycles"),
    ([_row(1, 1), _row(1, 3)], "unique consecutive cycles"),
    ([_row(1, 2)], "unique consecutive cycles"),
])
def test_invalid_trajectories(data_directory, rows, message):
    # Save ragged rows too, since malformed files must fail before adaptation.
    content = "\n".join(" ".join(map(str, row)) for row in rows)
    (data_directory / "train_FD001.txt").write_text(content)
    with pytest.raises(ValueError, match=message):
        load_turbofan_dataset(data_directory)


@pytest.mark.parametrize("content, message", [
    ("", "at least one observation"),
    ("   \n\t", "at least one observation"),
    ("7\n", "one RUL value per test unit"),
    ("7\n11\n12\n", "one RUL value per test unit"),
    ("7 11\n", "expected 1 columns"),
    ("-1\n11\n", "non-negative integer"),
    ("7.5\n11\n", "non-negative integer"),
    ("nan\n11\n", "finite"),
    ("seven\n11\n", "rectangular numeric table"),
])
def test_invalid_rul(data_directory, content, message):
    (data_directory / "RUL_FD001.txt").write_text(content)
    with pytest.raises(ValueError, match=message):
        load_turbofan_dataset(data_directory)


@pytest.mark.parametrize("subset, train_rows, test_rows, train_units, test_units", [
    ("FD001", 20631, 13096, 100, 100),
    ("FD002", 53759, 33991, 260, 259),
    ("FD003", 24720, 16596, 100, 100),
    ("FD004", 61249, 41214, 249, 248),
])
def test_committed_nasa_subsets(monkeypatch, tmp_path, subset, train_rows, test_rows,
                                train_units, test_units):
    directory = Path(__file__).resolve().parents[2] / "data/raw/turbofan_engine_degradation"
    if not (directory / f"train_{subset}.txt").is_file():
        pytest.skip("NASA C-MAPSS raw files are unavailable")
    monkeypatch.chdir(tmp_path)
    data = load_turbofan_dataset(dataset=subset)
    assert data.training.shape == (train_rows, 26)
    assert data.testing.shape == (test_rows, 26)
    assert data.training["unit"].nunique() == train_units
    assert data.testing["unit"].nunique() == test_units
    assert data.test_rul.size == test_units
    assert data.training_with_rul().groupby("unit")["rul"].last().eq(0).all()
    np.testing.assert_array_equal(
        data.testing_with_rul().groupby("unit")["rul"].last(), data.test_rul
    )
    for frame in (data.training_with_rul(), data.testing_with_rul()):
        assert frame.groupby("unit")["rul"].diff().dropna().eq(-1).all()
    # The function and class entry points return the same raw observations.
    pd.testing.assert_frame_equal(
        data.training, TurbofanDataset.load(directory, subset).training
    )
