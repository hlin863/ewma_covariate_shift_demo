"""Regression tests for the Park-inspired dataset inspection."""
import numpy as np
import pandas as pd
import pytest

from src.detection.multivariate_inspection import (
    demonstration_stream, describe_windows, load_multivariate_csv,
)


def test_demo_reproducible_and_cross_dependence_changes():
    frame = demonstration_stream()
    assert frame.equals(demonstration_stream())
    rows = describe_windows(frame, 40)
    assert len(rows) == 6
    assert np.mean([r["corr_12"] for r in rows[:3]]) > 0.5
    assert np.mean([r["corr_12"] for r in rows[3:]]) < -0.5


def test_csv_preserves_time_and_sensor_columns(tmp_path):
    path = tmp_path / "sensors.csv"
    pd.DataFrame({"time": np.arange(10) * .5, "a": np.arange(10),
                  "b": np.arange(10)[::-1]}).to_csv(path, index=False)
    loaded = load_multivariate_csv(path)
    assert loaded.columns.tolist() == ["time", "a", "b"]
    assert loaded.time.iloc[-1] == 4.5


@pytest.mark.parametrize("frame", [
    pd.DataFrame({"a": [1.0] * 10}),
    pd.DataFrame({"a": [1.0] * 10, "b": [np.nan] * 10}),
    pd.DataFrame({"time": [0.0] * 10, "a": [1.0] * 10, "b": [2.0] * 10}),
])
def test_bad_inputs_rejected(frame, tmp_path):
    path = tmp_path / "invalid.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_multivariate_csv(path)


def test_trailing_incomplete_window_excluded():
    assert len(describe_windows(demonstration_stream(size=26), window_size=8)) == 3
