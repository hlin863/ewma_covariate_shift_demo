"""Tests for CSV bridge shared by the Flask Park page and MATLAB."""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from scripts.export_park_matlab_features import main


def test_export_uses_selected_session_features(tmp_path):
    frame = pd.DataFrame({
        "time": [0., 1., 2.],
        "mu_power_uv2": [2., 3., 4.],
        "beta_power_uv2": [4., 3., 2.],
        "rms_uv": [1., 2., 3.],
    })
    out = tmp_path / "matlab" / "bci.csv"
    with patch("scripts.export_park_matlab_features._park_bci_features", return_value=(frame, "mock session")) as loader:
        main(["--dataset","2b","--subject","1","--session","4","--output",str(out)])
    loader.assert_called_once_with("2b", 1, 4)
    loaded = pd.read_csv(out)
    pd.testing.assert_frame_equal(loaded, frame)


@pytest.mark.parametrize("dataset,session", [("2a", 3), ("2b", 6)])
def test_export_rejects_invalid_session(dataset, session, tmp_path):
    with pytest.raises(ValueError, match="session"):
        main(["--dataset",dataset,"--subject","1","--session",str(session),
              "--output",str(tmp_path/"bad.csv")])
