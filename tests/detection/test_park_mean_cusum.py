"""Tests for paper-inspired mean-CUSUM candidate curves and Park UI."""
import numpy as np
import pytest

from app import app
from src.detection.stage1.park_mean_cusum import mean_cusum_curves


def test_mean_step_location_and_aggregation():
    rng = np.random.default_rng(12)
    x = np.maximum(rng.normal(8, 0.25, (120, 3)), 0.1)
    x[60:, :] += 3.0
    result = mean_cusum_curves(x)
    assert abs(result["b_max"] - 60) <= 3
    assert abs(result["b_avg"] - 60) <= 3
    assert len(result["positions"]) == len(result["maximum"])
    assert len(result["by_channel"]) == 3


def test_reject_zero_mean_signed_eeg():
    x = np.column_stack((np.ones(40), np.linspace(-1, 1, 40)))
    with pytest.raises(ValueError, match="positive"):
        mean_cusum_curves(x)


def test_park_mean_change_page_renders():
    app.config.pop("PARK_MULTIVARIATE_DATA_PATH", None)
    response = app.test_client().get("/data-structures/park")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'id="park-cusum-svg"' in html
    assert "Candidate mean-change locations" in html
    assert "not included" in html
    assert 'name="source"' in html


def test_park_invalid_source():
    assert app.test_client().get("/data-structures/park?source=invalid").status_code == 400
