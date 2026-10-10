"""Independent Park worked-example page and exact numerical regression."""
from math import sqrt
import pytest

from app import app
from src.detection.stage1.park_single_feature_example import single_feature_cusum, worked_example


def test_exact_quiz_example():
    result = worked_example()
    assert result["mean_before"] == 2
    assert result["mean_after"] == 6
    assert result["overall_mean"] == 4
    assert result["absolute_difference"] == 4
    assert result["weight"] == pytest.approx(sqrt(2))
    assert result["cusum"] == pytest.approx(sqrt(2))
    assert result["passes_example"] is True
    assert max(result["curve"], key=lambda v: v["cusum"])["split"] == 4


def test_invalid_split_and_mean_are_rejected():
    with pytest.raises(ValueError, match="split"):
        single_feature_cusum([2, 2, 6, 6], split=4)
    with pytest.raises(ValueError, match="positive"):
        single_feature_cusum([-2, -2, -6, -6], split=2)


def test_standalone_route_and_navigation():
    client = app.test_client()
    page = client.get("/learning/park-cusum-example")
    assert page.status_code == 200
    assert b"1.414214" in page.data
    assert b"Verified against" in page.data
    assert b"All candidate splits" in page.data
    overview = client.get("/data-structures")
    assert b"/learning/park-cusum-example" in overview.data
    existing = client.get("/data-structures/park")
    assert existing.status_code == 200
