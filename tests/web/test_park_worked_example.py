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
    learning = client.get("/learning")
    assert learning.status_code == 200
    assert b"/learning/park-cusum-example" in learning.data
    assert b"Step-by-step working" in page.data
    assert b"Substitute into Equation" in page.data
    home = client.get("/")
    assert home.status_code == 200
    assert b"Learning hub" in home.data
    existing = client.get("/data-structures/park")
    assert existing.status_code == 200


def test_learning_trailing_slash_and_progress_persistence(tmp_path):
    app.config["LEARNING_PROGRESS_PATH"] = str(tmp_path / "progress.json")
    try:
        client = app.test_client()
        for route in ("/learning", "/learning/"):
            response = client.get(route)
            assert response.status_code == 200
            assert b"Study progress" in response.data
            assert b"Park et al. (2023) learning roadmap" in response.data
        response = client.post("/learning/progress", data={
            "topic": "park-normalisation", "status": "completed"
        })
        assert response.status_code == 303
        assert b"1 / 6" in client.get("/learning/").data
        assert (tmp_path / "progress.json").is_file()
        assert client.post("/learning/progress", data={
            "topic": "invalid", "status": "completed"
        }).status_code == 400
    finally:
        app.config.pop("LEARNING_PROGRESS_PATH", None)
