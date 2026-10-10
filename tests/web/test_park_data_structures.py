"""Check that Park inspection is integrated without changing other dataset routes."""
from pathlib import Path

from app import app


def test_park_demo_in_existing_catalogue():
    app.config.pop("PARK_MULTIVARIATE_DATA_PATH", None)
    client = app.test_client()
    index = client.get("/data-structures")
    assert index.status_code == 200
    assert b"Park (2023)" in index.data
    page = client.get("/data-structures/park")
    assert page.status_code == 200
    assert b"Time-varying multivariate descriptors" in page.data
    assert b"Sensor 1" in page.data
    assert b"not the published CUSUM" in page.data


def test_park_custom_file_and_invalid_window(tmp_path):
    path = tmp_path / "two_channels.csv"
    path.write_text("time,c1,c2\n" + "".join(
        f"{i},{i/10},{i/5}\n" for i in range(16)
    ))
    app.config["PARK_MULTIVARIATE_DATA_PATH"] = str(path)
    try:
        client = app.test_client()
        response = client.get("/data-structures/park?source=csv&window=8&min_segment=4")
        assert response.status_code == 200
        assert b"c1" in response.data
        assert b"c2" in response.data
        assert client.get("/data-structures/park?source=csv&window=2&min_segment=4").status_code == 400
    finally:
        app.config.pop("PARK_MULTIVARIATE_DATA_PATH", None)
