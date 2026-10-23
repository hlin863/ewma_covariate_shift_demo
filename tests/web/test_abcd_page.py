"""Smoke tests for the new Flask Blueprint."""
from src.web import app


def test_abcd_page_empty_and_run():
    client = app.test_client()
    page = client.get("/results/abcd")
    assert page.status_code == 200
    assert b"Ready for a reproducible teaching run" in page.data

    response = client.post("/results/abcd", data={
        "scenario": "Normal-M", "seed": "1", "segment_size": "80"
    })
    assert response.status_code == 200
    assert b'abcd-data' in response.data
    assert b"Bernstein" in response.data
    assert b"not the authors" in response.data


def test_abcd_bad_inputs_are_rejected():
    client = app.test_client()
    for payload in (
        {"segment_size": "9999"}, {"scenario": "HAR"}, {"delta": "nan"}
    ):
        response = client.post("/results/abcd", data=payload)
        assert response.status_code == 400
        assert b'role="alert"' in response.data
        assert b'id="abcd-data"' not in response.data


def test_lab_is_discoverable_from_results_navigation():
    client = app.test_client()
    page = client.get("/results/abcd")
    assert b"ABCD" in page.data
    assert b"/results/abcd" in page.data
