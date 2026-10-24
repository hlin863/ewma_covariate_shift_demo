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


def test_shared_research_design_and_theme_controls():
    client = app.test_client()
    page = client.get("/results/abcd")
    assert page.status_code == 200
    assert b"dashboard.css" in page.data
    assert b"home.css" in page.data
    assert b"dashboard.js" in page.data
    assert b'data-theme-option="light"' in page.data
    assert b'data-theme-option="dark"' in page.data
    assert b"Research hub" in page.data
    assert b"Teaching baseline" in page.data


def test_run_retains_visual_evidence_and_accessible_split_picker():
    page = app.test_client().post("/results/abcd", data={"segment_size": "80"})
    assert page.status_code == 200
    assert b'id="loss-chart"' in page.data
    assert b'id="bound-chart"' in page.data
    assert b'id="inspect-k"' in page.data
    assert b'data-split-k="' in page.data
    assert b'id="download"' in page.data
    assert b"not stream-calibrated" in page.data
