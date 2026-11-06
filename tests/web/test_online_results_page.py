"""The added results page must not break existing research-page structure."""
from src.web import app


def test_online_page_get_does_not_invent_experiment_results():
    client = app.test_client()
    response = client.get("/results/online-learning")
    assert response.status_code == 200
    text = response.get_data(as_text=True)
    assert "Online adaptive learning" in text
    assert "No saved real-BCI online runs" in text or "BCI 2A / 2B results" in text
    assert 'id="online-experiment-results"' not in text
    assert 'href="/results"' in text
    assert 'href="/results/diethe"' in text


def test_online_page_runs_measured_synthetic_comparison():
    response = app.test_client().post(
        "/results/online-learning",
        data={"scenario":"feature_shift","seed":"9","trials":"80",
              "window_size":"5","interval":"10","label_delay":"1"},
    )
    assert response.status_code == 200
    text = response.get_data(as_text=True)
    assert 'id="online-experiment-results"' in text
    assert "Gain vs never" in text
    assert "Post-change accuracy" in text
    assert "Stage-I alarms" in text
    assert "validated" in text


def test_online_page_rejects_unbounded_or_invalid_inputs():
    client = app.test_client()
    for data in (
        {"trials":"100000"},
        {"trials":"not-number"},
        {"scenario":"unknown"},
        {"window_size":"0"},
        {"label_delay":"-2"},
    ):
        response = client.post("/results/online-learning", data=data)
        assert response.status_code == 400
        assert b'role="alert"' in response.data
        assert b'id="online-experiment-results"' not in response.data


def test_online_page_links_appear_in_existing_navigation():
    html = app.test_client().get("/results").get_data(as_text=True)
    assert 'href="/results/online-learning"' in html
    assert app.test_client().get("/results/diethe").status_code == 200
