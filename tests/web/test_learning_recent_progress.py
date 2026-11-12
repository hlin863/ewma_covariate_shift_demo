"""Regression checks for November 2026 paper and code learning updates."""
from app import app
from src.web.learning_progress import TOPICS, learning_model, read_progress, write_progress


def test_original_park_topics_and_new_benchmark_topics_coexist(tmp_path):
    config = {"LEARNING_PROGRESS_PATH": str(tmp_path / "progress.json")}
    topic_ids = {item["id"] for item in TOPICS}
    assert len(topic_ids) == len(TOPICS)
    assert {
        "park-segment-means",
        "bci-cue-provenance",
        "bci-visual-evidence",
        "moabb-benchmark",
        "cpd-benchmark",
        "stream-benchmark",
        "stage2-replay",
        "cost-aware-bci",
    } <= topic_ids
    write_progress(config, {"park-segment-means": "completed"})
    model = learning_model(config)
    assert model["completed"] == 1
    assert model["total"] == len(TOPICS)
    assert read_progress(config) == {"park-segment-means": "completed"}
    assert all(
        item["status"] == "not_started"
        for item in model["topics"]
        if item["id"] != "park-segment-means"
    )


def test_recent_commit_evidence_is_not_auto_marked_complete(tmp_path):
    app.config["LEARNING_PROGRESS_PATH"] = str(tmp_path / "learning.json")
    try:
        response = app.test_client().get("/learning/")
        assert response.status_code == 200
        html = response.get_data(as_text=True)
        assert "Recent code progress and learning checkpoints" in html
        assert "MOABB" in html
        assert "van den Burg" in html
        assert "Souza" in html
        assert "6cafc5ac" in html
        assert "14 warnings/10 confirmations" in html
        assert "17/0" in html
        assert "0 / 13" in html or "0<small> / 13</small>" in html
    finally:
        app.config.pop("LEARNING_PROGRESS_PATH", None)


def test_new_topic_can_be_saved_without_marking_other_topics(tmp_path):
    app.config["LEARNING_PROGRESS_PATH"] = str(tmp_path / "learning.json")
    try:
        client = app.test_client()
        response = client.post(
            "/learning/progress",
            data={"topic": "moabb-benchmark", "status": "in_progress"},
        )
        assert response.status_code == 303
        saved = read_progress(app.config)
        assert saved == {"moabb-benchmark": "in_progress"}
        assert client.post(
            "/learning/progress",
            data={"topic": "moabb-benchmark", "status": "mastered"},
        ).status_code == 400
    finally:
        app.config.pop("LEARNING_PROGRESS_PATH", None)
