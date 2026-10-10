"""Paper-linked learning hub and independent Park (2023) worked example."""
from flask import Blueprint, abort, current_app, redirect, render_template, request, url_for
from src.detection.stage1.park_single_feature_example import worked_example
from src.web.learning_progress import TOPICS, VALID_STATUS, learning_model, read_progress, write_progress

park_worked_example_bp = Blueprint("park_worked_example", __name__)


@park_worked_example_bp.get("/learning/")
@park_worked_example_bp.get("/learning")
def index():
    return render_template("learning.html", learning=learning_model(current_app.config))


@park_worked_example_bp.post("/learning/progress")
def save_progress():
    topic = request.form.get("topic")
    status = request.form.get("status")
    if topic not in {item["id"] for item in TOPICS} or status not in VALID_STATUS:
        abort(400, description="Choose a valid learning milestone and status.")
    progress = read_progress(current_app.config)
    progress[topic] = status
    write_progress(current_app.config, progress)
    return redirect(url_for(".index"), code=303)


@park_worked_example_bp.get("/learning/park-cusum-example")
def example():
    return render_template("park_worked_example.html", example=worked_example())
