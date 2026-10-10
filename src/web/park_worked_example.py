"""Independent learning page for Park (2023) Eq. (2.2); no raw EEG loading."""
from flask import Blueprint, render_template
from src.detection.stage1.park_single_feature_example import worked_example

park_worked_example_bp = Blueprint("park_worked_example", __name__)

@park_worked_example_bp.get("/learning/park-cusum-example")
def example():
    return render_template("park_worked_example.html", example=worked_example())
