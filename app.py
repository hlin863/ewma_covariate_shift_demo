"""Compatibility entry point for the structured Flask dashboard package."""

from src.web import app
from src.web.dashboard import (  # noqa: F401
    DEFAULT_CHOWDHURY_DATA_PATH,
    DEFAULT_OUTPUTS_ROOT,
    DEFAULT_RESULTS_PATH,
    PROJECT_ROOT,
    _build_results_catalog,
    _chart_rows,
    _dataset_summary,
    _load_results,
    chowdhury_demographics,
    dashboard,
    home,
    outputs_file,
    results_catalog,
)
from src.web.test_results import load_test_report, test_results_bp


# Keep blueprint registration idempotent so app.py remains safe even if
# src.web.__init__ has already registered the test-results blueprint.
if "test_results" not in app.blueprints:
    app.register_blueprint(test_results_bp)


if __name__ == "__main__":
    app.run(debug=True)
