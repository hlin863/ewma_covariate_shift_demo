"""Compatibility entry point for the structured Flask dashboard package."""

from src.web import app
from src.web.dashboard import (  # noqa: F401
    DEFAULT_OUTPUTS_ROOT,
    DEFAULT_RESULTS_PATH,
    PROJECT_ROOT,
    _build_results_catalog,
    _chart_rows,
    _dataset_summary,
    _load_results,
    dashboard,
    outputs_file,
    results_catalog,
)
from src.web.test_results import load_test_report


if __name__ == "__main__":
    app.run(debug=True)
