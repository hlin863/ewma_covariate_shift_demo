"""Compatibility entry point for the structured Flask dashboard package."""

from src.web.dashboard import (  # noqa: F401
    DEFAULT_OUTPUTS_ROOT,
    DEFAULT_RESULTS_PATH,
    PROJECT_ROOT,
    _build_results_catalog,
    _chart_rows,
    _dataset_summary,
    _load_results,
    app,
    dashboard,
    outputs_file,
    results_catalog,
)
from src.web.test_results import (
    DEFAULT_TEST_REFRESH_SECONDS,
    DEFAULT_TEST_REPORT_PATH,
    load_test_report,
    test_results_bp,
)


app.config.setdefault("TEST_RESULTS_PATH", str(DEFAULT_TEST_REPORT_PATH))
app.config.setdefault("TEST_RESULTS_REFRESH_SECONDS", DEFAULT_TEST_REFRESH_SECONDS)
app.config.setdefault("TEST_RESULTS_AUTO_RUN", True)
app.register_blueprint(test_results_bp)


if __name__ == "__main__":
    app.run(debug=True)
