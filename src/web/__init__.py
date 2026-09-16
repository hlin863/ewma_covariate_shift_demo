"""Web presentation layer for research-result visualisation."""

from src.web.dashboard import app
from src.web.test_results import (
    DEFAULT_TEST_REFRESH_SECONDS,
    DEFAULT_TEST_REPORT_PATH,
    test_results_bp,
)


app.config.setdefault("TEST_RESULTS_PATH", str(DEFAULT_TEST_REPORT_PATH))
app.config.setdefault("TEST_RESULTS_REFRESH_SECONDS", DEFAULT_TEST_REFRESH_SECONDS)
app.config.setdefault("TEST_RESULTS_AUTO_RUN", True)

if "test_results" not in app.blueprints:
    app.register_blueprint(test_results_bp)


__all__ = ["app"]
