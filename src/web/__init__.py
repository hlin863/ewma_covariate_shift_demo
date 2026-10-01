"""Web presentation layer for research-result visualisation."""

from src.web.dashboard import app
from src.web.diethe import diethe_bp
from src.web.test_results import (
    DEFAULT_TEST_REFRESH_SECONDS,
    DEFAULT_TEST_REPORT_PATH,
    test_results_bp,
)
from src.web.support import configure_support_defaults, support_bp


app.config.setdefault("TEST_RESULTS_PATH", str(DEFAULT_TEST_REPORT_PATH))
app.config.setdefault("TEST_RESULTS_REFRESH_SECONDS", DEFAULT_TEST_REFRESH_SECONDS)
app.config.setdefault("TEST_RESULTS_AUTO_RUN", True)
configure_support_defaults(app)

if "support" not in app.blueprints:
    app.register_blueprint(support_bp)

if "test_results" not in app.blueprints:
    app.register_blueprint(test_results_bp)


if "diethe" not in app.blueprints:
    app.register_blueprint(diethe_bp)


__all__ = ["app"]
