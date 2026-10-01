"""Keep the shared Flask application configuration isolated between tests."""

import pytest

from app import app
from src.web.support import clear_support_cache


@pytest.fixture(autouse=True)
def isolated_app_config():
    original = app.config.copy()
    app.config.update(TESTING=True, TEST_RESULTS_AUTO_RUN=False)
    clear_support_cache()
    try:
        yield
    finally:
        app.config.clear()
        app.config.update(original)
        clear_support_cache()
