"""Compatibility entry point for the structured Flask dashboard package."""

from src.web.dashboard import (  # noqa: F401
    DEFAULT_RESULTS_PATH,
    DEFAULT_A07_VISUALISATION_PATH,
    PROJECT_ROOT,
    _chart_rows,
    _dataset_summary,
    _load_results,
    app,
    covariate_shift,
    dashboard,
)


if __name__ == "__main__":
    app.run(debug=True)
