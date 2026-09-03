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
    results_catalog,
)


if __name__ == "__main__":
    app.run(debug=True)
