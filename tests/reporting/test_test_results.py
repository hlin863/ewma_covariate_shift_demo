from pathlib import Path

from app import app, load_test_report


JUNIT_XML = """<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <testsuite name="pytest" tests="4" failures="1" errors="0" skipped="1" time="1.25">
    <testcase classname="tests.bci.test_bci_data" name="test_loader" time="0.10" />
    <testcase classname="tests.detection.test_cse" name="test_cse" time="0.20" />
    <testcase classname="tests.reporting.test_dashboard" name="test_dashboard" time="0.30">
      <failure message="assert 1 == 2">assert 1 == 2</failure>
    </testcase>
    <testcase classname="tests.integration.test_project_structure" name="test_structure" time="0.05">
      <skipped message="optional dependency" />
    </testcase>
  </testsuite>
</testsuites>
"""


def test_load_test_report_calculates_distribution(tmp_path: Path) -> None:
    report_path = tmp_path / "pytest_results.xml"
    report_path.write_text(JUNIT_XML, encoding="utf-8")

    report = load_test_report(report_path)

    assert report["available"] is True
    assert report["total"] == 4
    assert report["passed"] == 2
    assert report["failed"] == 1
    assert report["skipped"] == 1
    assert report["pass_percent"] == 50.0
    assert {group["name"] for group in report["groups"]} == {
        "bci",
        "detection",
        "integration",
        "reporting",
    }


def test_test_results_page_renders_pie_and_cases(tmp_path: Path) -> None:
    report_path = tmp_path / "pytest_results.xml"
    report_path.write_text(JUNIT_XML, encoding="utf-8")
    app.config.update(
        TESTING=True,
        TEST_RESULTS_PATH=str(report_path),
        TEST_RESULTS_AUTO_RUN=False,
        TEST_RESULTS_REFRESH_SECONDS=60,
    )

    response = app.test_client().get("/tests")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Automated test case results" in html
    assert "Passed / failed / skipped" in html
    assert "test_loader" in html
    assert "test_dashboard" in html
    assert "50.0%" in html
    assert "Research results" in html


def test_test_results_api_returns_current_counts(tmp_path: Path) -> None:
    report_path = tmp_path / "pytest_results.xml"
    report_path.write_text(JUNIT_XML, encoding="utf-8")
    app.config.update(
        TESTING=True,
        TEST_RESULTS_PATH=str(report_path),
        TEST_RESULTS_AUTO_RUN=False,
        TEST_RESULTS_REFRESH_SECONDS=60,
    )

    response = app.test_client().get("/api/test-results")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["total"] == 4
    assert payload["passed"] == 2
    assert payload["failed"] == 1
    assert payload["skipped"] == 1
