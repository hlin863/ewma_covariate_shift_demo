"""Live pytest result reporting for the Flask research dashboard."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from flask import Blueprint, current_app, jsonify, render_template, request


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEST_REPORT_PATH = PROJECT_ROOT / "outputs" / "metrics" / "pytest_results.xml"
DEFAULT_TEST_REFRESH_SECONDS = 60


test_results_bp = Blueprint("test_results", __name__)


def _status_for_case(case: ET.Element) -> tuple[str, str]:
    failure = case.find("failure")
    if failure is not None:
        return "failed", (failure.get("message") or (failure.text or "")).strip()
    error = case.find("error")
    if error is not None:
        return "failed", (error.get("message") or (error.text or "")).strip()
    skipped = case.find("skipped")
    if skipped is not None:
        return "skipped", (skipped.get("message") or (skipped.text or "")).strip()
    return "passed", ""


def _group_from_classname(classname: str) -> str:
    parts = [part for part in classname.split(".") if part]
    if "tests" in parts:
        index = parts.index("tests")
        if index + 1 < len(parts):
            candidate = parts[index + 1]
            if not candidate.startswith("test_"):
                return candidate
    return "other"


def load_test_report(path: str | Path) -> dict[str, object]:
    """Parse pytest's built-in JUnit XML output into dashboard-ready data."""
    report_path = Path(path)
    if not report_path.is_file():
        return {
            "available": False,
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "pass_percent": 0.0,
            "fail_percent": 0.0,
            "skip_percent": 0.0,
            "duration_seconds": 0.0,
            "generated_at": None,
            "cases": [],
            "groups": [],
        }

    root = ET.parse(report_path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))

    cases: list[dict[str, object]] = []
    group_counts: dict[str, dict[str, int]] = {}
    duration = 0.0

    for suite in suites:
        duration += float(suite.get("time", "0") or 0)
        for case in suite.findall("testcase"):
            status, detail = _status_for_case(case)
            classname = case.get("classname", "")
            group = _group_from_classname(classname)
            group_entry = group_counts.setdefault(
                group,
                {"passed": 0, "failed": 0, "skipped": 0, "total": 0},
            )
            group_entry[status] += 1
            group_entry["total"] += 1
            cases.append(
                {
                    "group": group,
                    "classname": classname,
                    "name": case.get("name", "unnamed test"),
                    "status": status,
                    "time": float(case.get("time", "0") or 0),
                    "detail": detail[:800],
                }
            )

    total = len(cases)
    passed = sum(case["status"] == "passed" for case in cases)
    failed = sum(case["status"] == "failed" for case in cases)
    skipped = sum(case["status"] == "skipped" for case in cases)
    divisor = max(total, 1)
    groups = [
        {"name": name, **counts}
        for name, counts in sorted(group_counts.items())
    ]

    return {
        "available": True,
        "total": total,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "pass_percent": 100.0 * passed / divisor,
        "fail_percent": 100.0 * failed / divisor,
        "skip_percent": 100.0 * skipped / divisor,
        "duration_seconds": duration,
        "generated_at": datetime.fromtimestamp(
            report_path.stat().st_mtime, tz=timezone.utc
        ).isoformat(timespec="seconds"),
        "cases": cases,
        "groups": groups,
    }


def run_test_suite(report_path: str | Path) -> dict[str, object]:
    """Run the repository test suite and write a JUnit XML report."""
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests",
        "-q",
        "--disable-warnings",
        f"--junitxml={path}",
    ]
    started = time.perf_counter()
    try:
        process = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        return {
            "returncode": process.returncode,
            "duration_seconds": time.perf_counter() - started,
            "output": "\n".join(
                part for part in (process.stdout.strip(), process.stderr.strip()) if part
            )[-6000:],
            "command": " ".join(command),
        }
    except subprocess.TimeoutExpired as error:
        return {
            "returncode": 124,
            "duration_seconds": time.perf_counter() - started,
            "output": f"pytest timed out after {error.timeout} seconds",
            "command": " ".join(command),
        }


def _ensure_current_report(force: bool = False) -> tuple[dict[str, object], dict[str, object] | None]:
    path = Path(current_app.config["TEST_RESULTS_PATH"])
    refresh_seconds = int(current_app.config["TEST_RESULTS_REFRESH_SECONDS"])
    stale = (
        not path.is_file()
        or (time.time() - path.stat().st_mtime) >= refresh_seconds
    )

    run_info = None
    auto_run = bool(current_app.config.get("TEST_RESULTS_AUTO_RUN", True))
    if auto_run and (force or stale):
        run_info = run_test_suite(path)

    return load_test_report(path), run_info


@test_results_bp.get("/tests")
def test_results():
    force = request.args.get("refresh") == "1"
    report, run_info = _ensure_current_report(force=force)
    return render_template(
        "test_results.html",
        report=report,
        run_info=run_info,
        refresh_seconds=int(current_app.config["TEST_RESULTS_REFRESH_SECONDS"]),
    )


@test_results_bp.get("/api/test-results")
def test_results_api():
    report, run_info = _ensure_current_report(force=False)
    return jsonify(
        {
            "generated_at": report["generated_at"],
            "available": report["available"],
            "total": report["total"],
            "passed": report["passed"],
            "failed": report["failed"],
            "skipped": report["skipped"],
            "run_returncode": None if run_info is None else run_info["returncode"],
        }
    )
