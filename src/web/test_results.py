"""Live pytest result reporting for the Flask research dashboard."""

from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from flask import Blueprint, abort, current_app, jsonify, render_template, request


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


def _source_path_from_classname(classname: str) -> Path | None:
    """Map a pytest JUnit classname back to its repository Python file."""
    if not classname:
        return None
    module_parts = [part for part in classname.split(".") if part]
    if not module_parts or module_parts[0] != "tests":
        return None
    candidate = PROJECT_ROOT.joinpath(*module_parts).with_suffix(".py")
    try:
        candidate.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def _normalise_test_function_name(name: str) -> str:
    """Strip pytest parameter IDs from a JUnit testcase name."""
    return name.split("[", 1)[0]


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _comparison_contract(node: ast.AST, source: str) -> dict[str, str]:
    expression = ast.get_source_segment(source, node) or "assertion"
    actual = expression
    expected = "Assertion condition evaluates to true"

    if isinstance(node, ast.Compare) and len(node.ops) == 1 and len(node.comparators) == 1:
        left = ast.get_source_segment(source, node.left) or "left expression"
        right = ast.get_source_segment(source, node.comparators[0]) or "right expression"
        operator = node.ops[0]
        operator_text = {
            ast.Eq: "==",
            ast.NotEq: "!=",
            ast.Lt: "<",
            ast.LtE: "<=",
            ast.Gt: ">",
            ast.GtE: ">=",
            ast.Is: "is",
            ast.IsNot: "is not",
            ast.In: "in",
            ast.NotIn: "not in",
        }.get(type(operator), operator.__class__.__name__)
        actual = left
        expected = f"{operator_text} {right}"
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        actual = ast.get_source_segment(source, node.operand) or expression
        expected = "is false / empty"
    elif isinstance(node, ast.Call):
        name = _call_name(node.func)
        actual = expression
        if name == "isinstance" and len(node.args) >= 2:
            subject = ast.get_source_segment(source, node.args[0]) or "value"
            type_expr = ast.get_source_segment(source, node.args[1]) or "type"
            actual = subject
            expected = f"is instance of {type_expr}"

    return {
        "kind": "assert",
        "expression": expression,
        "actual_expression": actual,
        "expected": expected,
    }


def _extract_test_source_analysis(classname: str, case_name: str) -> dict[str, object]:
    """Extract the selected test's source and expectation contracts using Python AST."""
    path = _source_path_from_classname(classname)
    if path is None:
        return {"available": False, "source_path": None, "source": "", "assertions": []}

    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {
            "available": False,
            "source_path": str(path.relative_to(PROJECT_ROOT)),
            "source": "",
            "assertions": [],
        }

    function_name = _normalise_test_function_name(case_name)
    function: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            function = node
            break

    if function is None:
        return {
            "available": False,
            "source_path": str(path.relative_to(PROJECT_ROOT)),
            "source": "",
            "assertions": [],
        }

    function_source = ast.get_source_segment(source, function) or ""
    contracts: list[dict[str, str]] = []
    for node in ast.walk(function):
        if isinstance(node, ast.Assert):
            contracts.append(_comparison_contract(node.test, source))
            continue

        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name.startswith("np.testing.assert_") or name.startswith("numpy.testing.assert_"):
                expression = ast.get_source_segment(source, node) or name
                actual = ast.get_source_segment(source, node.args[0]) if node.args else "actual value"
                expected = ast.get_source_segment(source, node.args[1]) if len(node.args) > 1 else "assertion succeeds"
                contracts.append({
                    "kind": "numeric_assertion",
                    "expression": expression,
                    "actual_expression": actual or "actual value",
                    "expected": expected or "expected value",
                })
            elif name == "pytest.raises":
                expression = ast.get_source_segment(source, node) or name
                exception_type = ast.get_source_segment(source, node.args[0]) if node.args else "exception"
                match_value = None
                for keyword in node.keywords:
                    if keyword.arg == "match":
                        match_value = ast.get_source_segment(source, keyword.value)
                expected = f"raises {exception_type or 'exception'}"
                if match_value:
                    expected += f" matching {match_value}"
                contracts.append({
                    "kind": "exception_contract",
                    "expression": expression,
                    "actual_expression": "call inside pytest.raises block",
                    "expected": expected,
                })

    return {
        "available": True,
        "source_path": str(path.relative_to(PROJECT_ROOT)),
        "source": function_source,
        "assertions": contracts,
    }


def _parse_properties(case: ET.Element) -> dict[str, str]:
    properties: dict[str, str] = {}
    container = case.find("properties")
    if container is None:
        return properties
    for item in container.findall("property"):
        name = item.get("name")
        if not name:
            continue
        properties[name] = item.get("value", item.text or "")
    return properties


def _try_float(value: object) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _comparison_rows(case: dict[str, object], analysis: dict[str, object]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Pair JUnit-recorded actual/expected properties and prepare numeric plots."""
    properties = {str(k): str(v) for k, v in dict(case.get("properties", {})).items()}
    actuals = {key[7:]: value for key, value in properties.items() if key.startswith("actual.")}
    expected = {key[9:]: value for key, value in properties.items() if key.startswith("expected.")}

    keys: list[str] = []
    for key in (*actuals.keys(), *expected.keys()):
        if key not in keys:
            keys.append(key)

    rows: list[dict[str, object]] = []
    numeric: list[dict[str, object]] = []
    for key in keys:
        actual_value = actuals.get(key)
        expected_value = expected.get(key)
        actual_number = _try_float(actual_value) if actual_value is not None else None
        expected_number = _try_float(expected_value) if expected_value is not None else None
        row = {
            "label": key.replace("_", " ").replace(".", " · "),
            "actual": actual_value if actual_value is not None else "not recorded",
            "expected": expected_value if expected_value is not None else "not recorded",
            "numeric": actual_number is not None and expected_number is not None,
        }
        rows.append(row)
        if row["numeric"]:
            scale = max(abs(actual_number), abs(expected_number), 1e-12)
            numeric.append({
                "label": row["label"],
                "actual": actual_number,
                "expected": expected_number,
                "actual_width": 100.0 * abs(actual_number) / scale,
                "expected_width": 100.0 * abs(expected_number) / scale,
            })

    if not rows:
        rows.append({
            "label": "pytest outcome",
            "actual": str(case.get("status", "unknown")),
            "expected": "passed",
            "numeric": False,
        })
        for assertion in analysis.get("assertions", []):
            if assertion.get("kind") == "exception_contract":
                rows.append({
                    "label": "exception contract",
                    "actual": (
                        "contract completed successfully"
                        if case.get("status") == "passed"
                        else str(case.get("detail", "failure recorded"))
                    ),
                    "expected": str(assertion.get("expected", "exception expected")),
                    "numeric": False,
                })
                break

    return rows, numeric


def load_test_report(path: str | Path) -> dict[str, object]:
    """Parse pytest's built-in JUnit XML output into dashboard-ready data."""
    report_path = Path(path)
    if not report_path.is_file():
        return {
            "available": False, "total": 0, "passed": 0, "failed": 0, "skipped": 0,
            "pass_percent": 0.0, "fail_percent": 0.0, "skip_percent": 0.0,
            "duration_seconds": 0.0, "generated_at": None, "cases": [], "groups": [],
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
            group_entry = group_counts.setdefault(group, {"passed": 0, "failed": 0, "skipped": 0, "total": 0})
            group_entry[status] += 1
            group_entry["total"] += 1
            system_out = case.findtext("system-out", default="").strip()
            system_err = case.findtext("system-err", default="").strip()
            cases.append({
                "index": len(cases),
                "group": group,
                "classname": classname,
                "name": case.get("name", "unnamed test"),
                "status": status,
                "time": float(case.get("time", "0") or 0),
                "detail": detail[:6000],
                "file": case.get("file"),
                "line": case.get("line"),
                "properties": _parse_properties(case),
                "system_out": system_out[-6000:],
                "system_err": system_err[-6000:],
            })

    total = len(cases)
    passed = sum(case["status"] == "passed" for case in cases)
    failed = sum(case["status"] == "failed" for case in cases)
    skipped = sum(case["status"] == "skipped" for case in cases)
    divisor = max(total, 1)
    groups = [{"name": name, **counts} for name, counts in sorted(group_counts.items())]

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
        "generated_at": datetime.fromtimestamp(report_path.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds"),
        "cases": cases,
        "groups": groups,
    }


def run_test_suite(report_path: str | Path) -> dict[str, object]:
    """Run the repository test suite and write a JUnit XML report."""
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "pytest", "tests", "-q", "--disable-warnings", f"--junitxml={path}"]
    started = time.perf_counter()
    try:
        process = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=300, check=False)
        return {
            "returncode": process.returncode,
            "duration_seconds": time.perf_counter() - started,
            "output": "\n".join(part for part in (process.stdout.strip(), process.stderr.strip()) if part)[-6000:],
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
    stale = not path.is_file() or (time.time() - path.stat().st_mtime) >= refresh_seconds

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


@test_results_bp.get("/tests/case/<int:case_index>")
def test_case_detail(case_index: int):
    """Show one test case with source expectations and recorded runtime outputs."""
    report, _ = _ensure_current_report(force=False)
    cases = report.get("cases", [])
    if case_index < 0 or case_index >= len(cases):
        abort(404)

    case = cases[case_index]
    analysis = _extract_test_source_analysis(str(case.get("classname", "")), str(case.get("name", "")))
    comparisons, numeric_comparisons = _comparison_rows(case, analysis)
    return render_template(
        "test_case_detail.html",
        case=case,
        analysis=analysis,
        comparisons=comparisons,
        numeric_comparisons=numeric_comparisons,
        generated_at=report.get("generated_at"),
    )


@test_results_bp.get("/api/test-results")
def test_results_api():
    report, run_info = _ensure_current_report(force=False)
    return jsonify({
        "generated_at": report["generated_at"],
        "available": report["available"],
        "total": report["total"],
        "passed": report["passed"],
        "failed": report["failed"],
        "skipped": report["skipped"],
        "run_returncode": None if run_info is None else run_info["returncode"],
    })
