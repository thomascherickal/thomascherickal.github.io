#!/usr/bin/env python3
"""
run_tests.py - Comprehensive Unit Test Suite Runner.

Discovers and executes all unit tests under tests/ directory,
evaluates test outcomes, enforces data/*.json Single Source of Truth immutability,
and reports a structured summary.
Exits with code 0 on complete pass, or code 1 if any failure occurs.

Strict Repo Compliance:
- Zero emojis.
- Zero em-dashes (spaced hyphens ' - ' or plain hyphens '-' used).
"""

import os
import sys
import time
import json
import hashlib
import subprocess
import unittest
from datetime import datetime, timezone

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.join(REPO_DIR, "tests")
DATA_DIR = os.path.join(REPO_DIR, "data")

# Mark test environment to prevent recursive post-sync test triggering
os.environ["ANTIGRAVITY_IN_TEST"] = "1"


class RecordingTestResult(unittest.TextTestResult):
    """Custom TestResult that captures per-test metadata for commit report generation."""

    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.test_records = []
        self._test_start_times = {}

    def startTest(self, test):
        super().startTest(test)
        self._test_start_times[test.id()] = time.time()

    def addSuccess(self, test):
        super().addSuccess(test)
        elapsed = time.time() - self._test_start_times.get(test.id(), time.time())
        doc = (test.shortDescription() or "").strip()
        self.test_records.append({
            "id": test.id(),
            "module": getattr(test, "__class__", type(test)).__module__,
            "class": getattr(test, "__class__", type(test)).__name__,
            "method": getattr(test, "_testMethodName", str(test)),
            "description": doc,
            "status": "PASSED",
            "time_taken": round(elapsed, 4)
        })

    def addFailure(self, test, err):
        super().addFailure(test, err)
        elapsed = time.time() - self._test_start_times.get(test.id(), time.time())
        doc = (test.shortDescription() or "").strip()
        self.test_records.append({
            "id": test.id(),
            "module": getattr(test, "__class__", type(test)).__module__,
            "class": getattr(test, "__class__", type(test)).__name__,
            "method": getattr(test, "_testMethodName", str(test)),
            "description": doc,
            "status": "FAILED",
            "error": self._exc_info_to_string(err, test),
            "time_taken": round(elapsed, 4)
        })

    def addError(self, test, err):
        super().addError(test, err)
        elapsed = time.time() - self._test_start_times.get(test.id(), time.time())
        doc = (test.shortDescription() or "").strip()
        self.test_records.append({
            "id": test.id(),
            "module": getattr(test, "__class__", type(test)).__module__,
            "class": getattr(test, "__class__", type(test)).__name__,
            "method": getattr(test, "_testMethodName", str(test)),
            "description": doc,
            "status": "ERROR",
            "error": self._exc_info_to_string(err, test),
            "time_taken": round(elapsed, 4)
        })

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.test_records.append({
            "id": test.id(),
            "module": getattr(test, "__class__", type(test)).__module__,
            "class": getattr(test, "__class__", type(test)).__name__,
            "method": getattr(test, "_testMethodName", str(test)),
            "description": (test.shortDescription() or "").strip(),
            "status": "SKIPPED",
            "reason": str(reason),
            "time_taken": 0.0
        })

def get_data_dir_hashes():
    """Computes SHA-256 hashes for all files under data/."""
    hashes = {}
    if not os.path.exists(DATA_DIR):  # pragma: no cover
        return hashes
    for fname in sorted(os.listdir(DATA_DIR)):
        if fname.endswith(".json"):
            fpath = os.path.join(DATA_DIR, fname)
            with open(fpath, "rb") as f:
                hashes[fname] = hashlib.sha256(f.read()).hexdigest()
    return hashes

def run_all_tests():
    """Discovers and executes all test cases under tests/."""
    test_lock = os.path.join(REPO_DIR, ".test_lock")
    try:
        with open(test_lock, "w") as f:
            f.write("1")
    except OSError:  # pragma: no cover
        pass

    try:
        print("=" * 80)
        print("           UNIT TEST SUITE EXECUTION (tests/)")
        print("=" * 80)
        print(f"Target directory:  {TESTS_DIR}")
        print("Discovery pattern: test_*.py")
        print("-" * 80)

        # Snapshot data/*.json hashes before running tests
        pre_hashes = get_data_dir_hashes()

        loader = unittest.TestLoader()
        suite = loader.discover(start_dir=TESTS_DIR, pattern="test_*.py")

        start_time = time.time()
        is_summary_only = ("--summary-only" in sys.argv)
        stream_target = open(os.devnull, "w") if is_summary_only else None
        runner = unittest.TextTestRunner(resultclass=RecordingTestResult, verbosity=0 if is_summary_only else 2, stream=stream_target)
        result = runner.run(suite)
        elapsed = time.time() - start_time

        # Save comprehensive test report cache for LAST-COMMIT-REPORT.md
        try:
            modules_dict = {}
            for rec in getattr(result, "test_records", []):
                mod = rec["module"]
                if mod not in modules_dict:
                    modules_dict[mod] = {"total": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0, "tests": []}
                modules_dict[mod]["total"] += 1
                if rec["status"] == "PASSED":
                    modules_dict[mod]["passed"] += 1
                elif rec["status"] == "FAILED":
                    modules_dict[mod]["failed"] += 1
                elif rec["status"] == "ERROR":
                    modules_dict[mod]["errors"] += 1
                elif rec["status"] == "SKIPPED":
                    modules_dict[mod]["skipped"] += 1
                modules_dict[mod]["tests"].append(rec)

            cache_path = os.path.join(TESTS_DIR, ".test_reports_cache.json")
            cache_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tests_run": result.testsRun,
                "passed": result.testsRun - len(result.failures) - len(result.errors),
                "failures": len(result.failures),
                "errors": len(result.errors),
                "skipped": len(result.skipped),
                "time_taken": round(elapsed, 2),
                "status": "PASSED" if result.wasSuccessful() else "FAILED",
                "modules": modules_dict,
                "records": getattr(result, "test_records", [])
            }
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:  # pragma: no cover
            print(f"[run_tests] Note: test cache write exception: {e}")

        # Snapshot data/*.json hashes after running tests
        post_hashes = get_data_dir_hashes()
        data_tampered = False
        for fname, pre_hash in pre_hashes.items():
            post_hash = post_hashes.get(fname)
            if post_hash != pre_hash:
                print(f"CRITICAL ERROR: data/{fname} was mutated during test execution!")
                print(f"  Pre-test SHA-256:  {pre_hash}")
                print(f"  Post-test SHA-256: {post_hash}")
                data_tampered = True

        for fname in post_hashes:
            if fname not in pre_hashes:
                print(f"CRITICAL ERROR: New file data/{fname} was created during test execution!")
                data_tampered = True

        print("-" * 80)
        print("PYTHON UNIT TESTS CATEGORY SUMMARY (tests/)")
        print("-" * 80)
        cat_map = {
            "test_ssot": "SSOT Build & Validation",
            "test_pricing_and_policy_invariants": "Pricing & Policy Invariants",
            "test_check_links": "Link & Anchor Integrity",
            "test_html_ui": "HTML UI Verification",
            "test_python_tools": "Python Tooling & Build Scripts",
            "test_coverage_report": "Reports & Coverage",
            "test_last_commit_report": "Reports & Coverage",
        }
        cat_stats = {}
        for rec in getattr(result, "test_records", []):
            cat = cat_map.get(rec["module"], "Other Unit Tests")
            if cat not in cat_stats:
                cat_stats[cat] = {"total": 0, "passed": 0, "failed": 0}
            cat_stats[cat]["total"] += 1
            if rec["status"] == "PASSED":
                cat_stats[cat]["passed"] += 1
            elif rec["status"] in ("FAILED", "ERROR"):
                cat_stats[cat]["failed"] += 1
        if cat_stats:
            headers = ("Category", "Tests", "Passed", "Failed", "Status")
            rows = [
                (
                    c,
                    str(s["total"]),
                    str(s["passed"]),
                    str(s["failed"]),
                    "PASSED" if s["failed"] == 0 else "FAILED",
                )
                for c, s in cat_stats.items()
            ]
            rows.append((
                "TOTAL",
                str(result.testsRun),
                str(result.testsRun - len(result.failures) - len(result.errors)),
                str(len(result.failures) + len(result.errors)),
                "PASSED" if result.wasSuccessful() else "FAILED",
            ))
            widths = [max(len(headers[i]), max(len(r[i]) for r in rows)) for i in range(len(headers))]
            sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
            fmt = lambda r: "| " + " | ".join(r[i].ljust(widths[i]) if i == 0 else r[i].rjust(widths[i]) if i in (1, 2, 3) else r[i].ljust(widths[i]) for i in range(len(headers))) + " |"
            print(sep)
            print(fmt(headers))
            print(sep)
            for idx, r in enumerate(rows):
                if idx == len(rows) - 1:
                    print(sep)
                print(fmt(r))
            print(sep)
        print(f"Tests Run:   {result.testsRun} | Failures: {len(result.failures)} | Errors: {len(result.errors)} | Skipped: {len(result.skipped)} | Time: {elapsed:.2f}s")
        print("=" * 80)

        if data_tampered:
            print("RESULT: FAILED DUE TO DATA DIRECTORY MUTATION (SSOT VIOLATION).")
            print("Files under data/*.json are immutable and must never be modified by tests.")
            print("=" * 80)
            return 1

        failed_dir = os.path.join(REPO_DIR, "failed-tests")
        failed_report_path = os.path.join(failed_dir, "failed-python-tests.md")

        if result.wasSuccessful():
            print("RESULT: ALL TESTS PASSED.")
            print("=" * 80)
            if os.path.exists(failed_report_path):
                try:
                    os.remove(failed_report_path)
                except OSError:  # pragma: no cover
                    pass
            return 0
        else:
            print("RESULT: TEST SUITE FAILED.")
            print("=" * 80)
            try:
                os.makedirs(failed_dir, exist_ok=True)
                buf = []
                for test, err in result.failures:
                    buf.append(f"FAIL: {test}\n{err}")
                for test, err in result.errors:
                    buf.append(f"ERROR: {test}\n{err}")
                total_fails = len(result.failures) + len(result.errors)
                ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                report_lines = [
                    "# Failed Python Unit Tests Report",
                    "",
                    f"- **Timestamp**: {ts}",
                    f"- **Tests Run**: {result.testsRun}",
                    f"- **Failures**: {len(result.failures)}",
                    f"- **Errors**: {len(result.errors)}",
                    f"- **Total Failed**: {total_fails}",
                    "",
                    "## Failure Details",
                    "",
                    "```text",
                    "\n".join(buf),
                    "```",
                    "",
                ]
                with open(failed_report_path, "w", encoding="utf-8") as rf:
                    rf.write("\n".join(report_lines))
                print(f"[run_tests] Wrote failure report to: {os.path.relpath(failed_report_path, REPO_DIR)}")
            except Exception as e:  # pragma: no cover
                print(f"[run_tests] Note: failed report generation notice: {e}")
            return 1
    finally:
        if os.path.exists(test_lock):
            try:
                os.remove(test_lock)
            except OSError:  # pragma: no cover
                pass

def run_js_and_html_suites(summary_only=False):
    """Runs the JavaScript pricing test suite and HTML UI verification suite."""
    js_cmd = ["node", os.path.join(REPO_DIR, "pricing-tests.js")]
    html_cmd = [sys.executable, os.path.join(REPO_DIR, "python", "run_html_tests.py")]
    if summary_only:
        html_cmd.append("--summary-only")

    print("\n" + "=" * 80)
    print("           RUNNING JAVASCRIPT UNIT TEST SUITE (pricing-tests.js)")
    print("=" * 80)
    js_res = subprocess.run(js_cmd, cwd=REPO_DIR)

    print("\n" + "=" * 80)
    print("           RUNNING HTML UI TEST SUITE (python/run_html_tests.py)")
    print("=" * 80)
    html_res = subprocess.run(html_cmd, cwd=REPO_DIR)

    return js_res.returncode, html_res.returncode


def run_full_test_suite():
    """Executes Python, JavaScript, and HTML UI test suites and returns combined exit code."""
    summary_only = ("--summary-only" in sys.argv)
    py_code = run_all_tests()
    if "--python-only" in sys.argv:
        return py_code

    js_code, html_code = run_js_and_html_suites(summary_only=summary_only)

    print("\n" + "=" * 80)
    print("                 COMBINED TEST SUITE EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Python Unit Tests (tests/):          {'PASSED' if py_code == 0 else 'FAILED'}")
    print(f"JavaScript Tests (pricing-tests.js): {'PASSED' if js_code == 0 else 'FAILED'}")
    print(f"HTML UI Tests (run_html_tests.py):   {'PASSED' if html_code == 0 else 'FAILED'}")
    overall_ok = (py_code == 0 and js_code == 0 and html_code == 0)
    print("=" * 80)
    print(f"OVERALL RESULT: {'ALL SUITES PASSED.' if overall_ok else 'ONE OR MORE SUITES FAILED.'}")
    print("=" * 80)
    return 0 if overall_ok else 1


if __name__ == "__main__":  # pragma: no cover
    exit_code = run_full_test_suite()
    sys.exit(exit_code)
