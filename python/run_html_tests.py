#!/usr/bin/env python3
"""
run_html_tests.py - Dedicated HTML UI Test Suite Runner & Metrics Cache Generator.

Executes tests/test_html_ui.py using BeautifulSoup4 + lxml across all 85+ HTML pages,
prints a structured ASCII summary table to stdout, and persists results to
tests/.html_ui_test_reports_cache.json for METRICS.md reporting.

Strict Repo Compliance: Zero emojis, zero em-dashes.
"""

import json
import os
import sys
import time
import unittest
from datetime import datetime, timezone

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.join(REPO_DIR, "tests")
CACHE_PATH = os.path.join(TESTS_DIR, ".html_ui_test_reports_cache.json")

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)


class HTMLUITestResult(unittest.TextTestResult):
    """Captures per-test metadata for METRICS.md HTML UI summary table."""

    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.records = []
        self._starts = {}

    def startTest(self, test):
        super().startTest(test)
        self._starts[test.id()] = time.time()

    def _record(self, test, status, error=None):
        elapsed = round(time.time() - self._starts.get(test.id(), time.time()), 4)
        method = getattr(test, "_testMethodName", str(test))
        desc = (test.shortDescription() or "").strip()
        rec = {
            "method": method,
            "description": desc,
            "status": status,
            "time_taken": elapsed,
        }
        if error:
            rec["error"] = error
        self.records.append(rec)

    def addSuccess(self, test):
        super().addSuccess(test)
        self._record(test, "PASSED")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self._record(test, "FAILED", self._exc_info_to_string(err, test))

    def addError(self, test, err):
        super().addError(test, err)
        self._record(test, "ERROR", self._exc_info_to_string(err, test))

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self._record(test, "SKIPPED")


def run_html_ui_tests(quiet=False):
    start_time = time.time()
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=TESTS_DIR, pattern="test_html_ui.py")

    stream = open(os.devnull, "w") if quiet else sys.stdout
    runner = unittest.TextTestRunner(
        stream=stream,
        verbosity=0 if quiet else 2,
        resultclass=HTMLUITestResult,
    )
    result = runner.run(suite)
    if quiet:
        stream.close()

    elapsed = round(time.time() - start_time, 2)
    passed = sum(1 for r in result.records if r["status"] == "PASSED")
    failed = len(result.failures) + len(result.errors)
    total = result.testsRun
    status = "PASSED" if result.wasSuccessful() else "FAILED"

    # Discover page count and aggregate category-only summaries for reporting
    from tests.test_html_ui import (
        HTML_PARSER,
        HTML_TEST_CATEGORY_ORDER,
        discover_html_pages,
        get_html_test_category,
    )
    pages_scanned = len(discover_html_pages())
    framework_label = f"BeautifulSoup4 + {HTML_PARSER} (unittest)"

    categories = {}
    for cat in HTML_TEST_CATEGORY_ORDER:
        categories[cat] = {"total": 0, "passed": 0, "failed": 0, "skipped": 0}
    for rec in result.records:
        cat = get_html_test_category(rec["method"])
        rec["category"] = cat
        if cat not in categories:
            categories[cat] = {"total": 0, "passed": 0, "failed": 0, "skipped": 0}
        categories[cat]["total"] += 1
        if rec["status"] == "PASSED":
            categories[cat]["passed"] += 1
        elif rec["status"] == "SKIPPED":
            categories[cat]["skipped"] += 1
        else:
            categories[cat]["failed"] += 1
    categories = {k: v for k, v in categories.items() if v["total"] > 0}

    cache_payload = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "framework": framework_label,
        "pages_scanned": pages_scanned,
        "total": total,
        "passed": passed,
        "failed": failed,
        "elapsed_seconds": elapsed,
        "status": status,
        "categories": categories,
        "tests": result.records,
    }
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache_payload, f, indent=2)

    if not quiet:
        print("\n" + "=" * 80)
        print("            HTML UI VERIFICATION SUITE SUMMARY (test_html_ui.py)")
        print("=" * 80)
        print(f"Framework:     {framework_label}")
        print(f"Pages Scanned: {pages_scanned} HTML pages")
        headers = ("Category", "Total", "Passed", "Failed", "Status")
        rows = [
            (
                cat,
                str(stats["total"]),
                str(stats["passed"]),
                str(stats["failed"]),
                "PASS" if stats["failed"] == 0 else "FAIL",
            )
            for cat, stats in categories.items()
        ]
        rows.append(("TOTAL", str(total), str(passed), str(failed), status))
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
        print(f"Tests Run:     {total} | Passed: {passed} | Failed: {failed} | Time: {elapsed}s")
        print(f"Status:        {status}")
        print("=" * 80)

    failed_dir = os.path.join(REPO_DIR, "failed-tests")
    failed_report_path = os.path.join(failed_dir, "failed-html-tests.md")
    if result.wasSuccessful():
        if os.path.exists(failed_report_path):
            try:
                os.remove(failed_report_path)
            except OSError:
                pass
        return 0

    try:
        os.makedirs(failed_dir, exist_ok=True)
        buf = []
        for test, err in result.failures:
            buf.append(f"FAIL: {test}\n{err}")
        for test, err in result.errors:
            buf.append(f"ERROR: {test}\n{err}")
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        report_lines = [
            "# Failed HTML UI Tests Report",
            "",
            f"- **Timestamp**: {ts}",
            f"- **Tests Run**: {total}",
            f"- **Failed**: {failed}",
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
    except Exception:
        pass
    return 1


if __name__ == "__main__":
    sys.exit(run_html_ui_tests())
