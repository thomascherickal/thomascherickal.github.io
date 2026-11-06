#!/usr/bin/env python3
"""
test_coverage_report.py - Unit Tests for Test Coverage Reporter.

Strict Repo Compliance:
- Zero emojis.
- Zero em-dashes (spaced hyphens ' - ' or plain hyphens '-' used).
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_DIR = os.path.join(REPO_DIR, "python")
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)

import generate_test_coverage_report as cov_reporter


class TestCoverageReport(unittest.TestCase):
    """Verifies coverage formatting, metric computation, and report generation."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_format_line_ranges(self):
        """Verifies formatting line numbers into concise readable ranges."""
        self.assertEqual(cov_reporter.format_line_ranges([]), "-")
        self.assertEqual(cov_reporter.format_line_ranges([5]), "5")
        self.assertEqual(cov_reporter.format_line_ranges([1, 2, 3, 5, 7, 8, 9]), "1-3, 5, 7-9")
        self.assertEqual(cov_reporter.format_line_ranges([10, 11, 12]), "10-12")

    def test_get_git_info(self):
        """Verifies retrieving git commit and branch info, plus fallback on error."""
        commit_hash, branch = cov_reporter.get_git_info()
        self.assertIsInstance(commit_hash, str)
        self.assertIsInstance(branch, str)
        self.assertTrue(len(commit_hash) > 0)
        self.assertTrue(len(branch) > 0)

        with patch("subprocess.check_output", side_effect=RuntimeError("git err")):
            c_err, b_err = cov_reporter.get_git_info()
            self.assertEqual(c_err, "unknown")
            self.assertEqual(b_err, "main")

    def test_run_python_coverage_with_mocked_subprocess(self):
        """Verifies run_python_coverage handles fresh and missing .coverage files."""
        cov_json = os.path.join(self.temp_dir, "cov.json")

        def fake_subproc(cmd, **kwargs):
            if "json" in cmd:
                with open(cov_json, "w", encoding="utf-8") as f:
                    json.dump({"files": {"python/build.py": {"summary": {"num_statements": 10}}}}, f)

        # Case 1: no .coverage file -> runs coverage run + coverage json
        with patch.object(cov_reporter, "REPO_DIR", self.temp_dir), \
             patch("subprocess.run", side_effect=fake_subproc):
            res = cov_reporter.run_python_coverage(cov_json)
            self.assertIn("files", res)

        # Case 2: fresh .coverage exists, but json output fails to write
        fake_cov = os.path.join(self.temp_dir, ".coverage")
        with open(fake_cov, "w", encoding="utf-8") as f:
            f.write("cov")
        missing_json = os.path.join(self.temp_dir, "missing.json")
        with patch.object(cov_reporter, "REPO_DIR", self.temp_dir), \
             patch("subprocess.run"):
            res_empty = cov_reporter.run_python_coverage(missing_json)
            self.assertEqual(res_empty, {})

    def test_run_javascript_coverage_v8_parsing(self):
        """Verifies run_javascript_coverage parses V8 byte-offset ranges into line stats."""
        v8_dir = os.path.join(self.temp_dir, "v8")
        os.makedirs(v8_dir, exist_ok=True)

        # Create a tiny JS file in temp_dir
        js_file = os.path.join(self.temp_dir, "pricing-calculator.js")
        js_src = "// Comment line\nconst a = 1;\nconst b = 2;\n"
        with open(js_file, "w", encoding="utf-8") as f:
            f.write(js_src)

        def fake_node_run(cmd, **kwargs):
            # Write one valid V8 json covering only 'const a = 1;' and one corrupt json
            v8_valid = os.path.join(v8_dir, "cov1.json")
            with open(v8_valid, "w", encoding="utf-8") as f:
                json.dump({
                    "result": [{
                        "url": "file:///path/pricing-calculator.js",
                        "functions": [{
                            "ranges": [
                                {"startOffset": 0, "endOffset": 28, "count": 1},
                                {"startOffset": 28, "endOffset": len(js_src), "count": 0}
                            ]
                        }]
                    }]
                }, f)
            with open(os.path.join(v8_dir, "corrupt.json"), "w", encoding="utf-8") as f:
                f.write("{corrupt")

        with patch.object(cov_reporter, "REPO_DIR", self.temp_dir), \
             patch("subprocess.run", side_effect=fake_node_run):
            results = cov_reporter.run_javascript_coverage(v8_dir)
            self.assertIn("pricing-calculator.js", results)
            info = results["pricing-calculator.js"]
            self.assertEqual(info["executable_lines"], 2)
            self.assertEqual(info["covered_lines"], 1)
            self.assertEqual(info["missed_lines"], 1)

    def test_generate_coverage_report_mocked(self):
        """Verifies report generation with mocked coverage engines."""
        mock_py = {
            "files": {
                os.path.join(REPO_DIR, "python", "check_links.py"): {
                    "summary": {
                        "num_statements": 100,
                        "covered_lines": 95,
                        "missing_lines": 5,
                        "percent_covered": 95.0
                    },
                    "missing_lines": list(range(1, 25))
                }
            }
        }
        mock_js = {
            "pricing-calculator.js": {
                "path": os.path.join(REPO_DIR, "pricing-calculator.js"),
                "executable_lines": 600,
                "covered_lines": 590,
                "missed_lines": 10,
                "coverage_pct": 98.33,
                "missing": list(range(1, 25))
            }
        }

        report_file = os.path.join(self.temp_dir, "LAST-COMMIT-REPORT.md")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("# Git Commit Report\n\nSome initial commit details.\n")

        with patch("generate_test_coverage_report.run_python_coverage", return_value=mock_py):
            with patch("generate_test_coverage_report.run_javascript_coverage", return_value=mock_js):
                content = cov_reporter.generate_coverage_report(output_path=report_file)
                self.assertTrue(os.path.exists(report_file))
                self.assertIn("# Git Commit Report", content)
                self.assertIn("## Test Coverage Report (Python & JavaScript)", content)
                self.assertIn("### Coverage Executive Summary", content)
                self.assertIn("pricing-calculator.js", content)
                self.assertIn("check_links.py", content)
                self.assertNotIn(chr(0x2014), content)

                # Test idempotency (replacing coverage block without duplicating)
                content_second = cov_reporter.generate_coverage_report(output_path=report_file)
                self.assertEqual(content_second.count("## Test Coverage Report"), 1)

    def test_legacy_coverage_file_cleanup(self):
        """Verifies that legacy TEST_COVERAGE_REPORT.md is deleted when report runs."""
        legacy_file = os.path.join(self.temp_dir, "TEST_COVERAGE_REPORT.md")
        with open(legacy_file, "w", encoding="utf-8") as f:
            f.write("Old coverage report\n")

        report_file = os.path.join(self.temp_dir, "LAST-COMMIT-REPORT.md")
        with patch.object(cov_reporter, "LEGACY_COVERAGE_PATH", legacy_file):
            with patch("generate_test_coverage_report.run_python_coverage", return_value={"files": {}}):
                with patch("generate_test_coverage_report.run_javascript_coverage", return_value={}):
                    cov_reporter.generate_coverage_report(output_path=report_file)
                    self.assertFalse(os.path.exists(legacy_file))


if __name__ == "__main__":
    unittest.main()
