#!/usr/bin/env python3
"""
test_coverage_report.py - Unit Tests for Test Coverage Reporter.

Strict Repo Compliance:
- Zero emojis.
- Zero em-dashes (spaced hyphens ' - ' or plain hyphens '-' used).
"""

import os
import sys
import unittest
import tempfile
import shutil
from unittest.mock import patch, MagicMock

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_DIR = os.path.join(REPO_DIR, "python")
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
        """Verifies retrieving git commit and branch info."""
        commit_hash, branch = cov_reporter.get_git_info()
        self.assertIsInstance(commit_hash, str)
        self.assertIsInstance(branch, str)
        self.assertTrue(len(commit_hash) > 0)
        self.assertTrue(len(branch) > 0)

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
                    "missing_lines": [10, 11, 12, 13, 14]
                }
            }
        }
        mock_js = {
            "pricing-calculator.js": {
                "path": os.path.join(REPO_DIR, "pricing-calculator.js"),
                "executable_lines": 600,
                "covered_lines": 600,
                "missed_lines": 0,
                "coverage_pct": 100.0,
                "missing": []
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
                self.assertNotIn(chr(0x2014), content)  # Zero em-dashes

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
