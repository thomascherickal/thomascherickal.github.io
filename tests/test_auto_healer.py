#!/usr/bin/env python3
"""
test_auto_healer.py - Unit Tests for Auto-Healer Module.

Strict Repo Compliance:
- Zero emojis.
- Zero em-dashes (spaced hyphens ' - ' or plain hyphens '-' used).
"""

import os
import sys
import unittest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(TESTS_DIR)
NLP_EXPERT_DIR = os.path.join(REPO_DIR, "nlp-fuzzy-expert")

sys.path.insert(0, NLP_EXPERT_DIR)
from auto_healer import (
    compute_data_fingerprint,
    verify_data_untouched,
    generate_failed_tests_report,
    clean_stale_failure_reports,
)


class TestAutoHealer(unittest.TestCase):
    """Verifies auto-healer fingerprinting, immutability guards, and report generation."""

    def test_compute_data_fingerprint(self):
        """Verifies that fingerprints are computed for all data/*.json files."""
        fp = compute_data_fingerprint()
        self.assertIsInstance(fp, dict)
        self.assertIn("pricing.json", fp)
        self.assertIn("tiers.json", fp)
        self.assertIn("config.json", fp)
        self.assertEqual(len(fp["pricing.json"]), 64)  # SHA-256 hex length

    def test_verify_data_untouched_success(self):
        """Verifies that identical fingerprints pass verification without raising."""
        fp = compute_data_fingerprint()
        try:
            verify_data_untouched(fp)
        except PermissionError:
            self.fail("verify_data_untouched raised PermissionError unexpectedly.")

    def test_verify_data_untouched_detects_mutation(self):
        """Verifies that modified fingerprints raise a PermissionError."""
        tampered_fp = {"pricing.json": "0000000000000000000000000000000000000000000000000000000000000000"}
        with self.assertRaises(PermissionError):
            verify_data_untouched(tampered_fp)

    def test_generate_failed_tests_report_omits_dangerously_skip_permissions(self):
        """Verifies that failed_tests_report.md generates properly without --dangerously-skip-permissions."""
        report_path = os.path.join(REPO_DIR, "failed_tests_report.md")
        sample_stdout = "Ran 10 tests in 1.0s"
        sample_stderr = "FAIL: test_sample (tests.test_sample)\nAssertionError: 1 != 2"
        try:
            generate_failed_tests_report(sample_stdout, sample_stderr, 1)
            self.assertTrue(os.path.exists(report_path))
            with open(report_path, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertIn("# Failed Unit Tests Report", content)
            self.assertIn("agy --print", content)
            self.assertNotIn("--dangerously-skip-permissions", content)
            self.assertIn("AssertionError: 1 != 2", content)
        finally:
            clean_stale_failure_reports()

    def test_generate_failed_module_tests_report(self):
        """Verifies that failed_[module_name]_tests.report.md is created for failing test suites."""
        sample_stdout = "Ran 5 tests in 0.5s"
        sample_stderr = (
            "======================================================================\n"
            "FAIL: test_pricing (tests.test_sync_pricing.TestSyncPricing.test_pricing)\n"
            "----------------------------------------------------------------------\n"
            "Traceback (most recent call last):\n"
            "  File \"/path/tests/test_sync_pricing.py\", line 45, in test_pricing\n"
            "    self.assertEqual(2500, 3000)\n"
            "AssertionError: 2500 != 3000\n"
        )
        mod_report_path = os.path.join(REPO_DIR, "failed_test_sync_pricing_tests.report.md")
        clean_mod_report_path = os.path.join(REPO_DIR, "failed_sync_pricing_tests.report.md")
        try:
            reports = generate_failed_tests_report(sample_stdout, sample_stderr, 1)
            self.assertTrue(os.path.exists(mod_report_path), f"{mod_report_path} was not created.")
            self.assertTrue(os.path.exists(clean_mod_report_path), f"{clean_mod_report_path} was not created.")

            with open(mod_report_path, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertIn("# Failed test_sync_pricing Tests Report", content)
            self.assertIn("2500 != 3000", content)
            self.assertIn("agy --print", content)
            self.assertNotIn("--dangerously-skip-permissions", content)
        finally:
            clean_stale_failure_reports()


if __name__ == "__main__":
    unittest.main()

