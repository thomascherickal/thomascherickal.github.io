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
    invoke_agy_healer,
    commit_verified_state,
    handle_test_verification_and_commit,
    parse_test_failures,
)
from unittest.mock import patch, MagicMock


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

    def test_compute_data_fingerprint_missing_directory(self):
        """Verifies compute_data_fingerprint handles non-existent directory."""
        fp = compute_data_fingerprint("/non/existent/data/path/xyz")
        self.assertEqual(fp, {})

    @patch("shutil.which", return_value="/usr/local/bin/agy")
    @patch("os.path.exists", return_value=True)
    @patch("subprocess.run")
    def test_invoke_agy_healer(self, mock_run, mock_exists, mock_which):
        """Verifies invoke_agy_healer constructs correct prompt and command."""
        invoke_agy_healer("Ran 5 tests", "FAIL: test_x", iteration=1, max_iterations=3)
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        self.assertEqual(args[0], "/usr/local/bin/agy")
        self.assertEqual(args[1], "--print")
        self.assertIn("FAIL: test_x", args[2])

    @patch("shutil.which", return_value=None)
    @patch("os.path.exists", return_value=False)
    def test_invoke_agy_healer_not_found(self, mock_exists, mock_which):
        """Verifies invoke_agy_healer raises FileNotFoundError when agy is absent."""
        with self.assertRaises(FileNotFoundError):
            invoke_agy_healer("stdout", "stderr")

    @patch("subprocess.run")
    def test_commit_verified_state_clean(self, mock_run):
        """Verifies commit_verified_state does nothing when working tree is clean."""
        mock_run.return_value = MagicMock(stdout="", stderr="")
        commit_verified_state("test commit")
        # Only git status is executed
        self.assertEqual(mock_run.call_count, 1)

    @patch("subprocess.run")
    def test_commit_verified_state_dirty(self, mock_run):
        """Verifies commit_verified_state stages and commits when working tree has changes."""
        # 1st call: git status
        # 2nd call: git add
        # 3rd call: git commit
        # 4th call: generate_last_commit_report.py
        mock_status = MagicMock(stdout=" M test.html")
        mock_add = MagicMock(returncode=0)
        mock_commit = MagicMock(stdout="[main abc] commit")
        mock_report = MagicMock(returncode=0)
        mock_run.side_effect = [mock_status, mock_add, mock_commit, mock_report]

        commit_verified_state("feat: update test")
        self.assertGreaterEqual(mock_run.call_count, 3)

    @patch("auto_healer.run_unit_tests")
    @patch("auto_healer.commit_verified_state")
    @patch("auto_healer.clean_stale_failure_reports")
    def test_handle_test_verification_all_pass(self, mock_clean, mock_commit, mock_run_tests):
        """Verifies handle_test_verification_and_commit when all tests pass."""
        mock_run_tests.return_value = (True, "OK", "", 0)
        res = handle_test_verification_and_commit()
        self.assertTrue(res)
        mock_clean.assert_called_once()
        mock_commit.assert_called_once()

    @patch("auto_healer.run_unit_tests")
    @patch("auto_healer.generate_failed_tests_report", return_value=["failed_tests_report.md"])
    @patch("auto_healer.commit_verified_state")
    def test_handle_test_verification_few_failures(self, mock_commit, mock_gen_report, mock_run_tests):
        """Verifies handle_test_verification_and_commit when < 5 tests fail."""
        mock_run_tests.return_value = (False, "FAILED", "FAIL: 1", 2)
        res = handle_test_verification_and_commit()
        self.assertTrue(res)
        mock_gen_report.assert_called_once()
        mock_commit.assert_called_once()

    def test_parse_test_failures_variations(self):
        """Verifies parse_test_failures extracts modules and errors accurately."""
        sample_stderr = (
            "======================================================================\n"
            "FAIL: test_a (test_mod_a.TestA.test_a)\n"
            "----------------------------------------------------------------------\n"
            "AssertionError: 1 != 2\n"
            "======================================================================\n"
            "ERROR: test_b (tests.test_mod_b.TestB.test_b)\n"
            "----------------------------------------------------------------------\n"
            "ValueError: bad value\n"
        )
        failures = parse_test_failures("", sample_stderr)
        self.assertIn("test_mod_a", failures)
        self.assertIn("test_mod_b", failures)
        self.assertEqual(len(failures["test_mod_a"]), 1)
        self.assertEqual(len(failures["test_mod_b"]), 1)


if __name__ == "__main__":
    unittest.main()

