#!/usr/bin/env python3
"""
test_autonomous_commit_healer.py - Unit Tests for python/autonomous_commit_healer.py.

Verifies:
1. Multi-pass autonomous healing loop fixes errors and terminates on green.
2. Terminal ASCII tabular report formatting for both clean and healed runs.
3. Cache persistence to .auto_heal_report_cache.json and LAST-COMMIT-REPORT.md rendering.
4. Strict data/*.json SHA-256 immutability enforcement.
5. Python and JS failure output parsers.

Strict Repo Compliance: Zero emojis, zero em-dashes.
"""

import json
import os
import sys
import tempfile
import unittest

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(REPO_DIR, "python") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_DIR, "python"))

import autonomous_commit_healer as healer


class TestAutonomousCommitHealer(unittest.TestCase):
    """Tests for the autonomous self-healing pre-commit loop and tabular reporting."""

    def test_parse_python_and_js_failures(self):
        """Verifies failure parsers extract suite, test name, and error message."""
        py_out = (
            "FAIL: test_demo (test_sample.TestDemo)\n"
            "Traceback:\n"
            "AssertionError: expected 10 got 5\n"
        )
        py_fails = healer.parse_python_failures(py_out)
        self.assertEqual(len(py_fails), 1)
        self.assertEqual(py_fails[0]["suite"], "Python")
        self.assertIn("test_demo", py_fails[0]["test"])
        self.assertIn("expected 10 got 5", py_fails[0]["error"])

        js_out = (
            "- FAIL: [Flagship] Personal Flagship 10 seats\n"
            "  Reason: Expected $1000, got $2000\n"
        )
        js_fails = healer.parse_js_failures(js_out)
        self.assertEqual(len(js_fails), 1)
        self.assertEqual(js_fails[0]["suite"], "JavaScript")
        self.assertIn("Expected $1000", js_fails[0]["error"])

    def test_healing_loop_fixes_errors_across_passes_and_writes_cache(self):
        """Verifies healing loop runs until all suites pass and marks fixed errors as FIXED."""
        calls = {"n": 0}

        def fake_runner():
            calls["n"] += 1
            if calls["n"] == 1:
                return False, [{"suite": "HTML UI", "test": "test_44", "error": "stale 20% copy"}], "log1"
            return True, [], ""

        def fake_fixer(fails, logs):
            return True, "Updated parity-pricing.html"

        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "data"), exist_ok=True)
            cache_path = os.path.join(tmp, ".auto_heal_report_cache.json")
            passed, history = healer.run_healing_loop(
                max_passes=3,
                repo_dir=tmp,
                cache_path=cache_path,
                runner_fn=fake_runner,
                fixer_fn=fake_fixer,
            )
            self.assertTrue(passed)
            self.assertEqual(calls["n"], 2)
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]["status"], "FIXED")
            self.assertTrue(os.path.exists(cache_path))
            with open(cache_path, encoding="utf-8") as cf:
                data = json.load(cf)
            self.assertEqual(data["errors_fixed"], 1)
            self.assertEqual(data["errors_remaining"], 0)
            self.assertGreaterEqual(len(data["action_log"]), 4)

    def test_no_change_fast_path_skips_redundant_execution(self):
        """Verifies clean repo matching cached fingerprint takes instant fast-path."""
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = os.path.join(tmp, ".auto_heal_report_cache.json")
            fp, _ = healer.compute_repo_fingerprint(REPO_DIR)
            healer.save_heal_cache(
                [], True, 1,
                action_log=["Prior green pass"],
                repo_fingerprint=fp,
                elapsed_seconds=1.0,
                cache_path=cache_path,
            )
            passed, history = healer.run_healing_loop(
                max_passes=3,
                repo_dir=REPO_DIR,
                cache_path=cache_path,
                force=False,
            )
            self.assertTrue(passed)
            self.assertEqual(history, [])
            with open(cache_path, encoding="utf-8") as cf:
                data = json.load(cf)
            self.assertTrue(data["fast_path_used"])
            self.assertTrue(any("Fast-path cache hit" in a for a in data["action_log"]))

    def test_format_terminal_table_renders_clean_and_healed_rows(self):
        """Verifies ASCII table renders cleanly both with 0 errors and with healed errors."""
        clean_tbl = healer.format_terminal_table([])
        self.assertIn("0 errors detected", clean_tbl)
        self.assertIn("PASSED", clean_tbl)

        healed_tbl = healer.format_terminal_table([{
            "pass": 1,
            "suite": "Python",
            "test": "test_x",
            "error": "mismatch",
            "fix": "Fixed test_x",
            "status": "FIXED",
        }])
        self.assertIn("test_x", healed_tbl)
        self.assertIn("Fixed test_x", healed_tbl)
        self.assertIn("FIXED", healed_tbl)

    def test_data_immutability_blocks_mutation(self):
        """Verifies verify_data_immutable raises PermissionError if data/*.json changes."""
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "data")
            os.makedirs(data_dir)
            fpath = os.path.join(data_dir, "pricing.json")
            with open(fpath, "w", encoding="utf-8") as f:
                f.write('{"a": 1}')
            initial = healer.compute_data_hashes(data_dir)
            with open(fpath, "w", encoding="utf-8") as f:
                f.write('{"a": 2}')
            with self.assertRaises(PermissionError):
                healer.verify_data_immutable(initial, data_dir)


if __name__ == "__main__":
    unittest.main()
