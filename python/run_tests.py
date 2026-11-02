#!/usr/bin/env python3
"""
run_tests.py - Comprehensive Unit Test Suite Runner.

Discovers and executes all unit tests under tests/ directory,
evaluates test outcomes, and reports a structured summary.
Exits with code 0 on complete pass, or code 1 if any failure occurs.

Strict Repo Compliance:
- Zero emojis.
- Zero em-dashes (spaced hyphens ' - ' or plain hyphens '-' used).
"""

import os
import sys
import time
import unittest

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.join(REPO_DIR, "tests")

# Mark test environment to prevent recursive post-sync test triggering
os.environ["ANTIGRAVITY_IN_TEST"] = "1"

def run_all_tests():
    """Discovers and executes all test cases under tests/."""
    print("=" * 80)
    print("           UNIT TEST SUITE EXECUTION (tests/)")
    print("=" * 80)
    print(f"Target directory:  {TESTS_DIR}")
    print("Discovery pattern: test_*.py")
    print("-" * 80)

    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=TESTS_DIR, pattern="test_*.py")

    start_time = time.time()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    elapsed = time.time() - start_time

    print("-" * 80)
    print("TEST SUITE SUMMARY REPORT")
    print("-" * 80)
    print(f"Tests Run:   {result.testsRun}")
    print(f"Failures:    {len(result.failures)}")
    print(f"Errors:      {len(result.errors)}")
    print(f"Skipped:     {len(result.skipped)}")
    print(f"Time Taken:  {elapsed:.2f} seconds")
    print("=" * 80)

    if result.wasSuccessful():
        print("RESULT: ALL TESTS PASSED.")
        print("=" * 80)
        return 0
    else:
        print("RESULT: TEST SUITE FAILED.")
        print("=" * 80)
        return 1

if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)

