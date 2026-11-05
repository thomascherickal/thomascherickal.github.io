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
import hashlib
import unittest

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.join(REPO_DIR, "tests")
DATA_DIR = os.path.join(REPO_DIR, "data")

# Mark test environment to prevent recursive post-sync test triggering
os.environ["ANTIGRAVITY_IN_TEST"] = "1"

def get_data_dir_hashes():
    """Computes SHA-256 hashes for all files under data/."""
    hashes = {}
    if not os.path.exists(DATA_DIR):
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
    except OSError:
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
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        elapsed = time.time() - start_time

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
        print("TEST SUITE SUMMARY REPORT")
        print("-" * 80)
        print(f"Tests Run:   {result.testsRun}")
        print(f"Failures:    {len(result.failures)}")
        print(f"Errors:      {len(result.errors)}")
        print(f"Skipped:     {len(result.skipped)}")
        print(f"Time Taken:  {elapsed:.2f} seconds")
        print("=" * 80)

        if data_tampered:
            print("RESULT: FAILED DUE TO DATA DIRECTORY MUTATION (SSOT VIOLATION).")
            print("Files under data/*.json are immutable and must never be modified by tests.")
            print("=" * 80)
            return 1

        if result.wasSuccessful():
            print("RESULT: ALL TESTS PASSED.")
            print("=" * 80)
            try:
                sys.path.insert(0, os.path.join(REPO_DIR, "nlp-fuzzy-expert"))
                from auto_healer import clean_stale_failure_reports
                clean_stale_failure_reports()
            except Exception:
                pass
            return 0
        else:
            print("RESULT: TEST SUITE FAILED.")
            print("=" * 80)
            try:
                sys.path.insert(0, os.path.join(REPO_DIR, "nlp-fuzzy-expert"))
                from auto_healer import generate_failed_tests_report
                buf = []
                for test, err in result.failures:
                    buf.append(f"FAIL: {test}\n{err}")
                for test, err in result.errors:
                    buf.append(f"ERROR: {test}\n{err}")
                total_fails = len(result.failures) + len(result.errors)
                generate_failed_tests_report("", "\n".join(buf), total_fails)
            except Exception as e:
                print(f"[run_tests] Note: failed report generation notice: {e}")
            return 1
    finally:
        if os.path.exists(test_lock):
            try:
                os.remove(test_lock)
            except OSError:
                pass

if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
