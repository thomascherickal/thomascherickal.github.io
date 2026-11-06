#!/usr/bin/env python3
"""
auto_healer.py - Autonomous Test-Driven Auto-Healer for NLP-Fuzzy-Expert.

When post-synchronization tests fail:
- If fewer than 5 tests fail, does not abort the commit: generates failed_tests_report.md
  in the root directory, proceeds with committing the synchronized state, and prints
  instructions for the user along with the ready-to-run AGY prompt.
- If 5 or more tests fail, triggers the agy CLI auto-healing loop to heal regressions.

Strict Repo Compliance:
- Zero emojis.
- Zero em-dashes (spaced hyphens ' - ' or plain hyphens '-' used).
- Strict data/ directory immutability.
"""

import os
import sys
import subprocess
import shutil
import hashlib
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(REPO_DIR, "data")


def compute_data_fingerprint(data_dir=DATA_DIR):
    """Computes SHA-256 fingerprint of all data/*.json files to enforce immutability."""
    fingerprints = {}
    if not os.path.exists(data_dir):
        return fingerprints
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith(".json"):
            fpath = os.path.join(data_dir, fname)
            with open(fpath, "rb") as f:
                fingerprints[fname] = hashlib.sha256(f.read()).hexdigest()
    return fingerprints


def verify_data_untouched(initial_fingerprint, data_dir=DATA_DIR):
    """Verifies that no files in data/ were modified or reverted."""
    current_fingerprint = compute_data_fingerprint(data_dir)
    if initial_fingerprint != current_fingerprint:
        raise PermissionError(
            "DATA MUTATION BLOCKED: The data/ directory was altered during auto-healing. "
            "Files under data/ must remain strictly immutable and cannot be undone or mutated."
        )


def run_unit_tests():
    """Runs the python unit test suite and returns (passed, stdout, stderr, failure_count)."""
    test_cmd = [sys.executable, "-m", "unittest", "discover", "-s", os.path.join(REPO_DIR, "tests"), "-v"]
    env = os.environ.copy()
    env["ANTIGRAVITY_IN_TEST"] = "1"
    run = subprocess.run(test_cmd, cwd=REPO_DIR, capture_output=True, text=True, env=env)
    
    output = run.stdout + "\n" + run.stderr
    # Count failed / error tests
    fail_matches = re.findall(r"^(?:FAIL|ERROR):\s+([^\s]+)", output, re.MULTILINE)
    failure_count = len(fail_matches)
    if failure_count == 0 and run.returncode != 0:
        failure_count = 1

    return (run.returncode == 0), run.stdout, run.stderr, failure_count


def extract_failures_by_module(stdout, stderr):
    """
    Parses test runner output and groups failure/error blocks by test suite/module name.
    Returns {module_name: [failure_block_strings]}.
    """
    output = stdout + "\n" + stderr
    chunks = re.split(r"={50,}", output)
    by_mod = {}

    for chunk in chunks:
        chunk = chunk.strip()
        if not (chunk.startswith("FAIL:") or chunk.startswith("ERROR:")):
            continue
        first_line = chunk.splitlines()[0]
        # Match FAIL: test_name (module.class...)
        m = re.search(r"\(([^)]+)\)", first_line)
        mod_name = "unknown"
        if m:
            parts = m.group(1).split(".")
            for p in parts:
                if p.startswith("test_"):
                    mod_name = p
                    break
            if mod_name == "unknown" and len(parts) > 1:
                mod_name = parts[-2] if parts[-1].startswith("Test") else parts[-1]
            elif mod_name == "unknown" and len(parts) == 1:
                mod_name = parts[0]
        if mod_name == "unknown":
            fm = re.search(r'File ".*?/(test_[^\.]+)\.py"', chunk)
            if fm:
                mod_name = fm.group(1)
        by_mod.setdefault(mod_name, []).append(chunk)

    if not by_mod:
        # Fallback to searching lines
        lines = output.splitlines()
        for i, line in enumerate(lines):
            if line.startswith("FAIL:") or line.startswith("ERROR:"):
                m = re.search(r"\(([^)]+)\)", line)
                mod_name = "unknown"
                if m:
                    for p in m.group(1).split("."):
                        if p.startswith("test_"):
                            mod_name = p
                            break
                block = "\n".join(lines[i:min(i + 30, len(lines))])
                by_mod.setdefault(mod_name, []).append(block)

    if not by_mod:
        by_mod["unit"] = [output[-2000:]]

    return by_mod


# Alias for backwards compatibility
parse_test_failures = extract_failures_by_module


def clean_stale_failure_reports():
    """Removes any existing failed_*_tests.report.md, failed_*_tests_report.md, and failed_tests_report.md files."""
    for fname in os.listdir(REPO_DIR):
        if fname.startswith("failed_") and (
            fname.endswith("_tests.report.md")
            or fname.endswith("_tests_report.md")
            or fname == "failed_tests_report.md"
        ):
            try:
                os.remove(os.path.join(REPO_DIR, fname))
            except OSError:  # pragma: no cover
                pass


def generate_failed_tests_report(stdout, stderr, failure_count):
    """
    Generates failed_[module_name]_tests.report.md for each failing test suite,
    as well as failed_tests_report.md in the repository root directory.
    Instructs the user how to fix the tests and provides the exact ready-to-run AGY prompt.
    """
    by_mod = extract_failures_by_module(stdout, stderr)
    generated_reports = []

    for mod_name, blocks in by_mod.items():
        summary_text = "\n\n".join(blocks)
        agy_prompt = (
            f"Fix the {len(blocks)} unit test failure(s) in module '{mod_name}' in the repository.\n\n"
            f"FAILURES:\n```\n{summary_text}\n```\n\n"
            f"RULES TO FOLLOW:\n"
            f"1. Strictly do NOT modify or undo any files in the 'data/' directory. It is the immutable Single Source of Truth.\n"
            f"2. Ensure zero emojis and zero em-dashes across all files.\n"
            f"3. Verify all tests pass with 'python3 python/run_tests.py' after fixing.\n"
        )

        report_content = f"""# Failed {mod_name} Tests Report

Generated by `nlp-fuzzy-expert/auto_healer.py` post-synchronization test runner for test suite `{mod_name}`.

## Summary
- **Module**: `{mod_name}`
- **Failures in Module**: {len(blocks)}
- **Total Test Failures**: {failure_count}
- **Action Required**: The developer or AGY tool must resolve the failing assertions below.

---

## Failure Details

```text
{summary_text}
```

---

## Instructions for Developer

1. Inspect the failing test assertion(s) in `tests/` for module `{mod_name}`.
2. Do **NOT** modify or revert any files under `data/`. The `data/` directory is the immutable authoritative specification.
3. Update the matching assertions or regex rules in `nlp-fuzzy-expert/rules.py` or `tests/` to align with the authoritative data.
4. Maintain zero emojis and zero em-dashes across all files.
5. Run `python3 python/run_tests.py` to confirm the test suite is 100% green.

---

## Ready-to-Run AGY Tool Prompt

To automatically resolve these test failures using the Google Antigravity CLI, run:

```bash
agy --print "{agy_prompt.strip()}"
```

Or pass this prompt directly into your active Antigravity session:

> {agy_prompt.strip()}
"""
        primary_name = f"failed_{mod_name}_tests.report.md"
        primary_path = os.path.join(REPO_DIR, primary_name)
        with open(primary_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        generated_reports.append(primary_name)
        print(f"\n[Auto-Healer] Wrote failure report to: {primary_name}")

        if mod_name.startswith("test_"):
            clean_name = mod_name[5:]
            clean_filename = f"failed_{clean_name}_tests.report.md"
            clean_path = os.path.join(REPO_DIR, clean_filename)
            with open(clean_path, "w", encoding="utf-8") as f:
                f.write(report_content)
            generated_reports.append(clean_filename)

        compat_filename = f"failed_{mod_name}_tests_report.md"
        compat_path = os.path.join(REPO_DIR, compat_filename)
        with open(compat_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        generated_reports.append(compat_filename)

    aggregate_path = os.path.join(REPO_DIR, "failed_tests_report.md")
    total_summary = "\n\n".join(["\n\n".join(b) for b in by_mod.values()])
    agy_total_prompt = (
        f"Fix the following {failure_count} unit test failure(s) in the repository.\n\n"
        f"FAILURES:\n```\n{total_summary}\n```\n\n"
        f"RULES TO FOLLOW:\n"
        f"1. Strictly do NOT modify or undo any files in the 'data/' directory. It is the immutable Single Source of Truth.\n"
        f"2. Ensure zero emojis and zero em-dashes across all files.\n"
        f"3. Verify all tests pass with 'python3 python/run_tests.py' after fixing.\n"
    )

    aggregate_content = f"""# Failed Unit Tests Report

Generated by `nlp-fuzzy-expert/synchronizer.py` post-synchronization test runner.

## Summary
- **Status**: Synchronized with test warnings ({failure_count} failure(s) detected across {len(by_mod)} suite(s))
- **Failing Suites**: {', '.join(by_mod.keys())}
- **Action Required**: The developer or AGY tool must resolve the failing assertions below.

---

## Failure Details

```text
{total_summary}
```

---

## Instructions for Developer

1. Inspect the failing test assertion(s) in `tests/`.
2. Do **NOT** modify or revert any files under `data/`. The `data/` directory is the immutable authoritative specification.
3. Update the matching assertions or regex rules in `nlp-fuzzy-expert/rules.py` or `tests/` to align with the authoritative data.
4. Maintain zero emojis and zero em-dashes across all files.
5. Run `python3 python/run_tests.py` to confirm the test suite is 100% green.

---

## Ready-to-Run AGY Tool Prompt

To automatically resolve these test failures using the Google Antigravity CLI, run:

```bash
agy --print "{agy_total_prompt.strip()}"
```

Or pass this prompt directly into your active Antigravity session:

> {agy_total_prompt.strip()}
"""
    with open(aggregate_path, "w", encoding="utf-8") as f:
        f.write(aggregate_content)
    generated_reports.append("failed_tests_report.md")
    print(f"[Auto-Healer] Wrote aggregate report to: failed_tests_report.md")

    return generated_reports


def invoke_agy_healer(test_stdout, test_stderr, iteration=1, max_iterations=5):
    """Invokes the agy CLI non-interactively to inspect and correct test failures."""
    agy_bin = shutil.which("agy") or "/home/thomas/.local/bin/agy"
    if not os.path.exists(agy_bin):
        raise FileNotFoundError(f"Google Antigravity CLI ('agy') not found at '{agy_bin}'.")

    prompt = (
        f"You are the autonomous repair agent for the website repository. "
        f"A test failure occurred during post-synchronization verification (Iteration {iteration}/{max_iterations}).\n\n"
        f"TEST FAILURE OUTPUT:\n{test_stdout}\n{test_stderr}\n\n"
        f"INSTRUCTIONS:\n"
        f"1. Analyze the test failure above.\n"
        f"2. Inspect the failing test or rule file causing the assertion failure.\n"
        f"3. Correct the changes in code/tests/rules so that the tests match the authoritative data/ definitions.\n"
        f"4. CRITICAL RULE: Under NO circumstances are you allowed to modify, overwrite, undo, or revert any file in the 'data/' directory. "
        f"The 'data/' directory is the immutable source of truth.\n"
        f"5. Maintain zero emojis and zero em-dashes across all modified files.\n"
        f"6. Make the necessary surgical edits now."
    )

    print(f"\n[Auto-Healer] Invoking agy CLI to diagnose and heal failure (Attempt {iteration}/{max_iterations})...")
    agy_cmd = [
        agy_bin,
        "--print",
        prompt
    ]

    subprocess.run(agy_cmd, cwd=REPO_DIR, check=False)


def handle_test_verification_and_commit(max_iterations=5):
    """
    Handles post-synchronization test verification:
    - If all tests pass: commits verified state.
    - If fewer than 5 tests fail: does not abort. Generates failed_tests_report.md,
      proceeds with commit, and instructs the user with the AGY prompt.
    - If 5 or more tests fail: triggers the agy auto-healing loop.
    """
    initial_data_fp = compute_data_fingerprint()
    passed, stdout, stderr, failure_count = run_unit_tests()
    verify_data_untouched(initial_data_fp)

    if passed:
        print("\n[Auto-Healer] Post-sync verification: ALL UNIT TESTS PASSED.")
        clean_stale_failure_reports()
        commit_verified_state("feat(synchronizer): synchronize repository state with authoritative data")
        return True

    print(f"\n[Auto-Healer] Post-sync verification: {failure_count} UNIT TEST(S) FAILED.")

    if failure_count < 5:
        print(f"[Auto-Healer] Failure count ({failure_count}) is less than 5. Continuing with commit without aborting.")
        reports = generate_failed_tests_report(stdout, stderr, failure_count)
        first_rep = reports[0] if reports else "failed_tests_report.md"
        commit_verified_state(f"feat(synchronizer): synchronize repository state ({failure_count} test failure warnings - see {first_rep})")
        print("\n" + "=" * 80)
        print(" ACTION REQUIRED: UNIT TEST DISCREPANCY DETECTED")
        print("=" * 80)
        print(f"Less than 5 tests failed ({failure_count} failure(s)). The commit has proceeded.")
        print(f"Comprehensive report(s) created: {', '.join(reports)}")
        print("To fix the tests automatically with AGY, run:")
        print(f"  agy --print \"Fix the unit test failures reported in {first_rep}\"")
        print("=" * 80 + "\n")
        return True

    # 5 or more failures: run full auto-healing loop
    print(f"[Auto-Healer] Failure count ({failure_count}) >= 5. Initiating AGY auto-healing loop...")
    for iteration in range(1, max_iterations + 1):
        invoke_agy_healer(stdout, stderr, iteration=iteration, max_iterations=max_iterations)
        verify_data_untouched(initial_data_fp)

        passed, stdout, stderr, failure_count = run_unit_tests()
        verify_data_untouched(initial_data_fp)

        if passed:
            print(f"\n[Auto-Healer] Test suite is completely GREEN after {iteration} repair iteration(s)!")
            clean_stale_failure_reports()
            commit_verified_state("feat(synchronizer): auto-heal and synchronize repository state with authoritative data")
            return True

        if failure_count < 5:
            print(f"[Auto-Healer] Failures reduced to {failure_count} (< 5). Proceeding with commit and report.")
            reports = generate_failed_tests_report(stdout, stderr, failure_count)
            first_rep = reports[0] if reports else "failed_tests_report.md"
            commit_verified_state(f"feat(synchronizer): synchronize repository state ({failure_count} test failure warnings)")
            return True

    return False


def commit_verified_state(commit_msg="feat(synchronizer): synchronize repository state with authoritative data"):
    """Generates pre-commit reports, stages all verified changes, and commits them cleanly with zero post-commit modifications."""
    print(f"\n[Auto-Healer] Executing commit: '{commit_msg}'...")
    try:
        status = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_DIR, capture_output=True, text=True)
        if not status.stdout.strip():
            print("  -> Working tree is clean. Nothing to commit.")
            return

        subprocess.run(["git", "add", "-A"], cwd=REPO_DIR, check=True)

        # Generate LAST-COMMIT-REPORT.md and append coverage BEFORE committing so working tree stays clean after commit
        report_script = os.path.join(REPO_DIR, "python", "generate_last_commit_report.py")
        if os.path.exists(report_script):
            subprocess.run([sys.executable, report_script], cwd=REPO_DIR, capture_output=True, text=True)

        cov_script = os.path.join(REPO_DIR, "python", "generate_test_coverage_report.py")
        if os.path.exists(cov_script):
            subprocess.run([sys.executable, cov_script], cwd=REPO_DIR, capture_output=True, text=True)

        subprocess.run(["git", "add", "-A"], cwd=REPO_DIR, check=True)
        commit_res = subprocess.run(["git", "commit", "--no-verify", "-m", commit_msg], cwd=REPO_DIR, capture_output=True, text=True)
        print("  -> Git commit output:")
        print(commit_res.stdout.strip())
        if commit_res.returncode == 0:
            subprocess.run(["git", "push"], cwd=REPO_DIR, capture_output=True, text=True)
    except Exception as e:
        print(f"  -> Git commit notice: {e}")
