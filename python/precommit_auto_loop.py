#!/usr/bin/env python3
"""
precommit_auto_loop.py - Repo-Scoped Autonomous Pre-Commit Test & Fix Loop.

Runs Python unit tests, JavaScript pricing tests, and HTML UI tests in a loop
before every git commit. If any test fails:
1. Parses all failing test errors across suites.
2. Applies deterministic in-repo sync fixes (sync_schemas.py, md_html_sync.py).
3. Invokes `agy --print` strictly scoped to this repository directory to fix
   every failing test in a single pass.
4. Enforces strict SHA-256 immutability on data/*.json and blocks any edits
   outside this repository.
5. Loops until all tests pass (or --max-passes is exhausted).

Strict Repo Compliance: Zero emojis, zero em-dashes, strictly scoped to REPO_DIR.
"""

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_DIR, "data")


def compute_data_hashes():
    hashes = {}
    if not os.path.isdir(DATA_DIR):
        return hashes
    for fname in sorted(os.listdir(DATA_DIR)):
        if fname.endswith(".json"):
            with open(os.path.join(DATA_DIR, fname), "rb") as f:
                hashes[fname] = hashlib.sha256(f.read()).hexdigest()
    return hashes


def verify_data_immutable(initial_hashes):
    if compute_data_hashes() != initial_hashes:
        raise PermissionError("DATA MUTATION BLOCKED: data/*.json was modified during auto-fix loop.")


def parse_failures(py_out, js_out, html_out):
    failures = []
    for label, out in (("Python", py_out), ("HTML UI", html_out)):
        for m in re.finditer(r"^(?:FAIL|ERROR):\s+([^\s]+)\s+\(([^)]+)\)", out, re.MULTILINE):
            tail = out[m.end():m.end() + 800]
            em = re.search(r"(?:AssertionError|Error|Exception):\s*(.+)", tail)
            msg = em.group(1).strip()[:120] if em else "Assertion failed"
            failures.append({"suite": label, "test": f"{m.group(2)}.{m.group(1)}", "error": msg})
    lines = js_out.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"^\s*-\s*FAIL:\s*(.+)", line)
        if m:
            reason = lines[i + 1].split("Reason:", 1)[1].strip()[:120] if i + 1 < len(lines) and "Reason:" in lines[i + 1] else "JS assertion failed"
            failures.append({"suite": "JavaScript", "test": m.group(1).strip(), "error": reason})
    return failures


def run_all_tests():
    env = os.environ.copy()
    env["ANTIGRAVITY_IN_TEST"] = "1"
    py_r = subprocess.run([sys.executable, "python/run_tests.py"], cwd=REPO_DIR, capture_output=True, text=True, env=env)
    js_r = subprocess.run(["node", "pricing-tests.js"], cwd=REPO_DIR, capture_output=True, text=True, env=env)
    ht_r = subprocess.run([sys.executable, "python/run_html_tests.py"], cwd=REPO_DIR, capture_output=True, text=True, env=env)

    py_out = py_r.stdout + "\n" + py_r.stderr
    js_out = js_r.stdout + "\n" + js_r.stderr
    ht_out = ht_r.stdout + "\n" + ht_r.stderr
    all_ok = (py_r.returncode == 0) and (js_r.returncode == 0) and (ht_r.returncode == 0)
    failures = [] if all_ok else parse_failures(py_out, js_out, ht_out)
    raw = f"{py_out[-2500:]}\n{js_out[-1500:]}\n{ht_out[-1500:]}"
    return all_ok, failures, raw


def print_summary_table(history):
    headers = ("Pass", "Suite", "Test Error", "Fix Applied", "Status")
    rows = [
        (str(h["pass"]), h["suite"][:10], f"{h['test']}: {h['error']}"[:44], h["fix"][:36], h["status"])
        for h in history
    ] or [("1", "All Suites", "0 errors detected", "No fixes required", "PASSED")]
    widths = [max(len(headers[i]), max(len(r[i]) for r in rows)) for i in range(len(headers))]
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    fmt = lambda r: "| " + " | ".join(r[i].ljust(widths[i]) for i in range(len(headers))) + " |"
    print("\n" + sep)
    print(fmt(headers))
    print(sep)
    for r in rows:
        print(fmt(r))
    print(sep + "\n")


def main():
    parser = argparse.ArgumentParser(description="Repo-scoped autonomous pre-commit test and fix loop")
    parser.add_argument("--max-passes", type=int, default=5)
    args = parser.parse_args()

    initial_hashes = compute_data_hashes()
    history = []

    for pass_num in range(1, args.max_passes + 1):
        print(f"[pre-commit loop] Pass {pass_num}/{args.max_passes}: Running Python, JS, and HTML UI tests...", flush=True)
        all_ok, failures, raw = run_all_tests()
        if all_ok:
            for h in history:
                if h["status"] == "HEALING":
                    h["status"] = "FIXED"
            print(f"[pre-commit loop] Pass {pass_num}: ALL TESTS PASSED.", flush=True)
            print_summary_table(history)
            return 0

        print(f"[pre-commit loop] Pass {pass_num}: {len(failures)} failure(s) detected. Attempting autonomous fix...", flush=True)
        for script in ("python/sync_schemas.py", "python/md_html_sync.py"):
            if os.path.exists(os.path.join(REPO_DIR, script)):
                subprocess.run([sys.executable, script], cwd=REPO_DIR, capture_output=True, text=True)

        agy_bin = shutil.which("agy")
        fix_desc = "Deterministic sync only (agy not in PATH)"
        if agy_bin and os.environ.get("ANTIGRAVITY_NESTED_HEAL") != "1":
            env = os.environ.copy()
            env["ANTIGRAVITY_NESTED_HEAL"] = "1"
            fail_list = "\n".join(f"- [{f['suite']}] {f['test']}: {f['error']}" for f in failures[:20])
            prompt = (
                f"Fix ALL failing tests below strictly inside {REPO_DIR}. "
                "Do NOT modify any file outside this repository and NEVER modify data/*.json. "
                "Zero emojis, zero em-dashes.\n\n" + fail_list + "\n\n" + raw[-3000:]
            )
            subprocess.run([agy_bin, "--print", prompt], cwd=REPO_DIR, capture_output=True, text=True, env=env)
            fix_desc = "Autonomous agy fix + schema/MD sync"

        verify_data_immutable(initial_hashes)
        for f in failures:
            history.append({
                "pass": pass_num,
                "suite": f["suite"],
                "test": f["test"],
                "error": f["error"],
                "fix": fix_desc,
                "status": "HEALING",
            })

    for h in history:
        if h["status"] == "HEALING":
            h["status"] = "UNRESOLVED"
    print_summary_table(history)
    print("[pre-commit loop] ERROR: Tests still failing after max passes. Commit aborted.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
