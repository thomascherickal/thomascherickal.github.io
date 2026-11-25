#!/usr/bin/env python3
"""
autonomous_commit_healer.py - Autonomous Self-Healing Pre-Commit & Push Loop.

Runs all three verification suites (Python unit tests, JavaScript pricing tests,
and BeautifulSoup4 HTML UI tests) in a loop. Whenever any test fails:
1. Parses every failing test error across all suites.
2. Applies deterministic sync fixes (schemas, Markdown mirrors, stale reports)
   and invokes `agy --print` to autonomously fix every remaining error in a pass.
3. Enforces strict SHA-256 immutability on `data/*.json`.
4. Loops until all tests pass (or max passes reached).
5. Prints a tabular report of all errors and how they were fixed to the terminal
   and persists `tests/.auto_heal_report_cache.json` for `LAST-COMMIT-REPORT.md`.
6. Optionally commits and pushes automatically once all tests pass (`--push`).

Strict Repo Compliance: Zero emojis, zero em-dashes.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.join(REPO_DIR, "tests")
DATA_DIR = os.path.join(REPO_DIR, "data")
HEAL_CACHE_PATH = os.path.join(TESTS_DIR, ".auto_heal_report_cache.json")


def compute_data_hashes(data_dir=DATA_DIR):
    """Computes SHA-256 hashes of all data/*.json files."""
    hashes = {}
    if not os.path.isdir(data_dir):
        return hashes
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith(".json"):
            with open(os.path.join(data_dir, fname), "rb") as f:
                hashes[fname] = hashlib.sha256(f.read()).hexdigest()
    return hashes


def verify_data_immutable(initial_hashes, data_dir=DATA_DIR):
    """Raises PermissionError if any file in data/ was mutated during healing."""
    current = compute_data_hashes(data_dir)
    if current != initial_hashes:
        raise PermissionError(
            "DATA MUTATION BLOCKED: Files in data/ were modified during autonomous healing."
        )


def parse_python_failures(output):
    """Extracts (test_name, summary) pairs from Python unittest output."""
    failures = []
    for m in re.finditer(r"^(?:FAIL|ERROR):\s+([^\s]+)\s+\(([^)]+)\)", output, re.MULTILINE):
        test_id = f"{m.group(2)}.{m.group(1)}"
        # Grab first AssertionError / Exception line following this match
        tail = output[m.end():m.end() + 1200]
        err_m = re.search(r"(?:AssertionError|Error|Exception):\s*(.+)", tail)
        msg = err_m.group(1).strip()[:120] if err_m else "Test assertion failed"
        failures.append({"suite": "Python", "test": test_id, "error": msg})
    return failures


def parse_js_failures(output):
    """Extracts failing test names and reasons from pricing-tests.js output."""
    failures = []
    lines = output.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"^\s*-\s*FAIL:\s*(.+)", line)
        if m:
            test_name = m.group(1).strip()
            reason = "JS assertion failed"
            if i + 1 < len(lines) and "Reason:" in lines[i + 1]:
                reason = lines[i + 1].split("Reason:", 1)[1].strip()[:120]
            failures.append({"suite": "JavaScript", "test": test_name, "error": reason})
    return failures


def run_all_suites(repo_dir=REPO_DIR):
    """Runs Python, JavaScript, and HTML UI test suites and returns (all_passed, failures_list, raw_logs)."""
    env = os.environ.copy()
    env["ANTIGRAVITY_IN_TEST"] = "1"
    failures = []
    logs = []

    # 1. Python Unit Tests (includes test_html_ui.py)
    py_res = subprocess.run(
        [sys.executable, "python/run_tests.py"],
        cwd=repo_dir, capture_output=True, text=True, env=env
    )
    py_out = py_res.stdout + "\n" + py_res.stderr
    if py_res.returncode != 0:
        py_fails = parse_python_failures(py_out)
        if not py_fails:
            py_fails = [{"suite": "Python", "test": "python/run_tests.py", "error": f"Exited with code {py_res.returncode}"}]
        failures.extend(py_fails)
        logs.append("=== PYTHON FAILURES ===\n" + py_out[-3000:])

    # 2. JavaScript Unit Tests
    js_res = subprocess.run(
        ["node", "pricing-tests.js"],
        cwd=repo_dir, capture_output=True, text=True, env=env
    )
    js_out = js_res.stdout + "\n" + js_res.stderr
    if js_res.returncode != 0:
        js_fails = parse_js_failures(js_out)
        if not js_fails:
            js_fails = [{"suite": "JavaScript", "test": "pricing-tests.js", "error": f"Exited with code {js_res.returncode}"}]
        failures.extend(js_fails)
        logs.append("=== JS FAILURES ===\n" + js_out[-2000:])

    # 3. Dedicated HTML UI Verification Suite (refreshes .html_ui_test_reports_cache.json)
    html_res = subprocess.run(
        [sys.executable, "python/run_html_tests.py"],
        cwd=repo_dir, capture_output=True, text=True, env=env
    )
    html_out = html_res.stdout + "\n" + html_res.stderr
    if html_res.returncode != 0:
        html_fails = parse_python_failures(html_out)
        for hf in html_fails:
            hf["suite"] = "HTML UI"
        if not html_fails:
            html_fails = [{"suite": "HTML UI", "test": "python/run_html_tests.py", "error": f"Exited with code {html_res.returncode}"}]
        failures.extend(html_fails)
        logs.append("=== HTML UI FAILURES ===\n" + html_out[-2000:])

    return (len(failures) == 0), failures, "\n\n".join(logs)


def apply_deterministic_fixes(repo_dir=REPO_DIR):
    """Applies safe deterministic sync fixes (schemas, MD mirrors, stale reports)."""
    actions = []
    for script in ("python/sync_schemas.py", "python/md_html_sync.py"):
        if os.path.exists(os.path.join(repo_dir, script)):
            subprocess.run([sys.executable, script], cwd=repo_dir, capture_output=True, text=True)
            actions.append(os.path.basename(script))
    return actions


def invoke_agy_fix(failures, raw_logs, repo_dir=REPO_DIR):
    """Invokes `agy --print` to autonomously fix all failing tests in a single pass."""
    agy_bin = shutil.which("agy")
    if not agy_bin:
        return False, "agy CLI not found in PATH"

    summary_lines = [f"- [{f['suite']}] {f['test']}: {f['error']}" for f in failures[:20]]
    prompt = (
        "Autonomously fix ALL of the following failing tests in this single pass. "
        "Strict rules: zero emojis, zero em-dashes, NEVER modify any file in data/*.json. "
        "Fix the root cause cleanly.\n\n"
        "Failing tests:\n" + "\n".join(summary_lines) + "\n\n"
        "Recent test output:\n" + raw_logs[-4000:]
    )
    before_diff = subprocess.run(["git", "diff", "--name-only"], cwd=repo_dir, capture_output=True, text=True).stdout
    res = subprocess.run([agy_bin, "--print", prompt], cwd=repo_dir, capture_output=True, text=True)
    after_diff = subprocess.run(["git", "diff", "--name-only"], cwd=repo_dir, capture_output=True, text=True).stdout

    changed = sorted(set(after_diff.splitlines()) - set(before_diff.splitlines()))
    if not changed:
        changed = sorted(set(after_diff.splitlines()))[:5]
    fix_desc = f"Autonomous agy fix ({', '.join(changed[:4])})" if changed else "Autonomous agy fix applied"
    return (res.returncode == 0), fix_desc


def format_terminal_table(history):
    """Formats the healing history into a clean ASCII table for terminal output."""
    headers = ("Pass", "Suite", "Test Error", "How It Was Fixed", "Status")
    rows = []
    for item in history:
        rows.append((
            str(item.get("pass", 1)),
            str(item.get("suite", ""))[:12],
            f"{item.get('test', '')}: {item.get('error', '')}"[:46],
            str(item.get("fix", ""))[:40],
            str(item.get("status", "")),
        ))
    if not rows:
        rows.append(("1", "All Suites", "0 errors detected", "No fixes required", "PASSED"))

    widths = [max(len(headers[i]), max(len(r[i]) for r in rows)) for i in range(len(headers))]
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    fmt = lambda r: "| " + " | ".join(r[i].ljust(widths[i]) for i in range(len(headers))) + " |"

    lines = [
        "=" * len(sep),
        "AUTONOMOUS COMMIT HEALER - ERROR & FIX SUMMARY TABLE".center(len(sep)),
        "=" * len(sep),
        sep,
        fmt(headers),
        sep,
    ]
    for r in rows:
        lines.append(fmt(r))
    lines.append(sep)
    return "\n".join(lines)


def save_heal_cache(history, passed, total_passes, cache_path=HEAL_CACHE_PATH):
    """Persists healing history to tests/.auto_heal_report_cache.json for LAST-COMMIT-REPORT.md."""
    payload = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "passed": passed,
        "total_passes": total_passes,
        "errors_fixed": sum(1 for h in history if h.get("status") == "FIXED"),
        "errors_remaining": sum(1 for h in history if h.get("status") != "FIXED"),
        "history": history,
    }
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def run_healing_loop(max_passes=5, repo_dir=REPO_DIR, cache_path=HEAL_CACHE_PATH, runner_fn=None, fixer_fn=None):
    """Executes the self-healing test loop until all suites pass or max_passes is reached."""
    initial_hashes = compute_data_hashes(os.path.join(repo_dir, "data"))
    runner = runner_fn or (lambda: run_all_suites(repo_dir))
    fixer = fixer_fn or (lambda fails, logs: invoke_agy_fix(fails, logs, repo_dir))

    history = []
    all_passed = False
    passes_run = 0

    for pass_num in range(1, max_passes + 1):
        passes_run = pass_num
        all_passed, failures, raw_logs = runner()
        if all_passed:
            # Mark any previously open items as FIXED
            for h in history:
                if h["status"] == "HEALING":
                    h["status"] = "FIXED"
            break

        # Mark previously healing items that are no longer failing as FIXED
        current_keys = {(f["suite"], f["test"]) for f in failures}
        for h in history:
            if h["status"] == "HEALING" and (h["suite"], h["test"]) not in current_keys:
                h["status"] = "FIXED"

        # Apply deterministic sync fixes + autonomous agy fix
        det_actions = apply_deterministic_fixes(repo_dir)
        ok, fix_desc = fixer(failures, raw_logs)
        verify_data_immutable(initial_hashes, os.path.join(repo_dir, "data"))

        full_fix = fix_desc if not det_actions else f"{fix_desc}; synced {', '.join(det_actions)}"
        for f in failures:
            history.append({
                "pass": pass_num,
                "suite": f["suite"],
                "test": f["test"],
                "error": f["error"],
                "fix": full_fix,
                "status": "HEALING" if ok else "UNRESOLVED",
            })

    if not all_passed:
        for h in history:
            if h["status"] == "HEALING":
                h["status"] = "UNRESOLVED"

    save_heal_cache(history, all_passed, passes_run, cache_path=cache_path)
    print("\n" + format_terminal_table(history) + "\n")
    return all_passed, history


def main():
    parser = argparse.ArgumentParser(description="Autonomous Self-Healing Commit & Push Loop")
    parser.add_argument("--max-passes", type=int, default=5, help="Maximum self-healing passes")
    parser.add_argument("--commit", action="store_true", help="Stage and commit after all tests pass")
    parser.add_argument("-m", "--message", default="chore: autonomous self-healed commit", help="Commit message")
    parser.add_argument("--push", action="store_true", help="Push to remote after all tests pass")
    args = parser.parse_args()

    passed, _ = run_healing_loop(max_passes=args.max_passes)
    if not passed:
        print("[auto-healer] ERROR: Tests still failing after maximum healing passes.")
        return 1

    if args.commit:
        subprocess.run(["git", "add", "-A"], cwd=REPO_DIR, check=True)
        subprocess.run(["git", "commit", "-m", args.message], cwd=REPO_DIR, check=True)

    if args.push:
        subprocess.run(["git", "push"], cwd=REPO_DIR, check=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
