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
import json
import os
import re
import shutil
import subprocess
import sys

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_DIR, "data")
TESTS_DIR = os.path.join(REPO_DIR, "tests")

PYTHON_CATEGORY_ORDER = [
    "SSOT Build & Validation",
    "Core Documents & Pricing Invariants",
    "Python Tooling & Build Scripts",
    "Reports & Coverage",
]

PYTHON_MODULE_CATEGORIES = {
    "test_ssot": "SSOT Build & Validation",
    "test_pricing_and_policy_invariants": "Core Documents & Pricing Invariants",
    "test_check_links": "Core Documents & Pricing Invariants",
    "test_python_tools": "Python Tooling & Build Scripts",
    "test_coverage_report": "Reports & Coverage",
    "test_last_commit_report": "Reports & Coverage",
}


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


def _render_ascii_table(title, headers, rows, right_align_cols=()):
    """Formats a clean ASCII box table with a bold header and grand-total separator."""
    if not rows:
        rows = [("No data available", "0", "0", "0", "N/A")]
    widths = [max(len(headers[i]), max(len(str(r[i])) for r in rows)) for i in range(len(headers))]
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def fmt(row):
        cells = []
        for i, val in enumerate(row):
            s = str(val)
            cells.append(s.rjust(widths[i]) if i in right_align_cols else s.ljust(widths[i]))
        return "| " + " | ".join(cells) + " |"

    lines = [
        "",
        title,
        sep,
        fmt(headers),
        sep,
    ]
    for idx, r in enumerate(rows):
        if len(rows) > 1 and idx == len(rows) - 1 and str(r[0]).upper().startswith("TOTAL"):
            lines.append(sep)
        lines.append(fmt(r))
    lines.append(sep)
    return "\n".join(lines)


def _load_json_safe(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def print_category_test_summaries():
    """Prints Category-only tabular summaries for Python, JavaScript, and HTML UI tests."""
    headers = ("Category", "Total", "Passed", "Failed", "Status")
    overall_rows = []

    # 1. Python Unit Tests (Category-only, excluding test_html_ui which has its own table)
    py_cache = _load_json_safe(os.path.join(TESTS_DIR, ".test_reports_cache.json"))
    py_cats = {cat: {"total": 0, "passed": 0, "failed": 0} for cat in PYTHON_CATEGORY_ORDER}
    for mod, stats in (py_cache.get("modules") or {}).items():
        if mod == "test_html_ui":
            continue
        cat = PYTHON_MODULE_CATEGORIES.get(mod, "Other Python Tests")
        if cat not in py_cats:
            py_cats[cat] = {"total": 0, "passed": 0, "failed": 0}
        py_cats[cat]["total"] += int(stats.get("total", 0))
        py_cats[cat]["passed"] += int(stats.get("passed", 0))
        py_cats[cat]["failed"] += int(stats.get("failed", 0)) + int(stats.get("errors", 0))

    py_rows = []
    py_tot = py_pass = py_fail = 0
    for cat, st in py_cats.items():
        if st["total"] == 0:
            continue
        py_tot += st["total"]
        py_pass += st["passed"]
        py_fail += st["failed"]
        py_rows.append((cat, str(st["total"]), str(st["passed"]), str(st["failed"]), "PASS" if st["failed"] == 0 else "FAIL"))
    py_status = "PASS" if py_fail == 0 and py_tot > 0 else "FAIL"
    py_rows.append(("TOTAL (Python Unit Tests)", str(py_tot), str(py_pass), str(py_fail), py_status))
    print(_render_ascii_table("=== PYTHON UNIT TESTS SUMMARY (BY CATEGORY) ===", headers, py_rows, right_align_cols=(1, 2, 3)))
    overall_rows.append(("Python Unit Tests", str(py_tot), str(py_pass), str(py_fail), py_status))

    # 2. JavaScript Unit Tests (Category-only from runner.js sections)
    js_cache = _load_json_safe(os.path.join(TESTS_DIR, ".js_test_reports_cache.json"))
    js_sections = js_cache.get("sections") or {}
    js_rows = []
    js_tot = js_pass = js_fail = 0
    for cat, st in js_sections.items():
        t = int(st.get("total", 0))
        p = int(st.get("passed", 0))
        f = int(st.get("failed", 0))
        js_tot += t
        js_pass += p
        js_fail += f
        js_rows.append((cat, str(t), str(p), str(f), "PASS" if f == 0 else "FAIL"))
    js_status = "PASS" if js_fail == 0 and js_tot > 0 else "FAIL"
    js_rows.append(("TOTAL (JavaScript Unit Tests)", str(js_tot), str(js_pass), str(js_fail), js_status))
    print(_render_ascii_table("=== JAVASCRIPT UNIT TESTS SUMMARY (BY CATEGORY) ===", headers, js_rows, right_align_cols=(1, 2, 3)))
    overall_rows.append(("JavaScript Unit Tests", str(js_tot), str(js_pass), str(js_fail), js_status))

    # 3. HTML UI Tests (Category-only from run_html_tests.py categories)
    ht_cache = _load_json_safe(os.path.join(TESTS_DIR, ".html_ui_test_reports_cache.json"))
    ht_cats = ht_cache.get("categories") or {}
    ht_rows = []
    ht_tot = ht_pass = ht_fail = 0
    for cat, st in ht_cats.items():
        t = int(st.get("total", 0))
        p = int(st.get("passed", 0))
        f = int(st.get("failed", 0))
        ht_tot += t
        ht_pass += p
        ht_fail += f
        ht_rows.append((cat, str(t), str(p), str(f), "PASS" if f == 0 else "FAIL"))
    ht_status = "PASS" if ht_fail == 0 and ht_tot > 0 else "FAIL"
    ht_rows.append(("TOTAL (HTML UI Tests)", str(ht_tot), str(ht_pass), str(ht_fail), ht_status))
    print(_render_ascii_table("=== HTML UI TESTS SUMMARY (BY CATEGORY) ===", headers, ht_rows, right_align_cols=(1, 2, 3)))
    overall_rows.append(("HTML UI Tests", str(ht_tot), str(ht_pass), str(ht_fail), ht_status))

    # 4. Combined Overall Pre-Commit Summary
    g_tot = py_tot + js_tot + ht_tot
    g_pass = py_pass + js_pass + ht_pass
    g_fail = py_fail + js_fail + ht_fail
    g_status = "PASS" if g_fail == 0 and g_tot > 0 else "FAIL"
    overall_rows.append(("TOTAL (All Suites)", str(g_tot), str(g_pass), str(g_fail), g_status))
    print(_render_ascii_table("=== OVERALL PRE-COMMIT TEST SUMMARY ===", ("Suite", "Total", "Passed", "Failed", "Status"), overall_rows, right_align_cols=(1, 2, 3)))


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
            print_category_test_summaries()
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
    print_category_test_summaries()
    print_summary_table(history)
    print("[pre-commit loop] ERROR: Tests still failing after max passes. Commit aborted.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
