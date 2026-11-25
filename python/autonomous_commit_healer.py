#!/usr/bin/env python3
"""
autonomous_commit_healer.py - Fast, Autonomous Self-Healing Pre-Commit & Push Loop.

Optimizations & Features:
1. No-Change Fast-Path (< 0.2s): Computes a deterministic repository fingerprint
   (HEAD + working/staged file content hashes, ignoring volatile report/cache files).
   If nothing has changed since the last verified passing run, skips redundant test
   re-execution unless --force is specified.
2. Parallel Execution & Zero Duplicate Runs: Runs Python unit tests (which include
   tests/test_html_ui.py, optionally instrumented with coverage via --with-coverage)
   and JavaScript pricing tests in parallel, then synthesizes
   tests/.html_ui_test_reports_cache.json directly from .test_reports_cache.json
   instead of re-parsing all 85+ HTML files a second time.
3. Human-Readable Action Summary: Prints a clear, numbered log of every action taken
   during verification and self-healing, and saves it alongside the tabular error/fix
   history in tests/.auto_heal_report_cache.json for LAST-COMMIT-REPORT.md.
4. Strict data/*.json SHA-256 immutability guard across all healing passes.

Strict Repo Compliance: Zero emojis, zero em-dashes.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.join(REPO_DIR, "tests")
DATA_DIR = os.path.join(REPO_DIR, "data")
HEAL_CACHE_PATH = os.path.join(TESTS_DIR, ".auto_heal_report_cache.json")
PY_CACHE_PATH = os.path.join(TESTS_DIR, ".test_reports_cache.json")
HTML_CACHE_PATH = os.path.join(TESTS_DIR, ".html_ui_test_reports_cache.json")

IGNORED_FINGERPRINT_PATHS = {
    "METRICS.md",
    "LAST-COMMIT-REPORT.md",
    "tests/.test_reports_cache.json",
    "tests/.js_test_reports_cache.json",
    "tests/.html_ui_test_reports_cache.json",
    "tests/.auto_heal_report_cache.json",
    ".coverage",
}


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


def compute_repo_fingerprint(repo_dir=REPO_DIR):
    """Computes a fast SHA-256 fingerprint of HEAD + all dirty/staged non-volatile files."""
    h = hashlib.sha256()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True
    )
    h.update((head.stdout.strip() or "NO_HEAD").encode("utf-8"))

    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo_dir, capture_output=True, text=True
    )
    dirty_files = []
    for line in sorted(status.stdout.splitlines()):
        if not line.strip():
            continue
        rel = line[3:].strip()
        if " -> " in rel:
            rel = rel.split(" -> ", 1)[1].strip()
        if rel in IGNORED_FINGERPRINT_PATHS or rel.startswith("failed_"):
            continue
        dirty_files.append((line[:2], rel))
        h.update(f"{line[:2]}:{rel}\n".encode("utf-8"))
        full_path = os.path.join(repo_dir, rel)
        if os.path.isfile(full_path):
            try:
                with open(full_path, "rb") as fp:
                    h.update(hashlib.sha256(fp.read()).digest())
            except OSError:
                pass
    return h.hexdigest(), len(dirty_files)


def load_heal_cache(cache_path=HEAL_CACHE_PATH):
    """Loads existing heal cache if present and valid JSON."""
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def parse_python_failures(output):
    """Extracts failing test entries from Python unittest output."""
    failures = []
    for m in re.finditer(r"^(?:FAIL|ERROR):\s+([^\s]+)\s+\(([^)]+)\)", output, re.MULTILINE):
        test_id = f"{m.group(2)}.{m.group(1)}"
        suite = "HTML UI" if "test_html_ui" in test_id else "Python"
        tail = output[m.end():m.end() + 1200]
        err_m = re.search(r"(?:AssertionError|Error|Exception):\s*(.+)", tail)
        msg = err_m.group(1).strip()[:120] if err_m else "Test assertion failed"
        failures.append({"suite": suite, "test": test_id, "error": msg})
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


def sync_html_cache_from_python_cache(repo_dir=REPO_DIR):
    """Synthesizes .html_ui_test_reports_cache.json from .test_reports_cache.json without re-running HTML tests."""
    py_cache_file = os.path.join(repo_dir, "tests", ".test_reports_cache.json")
    html_cache_file = os.path.join(repo_dir, "tests", ".html_ui_test_reports_cache.json")
    if not os.path.exists(py_cache_file):
        return False, 0

    try:
        with open(py_cache_file, "r", encoding="utf-8") as f:
            py_data = json.load(f)
        html_mod = py_data.get("modules", {}).get("test_html_ui")
        if not html_mod or not html_mod.get("tests"):
            return False, 0

        tests = html_mod["tests"]
        total = len(tests)
        passed = sum(1 for t in tests if t.get("status") == "PASSED")
        failed = total - passed
        elapsed = round(sum(float(t.get("time_taken", 0)) for t in tests), 2)

        pages_scanned = 85
        try:
            if repo_dir not in sys.path:
                sys.path.insert(0, repo_dir)
            from tests.test_html_ui import discover_html_pages
            pages_scanned = len(discover_html_pages())
        except Exception:
            pass

        payload = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "framework": "BeautifulSoup4 + lxml (unittest)",
            "pages_scanned": pages_scanned,
            "total": total,
            "passed": passed,
            "failed": failed,
            "elapsed_seconds": elapsed,
            "status": "PASSED" if failed == 0 else "FAILED",
            "tests": tests,
        }
        with open(html_cache_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return True, total
    except Exception:
        return False, 0


def _coverage_available():
    res = subprocess.run([sys.executable, "-c", "import coverage"], capture_output=True)
    return res.returncode == 0


def run_all_suites(repo_dir=REPO_DIR, with_coverage=False):
    """Runs Python (+ HTML UI) and JavaScript suites in parallel to minimize wall-clock time."""
    env = os.environ.copy()
    env["ANTIGRAVITY_IN_TEST"] = "1"

    use_cov = with_coverage and _coverage_available()
    if use_cov:
        py_cmd = [
            sys.executable, "-m", "coverage", "run",
            f"--source={os.path.join(repo_dir, 'nlp-fuzzy-expert')},{os.path.join(repo_dir, 'python')}",
            "python/run_tests.py",
        ]
    else:
        py_cmd = [sys.executable, "python/run_tests.py"]

    js_cmd = ["node", "pricing-tests.js"]

    def _run(cmd):
        t0 = time.time()
        res = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True, env=env)
        return res, round(time.time() - t0, 2)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        f_py = pool.submit(_run, py_cmd)
        f_js = pool.submit(_run, js_cmd)
        py_res, py_sec = f_py.result()
        js_res, js_sec = f_js.result()

    failures = []
    logs = []

    py_out = py_res.stdout + "\n" + py_res.stderr
    if py_res.returncode != 0:
        py_fails = parse_python_failures(py_out)
        if not py_fails:
            py_fails = [{"suite": "Python", "test": "python/run_tests.py", "error": f"Exited with code {py_res.returncode}"}]
        failures.extend(py_fails)
        logs.append("=== PYTHON / HTML UI FAILURES ===\n" + py_out[-3000:])

    js_out = js_res.stdout + "\n" + js_res.stderr
    if js_res.returncode != 0:
        js_fails = parse_js_failures(js_out)
        if not js_fails:
            js_fails = [{"suite": "JavaScript", "test": "pricing-tests.js", "error": f"Exited with code {js_res.returncode}"}]
        failures.extend(js_fails)
        logs.append("=== JS FAILURES ===\n" + js_out[-2000:])

    # Sync HTML UI cache directly from Python test run (eliminates duplicate 5s HTML scan)
    synced, html_count = sync_html_cache_from_python_cache(repo_dir)
    if not synced and os.path.exists(os.path.join(repo_dir, "python", "run_html_tests.py")):
        html_res, _ = _run([sys.executable, "python/run_html_tests.py"])
        if html_res.returncode != 0:
            html_fails = parse_python_failures(html_res.stdout + "\n" + html_res.stderr)
            for hf in html_fails:
                hf["suite"] = "HTML UI"
            failures.extend(html_fails)

    meta = {
        "py_seconds": py_sec,
        "js_seconds": js_sec,
        "with_coverage": use_cov,
        "html_tests_synced": html_count,
    }
    return (len(failures) == 0), failures, "\n\n".join(logs), meta


def apply_deterministic_fixes(repo_dir=REPO_DIR):
    """Applies safe deterministic sync fixes (schemas, MD mirrors, stale failure reports)."""
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


def save_heal_cache(history, passed, total_passes, action_log=None, repo_fingerprint="", elapsed_seconds=0.0, fast_path=False, cache_path=HEAL_CACHE_PATH):
    """Persists healing history, action log, and repo fingerprint to tests/.auto_heal_report_cache.json."""
    payload = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "passed": passed,
        "total_passes": total_passes,
        "fast_path_used": fast_path,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "repo_fingerprint": repo_fingerprint,
        "errors_fixed": sum(1 for h in history if h.get("status") == "FIXED"),
        "errors_remaining": sum(1 for h in history if h.get("status") != "FIXED"),
        "action_log": action_log or [],
        "history": history,
    }
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def run_healing_loop(
    max_passes=5,
    repo_dir=REPO_DIR,
    cache_path=HEAL_CACHE_PATH,
    force=False,
    with_coverage=False,
    runner_fn=None,
    fixer_fn=None,
):
    """Executes the optimized verification and self-healing loop with human-readable action logging."""
    t_start = time.time()
    action_log = []

    def log_action(msg):
        step_str = f"[Action {len(action_log) + 1}] {msg}"
        action_log.append(msg)
        print(step_str, flush=True)

    # Step 1: Fast-path fingerprint check when no custom runner is injected
    fingerprint = ""
    dirty_count = 0
    if runner_fn is None:
        fingerprint, dirty_count = compute_repo_fingerprint(repo_dir)
        log_action(
            f"Scanned repository state: {dirty_count} changed non-volatile file(s) "
            f"(fingerprint {fingerprint[:12]})."
        )
        prev = load_heal_cache(cache_path)
        if (
            not force
            and prev
            and prev.get("passed") is True
            and prev.get("repo_fingerprint") == fingerprint
        ):
            elapsed = round(time.time() - t_start, 2)
            log_action(
                f"Fast-path cache hit: repository matches last verified passing state. "
                f"Skipped redundant full test re-execution ({elapsed}s)."
            )
            save_heal_cache(
                [], True, 0,
                action_log=action_log,
                repo_fingerprint=fingerprint,
                elapsed_seconds=elapsed,
                fast_path=True,
                cache_path=cache_path,
            )
            print("\n" + format_terminal_table([]) + "\n")
            return True, []

    # Step 2: Baseline data/*.json SHA-256 hashes
    initial_hashes = compute_data_hashes(os.path.join(repo_dir, "data"))
    log_action(f"Computed SHA-256 immutability baseline for {len(initial_hashes)} file(s) in data/.")

    if runner_fn is not None:
        def _wrapped_runner():
            res = runner_fn()
            if len(res) == 4:
                return res
            return res[0], res[1], res[2], {}
        runner = _wrapped_runner
    else:
        runner = lambda: run_all_suites(repo_dir, with_coverage=with_coverage)

    fixer = fixer_fn or (lambda fails, logs: invoke_agy_fix(fails, logs, repo_dir))

    history = []
    all_passed = False
    passes_run = 0

    for pass_num in range(1, max_passes + 1):
        passes_run = pass_num
        log_action(f"Pass {pass_num}/{max_passes}: Executing Python (+ HTML UI) and JavaScript suites in parallel...")
        all_passed, failures, raw_logs, meta = runner()

        if all_passed:
            for h in history:
                if h["status"] == "HEALING":
                    h["status"] = "FIXED"
            timing_note = ""
            if meta:
                timing_note = (
                    f" (Python+HTML: {meta.get('py_seconds', 0)}s, "
                    f"JS: {meta.get('js_seconds', 0)}s, "
                    f"HTML cache synced: {meta.get('html_tests_synced', 0)} tests)"
                )
            log_action(f"Pass {pass_num}: All verification suites PASSED{timing_note}.")
            break

        current_keys = {(f["suite"], f["test"]) for f in failures}
        for h in history:
            if h["status"] == "HEALING" and (h["suite"], h["test"]) not in current_keys:
                h["status"] = "FIXED"

        suites_hit = sorted({f["suite"] for f in failures})
        log_action(f"Pass {pass_num}: Detected {len(failures)} failure(s) in {', '.join(suites_hit)}.")

        det_actions = apply_deterministic_fixes(repo_dir)
        if det_actions:
            log_action(f"Pass {pass_num}: Ran deterministic sync utilities ({', '.join(det_actions)}).")

        ok, fix_desc = fixer(failures, raw_logs)
        log_action(f"Pass {pass_num}: Autonomous healing step completed -> {fix_desc}.")

        verify_data_immutable(initial_hashes, os.path.join(repo_dir, "data"))
        log_action(f"Pass {pass_num}: Verified data/*.json SHA-256 immutability (0 mutations).")

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
        log_action(f"Verification ended with unresolved failures after {passes_run} pass(es).")

    # Refresh post-healing fingerprint if healing modified files
    if runner_fn is None:
        fingerprint, _ = compute_repo_fingerprint(repo_dir)

    elapsed = round(time.time() - t_start, 2)
    log_action(f"Completed verification loop in {elapsed}s; saved report cache for LAST-COMMIT-REPORT.md.")
    save_heal_cache(
        history, all_passed, passes_run,
        action_log=action_log,
        repo_fingerprint=fingerprint,
        elapsed_seconds=elapsed,
        fast_path=False,
        cache_path=cache_path,
    )
    print("\n" + format_terminal_table(history) + "\n")
    return all_passed, history


def main():
    parser = argparse.ArgumentParser(description="Fast Autonomous Self-Healing Commit & Push Loop")
    parser.add_argument("--max-passes", type=int, default=5, help="Maximum self-healing passes")
    parser.add_argument("--force", action="store_true", help="Bypass no-change fast-path cache and run full suites")
    parser.add_argument("--with-coverage", action="store_true", help="Run Python suite under coverage instrumentation")
    parser.add_argument("--commit", action="store_true", help="Stage and commit after all tests pass")
    parser.add_argument("-m", "--message", default="chore: autonomous self-healed commit", help="Commit message")
    parser.add_argument("--push", action="store_true", help="Push to remote after all tests pass")
    args = parser.parse_args()

    passed, _ = run_healing_loop(
        max_passes=args.max_passes,
        force=args.force,
        with_coverage=args.with_coverage,
    )
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
