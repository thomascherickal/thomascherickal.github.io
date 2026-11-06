#!/usr/bin/env python3
"""
generate_test_coverage_report.py - Comprehensive Test Coverage Engine & Reporter.

Executes Python coverage analysis across nlp-fuzzy-expert/ and python/ modules,
and native Node.js V8 code coverage across pricing-calculator.js, components.js,
and mermaid.js. Computes granular line-level metrics, and generates a structured
TEST_COVERAGE_REPORT.md in the repository root.

Strict Repo Compliance:
- Zero emojis.
- Zero em-dashes (spaced hyphens ' - ' or plain hyphens '-' used).
"""

import os
import sys
import json
import glob
import time
import shutil
import tempfile
import subprocess
from datetime import datetime, timezone

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_REPORT_PATH = os.path.join(REPO_DIR, "TEST_COVERAGE_REPORT.md")


def get_git_info():
    """Retrieves current Git HEAD commit hash and short status."""
    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_DIR,
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
    except Exception:
        commit_hash = "unknown"

    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=REPO_DIR,
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
    except Exception:
        branch = "main"

    return commit_hash, branch


def run_python_coverage(cov_json_path):
    """
    Executes Python unit test suite under coverage and exports results to JSON.
    Returns parsed dictionary of file coverage statistics.
    """
    env = os.environ.copy()
    env["ANTIGRAVITY_IN_TEST"] = "1"

    # Run coverage
    run_cmd = [
        sys.executable,
        "-m",
        "coverage",
        "run",
        "-m",
        "unittest",
        "discover",
        "-s",
        os.path.join(REPO_DIR, "tests")
    ]
    subprocess.run(run_cmd, cwd=REPO_DIR, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Export to JSON
    export_cmd = [
        sys.executable,
        "-m",
        "coverage",
        "json",
        "-o",
        cov_json_path,
        "--include=nlp-fuzzy-expert/*,python/*"
    ]
    subprocess.run(export_cmd, cwd=REPO_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if not os.path.exists(cov_json_path):
        return {}

    with open(cov_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def format_line_ranges(lines):
    """Formats a sorted list of line numbers into concise ranges (e.g. '12-15, 22, 34-40')."""
    if not lines:
        return "-"
    sorted_lines = sorted(set(lines))
    ranges = []
    start = sorted_lines[0]
    end = start

    for line in sorted_lines[1:]:
        if line == end + 1:
            end = line
        else:
            if start == end:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{end}")
            start = line
            end = line

    if start == end:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{end}")

    return ", ".join(ranges)


def run_javascript_coverage(v8_dir):
    """
    Executes Node.js test suite with NODE_V8_COVERAGE and parses byte-level
    function/block coverage into line-level statistics for repository JS scripts.
    """
    if os.path.exists(v8_dir):
        shutil.rmtree(v8_dir)
    os.makedirs(v8_dir, exist_ok=True)

    env = os.environ.copy()
    env["NODE_V8_COVERAGE"] = v8_dir

    node_cmd = ["node", "pricing-tests.js"]
    subprocess.run(node_cmd, cwd=REPO_DIR, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    target_files = {
        "pricing-calculator.js": os.path.join(REPO_DIR, "pricing-calculator.js"),
        "components.js": os.path.join(REPO_DIR, "components.js"),
        "mermaid.js": os.path.join(REPO_DIR, "mermaid.js")
    }

    results = {}
    v8_files = glob.glob(os.path.join(v8_dir, "*.json"))

    for fname, fpath in target_files.items():
        if not os.path.exists(fpath):
            continue

        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            src = f.read()

        cov_map = [0] * len(src)

        for v8_file in v8_files:
            try:
                with open(v8_file, "r", encoding="utf-8") as f_v8:
                    v8_data = json.load(f_v8)
                for entry in v8_data.get("result", []):
                    url = entry.get("url", "")
                    if fname in url:
                        for fn in entry.get("functions", []):
                            for r in fn.get("ranges", []):
                                s = r.get("startOffset", 0)
                                e = r.get("endOffset", 0)
                                c = r.get("count", 0)
                                for i in range(s, min(e, len(cov_map))):
                                    cov_map[i] = max(cov_map[i], c)
            except Exception:
                continue

        lines = src.splitlines(keepends=True)
        executable_lines = 0
        covered_lines = 0
        missed_line_numbers = []
        curr = 0

        for line_idx, l in enumerate(lines, start=1):
            s = curr
            e = curr + len(l)
            curr = e
            t = l.strip()
            # Ignore empty lines and comment lines
            if not t or t.startswith("//") or t.startswith("/*") or t.startswith("*"):
                continue

            executable_lines += 1
            has_coverage = any(cov_map[i] > 0 for i in range(s, e) if not src[i].isspace())
            if has_coverage:
                covered_lines += 1
            else:
                missed_line_numbers.append(line_idx)

        pct = (covered_lines / executable_lines * 100.0) if executable_lines > 0 else 100.0
        results[fname] = {
            "path": fpath,
            "executable_lines": executable_lines,
            "covered_lines": covered_lines,
            "missed_lines": len(missed_line_numbers),
            "coverage_pct": round(pct, 2),
            "missing": missed_line_numbers
        }

    return results


def generate_coverage_report(output_path=OUTPUT_REPORT_PATH):
    """Orchestrates test coverage collection and writes TEST_COVERAGE_REPORT.md."""
    start_time = time.time()
    commit_hash, branch = get_git_info()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    temp_dir = tempfile.mkdtemp(prefix="test_cov_")
    try:
        py_cov_json = os.path.join(temp_dir, "python_coverage.json")
        v8_cov_dir = os.path.join(temp_dir, "v8cov")

        py_data = run_python_coverage(py_cov_json)
        js_data = run_javascript_coverage(v8_cov_dir)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Process Python Metrics
    py_files = {}
    py_total_stmts = 0
    py_total_covered = 0
    py_total_missed = 0

    if py_data and "files" in py_data:
        for fpath, info in py_data["files"].items():
            rel_path = os.path.relpath(fpath, REPO_DIR)
            stmts = info["summary"]["num_statements"]
            covered = info["summary"]["covered_lines"]
            missed = info["summary"]["missing_lines"]
            pct = info["summary"]["percent_covered"]
            py_total_stmts += stmts
            py_total_covered += covered
            py_total_missed += missed
            py_files[rel_path] = {
                "statements": stmts,
                "covered": covered,
                "missed": missed,
                "coverage_pct": round(pct, 2),
                "missing_lines": info.get("missing_lines", [])
            }

    py_overall_pct = (py_total_covered / py_total_stmts * 100.0) if py_total_stmts > 0 else 100.0

    # Process JavaScript Metrics
    js_total_lines = 0
    js_total_covered = 0
    js_total_missed = 0

    for fname, info in js_data.items():
        js_total_lines += info["executable_lines"]
        js_total_covered += info["covered_lines"]
        js_total_missed += info["missed_lines"]

    js_overall_pct = (js_total_covered / js_total_lines * 100.0) if js_total_lines > 0 else 100.0

    # Combined Repository Metrics
    comb_total = py_total_stmts + js_total_lines
    comb_covered = py_total_covered + js_total_covered
    comb_missed = py_total_missed + js_total_missed
    comb_overall_pct = (comb_covered / comb_total * 100.0) if comb_total > 0 else 100.0
    elapsed = round(time.time() - start_time, 2)

    # Build Markdown Content
    lines = []
    lines.append("# Test Coverage Report")
    lines.append("")
    lines.append(f"Recorded at: **{timestamp}** | Commit: **{commit_hash}** (`{branch}`) | Elapsed: **{elapsed}s**")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append("| Language | Tracked Files | Total Statements / Lines | Covered | Missed | Coverage % | Status |")
    lines.append("|:---|---:|---:|---:|---:|---:|:---|")

    py_status = "PASS" if py_overall_pct >= 85.0 else "WARN"
    js_status = "PASS" if js_overall_pct >= 90.0 else "WARN"
    comb_status = "PASS" if comb_overall_pct >= 88.0 else "WARN"

    lines.append(f"| Python (`nlp-fuzzy-expert/`, `python/`) | {len(py_files)} | {py_total_stmts:,} | {py_total_covered:,} | {py_total_missed:,} | **{py_overall_pct:.2f}%** | {py_status} |")
    lines.append(f"| JavaScript (Root scripts) | {len(js_data)} | {js_total_lines:,} | {js_total_covered:,} | {js_total_missed:,} | **{js_overall_pct:.2f}%** | {js_status} |")
    lines.append(f"| **Repository Combined Total** | **{len(py_files) + len(js_data)}** | **{comb_total:,}** | **{comb_covered:,}** | **{comb_missed:,}** | **{comb_overall_pct:.2f}%** | **{comb_status}** |")
    lines.append("")

    # Python Module Coverage Table
    lines.append("## Python Test Coverage")
    lines.append("")
    lines.append("| File | Statements | Covered | Missed | Coverage % | Missing Lines |")
    lines.append("|:---|---:|---:|---:|---:|:---|")

    for rel_path, info in sorted(py_files.items()):
        pct = info["coverage_pct"]
        ranges_str = format_line_ranges(info["missing_lines"])
        if len(ranges_str) > 35:
            ranges_str = ranges_str[:32] + "..."
        lines.append(f"| `{rel_path}` | {info['statements']} | {info['covered']} | {info['missed']} | **{pct:.1f}%** | `{ranges_str}` |")

    lines.append("")

    # JavaScript File Coverage Table
    lines.append("## JavaScript Test Coverage")
    lines.append("")
    lines.append("| File | Executable Lines | Covered | Missed | Coverage % | Missing Lines |")
    lines.append("|:---|---:|---:|---:|---:|:---|")

    for fname, info in sorted(js_data.items()):
        pct = info["coverage_pct"]
        ranges_str = format_line_ranges(info["missing"])
        if len(ranges_str) > 35:
            ranges_str = ranges_str[:32] + "..."
        lines.append(f"| `{fname}` | {info['executable_lines']} | {info['covered_lines']} | {info['missed_lines']} | **{pct:.1f}%** | `{ranges_str}` |")

    lines.append("")

    # Detailed Missing Lines Breakdown
    lines.append("## Detailed Uncovered Lines Breakdown")
    lines.append("")
    lines.append("The following files contain uncovered branches and line numbers identified for ongoing test expansion:")
    lines.append("")

    has_uncovered = False
    for rel_path, info in sorted(py_files.items()):
        if info["missing_lines"]:
            has_uncovered = True
            lines.append(f"- **`{rel_path}`** ({info['coverage_pct']:.1f}% covered):")
            lines.append(f"  - Missing Line(s): `{format_line_ranges(info['missing_lines'])}`")

    for fname, info in sorted(js_data.items()):
        if info["missing"]:
            has_uncovered = True
            lines.append(f"- **`{fname}`** ({info['coverage_pct']:.1f}% covered):")
            lines.append(f"  - Missing Line(s): `{format_line_ranges(info['missing'])}`")

    if not has_uncovered:
        lines.append("- Zero uncovered lines across all tracked files (100% complete coverage).")

    lines.append("")
    lines.append("---")
    lines.append("*Automated test coverage generated across Python unittest and Node.js V8 suites.*")
    lines.append("")

    report_content = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"[Coverage Report] Successfully generated {output_path} (Combined: {comb_overall_pct:.2f}% | Elapsed: {elapsed}s)")
    return report_content


if __name__ == "__main__":
    generate_coverage_report()

