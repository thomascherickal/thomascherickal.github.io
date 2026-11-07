#!/usr/bin/env python3
"""
update_readme_commit_summary.py - Extracts the summary of updates from
LAST-COMMIT-REPORT.md and metrics.md, and appends/updates them in README.md
just before commit.

Strict Repo Compliance:
- Zero emojis.
- Zero em-dashes (spaced hyphens ' - ' or plain hyphens '-' used).
- Idempotent update mode (replaces existing ## Latest Pre-Commit Report Summary section or appends).
"""

import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAST_COMMIT_REPORT_PATH = os.path.join(REPO_ROOT, "LAST-COMMIT-REPORT.md")
METRICS_PATH = os.path.join(REPO_ROOT, "metrics.md")
README_PATH = os.path.join(REPO_ROOT, "README.md")

SECTION_HEADER = "## Latest Pre-Commit Report Summary"


def extract_section(content, heading):
    """Extracts a markdown section starting at `heading` up to the next same-or-higher level heading or `---`."""
    pattern = re.compile(
        rf"({re.escape(heading)}\n[\s\S]*?)(?=\n## |\n---|\Z)"
    )
    match = pattern.search(content)
    if match:
        return match.group(1).strip()
    return ""


def extract_subsection(content, heading):
    """Extracts a markdown `###` subsection starting at `heading` up to the next `###`, `##`, or `---`."""
    pattern = re.compile(
        rf"({re.escape(heading)}\n[\s\S]*?)(?=\n### |\n## |\n---|\Z)"
    )
    match = pattern.search(content)
    if match:
        return match.group(1).strip()
    return ""


def build_summary_from_report(report_content, metrics_content=""):
    """Parses LAST-COMMIT-REPORT.md and metrics.md and builds a concise markdown summary block for README.md."""
    gen_match = re.search(r"Generated on:\s*`([^`]+)`", report_content)
    generated_on = gen_match.group(1) if gen_match else "Latest Commit"

    overview_sec = extract_section(report_content, "## Commit Overview")
    quality_sec = extract_section(report_content, "## Quality Gate & Verification Summary")
    coverage_sec = extract_subsection(report_content, "### Coverage Executive Summary")

    metrics_exec_sec = extract_section(metrics_content, "## 1. Executive Summary")
    metrics_arch_sec = extract_section(metrics_content, "## 2. Architecture & File Category Breakdown")

    lines = [
        SECTION_HEADER,
        "",
        f"> **Synchronized from [`LAST-COMMIT-REPORT.md`](LAST-COMMIT-REPORT.md) and [`metrics.md`](metrics.md)** | Recorded on: `{generated_on}`",
        "",
    ]

    if overview_sec:
        lines.append(overview_sec.replace("## Commit Overview", "### Commit Overview", 1))
        lines.append("")

    if quality_sec:
        lines.append(
            quality_sec.replace(
                "## Quality Gate & Verification Summary",
                "### Quality Gate & Verification Summary",
                1,
            )
        )
        lines.append("")

    if coverage_sec:
        lines.append(coverage_sec)
        lines.append("")

    if metrics_exec_sec:
        lines.append(
            metrics_exec_sec.replace(
                "## 1. Executive Summary",
                "### Repository Code & Engineering Metrics Summary (`metrics.md`)",
                1,
            )
        )
        lines.append("")

    if metrics_arch_sec:
        lines.append(
            metrics_arch_sec.replace(
                "## 2. Architecture & File Category Breakdown",
                "### Architecture & File Category Breakdown (`metrics.md`)",
                1,
            )
        )
        lines.append("")

    summary_block = "\n".join(lines).rstrip() + "\n"
    # Enforce zero em-dashes
    summary_block = summary_block.replace("\u2014", " - ").replace("\u2013", "-")
    return summary_block


def update_readme_with_commit_summary(report_path=None, readme_path=None, metrics_path=None):
    """Reads LAST-COMMIT-REPORT.md and metrics.md, extracts the update summary, and appends/updates it in README.md."""
    rep_file = report_path or LAST_COMMIT_REPORT_PATH
    rdm_file = readme_path or README_PATH
    met_file = metrics_path or METRICS_PATH

    if not os.path.exists(rep_file) or not os.path.exists(rdm_file):
        return ""

    with open(rep_file, "r", encoding="utf-8") as f:
        report_content = f.read()

    metrics_content = ""
    if os.path.exists(met_file):
        with open(met_file, "r", encoding="utf-8") as f:
            metrics_content = f.read()

    with open(rdm_file, "r", encoding="utf-8") as f:
        readme_content = f.read()

    summary_block = build_summary_from_report(report_content, metrics_content)

    if SECTION_HEADER in readme_content:
        before_sec = readme_content.split(SECTION_HEADER, 1)[0].rstrip()
        if before_sec.endswith("---"):
            before_sec = before_sec[:-3].rstrip()
        updated_readme = before_sec + "\n\n---\n\n" + summary_block
    else:
        base_readme = readme_content.rstrip()
        if base_readme.endswith("---"):
            base_readme = base_readme[:-3].rstrip()
        updated_readme = base_readme + "\n\n---\n\n" + summary_block

    with open(rdm_file, "w", encoding="utf-8") as f:
        f.write(updated_readme)

    print(f"[README Commit Summary] Successfully updated {rdm_file} from {rep_file} and {met_file}")
    return updated_readme


if __name__ == "__main__":  # pragma: no cover
    update_readme_with_commit_summary()
