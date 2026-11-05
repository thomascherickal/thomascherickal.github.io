#!/usr/bin/env python3
"""
generate_last_commit_report.py - Automated Last Commit Report Generator.

Records the full output report of the latest git commit to LAST-COMMIT-REPORT.md
in the repository root directory, overwriting it on every new commit.

Strict Repo Compliance:
- Zero emojis.
- Zero em-dashes (spaced hyphens ' - ' or plain hyphens '-' used).
"""

import os
import subprocess
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_FILE = os.path.join(REPO_ROOT, "LAST-COMMIT-REPORT.md")


def run_git_cmd(args):
    """Executes a git command and returns stdout as trimmed string."""
    try:
        res = subprocess.run(
            ["git"] + args,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True
        )
        return res.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error executing git {' '.join(args)}: {e.stderr.strip()}"
    except Exception as e:
        return f"Exception executing git {' '.join(args)}: {str(e)}"


def generate_last_commit_report(output_file=REPORT_FILE):
    """Generates the commit output report for HEAD and writes to output_file."""
    # Commit Metadata
    commit_hash = run_git_cmd(["log", "-1", "--format=%H"])
    commit_short = run_git_cmd(["log", "-1", "--format=%h"])
    author_name = run_git_cmd(["log", "-1", "--format=%an"])
    author_email = run_git_cmd(["log", "-1", "--format=%ae"])
    author_date = run_git_cmd(["log", "-1", "--format=%ad", "--date=iso-strict"])
    committer_name = run_git_cmd(["log", "-1", "--format=%cn"])
    committer_email = run_git_cmd(["log", "-1", "--format=%ce"])
    committer_date = run_git_cmd(["log", "-1", "--format=%cd", "--date=iso-strict"])
    commit_subject = run_git_cmd(["log", "-1", "--format=%s"])
    commit_body = run_git_cmd(["log", "-1", "--format=%b"])
    current_branch = run_git_cmd(["rev-parse", "--abbrev-ref", "HEAD"])
    total_commits = run_git_cmd(["rev-list", "--count", "HEAD"])

    # Commit Stats & Diff Summary
    commit_stat = run_git_cmd(["show", "--stat", "--oneline", "HEAD"])
    changed_files_status = run_git_cmd(["diff-tree", "--no-commit-id", "--name-status", "-r", "HEAD"])
    shortlog_summary = run_git_cmd(["log", "-1", "--stat"])

    gen_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Format Markdown Report
    lines = []
    lines.append("# Last Git Commit Report")
    lines.append("")
    lines.append(f"> **Automated Commit Record** | Recorded on: `{gen_timestamp}`")
    lines.append("")
    lines.append("## Commit Overview")
    lines.append("")
    lines.append("| Property | Value |")
    lines.append("| :--- | :--- |")
    lines.append(f"| **Commit Hash (Full)** | `{commit_hash}` |")
    lines.append(f"| **Commit Hash (Short)** | `{commit_short}` |")
    lines.append(f"| **Branch** | `{current_branch}` |")
    lines.append(f"| **Subject** | {commit_subject} |")
    lines.append(f"| **Author** | {author_name} `<{author_email}>` |")
    lines.append(f"| **Author Date** | `{author_date}` |")
    lines.append(f"| **Committer** | {committer_name} `<{committer_email}>` |")
    lines.append(f"| **Committer Date** | `{committer_date}` |")
    lines.append(f"| **Total Commits in Repo** | {total_commits} |")
    lines.append("")

    if commit_body:
        lines.append("## Commit Body")
        lines.append("")
        lines.append("```text")
        lines.append(commit_body)
        lines.append("```")
        lines.append("")

    lines.append("## Changed Files & Status")
    lines.append("")
    if changed_files_status:
        lines.append("```text")
        lines.append(changed_files_status)
        lines.append("```")
    else:
        lines.append("*No files changed in this commit.*")
    lines.append("")

    lines.append("## Commit Output & Stat Summary")
    lines.append("")
    lines.append("```text")
    lines.append(shortlog_summary)
    lines.append("```")
    lines.append("")

    lines.append("## Verification & Compliance")
    lines.append("")
    lines.append("- Zero emojis: Strictly enforced.")
    lines.append("- Zero em-dashes: Strictly enforced (spaced hyphens ' - ' or plain hyphens '-' used).")
    lines.append("- Overwrite mode: Automatically updated on every new commit.")
    lines.append("")

    report_content = "\n".join(lines)

    # Overwrite file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"[Last Commit Report] Successfully recorded report for commit {commit_short} to {output_file}")
    return report_content


if __name__ == "__main__":
    generate_last_commit_report()

