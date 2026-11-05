#!/usr/bin/env python3
"""
test_last_commit_report.py - Unit Tests for Last Commit Report Generator.

Strict Repo Compliance:
- Zero emojis.
- Zero em-dashes (spaced hyphens ' - ' or plain hyphens '-' used).
"""

import os
import sys
import unittest
import tempfile
import subprocess

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(TESTS_DIR)
PYTHON_DIR = os.path.join(REPO_DIR, "python")

sys.path.insert(0, PYTHON_DIR)
from generate_last_commit_report import generate_last_commit_report, REPORT_FILE


class TestLastCommitReport(unittest.TestCase):
    """Verifies generation, format, and content of LAST-COMMIT-REPORT.md."""

    def test_generate_report_creates_file(self):
        """Verifies report is generated and contains accurate HEAD information."""
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            content = generate_last_commit_report(output_file=tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            self.assertIn("# Last Git Commit Report", content)
            self.assertIn("## Commit Overview", content)
            self.assertIn("## Commit Output & Stat Summary", content)

            # Check that HEAD commit short hash is in the report
            head_hash = subprocess.run(
                ["git", "log", "-1", "--format=%h"],
                cwd=REPO_DIR,
                capture_output=True,
                text=True,
                check=True
            ).stdout.strip()
            self.assertIn(head_hash, content)

            # Verify no em-dashes
            self.assertNotIn("—", content)
            self.assertNotIn("&mdash;", content)
            self.assertNotIn("&#8212;", content)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_post_commit_hook_exists_and_executable(self):
        """Verifies that post-commit hooks exist in python/hooks and .git/hooks and are executable."""
        repo_hook = os.path.join(REPO_DIR, "python", "hooks", "post-commit")
        git_hook = os.path.join(REPO_DIR, ".git", "hooks", "post-commit")

        self.assertTrue(os.path.exists(repo_hook), "python/hooks/post-commit should exist")
        self.assertTrue(os.access(repo_hook, os.X_OK), "python/hooks/post-commit should be executable")

        if os.path.exists(os.path.join(REPO_DIR, ".git", "hooks")):
            self.assertTrue(os.path.exists(git_hook), ".git/hooks/post-commit should exist")
            self.assertTrue(os.access(git_hook, os.X_OK), ".git/hooks/post-commit should be executable")


if __name__ == "__main__":
    unittest.main()

