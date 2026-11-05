#!/usr/bin/env python3
"""
test_last_commit_report.py - Unit Tests for Last Commit Report Generator.

Verifies:
- Accurate metadata recording for git HEAD.
- Quality Gate & Verification Summary table.
- Python Unit Test Suite Details table & itemized test cases.
- JavaScript Unit Test Suite Details table.
- NLP-Fuzzy-Expert Knowledge Synchronization Report & domain tables.
- Link, Schema & Markdown Verification table.
- Strict compliance: zero emojis, zero em-dashes.
- Post-commit hook presence and executable permissions.

Strict Repo Compliance:
- Zero emojis.
- Zero em-dashes (spaced hyphens ' - ' or plain hyphens '-' used).
"""

import os
import re
import subprocess
import sys
import tempfile
import unittest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(TESTS_DIR)
PYTHON_DIR = os.path.join(REPO_DIR, "python")

sys.path.insert(0, PYTHON_DIR)
from generate_last_commit_report import generate_last_commit_report, REPORT_FILE


class TestLastCommitReport(unittest.TestCase):
    """Verifies generation, format, and content of LAST-COMMIT-REPORT.md."""

    def setUp(self):
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tmp:
            self.tmp_path = tmp.name
        self.content = generate_last_commit_report(output_file=self.tmp_path)

    def tearDown(self):
        if os.path.exists(self.tmp_path):
            os.remove(self.tmp_path)

    def test_generate_report_creates_file(self):
        """Verifies report is generated and contains accurate HEAD information."""
        self.assertTrue(os.path.exists(self.tmp_path))
        self.assertIn("# Last Git Commit Report", self.content)
        self.assertIn("## Commit Overview", self.content)
        self.assertIn("## Commit Output & Stat Summary", self.content)

        head_hash = subprocess.run(
            ["git", "log", "-1", "--format=%h"],
            cwd=REPO_DIR,
            capture_output=True,
            text=True,
            check=True
        ).stdout.strip()
        self.assertIn(head_hash, self.content)

    def test_report_includes_quality_gate_summary(self):
        """Verifies report includes the Quality Gate & Verification Summary table."""
        self.assertIn("## Quality Gate & Verification Summary", self.content)
        self.assertIn("| Quality Gate / Test Dimension | Total Scope | Passed | Failed | Status | Runtime |", self.content)
        self.assertIn("Python Unit Test Suite", self.content)
        self.assertIn("JavaScript Unit Test Suite", self.content)
        self.assertIn("NLP-Fuzzy-Expert Sync Engine", self.content)
        self.assertIn("SSOT Data Immutability Guard", self.content)
        self.assertIn("Internal Link & Anchor Integrity", self.content)
        self.assertIn("JSON-LD Schema Synchronization", self.content)
        self.assertIn("Markdown Sync Verification", self.content)
        self.assertIn("Overall Quality Gate Status", self.content)

    def test_report_includes_python_test_suite_details(self):
        """Verifies report includes the Python Unit Test Suite Details table and itemized list."""
        self.assertIn("## Python Unit Test Suite Details (`tests/`)", self.content)
        self.assertIn("| Test Module | Total Tests | Passed | Failed | Errors | Skipped | Status |", self.content)
        self.assertIn("`test_auto_healer`", self.content)
        self.assertIn("`test_code_test_syncer`", self.content)
        self.assertIn("`test_sync_guarantees`", self.content)
        self.assertIn("`test_unidirectional_sync`", self.content)
        self.assertIn("<details>", self.content)
        self.assertIn("Itemized Python Test Cases", self.content)

    def test_report_includes_javascript_test_suite_details(self):
        """Verifies report includes the JavaScript Unit Test Suite Details table."""
        self.assertIn("## JavaScript Unit Test Suite Details (`pricing-tests.js`)", self.content)
        self.assertIn("| Test Section | Tests | Passed | Failed | Status |", self.content)
        self.assertIn("Data Integrity & Metadata Tests", self.content)
        self.assertIn("Nothing Selected (0 USD) Tests", self.content)
        self.assertIn("Individual Pick-Services Tests", self.content)
        self.assertIn("Business Pick-Services Tests", self.content)
        self.assertIn("Flagship Package Tests", self.content)
        self.assertIn("DOM Controller Simulated Live-Update Tests", self.content)

    def test_report_includes_nlp_fuzzy_expert_report(self):
        """Verifies report includes the NLP-Fuzzy-Expert Knowledge Synchronization Report."""
        self.assertIn("## NLP-Fuzzy-Expert Knowledge Synchronization Report", self.content)
        self.assertIn("Expert System (Forward Chaining) + Fuzzy Logic + NLP", self.content)
        self.assertIn("Strictly Unidirectional", self.content)
        self.assertIn("### Master Reference Occurrences by Domain", self.content)
        self.assertIn("| Knowledge Domain | Tracked Occurrences | Primary Data Source | Domain Sync Report |", self.content)
        self.assertIn("`pricing`", self.content)
        self.assertIn("`business_services`", self.content)
        self.assertIn("`publications`", self.content)
        self.assertIn("### Core Target Documents Synchronization Status", self.content)
        self.assertIn("`guarantees.html`", self.content)
        self.assertIn("`pricing.html`", self.content)

    def test_report_includes_link_schema_verification(self):
        """Verifies report includes the Link, Schema & Markdown Verification Details table."""
        self.assertIn("## Link, Schema & Markdown Verification Details", self.content)
        self.assertIn("| Verification Scope | Evaluated Target | Passed / Matched | Broken / Errors | Verification Status |", self.content)
        self.assertIn("Internal Links & Anchors", self.content)
        self.assertIn("Structured JSON-LD Schemas", self.content)
        self.assertIn("Markdown Page Mirrors", self.content)
        self.assertIn("SSOT Data Immutability", self.content)

    def test_report_zero_emojis_and_zero_em_dashes(self):
        """Strictly verifies zero emojis and zero em-dashes throughout report."""
        # Check no em-dashes
        self.assertNotIn(chr(0x2014), self.content)
        self.assertNotIn("&" + "mdash;", self.content)
        self.assertNotIn("&" + "#8212;", self.content)

        # Check no emojis
        emoji_pattern = re.compile(r"[\U00010000-\U0010ffff]", flags=re.UNICODE)
        emojis_found = emoji_pattern.findall(self.content)
        self.assertEqual(len(emojis_found), 0, f"Emojis found in report: {emojis_found}")

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
