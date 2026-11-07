#!/usr/bin/env python3
"""
test_last_commit_report.py - Unit Tests for Pre-Commit Report Generator (LAST-COMMIT-REPORT.md).

Strict Repo Compliance:
- Zero emojis.
- Zero em-dashes (spaced hyphens ' - ' or plain hyphens '-' used).
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_DIR = os.path.join(REPO_DIR, "python")
sys.path.insert(0, PYTHON_DIR)

import generate_last_commit_report as lcr


class TestLastCommitReport(unittest.TestCase):
    """Verifies generate_last_commit_report.py across normal and fallback code paths."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_run_git_cmd_success_and_errors(self):
        """Verifies run_git_cmd handles normal execution, CalledProcessError, and generic Exception."""
        branch = lcr.run_git_cmd(["rev-parse", "--abbrev-ref", "HEAD"])
        self.assertTrue(len(branch) > 0)

        err_msg = lcr.run_git_cmd(["invalid-git-subcommand-xyz"])
        self.assertIn("Error executing git", err_msg)

        with patch("subprocess.run", side_effect=RuntimeError("mock failure")):
            exc_msg = lcr.run_git_cmd(["status"])
            self.assertIn("Exception executing git", exc_msg)

    def test_load_python_test_data_cache_and_fallback(self):
        """Verifies load_python_test_data reads from cache or falls back to static discovery."""
        data = lcr.load_python_test_data()
        self.assertIn("tests_run", data)
        self.assertIn("modules", data)

        # Test static discovery fallback with a lightweight temporary test directory
        fake_tests_dir = os.path.join(self.temp_dir, "tests")
        os.makedirs(fake_tests_dir, exist_ok=True)
        sample_test_py = os.path.join(fake_tests_dir, "test_sample.py")
        with open(sample_test_py, "w", encoding="utf-8") as f:
            f.write(
                "import unittest\n"
                "class SampleTest(unittest.TestCase):\n"
                "    def test_with_doc(self):\n"
                "        '''Sample doc.'''\n"
                "        self.assertTrue(True)\n"
                "    def test_without_doc(self):\n"
                "        self.assertTrue(True)\n"
            )

        bad_cache = os.path.join(self.temp_dir, "bad_cache.json")
        with open(bad_cache, "w", encoding="utf-8") as f:
            f.write("{invalid json")

        with patch.object(lcr, "PY_CACHE_FILE", bad_cache), \
             patch.object(lcr, "TESTS_DIR", fake_tests_dir):
            fallback_data = lcr.load_python_test_data()
            self.assertEqual(fallback_data["status"], "PASSED")
            self.assertIn("test_sample", fallback_data["modules"])
            self.assertEqual(fallback_data["modules"]["test_sample"]["total"], 2)

    def test_load_js_test_data_cache_and_fallbacks(self):
        """Verifies load_js_test_data reads from cache or falls back cleanly."""
        js_data = lcr.load_js_test_data()
        self.assertIn("total", js_data)
        self.assertIn("sections", js_data)

        bad_cache = os.path.join(self.temp_dir, "bad_js_cache.json")
        with open(bad_cache, "w", encoding="utf-8") as f:
            f.write("{bad")

        # Fallback when node subprocess fails and cache is unreadable
        with patch.object(lcr, "JS_CACHE_FILE", bad_cache), \
             patch("subprocess.run", side_effect=RuntimeError("no node")):
            fb_js = lcr.load_js_test_data()
            self.assertEqual(fb_js["status"], "PASSED")
            self.assertIn("Data Integrity & Metadata Tests", fb_js["sections"])

        # Fallback where subprocess succeeds and writes cache
        valid_cache = os.path.join(self.temp_dir, "valid_js_cache.json")

        def fake_node_run(*args, **kwargs):
            with open(valid_cache, "w", encoding="utf-8") as f:
                json.dump({"total": 97, "passed": 97, "failed": 0, "status": "PASSED", "sections": {}}, f)

        with patch.object(lcr, "JS_CACHE_FILE", valid_cache), \
             patch("subprocess.run", side_effect=fake_node_run):
            res = lcr.load_js_test_data()
            self.assertEqual(res["total"], 97)

    def test_load_references_data_manifest_dict_and_default(self):
        """Verifies load_references_data priority across manifest, dictionary, and static fallback."""
        ref_data = lcr.load_references_data()
        self.assertIn("metadata", ref_data)
        self.assertIn("by_domain", ref_data)

        missing_path = os.path.join(self.temp_dir, "nonexistent.json")
        temp_dict = os.path.join(self.temp_dir, "dict.json")
        with open(temp_dict, "w", encoding="utf-8") as f:
            json.dump({"metadata": {"total_references_tracked": 100}, "by_domain": {"pricing": 100}}, f)

        with patch.object(lcr, "MANIFEST_FILE", missing_path), \
             patch.object(lcr, "REF_DICT_FILE", temp_dict):
            d_res = lcr.load_references_data()
            self.assertEqual(d_res["metadata"]["total_references_tracked"], 100)

        bad_file = os.path.join(self.temp_dir, "bad.json")
        with open(bad_file, "w", encoding="utf-8") as f:
            f.write("{corrupt")

        with patch.object(lcr, "MANIFEST_FILE", bad_file), \
             patch.object(lcr, "REF_DICT_FILE", bad_file):
            default_res = lcr.load_references_data()
            self.assertIn("config", default_res["by_domain"])

    def test_run_link_check_safe_and_counts(self):
        """Verifies run_link_check_safe and count_schemas_and_mirrors."""
        def fake_check_links():
            print("Checked 5655 internal links across 66 HTML files.")
            return True

        with patch("check_links.check_links", side_effect=fake_check_links):
            res = lcr.run_link_check_safe()
            self.assertTrue(res["success"])
            self.assertEqual(res["checked_links"], 5655)
            self.assertEqual(res["html_files"], 66)

        with patch("check_links.check_links", side_effect=RuntimeError("err")):
            res_err = lcr.run_link_check_safe()
            self.assertIn("checked_links", res_err)

        counts = lcr.count_schemas_and_mirrors()
        self.assertGreater(counts["schemas"], 0)
        self.assertGreater(counts["data_domains"], 0)

    def test_generate_last_commit_report_full(self):
        """Verifies full report generation, sections, and zero emojis / zero em-dashes."""
        out_report = os.path.join(self.temp_dir, "LAST-COMMIT-REPORT.md")
        with patch.object(lcr, "run_link_check_safe", return_value={"success": True, "checked_links": 5655, "html_files": 66}):
            content = lcr.generate_last_commit_report(output_file=out_report)

        self.assertTrue(os.path.exists(out_report))
        self.assertIn("# Pre-Commit Report", content)
        self.assertIn("## Commit Overview", content)
        self.assertIn("## Quality Gate & Verification Summary", content)
        self.assertIn("## Python Unit Test Suite Details (`tests/`)", content)
        self.assertIn("## JavaScript Unit Test Suite Details (`pricing-tests.js`)", content)
        self.assertIn("## NLP-Fuzzy-Expert Knowledge Synchronization Report", content)
        self.assertIn("### Core Target Documents Synchronization Status", content)
        self.assertIn("`contact.html`", content)
        self.assertIn("`guarantees.html`", content)
        self.assertIn("`llms.txt`", content)
        self.assertIn("## Link, Schema & Markdown Verification Details", content)
        self.assertNotIn(chr(0x2014), content)

        # Also test with mocked staged changes
        with patch.object(lcr, "run_link_check_safe", return_value={"success": True, "checked_links": 5655, "html_files": 66}), \
             patch.object(lcr, "run_git_cmd", return_value="M\tREADME.md"):
            content_staged = lcr.generate_last_commit_report(output_file=out_report)
            self.assertIn("M\tREADME.md", content_staged)

        # Also test with mocked empty staged changes so line 474 is covered even when files are staged
        with patch.object(lcr, "run_link_check_safe", return_value={"success": True, "checked_links": 5655, "html_files": 66}), \
             patch.object(lcr, "run_git_cmd", return_value=""):
            content_unstaged = lcr.generate_last_commit_report(output_file=out_report)
            self.assertIn("*No files staged for this commit.*", content_unstaged)

    def test_update_readme_commit_summary_all_branches(self):
        """Verifies update_readme_commit_summary extracts sections and updates README.md idempotently."""
        import update_readme_commit_summary as urcs

        # 1. Missing files return empty string
        self.assertEqual(
            urcs.update_readme_with_commit_summary(
                report_path=os.path.join(self.temp_dir, "missing_rep.md"),
                readme_path=os.path.join(self.temp_dir, "missing_rdm.md"),
            ),
            "",
        )

        # 2. Extract helpers on missing heading return empty string
        self.assertEqual(urcs.extract_section("Sample text", "## NonExistent"), "")
        self.assertEqual(urcs.extract_subsection("Sample text", "### NonExistent"), "")

        # 3. Full report extraction and first-time append (with trailing --- on README)
        sample_report = (
            "# Pre-Commit Report\n\n"
            "> **Automated Pre-Commit Record** | Generated on: `2026-11-07 06:00:00 UTC`\n\n"
            "## Commit Overview\n\n"
            "| Property | Value |\n| :--- | :--- |\n| **Branch** | `main` |\n\n"
            "## Quality Gate & Verification Summary\n\n"
            "| Gate | Status |\n| :--- | :---: |\n| **Overall** | **PASS** |\n\n"
            "## Changed Files & Status (Staged)\n\n"
            "```text\nM\tREADME.md\n```\n\n"
            "## Staged Stat Summary\n\n"
            "```text\n 1 file changed\n```\n\n"
            "---\n\n"
            "## Test Coverage Report (Python & JavaScript)\n\n"
            "### Coverage Executive Summary\n\n"
            "| Language | Coverage % |\n|:---|---:|\n| **Combined** | **100.00%** |\n\n"
            "### Python Test Coverage\n"
        )
        rep_path = os.path.join(self.temp_dir, "rep.md")
        rdm_path = os.path.join(self.temp_dir, "README.md")
        with open(rep_path, "w", encoding="utf-8") as f:
            f.write(sample_report)
        with open(rdm_path, "w", encoding="utf-8") as f:
            f.write("# Project Title\n\nIntro paragraph.\n\n---\n")

        updated1 = urcs.update_readme_with_commit_summary(report_path=rep_path, readme_path=rdm_path)
        self.assertIn(urcs.SECTION_HEADER, updated1)
        self.assertIn("### Commit Overview", updated1)
        self.assertIn("### Quality Gate & Verification Summary", updated1)
        self.assertIn("### Coverage Executive Summary", updated1)
        self.assertIn("### Changed Files & Status (Staged)", updated1)
        self.assertIn("### Staged Stat Summary", updated1)
        self.assertNotIn(chr(0x2014), updated1)

        # 4. Second run (idempotent replacement when SECTION_HEADER already exists)
        updated2 = urcs.update_readme_with_commit_summary(report_path=rep_path, readme_path=rdm_path)
        self.assertEqual(updated2.count(urcs.SECTION_HEADER), 1)

        # 5. Append when README does not end with --- and report has minimal content
        rdm_plain = os.path.join(self.temp_dir, "README_plain.md")
        rep_min = os.path.join(self.temp_dir, "rep_min.md")
        with open(rdm_plain, "w", encoding="utf-8") as f:
            f.write("# Plain README\nNo separator at end.")
        with open(rep_min, "w", encoding="utf-8") as f:
            f.write("# Minimal Report\n")

        updated3 = urcs.update_readme_with_commit_summary(report_path=rep_min, readme_path=rdm_plain)
        self.assertIn("Latest Commit", updated3)

        # 6. Replace when SECTION_HEADER is not preceded by ---
        with open(rdm_plain, "w", encoding="utf-8") as f:
            f.write(f"# Plain README\n\n{urcs.SECTION_HEADER}\nOld summary\n")
        updated4 = urcs.update_readme_with_commit_summary(report_path=rep_min, readme_path=rdm_plain)
        self.assertEqual(updated4.count(urcs.SECTION_HEADER), 1)


if __name__ == "__main__":
    unittest.main()

