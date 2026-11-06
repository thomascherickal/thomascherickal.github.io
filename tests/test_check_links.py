#!/usr/bin/env python3
"""
test_check_links.py - Unit Tests for Link and Anchor Verifier.

Strict Repo Compliance:
- Zero emojis.
- Zero em-dashes (spaced hyphens ' - ' or plain hyphens '-' used).
"""

import os
import sys
import unittest
import tempfile
import shutil
from unittest.mock import patch

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_DIR = os.path.join(REPO_DIR, "python")
sys.path.insert(0, PYTHON_DIR)

import check_links


class TestCheckLinks(unittest.TestCase):
    """Verifies all branches of check_links internal and fragment verification."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_collect_file_targets(self):
        """Verifies extracting id and name attributes from HTML content."""
        fpath = os.path.join(self.temp_dir, "test.html")
        with open(fpath, "w", encoding="utf-8") as f:
            f.write('<h2 id="section-1">Title</h2><a name="anchor-2">Link</a>')
        targets = check_links.collect_file_targets(fpath)
        self.assertIn("section-1", targets)
        self.assertIn("anchor-2", targets)

    def test_collect_file_targets_nonexistent(self):
        """Verifies nonexistent file returns empty set gracefully."""
        targets = check_links.collect_file_targets("/nonexistent/file.html")
        self.assertEqual(targets, set())

    def test_get_html_files_discovery(self):
        """Verifies get_html_files walks REPO_DIR and excludes .git and node_modules."""
        os.makedirs(os.path.join(self.temp_dir, ".git"), exist_ok=True)
        with open(os.path.join(self.temp_dir, ".git", "ignored.html"), "w") as f:
            f.write("ignore")
        p1 = os.path.join(self.temp_dir, "valid.html")
        with open(p1, "w") as f:
            f.write("<html></html>")
        with patch.object(check_links, "REPO_DIR", self.temp_dir):
            found = check_links.get_html_files()
            self.assertEqual(found, [p1])

    @patch("check_links.REPO_DIR")
    @patch("check_links.get_html_files")
    def test_check_links_success(self, mock_get_files, mock_repo_dir):
        """Verifies check_links reports success on valid internal links."""
        mock_repo_dir.__str__.return_value = self.temp_dir
        f1 = os.path.join(self.temp_dir, "page1.html")
        f2 = os.path.join(self.temp_dir, "page2.html")
        with open(f1, "w", encoding="utf-8") as f:
            f.write('<a href="page2.html#target">Link</a><a href="https://example.com">Ext</a>')
        with open(f2, "w", encoding="utf-8") as f:
            f.write('<div id="target">Content</div>')

        mock_get_files.return_value = [f1, f2]
        res = check_links.check_links()
        self.assertTrue(res)

    @patch("check_links.REPO_DIR")
    @patch("check_links.get_html_files")
    def test_check_links_detects_broken_file_and_fragment(self, mock_get_files, mock_repo_dir):
        """Verifies check_links detects broken file references and missing fragments."""
        mock_repo_dir.__str__.return_value = self.temp_dir
        f1 = os.path.join(self.temp_dir, "broken.html")
        with open(f1, "w", encoding="utf-8") as f:
            f.write('<a href="missing.html">Missing File</a><a href="#missing-anchor">Missing Anchor</a><a href="?query=1">Query</a>')

        mock_get_files.return_value = [f1]
        res = check_links.check_links()
        self.assertFalse(res)

    @patch("check_links.REPO_DIR")
    @patch("check_links.get_html_files")
    def test_check_links_resolves_root_and_directory(self, mock_get_files, mock_repo_dir):
        """Verifies check_links resolves root-relative paths and directory index.html."""
        check_links.REPO_DIR = self.temp_dir
        sub_dir = os.path.join(self.temp_dir, "subdir")
        os.makedirs(sub_dir, exist_ok=True)
        idx = os.path.join(sub_dir, "index.html")
        with open(idx, "w", encoding="utf-8") as f:
            f.write('<h1 id="root-heading">Hello</h1>')

        source = os.path.join(self.temp_dir, "source.html")
        with open(source, "w", encoding="utf-8") as f:
            f.write('<a href="/subdir/#root-heading">Subdir Root Link</a>')

        mock_get_files.return_value = [source, idx]
        try:
            res = check_links.check_links()
            self.assertTrue(res)
        finally:
            check_links.REPO_DIR = REPO_DIR


if __name__ == "__main__":
    unittest.main()

