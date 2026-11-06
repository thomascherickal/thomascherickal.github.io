#!/usr/bin/env python3
"""
test_python_tools.py - Unit Tests for Python Utility Tools.
Covers build_components.py, sync_schemas.py, generate_sitemap.py, generate_metrics.py, and delete_page.py.

Strict Repo Compliance:
- Zero emojis.
- Zero em-dashes (spaced hyphens ' - ' or plain hyphens '-' used).
"""

import os
import sys
import unittest
import tempfile
import shutil
from unittest.mock import patch, MagicMock

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_DIR = os.path.join(REPO_DIR, "python")
sys.path.insert(0, PYTHON_DIR)

import build_components
import sync_schemas
import generate_sitemap
import generate_metrics
import delete_page
import md_html_sync
import build_md_files


class TestPythonTools(unittest.TestCase):
    """Verifies core build, synchronization, and maintenance scripts in python/."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # 1. build_components.py
    def test_build_components_extract_nav_and_footer(self):
        """Verifies extracting nav and footer tags and handling missing tags."""
        header_sample = '<header><nav class="nav">Nav Content</nav></header>'
        footer_sample = '<div class="wrap"><footer>Footer Content</footer></div>'
        nav = build_components.extract_nav(header_sample)
        footer = build_components.extract_footer(footer_sample)
        self.assertEqual(nav, '<nav class="nav">Nav Content</nav>')
        self.assertEqual(footer, '<footer>Footer Content</footer>')

        with self.assertRaises(ValueError):
            build_components.extract_nav("<div>No nav</div>")
        with self.assertRaises(ValueError):
            build_components.extract_footer("<div>No footer</div>")

    def test_build_components_build_execution(self):
        """Verifies build execution runs across repository without raising errors."""
        with patch("sys.stdout"):
            build_components.build()

    # 2. sync_schemas.py
    def test_sync_schemas_target_filename_mapping(self):
        """Verifies schema target filename mapping conventions."""
        self.assertEqual(sync_schemas.get_target_schema_filename("index.html"), "home-schema.json")
        self.assertEqual(sync_schemas.get_target_schema_filename("contact.html"), "contact-schema.json")
        self.assertEqual(sync_schemas.get_target_schema_filename("pricing.html"), "pricing-schema.json")

    def test_sync_schemas_check_only_mode(self):
        """Verifies check_only mode evaluates all pages without modifying schemas."""
        with patch("sys.stdout"):
            summary = sync_schemas.sync_html_to_schemas(check_only=True)
            self.assertIn("total", summary)
            self.assertGreaterEqual(summary["total"], 20)

    # 3. generate_metrics.py
    def test_generate_metrics_format_bytes(self):
        """Verifies format_bytes human-readable byte size representations."""
        self.assertIn("500 bytes", generate_metrics.format_bytes(500))
        self.assertIn("KB", generate_metrics.format_bytes(2048))
        self.assertIn("MB", generate_metrics.format_bytes(2 * 1024 * 1024))

    def test_generate_metrics_git_output(self):
        """Verifies get_git_output executes git command or returns empty string on error."""
        out = generate_metrics.get_git_output(["--version"])
        self.assertIn("git version", out)
        err_out = generate_metrics.get_git_output(["nonexistent-subcommand-xyz"])
        self.assertEqual(err_out, "")

    def test_generate_metrics_execution(self):
        """Verifies generate_metrics creates or updates metrics.md successfully."""
        with patch("sys.stdout"):
            generate_metrics.generate_metrics()
            self.assertTrue(os.path.exists(os.path.join(REPO_DIR, "metrics.md")))

    # 4. generate_sitemap.py
    def test_generate_sitemap_core_pages_exist(self):
        """Verifies core navigation pages defined in generate_sitemap exist in repository."""
        self.assertTrue(len(generate_sitemap.CORE_PAGES) >= 10)
        self.assertIn(("", "index.html"), generate_sitemap.CORE_PAGES)
        self.assertIn(("pricing.html", "pricing.html"), generate_sitemap.CORE_PAGES)
        self.assertIn(("contact.html", "contact.html"), generate_sitemap.CORE_PAGES)

    def test_generate_sitemap_execution(self):
        """Verifies generate_sitemap creates valid XML output."""
        temp_sitemap = os.path.join(self.temp_dir, "sitemap.xml")
        with patch("generate_sitemap.SITEMAP_FILE", temp_sitemap):
            with patch("sys.stdout"):
                generate_sitemap.generate_sitemap()
                self.assertTrue(os.path.exists(temp_sitemap))
                with open(temp_sitemap, "r", encoding="utf-8") as f:
                    xml = f.read()
                self.assertIn("<urlset", xml)
                self.assertIn("https://thedigitalfuturist.com", xml)

    # 5. delete_page.py
    def test_delete_page_detects_nonexistent_target(self):
        """Verifies delete_page aborts gracefully when target does not exist."""
        with patch("sys.stdout"):
            deleted = delete_page.delete_page("nonexistent-page-xyz-123.html", dry_run=True, verify=False)
            self.assertFalse(deleted)

    def test_delete_page_dry_run_on_existing_page(self):
        """Verifies delete_page dry run evaluates all targets without deleting any files."""
        mock_page = os.path.join(REPO_DIR, "tmp_test_delete_sample.html")
        try:
            with open(mock_page, "w", encoding="utf-8") as f:
                f.write("<html><body><h1>Temporary Test</h1></body></html>")
            with patch("sys.stdout"):
                delete_page.delete_page("tmp_test_delete_sample.html", dry_run=True, verify=False)
                self.assertTrue(os.path.exists(mock_page))  # Dry-run must preserve file
        finally:
            if os.path.exists(mock_page):
                os.remove(mock_page)

    # 6. md_html_sync.py
    def test_md_html_sync_helpers(self):
        """Verifies hash generation, text cleaning, and state serialization."""
        test_file = os.path.join(self.temp_dir, "test.txt")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("hello world")
        h = md_html_sync.get_file_hash(test_file)
        self.assertEqual(len(h), 64)  # SHA-256
        self.assertEqual(md_html_sync.clean_text("  hello   world  "), "hello world")

        state_path = os.path.join(self.temp_dir, ".sync-state.json")
        with patch("md_html_sync.STATE_FILE", state_path):
            sample_state = {"sample.html": {"hash": "abc"}}
            md_html_sync.save_state(sample_state)
            loaded = md_html_sync.load_state()
            self.assertEqual(loaded, sample_state)

    # 7. build_md_files.py
    def test_build_md_files_converters(self):
        """Verifies build_md_files document converters generate clean markdown."""
        contact_md = build_md_files.convert_contact()
        self.assertIn("Communication Channels", contact_md)
        self.assertIn("thomascherickal@gmail.com", contact_md)

        faqs_md = build_md_files.convert_faqs()
        self.assertIn("Frequently Asked Questions", faqs_md)

        err_md = build_md_files.convert_404()
        self.assertIn("404", err_md)

        pricing_md = build_md_files.convert_pricing()
        self.assertIn("Pricing", pricing_md)
        self.assertIn("Purchasing Power Parity", pricing_md)


if __name__ == "__main__":
    unittest.main()
