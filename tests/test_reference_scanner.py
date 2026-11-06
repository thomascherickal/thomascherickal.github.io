#!/usr/bin/env python3
"""
test_reference_scanner.py - Unit tests for the exhaustive tracked-term scanner.

Strict Repo Compliance:
- Zero emojis.
- Zero em-dashes (spaced hyphens ' - ' or plain hyphens '-' used).
"""

import hashlib
import json
import os
import sys
import unittest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(TESTS_DIR)
NLP_EXPERT_DIR = os.path.join(REPO_DIR, "nlp-fuzzy-expert")
DATA_DIR = os.path.join(REPO_DIR, "data")

sys.path.insert(0, NLP_EXPERT_DIR)
from reference_scanner import build_terms, scan_file, scan_repository


def _digest():
    h = hashlib.sha256()
    for fname in sorted(os.listdir(DATA_DIR)):
        with open(os.path.join(DATA_DIR, fname), "rb") as f:
            h.update(f.read())
    return h.hexdigest()


class TestReferenceScanner(unittest.TestCase):

    def test_terms_cover_all_data_domains(self):
        domains = {t["domain"] for t in build_terms(DATA_DIR)}
        self.assertEqual(
            domains,
            {"pricing", "business_services", "personal_services", "tiers",
             "claude", "gemini", "publications", "config", "faqs", "contact", "guarantees"},
        )

    def test_scan_records_schema_and_is_read_only(self):
        before = _digest()
        recs = scan_repository([os.path.join(REPO_DIR, "contact.html")], DATA_DIR)
        self.assertEqual(before, _digest(), "scanner must never modify data/")
        flat = [r for v in recs.values() for r in v]
        self.assertGreater(len(flat), 0)
        required = {"file", "page", "section", "line_number", "field_key",
                    "old_value", "new_value", "context", "match_type", "data_source"}
        for r in flat:
            self.assertTrue(required.issubset(r), r)
            self.assertIn(r["match_type"], ("exact", "fuzzy"))
            self.assertTrue(r["data_source"].startswith("data/"))
        self.assertTrue(any(r["field_key"] == "config:email" for r in flat))

    def test_reference_files_include_scanned_and_rule_records(self):
        with open(os.path.join(NLP_EXPERT_DIR, "references_dictionary.json"), encoding="utf-8") as f:
            data = json.load(f)
        kinds = {r.get("match_type") for r in data["all_references"]}
        self.assertIn("rule", kinds)
        self.assertTrue(kinds & {"exact", "fuzzy"})

    def test_scanner_cache_helpers_and_parallel_fallback(self):
        """Verifies prune_scanner_cache, load/save cache, >12 files batching, and main()."""
        import tempfile
        import shutil
        from unittest.mock import patch
        import reference_scanner as rs

        temp_dir = tempfile.mkdtemp()
        try:
            # Create a dummy file in REPO_DIR-relative path or temp_dir
            f_path = os.path.join(temp_dir, "sample.html")
            with open(f_path, "w", encoding="utf-8") as f:
                f.write("<html><body>thomascherickal@gmail.com</body></html>")
            st = os.stat(f_path)

            with patch.object(rs, "REPO_DIR", temp_dir):
                key_valid = f"sample.html:{st.st_mtime_ns}:{st.st_size}:digest1"
                cache = {
                    key_valid: [{"field_key": "config:email"}],
                    "short:key": [],
                    "missing.html:1:2:3": [],
                }
                pruned = rs.prune_scanner_cache(cache, active_keys={key_valid})
                self.assertIn(key_valid, pruned)

                # Test without active_keys (disk stat fallback)
                pruned2 = rs.prune_scanner_cache(cache, active_keys=None)
                self.assertIn(key_valid, pruned2)

            # Test load_scanner_cache with missing and corrupt cache file
            missing_cache = os.path.join(temp_dir, "missing_cache.json")
            with patch.object(rs, "CACHE_FILE", missing_cache):
                self.assertEqual(rs.load_scanner_cache(), {})

            corrupt_cache = os.path.join(temp_dir, "corrupt_cache.json")
            with open(corrupt_cache, "w", encoding="utf-8") as f:
                f.write("{corrupt")
            with patch.object(rs, "CACHE_FILE", corrupt_cache):
                rs._MEMORY_SCANNER_CACHE = None
                self.assertEqual(rs.load_scanner_cache(), {})

            # Test >12 uncached files with ProcessPoolExecutor fallback
            many_files = []
            for i in range(13):
                p = os.path.join(temp_dir, f"doc_{i}.html")
                with open(p, "w", encoding="utf-8") as f:
                    f.write(f"<html><body>Doc {i} thomascherickal@gmail.com</body></html>")
                many_files.append(p)

            temp_cache_file = os.path.join(temp_dir, ".scanner_cache.json")
            with patch.object(rs, "CACHE_FILE", temp_cache_file), \
                 patch.object(rs, "ProcessPoolExecutor", side_effect=RuntimeError("no pool")):
                rs._MEMORY_SCANNER_CACHE = {}
                res = rs.scan_repository(many_files, DATA_DIR)
                self.assertIn("config", res)

            # Test main()
            with patch("synchronizer.ExpertSynchronizer.refresh_references_only") as mock_ref:
                rs.main()
                mock_ref.assert_called_once()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
