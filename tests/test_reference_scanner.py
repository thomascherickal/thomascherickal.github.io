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
             "claude", "gemini", "publications", "config", "faqs"},
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


if __name__ == "__main__":
    unittest.main()
