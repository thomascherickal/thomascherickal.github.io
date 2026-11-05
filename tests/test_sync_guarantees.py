#!/usr/bin/env python3
"""
test_sync_guarantees.py - Unit Tests for Guarantees Page Renaming & Knowledge Synchronization.

Verifies:
- File renaming from guarantee.html to guarantees.html across all components, schemas, and markdown.
- Fixes and resolves all internal links to guarantees.html across the repository.
- GuaranteesPageRule registration and execution within the NLP-Fuzzy-Expert system.
- Authoritative pricing, schedule, and support figures correctly tracked in guarantees.html.
- Reference dictionary and domain reference files properly record guarantees.html.

Strict Repo Compliance:
- Zero emojis.
- Zero em-dashes (spaced hyphens ' - ' or plain hyphens '-' used).
"""

import os
import sys
import json
import unittest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(TESTS_DIR)
NLP_EXPERT_DIR = os.path.join(REPO_DIR, "nlp-fuzzy-expert")

sys.path.insert(0, NLP_EXPERT_DIR)
from synchronizer import ExpertSynchronizer
from expert_system import load_json
from rules import build_rules


class TestSyncGuarantees(unittest.TestCase):
    """Unit tests for guarantees page renaming and expert system tracking."""

    def setUp(self):
        self.orchestrator = ExpertSynchronizer()
        self.kb = self.orchestrator.kb

    def test_guarantees_page_exists_and_old_page_removed(self):
        """Verifies guarantees.html exists and obsolete guarantee.html is deleted."""
        new_html = os.path.join(REPO_DIR, "guarantees.html")
        old_html = os.path.join(REPO_DIR, "guarantee.html")
        new_schema = os.path.join(REPO_DIR, "schema", "guarantees-schema.json")
        old_schema = os.path.join(REPO_DIR, "schema", "guarantee-schema.json")
        new_md = os.path.join(REPO_DIR, "md-html-sync", "guarantees.md")
        old_md = os.path.join(REPO_DIR, "md-html-sync", "guarantee.md")

        self.assertTrue(os.path.exists(new_html), "guarantees.html must exist")
        self.assertFalse(os.path.exists(old_html), "guarantee.html must not exist")

        self.assertTrue(os.path.exists(new_schema), "schema/guarantees-schema.json must exist")
        self.assertFalse(os.path.exists(old_schema), "schema/guarantee-schema.json must not exist")

        self.assertTrue(os.path.exists(new_md), "md-html-sync/guarantees.md must exist")
        self.assertFalse(os.path.exists(old_md), "md-html-sync/guarantee.md must not exist")

    def test_guarantees_internal_links_across_repo(self):
        """Verifies all internal links point to guarantees.html and none to guarantee.html."""
        header_path = os.path.join(REPO_DIR, "header.html")
        with open(header_path, "r", encoding="utf-8") as f:
            header_content = f.read()
        self.assertIn('href="guarantees.html"', header_content)
        self.assertIn('data-nav="guarantees.html"', header_content)
        self.assertNotIn('href="guarantee.html"', header_content)

        index_path = os.path.join(REPO_DIR, "index.html")
        with open(index_path, "r", encoding="utf-8") as f:
            index_content = f.read()
        self.assertIn('href="guarantees.html"', index_content)
        self.assertNotIn('href="guarantee.html"', index_content)

        pricing_path = os.path.join(REPO_DIR, "pricing.html")
        with open(pricing_path, "r", encoding="utf-8") as f:
            pricing_content = f.read()
        self.assertIn('href="guarantees.html"', pricing_content)
        self.assertNotIn('href="guarantee.html"', pricing_content)

        contact_path = os.path.join(REPO_DIR, "contact.html")
        with open(contact_path, "r", encoding="utf-8") as f:
            contact_content = f.read()
        self.assertIn('href="guarantees.html"', contact_content)
        self.assertNotIn('href="guarantee.html"', contact_content)

    def test_guarantees_rule_in_expert_system(self):
        """Verifies GuaranteesPageRule is registered in the production rules queue."""
        rules = build_rules()
        rule_names = [r.name for r in rules]
        self.assertIn("GuaranteesPageRule", rule_names)

        guarantee_rule = next(r for r in rules if r.name == "GuaranteesPageRule")
        self.assertEqual(guarantee_rule.salience, 87)
        self.assertEqual(guarantee_rule.domain, "pricing")

        matched, _ = guarantee_rule.condition_fn("guarantees.html", "", self.kb, None, None)
        self.assertTrue(matched)
        matched_false, _ = guarantee_rule.condition_fn("pricing.html", "", self.kb, None, None)
        self.assertFalse(matched_false)

    def test_guarantees_page_content_in_sync(self):
        """Verifies guarantees.html content reflects authoritative pricing facts."""
        guarantees_path = os.path.join(REPO_DIR, "guarantees.html")
        with open(guarantees_path, "r", encoding="utf-8") as f:
            content = f.read()

        b_price = self.kb.get_fact("pricing", "b_flag_price")
        p_price = self.kb.get_fact("pricing", "p_flag_price")
        b_seats = self.kb.get_fact("pricing", "b_flag_seats")
        b_days = self.kb.get_fact("pricing", "b_flag_days")

        self.assertIn(f"${b_price:,}", content)
        self.assertIn(f"${p_price:,}", content)
        self.assertIn(f"Up to {b_seats} people included", content)
        self.assertIn(f"{b_days} Days Total", content)
        self.assertIn("Apache 2.0", content)

    def test_guarantees_tracked_in_references_dictionary(self):
        """Verifies guarantees.html occurrences are tracked in references_dictionary.json."""
        ref_dict_path = os.path.join(NLP_EXPERT_DIR, "references_dictionary.json")
        self.assertTrue(os.path.exists(ref_dict_path), "references_dictionary.json must exist")

        with open(ref_dict_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        all_refs = data.get("all_references", [])
        guarantees_refs = [r for r in all_refs if r.get("file") == "guarantees.html"]
        old_guarantee_refs = [r for r in all_refs if r.get("file") == "guarantee.html"]

        self.assertGreater(len(guarantees_refs), 0, "guarantees.html must have tracked occurrences")
        self.assertEqual(len(old_guarantee_refs), 0, "guarantee.html must have zero references")


if __name__ == "__main__":
    unittest.main()
