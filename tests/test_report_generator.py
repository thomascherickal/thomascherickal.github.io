#!/usr/bin/env python3
"""
test_report_generator.py - Unit Tests for Report Generator Module.

Strict Repo Compliance:
- Zero emojis.
- Zero em-dashes (spaced hyphens ' - ' or plain hyphens '-' used).
"""

import os
import sys
import unittest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(TESTS_DIR)
NLP_EXPERT_DIR = os.path.join(REPO_DIR, "nlp-fuzzy-expert")

sys.path.insert(0, NLP_EXPERT_DIR)
from report_generator import (
    DOMAIN_FILES,
    DOMAIN_DATA_SOURCE,
    classify_rule_field,
)


class TestReportGenerator(unittest.TestCase):
    """Verifies report generator mappings and classification logic."""

    def test_domain_mappings_completeness(self):
        """Verifies that all 11 expected domains have corresponding reference and source files."""
        expected_domains = [
            "pricing", "business_services", "personal_services", "tiers",
            "claude", "gemini", "publications", "config", "faqs", "contact", "guarantees"
        ]
        for d in expected_domains:
            self.assertIn(d, DOMAIN_FILES)
            self.assertIn(d, DOMAIN_DATA_SOURCE)
            self.assertTrue(DOMAIN_FILES[d].endswith("_references.json"))
            self.assertTrue(DOMAIN_DATA_SOURCE[d].endswith(".json"))

    def test_classify_rule_field_domains(self):
        """Verifies field classification assigns correct domain keys."""
        self.assertEqual(classify_rule_field("business_flagship_price"), "pricing")
        self.assertEqual(classify_rule_field("personal_flagship_days"), "pricing")
        self.assertEqual(classify_rule_field("calculator_options"), "pricing")
        self.assertEqual(classify_rule_field("tier_1_parity"), "tiers")
        self.assertEqual(classify_rule_field("personal_service_04_name"), "personal_services")
        self.assertEqual(classify_rule_field("business_service_01_card"), "business_services")
        self.assertEqual(classify_rule_field("claude_code_title"), "claude")
        self.assertEqual(classify_rule_field("gemini_argon_title"), "gemini")
        self.assertEqual(classify_rule_field("publication_article_url"), "publications")
        self.assertEqual(classify_rule_field("canonical_email"), "config")
        self.assertEqual(classify_rule_field("faq_walkaway"), "faqs")
        self.assertEqual(classify_rule_field("contact_form_endpoint"), "contact")
        self.assertEqual(classify_rule_field("walkaway_guarantee_terms"), "guarantees")

    def test_write_and_markdown_edge_cases(self):
        """Verifies write_reference_files cache hit and generate_markdown_reports empty domain."""
        import report_generator
        from unittest.mock import patch
        with patch("report_generator.save_json"), \
             patch("report_generator.load_json", return_value={"all_references": [{"a": 1}]}), \
             patch("os.path.exists", return_value=True):
            report_generator._DOMAIN_REFS_CACHE.clear()
            report_generator.write_reference_files({"pricing": [{"a": 1}]}, force=False)
            report_generator.write_reference_files({"pricing": [{"a": 1}]}, force=True)

        with patch("builtins.open", unittest.mock.mock_open()), \
             patch("os.path.exists", return_value=False):
            report_generator._DOMAIN_MD_CACHE.clear()
            report_generator.generate_markdown_reports({}, force=True)


if __name__ == "__main__":
    unittest.main()

