#!/usr/bin/env python3
"""
test_sync_pricing_tiers_and_calculators.py - Unit Tests for Pricing Tiers and Calculators Synchronization.

Verifies:
- Discovery of all new tier HTML pages, tier calculator JS files, and payment schedule widget.
- Registration and tracking within nlp-fuzzy-expert synchronization system and tiers_references.json.
- CodeTestSynchronizer discovery and synchronization of new scripts and test suites.
- Script tag linking on all tier-specific HTML pages.
- Strict zero-emojis and zero-em-dash compliance across all new files.

Strict Repo Compliance:
- Zero emojis.
- Zero em-dashes (spaced hyphens ' - ' or plain hyphens '-' used).
"""

import os
import re
import sys
import unittest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(TESTS_DIR)
NLP_EXPERT_DIR = os.path.join(REPO_DIR, "nlp-fuzzy-expert")

sys.path.insert(0, NLP_EXPERT_DIR)
from synchronizer import ExpertSynchronizer
from code_test_syncer import CodeTestSynchronizer
from expert_system import KnowledgeBase, load_json


class TestSyncPricingTiersAndCalculators(unittest.TestCase):
    """Test suite verifying synchronization of new pricing tier pages and calculators."""

    @classmethod
    def setUpClass(cls):
        cls.kb = KnowledgeBase(data_dir=os.path.join(REPO_DIR, "data"))
        cls.synchronizer = ExpertSynchronizer()
        cls.code_syncer = CodeTestSynchronizer(kb=cls.kb, repo_dir=REPO_DIR)
        cls.tier_pages = [
            "pricing/pricing-tier-1-usa-china-eu-100.html",
            "pricing/pricing-tier-2-brazil-malaysia-75.html",
            "pricing/pricing-tier-3-india-egypt-50.html",
            "pricing/pricing-tier-4-tech-startups-50.html",
            "pricing/pricing-tier-5-colleges-non-profits-25.html",
            "pricing/pricing-tier-6-crisis-affected-10.html"
        ]
        cls.tier_calcs = [
            f"pricing-calculator/pricing-calculator-tier-{i}.js" for i in range(1, 7)
        ]
        cls.tier_tests = [
            f"pricing-calculator/pricing-calculator-tier-{i}-tests.js" for i in range(1, 7)
        ]

    def test_discover_target_files_includes_new_files(self):
        """Verifies synchronizer.discover_target_files includes all 6 tier pages, 6 calculators, and widget."""
        targets = self.synchronizer.discover_target_files()
        target_basenames = [os.path.basename(t) for t in targets]

        self.assertIn("payment-schedule.js", target_basenames)
        for page in self.tier_pages:
            self.assertIn(os.path.basename(page), target_basenames)
        for calc in self.tier_calcs:
            self.assertIn(os.path.basename(calc), target_basenames)
        self.assertGreaterEqual(len(targets), 80)

    def test_code_syncer_discovers_all_new_files(self):
        """Verifies CodeTestSynchronizer discovers new scripts and new test suites."""
        sources = self.code_syncer.discover_source_files()
        source_basenames = [os.path.basename(s) for s in sources]
        self.assertIn("payment-schedule.js", source_basenames)
        for calc in self.tier_calcs:
            self.assertIn(os.path.basename(calc), source_basenames)

        tests = self.code_syncer.discover_test_files()
        test_basenames = [os.path.basename(t) for t in tests]
        self.assertIn("test_payment_schedule.js", test_basenames)
        for t_file in self.tier_tests:
            self.assertIn(os.path.basename(t_file), test_basenames)

    def test_tier_pages_script_linking(self):
        """Verifies each tier HTML page correctly links to its corresponding calculator script."""
        for i, page_rel in enumerate(self.tier_pages, start=1):
            full_path = os.path.join(REPO_DIR, page_rel)
            self.assertTrue(os.path.exists(full_path), f"File {page_rel} must exist")
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
            expected_script = f'pricing-calculator/pricing-calculator-tier-{i}.js'
            self.assertIn(expected_script, content, f"{page_rel} must link to {expected_script}")

    def test_tiers_references_tracking(self):
        """Verifies that nlp-fuzzy-expert/tiers_references.json tracks references in the new files."""
        ref_file = os.path.join(NLP_EXPERT_DIR, "tiers_references.json")
        self.assertTrue(os.path.exists(ref_file), "tiers_references.json must exist")
        data = load_json(ref_file)
        all_refs = data.get("all_references", [])
        tracked_files = {r.get("file") for r in all_refs}

        self.assertIn("payment-schedule.js", tracked_files)
        for page in self.tier_pages:
            self.assertIn(page, tracked_files)
        for calc in self.tier_calcs:
            self.assertIn(calc, tracked_files)

    def test_zero_emojis_and_em_dashes_in_new_files(self):
        """Verifies strict zero-emojis and zero-em-dash compliance across all new files."""
        emoji_pattern = re.compile(r"[\U00010000-\U0010ffff]", flags=re.UNICODE)
        em_dash_pattern = re.compile(r"(\u2014|&mdash;|&#8212;)")

        all_files = (
            self.tier_pages
            + self.tier_calcs
            + self.tier_tests
            + ["payment-schedule.js", "tests/js/test_payment_schedule.js"]
        )

        for rel_path in all_files:
            full_path = os.path.join(REPO_DIR, rel_path)
            self.assertTrue(os.path.exists(full_path), f"File {rel_path} must exist")
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertFalse(
                bool(emoji_pattern.search(content)),
                f"File {rel_path} contains forbidden emojis"
            )
            self.assertFalse(
                bool(em_dash_pattern.search(content)),
                f"File {rel_path} contains forbidden em-dash"
            )


if __name__ == "__main__":
    unittest.main()
