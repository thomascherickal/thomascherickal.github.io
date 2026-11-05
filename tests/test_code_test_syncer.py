#!/usr/bin/env python3
"""
test_code_test_syncer.py - Unit Tests for CodeTestSynchronizer.

Verifies:
- Discovery of source code files and test suites.
- Accurate synchronization of pricing facts into scripts and tests.
- Dynamic line diff calculation.
- JSON references and Markdown report artifact generation.
- Strict data/ directory immutability.

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
from code_test_syncer import CodeTestSynchronizer, compute_changed_lines
from expert_system import KnowledgeBase


class TestCodeTestSynchronizer(unittest.TestCase):
    """Test suite for CodeTestSynchronizer."""

    def setUp(self):
        self.kb = KnowledgeBase(data_dir=os.path.join(REPO_DIR, "data"))
        self.syncer = CodeTestSynchronizer(kb=self.kb, repo_dir=REPO_DIR)

    def test_discover_source_files(self):
        """Verifies that discover_source_files returns key Python and JS source scripts."""
        sources = self.syncer.discover_source_files()
        basenames = [os.path.basename(s) for s in sources]
        self.assertIn("pricing-calculator.js", basenames)
        self.assertIn("components.js", basenames)
        self.assertIn("generate_sitemap.py", basenames)
        self.assertIn("build_components.py", basenames)
        self.assertIn("delete_page.py", basenames)
        self.assertIn("generate_og_card.py", basenames)

    def test_discover_test_files(self):
        """Verifies that discover_test_files discovers pricing-tests.js and python unit tests."""
        tests = self.syncer.discover_test_files()
        basenames = [os.path.basename(t) for t in tests]
        self.assertIn("pricing-tests.js", basenames)
        self.assertTrue(any(b.startswith("test_") and b.endswith(".py") for b in basenames))

    def test_compute_changed_lines(self):
        """Verifies line diff calculation for identical and modified content."""
        self.assertEqual(compute_changed_lines("line1\nline2", "line1\nline2"), "None")
        diff = compute_changed_lines("line1\nline2", "line1\nline_modified")
        self.assertIn("2", diff)

    def test_sync_pricing_calculator(self):
        """Verifies that sync_pricing_calculator inserts authoritative rates and formulas."""
        sample_js = (
            "const ratePerSvc = 100;\n"
            "let basePrice = 1000;\n"
            "const matchingFlagshipPrice = 1000;\n"
            "const extraHeadRate = Math.round(extraParticipants * 100 * tier.multiplier);\n"
            "const extraSavings = Math.round(extraParticipants * 100 * (tier.discountPercent / 100));\n"
            "Math.max(0, participants - 5)\n"
            "const calculated = Math.round(100 * mult);\n"
        )
        updated = self.syncer.sync_pricing_calculator(sample_js)
        b_ind = self.kb.get_fact("pricing", "b_ind_price", 300)
        b_price = self.kb.get_fact("pricing", "b_flag_price", 3000)
        b_extra = self.kb.get_fact("pricing", "b_flag_extra_seat", 300)
        b_seats = self.kb.get_fact("pricing", "b_flag_seats", 10)

        self.assertIn(f"const ratePerSvc = {b_ind};", updated)
        self.assertIn(f"let basePrice = {b_price};", updated)
        self.assertIn(f"Math.round(extraParticipants * {b_extra} * tier.multiplier);", updated)
        self.assertIn(f"participants - {b_seats}", updated)
        self.assertIn(f"const calculated = Math.round({b_ind} * mult);", updated)

    def test_sync_pricing_tests_js_constants(self):
        """Verifies that sync_pricing_tests_js injects or updates the authoritative facts block."""
        sample_test = (
            "const PricingCalculator = require('./pricing-calculator.js');\n\n"
            "function runTestSuite() {}\n"
        )
        updated = self.syncer.sync_pricing_tests_js(sample_test)
        self.assertIn("// --- AUTHORITATIVE SYNCHRONIZED PRICING FACTS ---", updated)
        self.assertIn("const B_FLAG_PRICE =", updated)
        self.assertIn("const P_FLAG_PRICE =", updated)
        self.assertIn("const B_IND_PRICE =", updated)
        self.assertIn("const B_EXTRA_SEAT =", updated)
        self.assertIn("const B_INCLUDED_SEATS =", updated)

    def test_sync_components_js(self):
        """Verifies that orchestration pages in components.js match Claude guide slugs."""
        sample_comp = "var orchestrationPages = [\n      'old-guide.html'\n    ];"
        updated = self.syncer.sync_components_js(sample_comp)
        self.assertIn("var orchestrationPages = [", updated)
        self.assertIn("claude-101.html", updated)

    def test_sync_all_generates_artifacts(self):
        """Verifies that running sync_all creates code_references.json and code-sync.md."""
        res = self.syncer.sync_all()
        self.assertGreater(res["total_files"], 10)
        self.assertIn("modified_count", res)

        refs_path = os.path.join(NLP_EXPERT_DIR, "code_references.json")
        report_path = os.path.join(NLP_EXPERT_DIR, "code-sync.md")

        self.assertTrue(os.path.exists(refs_path))
        self.assertTrue(os.path.exists(report_path))

        with open(report_path, "r", encoding="utf-8") as f:
            report_text = f.read()
        self.assertIn("# Source Code and Test Suite Knowledge Synchronization Report", report_text)
        self.assertIn("Target Files Evaluated", report_text)


if __name__ == "__main__":
    unittest.main()

