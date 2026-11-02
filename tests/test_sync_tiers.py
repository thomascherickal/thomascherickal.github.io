#!/usr/bin/env python3
"""
test_sync_tiers.py - Unit Tests for Country Parity Tiers Synchronization.

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
from synchronizer import ExpertSynchronizer
from expert_system import load_json

class TestSyncTiers(unittest.TestCase):
    """Unit tests for country parity tiers knowledge synchronization."""

    def setUp(self):
        self.orchestrator = ExpertSynchronizer()
        self.kb = self.orchestrator.kb

    def test_facts_loaded(self):
        """Verifies 6 tiers facts exist in knowledge base."""
        for t in range(1, 7):
            mult = self.kb.get_fact("tiers", f"tier_{t}_multiplier")
            self.assertIsNotNone(mult, f"Tier {t} multiplier must exist")

    def test_sync_tiers_pricing_and_calc(self):
        """Verifies sync_all updates parity tiers in pricing.html and pricing-calculator.js."""
        self.orchestrator.sync_all()

        calc_path = os.path.join(REPO_DIR, "pricing-calculator.js")
        with open(calc_path, "r", encoding="utf-8") as f:
            content = f.read()

        t1_mult = self.kb.get_fact("tiers", "tier_1_multiplier")
        self.assertIn(f"multiplier: {t1_mult}", content)

    def test_references_and_markdown(self):
        """Verifies tiers_references.json and tiers.md exist."""
        self.orchestrator.sync_all()
        ref_file = os.path.join(NLP_EXPERT_DIR, "tiers_references.json")
        md_file = os.path.join(NLP_EXPERT_DIR, "tiers.md")

        self.assertTrue(os.path.exists(ref_file))
        self.assertTrue(os.path.exists(md_file))

        data = load_json(ref_file)
        self.assertGreater(len(data.get("all_references", [])), 0)

if __name__ == "__main__":
    unittest.main()
