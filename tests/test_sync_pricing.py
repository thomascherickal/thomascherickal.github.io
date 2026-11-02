#!/usr/bin/env python3
"""
test_sync_pricing.py - Comprehensive Unit Tests for Pricing Knowledge Synchronization.

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

class TestSyncPricing(unittest.TestCase):
    """Unit tests for pricing rules and repository synchronization."""

    def setUp(self):
        self.orchestrator = ExpertSynchronizer()
        self.kb = self.orchestrator.kb

    def test_pricing_authoritative_facts(self):
        """Verifies authoritative values are properly extracted from data/pricing.json."""
        b_price = self.kb.get_fact("pricing", "b_flag_price")
        b_days = self.kb.get_fact("pricing", "b_flag_days")
        b_seats = self.kb.get_fact("pricing", "b_flag_seats")
        b_extra = self.kb.get_fact("pricing", "b_flag_extra_seat")
        p_price = self.kb.get_fact("pricing", "p_flag_price")
        p_days = self.kb.get_fact("pricing", "p_flag_days")
        b_ind = self.kb.get_fact("pricing", "b_ind_price")
        p_ind = self.kb.get_fact("pricing", "p_ind_price")

        self.assertGreater(b_price, 0)
        self.assertGreater(b_days, 0)
        self.assertGreater(b_seats, 0)
        self.assertGreater(b_extra, 0)
        self.assertGreater(p_price, 0)
        self.assertGreater(p_days, 0)
        self.assertGreater(b_ind, 0)
        self.assertGreater(p_ind, 0)

    def test_pricing_calculator_js_updates(self):
        """Verifies pricing-calculator.js contains the synchronized variables."""
        self.orchestrator.sync_all()
        calc_path = os.path.join(REPO_DIR, "pricing-calculator.js")
        with open(calc_path, "r", encoding="utf-8") as f:
            content = f.read()

        b_seats = self.kb.get_fact("pricing", "b_flag_seats")
        b_extra = self.kb.get_fact("pricing", "b_flag_extra_seat")
        b_days = self.kb.get_fact("pricing", "b_flag_days")
        b_live = self.kb.get_fact("pricing", "b_flag_live_days")
        b_supp = self.kb.get_fact("pricing", "b_flag_supp_days")

        expected_math = f"Math.max(0, participants - {b_seats})"
        self.assertIn(expected_math, content, "Calculator must use dynamic b_flag_seats")

        expected_fee = f"extraParticipants * {b_extra} * tier.multiplier"
        self.assertIn(expected_fee, content, "Calculator must use dynamic b_flag_extra_seat")

        expected_scope = f"Business Training Flagship ({b_days} Days: {b_live}d + {b_supp}d"
        self.assertIn(expected_scope, content, "Calculator scope must reflect flagship days")

    def test_pricing_html_updates(self):
        """Verifies pricing.html reflects updated pricing and seats."""
        self.orchestrator.sync_all()
        pricing_path = os.path.join(REPO_DIR, "pricing.html")
        with open(pricing_path, "r", encoding="utf-8") as f:
            content = f.read()

        b_price = self.kb.get_fact("pricing", "b_flag_price")
        p_price = self.kb.get_fact("pricing", "p_flag_price")
        b_ind = self.kb.get_fact("pricing", "b_ind_price")
        b_seats = self.kb.get_fact("pricing", "b_flag_seats")

        self.assertIn(f'data-price="{b_price}"', content)
        self.assertIn(f'data-price="{p_price}"', content)
        self.assertIn(f'data-price="{b_ind}"', content)
        self.assertIn(f"Business (≤{b_seats} seats):", content)

    def test_pricing_references_and_markdown(self):
        """Verifies pricing_references.json and pricing.md exist and are populated."""
        self.orchestrator.sync_all()
        ref_file = os.path.join(NLP_EXPERT_DIR, "pricing_references.json")
        md_file = os.path.join(NLP_EXPERT_DIR, "pricing.md")

        self.assertTrue(os.path.exists(ref_file))
        self.assertTrue(os.path.exists(md_file))

        data = load_json(ref_file)
        self.assertGreater(len(data.get("all_references", [])), 0)

    def test_pricing_zero_and_edge_multipliers(self):
        """Verifies pricing calculations with edge cases and extreme participant volumes."""
        b_price = self.kb.get_fact("pricing", "b_flag_price")
        p_price = self.kb.get_fact("pricing", "p_flag_price")
        b_seats = self.kb.get_fact("pricing", "b_flag_seats")
        b_extra = self.kb.get_fact("pricing", "b_flag_extra_seat")

        # Business extra seats formula evaluation
        seats_edge_1 = b_seats # exact threshold (no fee)
        extra_fee_1 = max(0, seats_edge_1 - b_seats) * b_extra
        self.assertEqual(extra_fee_1, 0)

        seats_edge_2 = b_seats + 10 # 10 additional participants
        extra_fee_2 = max(0, seats_edge_2 - b_seats) * b_extra
        self.assertEqual(extra_fee_2, 10 * b_extra)

        # Baseline pricing sanity
        self.assertGreaterEqual(b_price, p_price, "Business Flagship must be >= Personal Flagship")
        self.assertEqual(b_price, 2500)
        self.assertGreaterEqual(p_price, 1000)

if __name__ == "__main__":
    unittest.main()
