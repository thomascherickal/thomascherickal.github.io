#!/usr/bin/env python3
"""
test_expert_system.py - Unit Tests for KnowledgeBase & InferenceEngine.

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
from expert_system import (
    KnowledgeBase,
    InferenceEngine,
    Rule,
    Fact,
    load_json,
    save_json
)

class TestExpertSystem(unittest.TestCase):
    """Unit tests for the expert system knowledge base and inference engine."""

    def setUp(self):
        self.kb = KnowledgeBase()
        self.engine = InferenceEngine(self.kb)

    def test_knowledge_base_loading(self):
        """Verifies KnowledgeBase loads facts from all data/*.json files."""
        # Pricing facts
        self.assertIsNotNone(self.kb.get_fact("pricing", "b_flag_price"))
        self.assertIsNotNone(self.kb.get_fact("pricing", "p_flag_price"))
        self.assertIsNotNone(self.kb.get_fact("pricing", "b_flag_seats"))

        # Config facts
        self.assertEqual(self.kb.get_fact("config", "author"), "Thomas Cherickal")
        self.assertEqual(self.kb.get_fact("config", "email"), "thomascherickal@gmail.com")

        # Tiers facts
        self.assertIsNotNone(self.kb.get_fact("tiers", "tier_1_multiplier"))

    def test_rule_salience_ordering(self):
        """Verifies production rules are resolved and fired in salience order."""
        r_low = Rule("LowRule", 10, lambda *args: (True, {}), lambda p, c, *args: (c, []))
        r_high = Rule("HighRule", 100, lambda *args: (True, {}), lambda p, c, *args: (c, []))

        self.engine.register_rule(r_low)
        self.engine.register_rule(r_high)

        self.assertEqual(self.engine.rules[0].name, "HighRule")
        self.assertEqual(self.engine.rules[1].name, "LowRule")

    def test_record_occurrence(self):
        """Verifies occurrence tracking records file, page, section, and line."""
        self.engine.record_occurrence("pricing.html", "Pricing", "Hero", 100, "flagship_price", "$1,000", "$2,000", "test context")
        self.assertEqual(len(self.engine.tracked_occurrences), 1)
        occ = self.engine.tracked_occurrences[0]
        self.assertEqual(occ["file"], "pricing.html")
        self.assertEqual(occ["field_key"], "flagship_price")
        self.assertEqual(occ["new_value"], "$2,000")

if __name__ == "__main__":
    unittest.main()

