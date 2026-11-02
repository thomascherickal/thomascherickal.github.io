#!/usr/bin/env python3
"""
test_expert_system.py - Unit Tests for KnowledgeBase & InferenceEngine.

Strict Repo Compliance:
- Zero emojis.
- Zero em-dashes (spaced hyphens ' - ' or plain hyphens '-' used).
"""

import os
import sys
import tempfile
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

    def test_expert_system_edge_cases(self):
        """Verifies KnowledgeBase and InferenceEngine error handling and edge cases."""
        # Non-existent domain or fact
        self.assertIsNone(self.kb.get_fact("non_existent_domain", "missing_key"))
        self.assertIsNone(self.kb.get_fact("pricing", "completely_bogus_fact_key"))

        # Empty rule execution
        empty_rule = Rule("NoopRule", 50, lambda p, c, *args: (False, {}), lambda p, c, *args: (c, []))
        self.engine.register_rule(empty_rule)
        cond, binds = empty_rule.condition_fn("sample_page.html", "sample content")
        self.assertFalse(cond)

        # Document discovery via ExpertSynchronizer
        from synchronizer import ExpertSynchronizer
        sync = ExpertSynchronizer()
        docs = sync.discover_target_files()
        self.assertGreater(len(docs), 50)
        self.assertTrue(any("index.html" in d for d in docs))
        self.assertTrue(any("pricing.html" in d for d in docs))

    def test_save_and_load_json_atomic(self):
        """Verifies atomic json writing and corruption-free reading."""
        test_file = os.path.join(tempfile.gettempdir(), f".test_atomic_{os.getpid()}.json")
        try:
            payload = {"status": "ok", "count": 42, "items": ["a", "b", "c"]}
            save_json(test_file, payload)
            loaded = load_json(test_file)
            self.assertEqual(loaded, payload)
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)

    def test_save_json_blocks_data_dir_mutation(self):
        """Verifies save_json strictly raises PermissionError when targeting data/."""
        data_dir = os.path.join(REPO_DIR, "data")
        pricing_file = os.path.join(data_dir, "pricing.json")
        with self.assertRaises(PermissionError):
            save_json(pricing_file, {"hack": "forbidden"})

        new_forbidden_file = os.path.join(data_dir, "new_file.json")
        with self.assertRaises(PermissionError):
            save_json(new_forbidden_file, {"hack": "forbidden"})

        nested_forbidden = os.path.join(data_dir, "nested", "file.json")
        with self.assertRaises(PermissionError):
            save_json(nested_forbidden, {"hack": "forbidden"})

if __name__ == "__main__":
    unittest.main()

