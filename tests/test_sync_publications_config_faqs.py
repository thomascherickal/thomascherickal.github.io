#!/usr/bin/env python3
"""
test_sync_publications_config_faqs.py - Unit Tests for Publications, Config, and FAQs.

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

class TestSyncPublicationsConfigFaqs(unittest.TestCase):
    """Unit tests for Publications, Config, and FAQs knowledge synchronization."""

    def setUp(self):
        self.orchestrator = ExpertSynchronizer()
        self.kb = self.orchestrator.kb

    def test_facts_loaded(self):
        """Verifies publications, config, and FAQs facts exist in knowledge base."""
        # Config
        self.assertEqual(self.kb.get_fact("config", "email"), "thomascherickal@gmail.com")
        self.assertEqual(self.kb.get_fact("config", "brand"), "The Digital Futurist")

        # FAQs
        self.assertIsNotNone(self.kb.get_fact("faqs", "walkaway_summary"))

        # Publications
        pubs = self.kb.raw_data.get("publications", [])
        self.assertGreater(len(pubs), 0)

    def test_sync_execution(self):
        """Verifies sync_all generates reference dictionaries and markdown files."""
        self.orchestrator.sync_all()

        for domain in ["publications", "config", "faqs"]:
            ref_path = os.path.join(NLP_EXPERT_DIR, f"{domain}_references.json")
            md_path = os.path.join(NLP_EXPERT_DIR, f"{domain}.md")
            self.assertTrue(os.path.exists(ref_path), f"{ref_path} must exist")
            self.assertTrue(os.path.exists(md_path), f"{md_path} must exist")

if __name__ == "__main__":
    unittest.main()
