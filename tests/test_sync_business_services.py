#!/usr/bin/env python3
"""
test_sync_business_services.py - Unit Tests for Business Services Synchronization.

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

class TestSyncBusinessServices(unittest.TestCase):
    """Unit tests for business services knowledge synchronization."""

    def setUp(self):
        self.orchestrator = ExpertSynchronizer()
        self.kb = self.orchestrator.kb

    def test_facts_loaded(self):
        """Verifies all 12 business services facts are in knowledge base."""
        for i in range(1, 13):
            num = f"{i:02d}"
            name = self.kb.get_fact("business_services", f"{num}_name")
            slug = self.kb.get_fact("business_services", f"{num}_slug")
            self.assertIsNotNone(name, f"Service {num} name must exist")
            self.assertIsNotNone(slug, f"Service {num} slug must exist")

    def test_sync_business_cards_and_curriculum(self):
        """Verifies sync_all updates all-business-services.html and all-business-services.html."""
        self.orchestrator.sync_all()

        biz_cards_path = os.path.join(REPO_DIR, "all-business-services.html")
        with open(biz_cards_path, "r", encoding="utf-8") as f:
            content = f.read()

        svc01_name = self.kb.get_fact("business_services", "01_name")
        self.assertIn(svc01_name, content)

        curriculum_path = os.path.join(REPO_DIR, "all-business-services.html")
        with open(curriculum_path, "r", encoding="utf-8") as f:
            c_content = f.read()
        self.assertIn(svc01_name, c_content)

    def test_references_and_markdown(self):
        """Verifies business_services_references.json and business-services.md exist."""
        self.orchestrator.sync_all()
        ref_file = os.path.join(NLP_EXPERT_DIR, "business_services_references.json")
        md_file = os.path.join(NLP_EXPERT_DIR, "business-services.md")

        self.assertTrue(os.path.exists(ref_file))
        self.assertTrue(os.path.exists(md_file))

        data = load_json(ref_file)
        self.assertGreater(len(data.get("all_references", [])), 0)

if __name__ == "__main__":
    unittest.main()
