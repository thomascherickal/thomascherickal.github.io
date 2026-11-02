#!/usr/bin/env python3
"""
test_sync_personal_services.py - Unit Tests for Personal Services Synchronization.

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

class TestSyncPersonalServices(unittest.TestCase):
    """Unit tests for personal services knowledge synchronization."""

    def setUp(self):
        self.orchestrator = ExpertSynchronizer()
        self.kb = self.orchestrator.kb

    def test_facts_loaded(self):
        """Verifies all 12 personal services facts are in knowledge base."""
        for i in range(1, 13):
            num = f"{i:02d}"
            name = self.kb.get_fact("personal_services", f"{num}_name")
            slug = self.kb.get_fact("personal_services", f"{num}_slug")
            self.assertIsNotNone(name, f"Personal Service {num} name must exist")
            self.assertIsNotNone(slug, f"Personal Service {num} slug must exist")

    def test_sync_personal_cards_and_curriculum(self):
        """Verifies sync_all updates all-personal-services.html and personal-ai-training.html."""
        self.orchestrator.sync_all()

        pers_cards_path = os.path.join(REPO_DIR, "all-personal-services.html")
        with open(pers_cards_path, "r", encoding="utf-8") as f:
            content = f.read()

        svc01_name = self.kb.get_fact("personal_services", "01_name")
        self.assertIn(svc01_name, content)

        curriculum_path = os.path.join(REPO_DIR, "personal-ai-training.html")
        with open(curriculum_path, "r", encoding="utf-8") as f:
            c_content = f.read()
        self.assertIn(f"Service: 01 ({svc01_name})", c_content)

    def test_references_and_markdown(self):
        """Verifies personal_services_references.json and personal-services.md exist."""
        self.orchestrator.sync_all()
        ref_file = os.path.join(NLP_EXPERT_DIR, "personal_services_references.json")
        md_file = os.path.join(NLP_EXPERT_DIR, "personal-services.md")

        self.assertTrue(os.path.exists(ref_file))
        self.assertTrue(os.path.exists(md_file))

        data = load_json(ref_file)
        self.assertGreater(len(data.get("all_references", [])), 0)

if __name__ == "__main__":
    unittest.main()
