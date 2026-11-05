#!/usr/bin/env python3
"""
test_sync_claude_and_gemini.py - Unit Tests for Claude and Gemini Guides Sync.

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

class TestSyncClaudeAndGemini(unittest.TestCase):
    """Unit tests for Claude and Gemini guides knowledge synchronization."""

    def setUp(self):
        self.orchestrator = ExpertSynchronizer()
        self.kb = self.orchestrator.kb

    def test_facts_loaded(self):
        """Verifies guides facts exist in knowledge base."""
        cl_raw = self.kb.raw_data.get("claude", {}).get("guides", [])
        gem_raw = self.kb.raw_data.get("gemini", {}).get("guides", [])

        self.assertGreaterEqual(len(cl_raw), 10)
        self.assertGreaterEqual(len(gem_raw), 10)

    def test_sync_guides_titles(self):
        """Verifies sync_all synchronizes guide titles in HTML files."""
        self.orchestrator.sync_all()

        guide_path = os.path.join(REPO_DIR, "claude-training", "claude-code.html")
        if os.path.exists(guide_path):
            with open(guide_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("<title>", content)
            self.assertIn("Thomas Cherickal</title>", content)

    def test_references_and_markdown(self):
        """Verifies claude_references.json, gemini_references.json, claude.md, and gemini.md exist."""
        self.orchestrator.sync_all()

        self.assertTrue(os.path.exists(os.path.join(NLP_EXPERT_DIR, "claude_references.json")))
        self.assertTrue(os.path.exists(os.path.join(NLP_EXPERT_DIR, "gemini_references.json")))
        self.assertTrue(os.path.exists(os.path.join(NLP_EXPERT_DIR, "claude.md")))
        self.assertTrue(os.path.exists(os.path.join(NLP_EXPERT_DIR, "gemini.md")))

    def test_deepmind_guide_facts_and_sync(self):
        """Verifies Google DeepMind guide facts and synchronization in target files."""
        gem_guides = self.kb.raw_data.get("gemini", {}).get("guides", [])
        dm_entry = next((g for g in gem_guides if g.get("id") == "google-deepmind"), None)
        self.assertIsNotNone(dm_entry, "google-deepmind must exist in data/gemini.json")
        self.assertEqual(dm_entry["slug"], "google-deepmind.html")
        self.assertEqual(dm_entry["short_name"], "Google DeepMind")

        dm_html = os.path.join(REPO_DIR, "google-training", "google-deepmind.html")
        self.assertTrue(os.path.exists(dm_html), "google-deepmind.html must exist")
        with open(dm_html, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("Google DeepMind: Frontier AI Research", content)
        self.assertIn("AlphaFold", content)
        self.assertIn("Argon", content)

    def test_claude_guides_completeness(self):
        """Verifies all 12 Claude guides exist on disk and have proper structure."""
        cl_guides = self.kb.raw_data.get("claude", {}).get("guides", [])
        self.assertEqual(len(cl_guides), 12, "Must track exactly 12 Claude guides")
        for g in cl_guides:
            slug = g["slug"]
            path = os.path.join(REPO_DIR, "claude-training", slug)
            self.assertTrue(os.path.exists(path), f"Claude guide {slug} must exist on disk")

if __name__ == "__main__":
    unittest.main()
