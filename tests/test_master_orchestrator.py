#!/usr/bin/env python3
"""
test_master_orchestrator.py - Unit Tests for NLP-Fuzzy-Expert Master Synchronizer.

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

class TestMasterOrchestrator(unittest.TestCase):
    """Unit tests for ExpertSynchronizer."""

    def setUp(self):
        self.orchestrator = ExpertSynchronizer()

    def test_discover_target_files(self):
        """Verifies discover_target_files finds root HTMLs, JS, and subdirectories."""
        targets = self.orchestrator.discover_target_files()
        self.assertGreaterEqual(len(targets), 60, "Should discover at least 60 target files")
        target_basenames = [os.path.basename(t) for t in targets]
        self.assertIn("pricing.html", target_basenames)
        self.assertIn("index.html", target_basenames)
        self.assertIn("pricing-calculator.js", target_basenames)
        self.assertIn("contact.html", target_basenames)
        self.assertIn("llms.txt", target_basenames)

    def test_sync_all_and_aggregate_dictionary(self):
        """Verifies full sync produces references_dictionary.json with 200+ references."""
        self.orchestrator.sync_all()

        dict_path = os.path.join(NLP_EXPERT_DIR, "references_dictionary.json")
        self.assertTrue(os.path.exists(dict_path), "Master dictionary must exist")

        data = load_json(dict_path)
        self.assertIn("by_domain", data)
        self.assertEqual(len(data["by_domain"]), 11, "All 11 domains must be present in master dictionary")

        total_refs = data["metadata"]["total_references_tracked"]
        self.assertGreaterEqual(total_refs, 200, "Should aggregate 200+ references across repo")
        self.assertEqual(total_refs, len(data["all_references"]))

        # Verify lightweight manifest
        manifest_path = os.path.join(NLP_EXPERT_DIR, "references_manifest.json")
        self.assertTrue(os.path.exists(manifest_path), "Lightweight manifest must exist")
        m_data = load_json(manifest_path)
        self.assertIn("by_domain", m_data)
        self.assertIn("by_document", m_data)
        self.assertEqual(m_data["metadata"]["total_references_tracked"], total_refs)

    def test_markdown_reports_generated(self):
        """Verifies that all 11 domain markdown reports exist under nlp-fuzzy-expert/."""
        reports = [
            "pricing.md", "business-services.md", "personal-services.md",
            "tiers.md", "claude.md", "gemini.md",
            "publications.md", "config.md", "faqs.md", "contact.md", "guarantees.md"
        ]
        for r_name in reports:
            r_path = os.path.join(NLP_EXPERT_DIR, r_name)
            self.assertTrue(os.path.exists(r_path), f"Report {r_name} must exist")
            with open(r_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("NLP-Fuzzy-Expert Synchronization Report", content)
            self.assertIn("Tracked Occurrences Table", content)

    def test_sync_file_single(self):
        """Verifies sync_file executes cleanly for a target file."""
        res = self.orchestrator.sync_file("pricing.json")
        self.assertTrue(res)

        res_unknown = self.orchestrator.sync_file("non_existent_extension.xyz")
        self.assertFalse(res_unknown)

    def test_refresh_references_and_main(self):
        """Verifies refresh_references_only and synchronizer.main()."""
        from unittest.mock import patch
        import synchronizer

        with patch.object(self.orchestrator, "scan_all_locations", return_value={"pricing": []}), \
             patch("synchronizer.write_reference_files"), \
             patch.object(self.orchestrator, "generate_markdown_reports"):
            self.orchestrator.refresh_references_only()

        with patch.object(synchronizer.ExpertSynchronizer, "sync_all", return_value=0) as mock_sync:
            synchronizer.main()
            mock_sync.assert_called_once()


if __name__ == "__main__":
    unittest.main()
