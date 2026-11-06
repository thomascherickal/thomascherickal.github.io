#!/usr/bin/env python3
"""
test_core_target_documents.py - Unit Tests for Core Target Documents and References Manifest.

Verifies:
- Inclusion of contact.html and llms.txt in discover_target_files().
- Presence of contact.html and llms.txt in Core Target Documents summary.
- LLMsTxtRule registration and execution.
- ContactPagePricingRule registration and execution.
- Generation and integrity of references_manifest.json.
- Preservation and integrity of references_dictionary.json.
- Strict compliance: zero emojis, zero em-dashes.

Strict Repo Compliance:
- Zero emojis.
- Zero em-dashes (spaced hyphens ' - ' or plain hyphens '-' used).
"""

import json
import os
import re
import sys
import unittest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(TESTS_DIR)
NLP_EXPERT_DIR = os.path.join(REPO_DIR, "nlp-fuzzy-expert")

sys.path.insert(0, NLP_EXPERT_DIR)
from expert_system import KnowledgeBase, InferenceEngine, load_json
from rules import build_rules
from synchronizer import ExpertSynchronizer


class TestCoreTargetDocuments(unittest.TestCase):
    """Verifies that contact.html and llms.txt are first-class Core Target Documents."""

    def setUp(self):
        self.orchestrator = ExpertSynchronizer()
        self.kb = KnowledgeBase()
        self.rules = build_rules()

    def test_discover_target_files_includes_contact_and_llms(self):
        """Verifies that discover_target_files includes contact.html and llms.txt."""
        targets = self.orchestrator.discover_target_files()
        basenames = [os.path.basename(t) for t in targets]
        self.assertIn("contact.html", basenames, "contact.html must be in discovered target files")
        self.assertIn("llms.txt", basenames, "llms.txt must be in discovered target files")

    def test_rules_include_llms_txt_and_contact_rules(self):
        """Verifies that LLMsTxtRule and ContactPagePricingRule are registered."""
        rule_names = [r.name for r in self.rules]
        self.assertIn("LLMsTxtRule", rule_names, "LLMsTxtRule must be registered in production rules")
        self.assertIn("ContactPagePricingRule", rule_names, "ContactPagePricingRule must be registered")

        llms_rule = next(r for r in self.rules if r.name == "LLMsTxtRule")
        self.assertEqual(llms_rule.domain, "pricing")
        self.assertEqual(llms_rule.salience, 86)

        contact_rule = next(r for r in self.rules if r.name == "ContactPagePricingRule")
        self.assertEqual(contact_rule.domain, "pricing")
        self.assertEqual(contact_rule.salience, 87)

    def test_llms_txt_rule_execution(self):
        """Verifies that LLMsTxtRule matches llms.txt and generates occurrences."""
        llms_rule = next(r for r in self.rules if r.name == "LLMsTxtRule")
        llms_path = os.path.join(REPO_DIR, "llms.txt")

        with open(llms_path, "r", encoding="utf-8") as f:
            content = f.read()

        matched, _ = llms_rule.condition_fn(llms_path, content, self.kb, None, None)
        self.assertTrue(matched, "LLMsTxtRule must match llms.txt")

        new_content, occs = llms_rule.action_fn(llms_path, content, {}, self.kb, None, None)
        self.assertIsInstance(new_content, str)
        self.assertGreaterEqual(len(occs), 3, "LLMsTxtRule must produce at least 3 tracked occurrences")

        field_names = [o["field"] for o in occs]
        self.assertIn("llms_business_flagship_price", field_names)
        self.assertIn("llms_personal_flagship_price", field_names)
        self.assertIn("llms_individual_service_price", field_names)

    def test_contact_page_rule_execution(self):
        """Verifies that ContactPagePricingRule matches contact.html and generates occurrences."""
        contact_rule = next(r for r in self.rules if r.name == "ContactPagePricingRule")
        contact_path = os.path.join(REPO_DIR, "contact.html")

        with open(contact_path, "r", encoding="utf-8") as f:
            content = f.read()

        matched, _ = contact_rule.condition_fn(contact_path, content, self.kb, None, None)
        self.assertTrue(matched, "ContactPagePricingRule must match contact.html")

        new_content, occs = contact_rule.action_fn(contact_path, content, {}, self.kb, None, None)
        self.assertIsInstance(new_content, str)
        self.assertGreaterEqual(len(occs), 1)

    def test_references_manifest_structure_and_counts(self):
        """Verifies that references_manifest.json contains lightweight indexing."""
        manifest_path = os.path.join(NLP_EXPERT_DIR, "references_manifest.json")
        self.assertTrue(os.path.exists(manifest_path), "references_manifest.json must exist")

        data = load_json(manifest_path)
        self.assertEqual(data["metadata"]["manifest_type"], "lightweight_index")
        self.assertIn("by_domain", data)
        self.assertIn("by_document", data)
        self.assertIn("domain_files", data)

        # Confirm contact.html and llms.txt are indexed in document breakdown
        self.assertIn("contact.html", data["by_document"])
        self.assertIn("llms.txt", data["by_document"])
        self.assertGreater(data["by_document"]["contact.html"], 0)
        self.assertGreater(data["by_document"]["llms.txt"], 0)

        # Confirm lightweight footprint (< 50 KB)
        file_size = os.path.getsize(manifest_path)
        self.assertLess(file_size, 50 * 1024, "Manifest must be lightweight (under 50 KB)")

    def test_sync_manifest_structure_and_counts(self):
        """Verifies that sync_manifest.json exists and tracks data digests and targets."""
        sync_manifest_path = os.path.join(NLP_EXPERT_DIR, "sync_manifest.json")
        self.assertTrue(os.path.exists(sync_manifest_path), "sync_manifest.json must exist")

        data = load_json(sync_manifest_path)
        self.assertEqual(data["metadata"]["manifest_type"], "synchronization_manifest")
        self.assertIn("data_hashes", data)
        self.assertIn("targets", data)
        self.assertEqual(len(data["data_hashes"]), 11, "Must track all 11 data files")
        self.assertGreaterEqual(len(data["targets"]), 67, "Must track all target documents")

    def test_references_dictionary_preserved_and_intact(self):
        """Verifies that master references_dictionary.json is preserved and intact."""
        master_path = os.path.join(NLP_EXPERT_DIR, "references_dictionary.json")
        self.assertTrue(os.path.exists(master_path), "references_dictionary.json must be preserved")

        data = load_json(master_path)
        self.assertIn("all_references", data)
        self.assertIn("by_domain", data)
        self.assertGreaterEqual(len(data["all_references"]), 10000)

    def test_contact_json_schema_and_facts(self):
        """Verifies that data/contact.json exists and registers facts in knowledge base."""
        contact_path = os.path.join(REPO_DIR, "data", "contact.json")
        self.assertTrue(os.path.exists(contact_path), "data/contact.json must exist")

        data = load_json(contact_path)
        expected_keys = [
            "contact_metadata", "config_references", "direct_communication",
            "engagement_standards", "intake_workflow_steps", "form_configuration"
        ]
        for key in expected_keys:
            self.assertIn(key, data, f"Key '{key}' must be in data/contact.json")

        # Verify KnowledgeBase loaded facts
        self.assertEqual(self.kb.get_fact("contact", "page_title"), "Contact | Thomas Cherickal")
        self.assertEqual(self.kb.get_fact("contact", "location"), "Remote Worldwide")
        self.assertEqual(
            self.kb.get_fact("contact", "action_endpoint"),
            "https://formsubmit.co/ajax/thomascherickal@gmail.com"
        )

    def test_contact_json_no_duplication(self):
        """Verifies data/contact.json avoids duplication by referencing config and pricing."""
        contact_path = os.path.join(REPO_DIR, "data", "contact.json")
        data = load_json(contact_path)

        refs = data.get("config_references", {})
        self.assertEqual(refs.get("author_ref"), "config.json#author")
        self.assertEqual(refs.get("email_ref"), "config.json#email")
        self.assertEqual(refs.get("pricing_ref"), "pricing.json")
        self.assertEqual(refs.get("tiers_ref"), "tiers.json")

        # Confirm contact.json does not duplicate full tier objects or full offering objects
        self.assertNotIn("tiers", data)
        self.assertNotIn("offerings", data)
        self.assertNotIn("general_payment_rules", data)

    def test_guarantees_json_schema_and_facts(self):
        """Verifies that data/guarantees.json exists and registers facts in knowledge base."""
        g_path = os.path.join(REPO_DIR, "data", "guarantees.json")
        self.assertTrue(os.path.exists(g_path), "data/guarantees.json must exist")

        data = load_json(g_path)
        expected_keys = [
            "guarantees_metadata", "config_references", "trust_pillars",
            "agency_comparison", "apache_license_rights"
        ]
        for key in expected_keys:
            self.assertIn(key, data, f"Key '{key}' must be in data/guarantees.json")

        # Verify KnowledgeBase loaded facts
        self.assertEqual(
            self.kb.get_fact("guarantees", "page_title"),
            "Trust Guarantees & Payment Structure | Thomas Cherickal"
        )
        self.assertEqual(
            self.kb.get_fact("guarantees", "hero_badge"),
            "// 100% RISK-FREE & SCAM-FREE ENGAGEMENT"
        )
        self.assertEqual(
            self.kb.get_fact("guarantees", "walkaway_title"),
            "Walk Away Free Guarantee"
        )

    def test_guarantees_json_no_duplication(self):
        """Verifies data/guarantees.json avoids duplication by referencing config, faqs, and pricing."""
        g_path = os.path.join(REPO_DIR, "data", "guarantees.json")
        data = load_json(g_path)

        refs = data.get("config_references", {})
        self.assertEqual(refs.get("author_ref"), "config.json#author")
        self.assertEqual(refs.get("email_ref"), "config.json#email")
        self.assertEqual(refs.get("pricing_ref"), "pricing.json")
        self.assertEqual(refs.get("faqs_ref"), "faqs.json")
        self.assertEqual(refs.get("tiers_ref"), "tiers.json")

        # Confirm guarantees.json does not duplicate full offerings or tiers schemas
        self.assertNotIn("tiers", data)
        self.assertNotIn("offerings", data)
        self.assertNotIn("core_policies", data)

    def test_compliance_zero_emojis_and_zero_em_dashes(self):
        """Verifies zero emojis and zero em-dashes in manifests, contact.json, guarantees.json, and llms.txt."""
        manifest_path = os.path.join(NLP_EXPERT_DIR, "references_manifest.json")
        with open(manifest_path, "r", encoding="utf-8") as f:
            m_text = f.read()

        sync_manifest_path = os.path.join(NLP_EXPERT_DIR, "sync_manifest.json")
        with open(sync_manifest_path, "r", encoding="utf-8") as f:
            sm_text = f.read()

        llms_path = os.path.join(REPO_DIR, "llms.txt")
        with open(llms_path, "r", encoding="utf-8") as f:
            l_text = f.read()

        contact_path = os.path.join(REPO_DIR, "data", "contact.json")
        with open(contact_path, "r", encoding="utf-8") as f:
            c_text = f.read()

        guarantees_path = os.path.join(REPO_DIR, "data", "guarantees.json")
        with open(guarantees_path, "r", encoding="utf-8") as f:
            g_text = f.read()

        for text in [m_text, sm_text, l_text, c_text, g_text]:
            self.assertNotIn(chr(0x2014), text)
            self.assertNotIn("&" + "mdash;", text)
            self.assertNotIn("&" + "#8212;", text)

            emoji_pattern = re.compile(r"[\U00010000-\U0010ffff]", flags=re.UNICODE)
            self.assertEqual(len(emoji_pattern.findall(text)), 0)


if __name__ == "__main__":
    unittest.main()

