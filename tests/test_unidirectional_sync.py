#!/usr/bin/env python3
"""
test_unidirectional_sync.py - Unit Tests for Strict Unidirectional Data Synchronization.

Verifies that the NLP-Fuzzy-Expert system enforces strictly one-way data propagation:
- Changes in data/*.json flow outward to HTML, JS, Markdown, and test files.
- Website changes never propagate back to data/*.json.
- Attempts to mutate data/*.json via save_json(), InferenceEngine, or CodeTestSynchronizer are blocked.
- Cryptographic SHA-256 hashes of data/*.json remain strictly immutable throughout.

Strict Repo Compliance:
- Zero emojis.
- Zero em-dashes (spaced hyphens ' - ' or plain hyphens '-' used).
"""

import os
import sys
import shutil
import hashlib
import tempfile
import unittest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(TESTS_DIR)
NLP_EXPERT_DIR = os.path.join(REPO_DIR, "nlp-fuzzy-expert")
DATA_DIR = os.path.join(REPO_DIR, "data")

sys.path.insert(0, NLP_EXPERT_DIR)
from expert_system import (
    KnowledgeBase,
    InferenceEngine,
    load_json,
    save_json,
    verify_unidirectional_sync,
)
from synchronizer import ExpertSynchronizer
from code_test_syncer import CodeTestSynchronizer
from auto_healer import compute_data_fingerprint, verify_data_untouched


def compute_hashes(directory):
    """Computes SHA-256 hashes for all JSON files in a directory."""
    hashes = {}
    if not os.path.exists(directory):
        return hashes
    for fname in sorted(os.listdir(directory)):
        if fname.endswith(".json"):
            fpath = os.path.join(directory, fname)
            with open(fpath, "rb") as f:
                hashes[fname] = hashlib.sha256(f.read()).hexdigest()
    return hashes


class TestUnidirectionalSync(unittest.TestCase):
    """Verifies strictly one-way data synchronization from data/*.json to the website."""

    def setUp(self):
        self.initial_hashes = compute_hashes(DATA_DIR)

    def tearDown(self):
        # Enforce that repository data/ was never mutated by any test
        current_hashes = compute_hashes(DATA_DIR)
        self.assertEqual(
            self.initial_hashes,
            current_hashes,
            "CRITICAL: data/*.json was mutated during test execution! Synchronization must be strictly unidirectional."
        )

    def test_save_json_blocks_mutation_of_data_dir(self):
        """Verifies save_json raises PermissionError when targeted at data/*.json."""
        target_path = os.path.join(DATA_DIR, "pricing.json")
        with self.assertRaises(PermissionError) as ctx:
            save_json(target_path, {"test": "unauthorized_write"})
        self.assertIn("MUTATION BLOCKED", str(ctx.exception))
        self.assertIn("unidirectional", str(ctx.exception).lower())

    def test_inference_engine_blocks_data_file_processing(self):
        """Verifies InferenceEngine.process_file raises PermissionError for any data/*.json file."""
        kb = KnowledgeBase(data_dir=DATA_DIR)
        engine = InferenceEngine(kb)
        data_file = os.path.join(DATA_DIR, "tiers.json")
        with self.assertRaises(PermissionError) as ctx:
            engine.process_file(data_file)
        self.assertIn("MUTATION BLOCKED", str(ctx.exception))
        self.assertIn("unidirectional", str(ctx.exception).lower())

    def test_discover_target_files_strictly_excludes_data_dir(self):
        """Verifies that discover_target_files never includes any file inside data/."""
        syncer = ExpertSynchronizer()
        targets = syncer.discover_target_files()
        abs_data_dir = os.path.abspath(DATA_DIR)

        for target in targets:
            abs_target = os.path.abspath(target)
            self.assertFalse(
                abs_target.startswith(abs_data_dir),
                f"Target file '{target}' is inside data directory! Targets must exclude data/."
            )

    def test_code_test_syncer_strictly_excludes_data_dir(self):
        """Verifies that CodeTestSynchronizer never targets files inside data/."""
        syncer = CodeTestSynchronizer()
        sources = syncer.discover_source_files()
        tests = syncer.discover_test_files()
        abs_data_dir = os.path.abspath(DATA_DIR)

        for f in sources + tests:
            self.assertFalse(
                os.path.abspath(f).startswith(abs_data_dir),
                f"Code syncer discovered file inside data directory: '{f}'"
            )

    def test_verify_unidirectional_sync_helper(self):
        """Verifies verify_unidirectional_sync validates cleanly and flags violations."""
        valid_targets = [
            os.path.join(REPO_DIR, "index.html"),
            os.path.join(REPO_DIR, "pricing.html"),
        ]
        self.assertTrue(verify_unidirectional_sync(DATA_DIR, valid_targets))

        invalid_targets = [
            os.path.join(REPO_DIR, "pricing.html"),
            os.path.join(DATA_DIR, "config.json"),
        ]
        with self.assertRaises(PermissionError):
            verify_unidirectional_sync(DATA_DIR, invalid_targets)

    def test_website_changes_do_not_propagate_to_data_dir(self):
        """Verifies that manual/external edits to HTML files do not alter data/*.json.

        Simulates modifying an HTML file, runs synchronizer, and confirms:
        1. data/*.json SHA-256 hashes remain 100% untouched.
        2. The HTML file is restored to authoritative facts from data/*.json.
        """
        pricing_html = os.path.join(REPO_DIR, "pricing.html")
        with open(pricing_html, "r", encoding="utf-8") as f:
            orig_content = f.read()

        try:
            # Simulate website modification
            corrupted_content = orig_content.replace(
                "$3,000 USD", "$9,999 USD"
            ).replace(
                "30 days live training", "99 days live training"
            )
            with open(pricing_html, "w", encoding="utf-8") as f:
                f.write(corrupted_content)

            # Pre-sync hashes
            pre_sync_hashes = compute_hashes(DATA_DIR)

            # Run synchronizer
            syncer = ExpertSynchronizer()
            syncer.sync_all()

            # Post-sync hashes must be identical
            post_sync_hashes = compute_hashes(DATA_DIR)
            self.assertEqual(
                pre_sync_hashes,
                post_sync_hashes,
                "data/*.json was modified by website changes! Sync must be strictly unidirectional."
            )

            # Check that website was restored from authoritative data
            with open(pricing_html, "r", encoding="utf-8") as f:
                restored_content = f.read()

            self.assertNotIn("$9,999 USD", restored_content)
            self.assertNotIn("99 days live training", restored_content)
            self.assertIn("$3,000 USD", restored_content)
        finally:
            with open(pricing_html, "w", encoding="utf-8") as f:
                f.write(orig_content)
            # Re-sync to ensure clean state
            ExpertSynchronizer().sync_all()

    def test_data_changes_propagate_outward_to_website_in_sandbox(self):
        """Verifies that data changes propagate outward to the website (unidirectional).

        Uses an isolated sandbox data directory to verify that changes made to data files
        propagate forward to website documents without altering the real data/ directory.
        """
        temp_dir = tempfile.mkdtemp()
        try:
            for fname in os.listdir(DATA_DIR):
                if fname.endswith(".json"):
                    shutil.copy2(os.path.join(DATA_DIR, fname), os.path.join(temp_dir, fname))

            # Modify a value in sandbox data
            sandbox_pricing = os.path.join(temp_dir, "pricing.json")
            data = load_json(sandbox_pricing)
            for off in data["offerings"]:
                if off["id"] == "flagship-business-training":
                    off["cohort_capacity"]["included_seats"] = 14

            # Write modified data to sandbox (using direct open since sandbox is not repo DATA_DIR)
            import json
            with open(sandbox_pricing, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            # Snapshot pricing.html
            pricing_html = os.path.join(REPO_DIR, "pricing.html")
            with open(pricing_html, "r", encoding="utf-8") as f:
                orig_pricing = f.read()

            try:
                # Synchronize from sandbox data directory
                syncer = ExpertSynchronizer(data_dir=temp_dir)
                syncer.sync_all()

                with open(pricing_html, "r", encoding="utf-8") as f:
                    updated_pricing = f.read()

                # Verify outward propagation occurred
                self.assertIn("14 participants", updated_pricing)

                # Verify real repo data/ directory was NOT touched
                real_hashes = compute_hashes(DATA_DIR)
                self.assertEqual(self.initial_hashes, real_hashes)
            finally:
                # Restore original pricing.html and re-sync with canonical data
                with open(pricing_html, "w", encoding="utf-8") as f:
                    f.write(orig_pricing)
                ExpertSynchronizer().sync_all()
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()
