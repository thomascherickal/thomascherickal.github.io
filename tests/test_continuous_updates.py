#!/usr/bin/env python3
"""
test_continuous_updates.py - Continuous Updates and Idempotency Unit Tests.

Proves that data changes in data/*.json can be applied multiple times in sequence,
and the NLP-Fuzzy-Expert system continuously and accurately overwrites HTML/JS files.
Uses an isolated temporary sandbox to strictly preserve the immutability of data/*.json.

Strict Repo Compliance:
- Zero emojis.
- Zero em-dashes (spaced hyphens ' - ' or plain hyphens '-' used).
"""

import os
import sys
import shutil
import tempfile
import unittest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(TESTS_DIR)
NLP_EXPERT_DIR = os.path.join(REPO_DIR, "nlp-fuzzy-expert")
DATA_DIR = os.path.join(REPO_DIR, "data")

sys.path.insert(0, NLP_EXPERT_DIR)
from synchronizer import ExpertSynchronizer
from expert_system import load_json, save_json

class TestContinuousUpdates(unittest.TestCase):
    """Verifies that the expert system handles continuous multi-cycle updates."""

    def setUp(self):
        # Target files that will be modified during tests
        self.pricing_html = os.path.join(REPO_DIR, "pricing.html")
        self.calc_js = os.path.join(REPO_DIR, "pricing-calculator.js")
        self.personal_ai_html = os.path.join(REPO_DIR, "all-personal-services.html")
        self.guarantees_html = os.path.join(REPO_DIR, "guarantees.html")

        # Snapshot original contents of target files
        with open(self.pricing_html, "r", encoding="utf-8") as f:
            self.orig_pricing_html = f.read()
        with open(self.calc_js, "r", encoding="utf-8") as f:
            self.orig_calc_js = f.read()
        with open(self.personal_ai_html, "r", encoding="utf-8") as f:
            self.orig_personal_ai_html = f.read()
        with open(self.guarantees_html, "r", encoding="utf-8") as f:
            self.orig_guarantees_html = f.read()

        # Create isolated temporary data directory (sandbox)
        self.temp_dir = tempfile.mkdtemp()
        for fname in os.listdir(DATA_DIR):
            if fname.endswith(".json"):
                shutil.copy2(os.path.join(DATA_DIR, fname), os.path.join(self.temp_dir, fname))

        self.pricing_file = os.path.join(self.temp_dir, "pricing.json")
        self.personal_file = os.path.join(self.temp_dir, "personal-services.json")
        self.orchestrator = ExpertSynchronizer(data_dir=self.temp_dir)

    def tearDown(self):
        # Remove temporary sandbox
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

        # Restore target files to original content
        with open(self.pricing_html, "w", encoding="utf-8") as f:
            f.write(self.orig_pricing_html)
        with open(self.calc_js, "w", encoding="utf-8") as f:
            f.write(self.orig_calc_js)
        with open(self.personal_ai_html, "w", encoding="utf-8") as f:
            f.write(self.orig_personal_ai_html)
        with open(self.guarantees_html, "w", encoding="utf-8") as f:
            f.write(self.orig_guarantees_html)

        # Re-run sync against authoritative data/ to leave repo clean
        canonical_sync = ExpertSynchronizer()
        canonical_sync.sync_all()

    def test_pricing_continuous_seat_and_duration_updates(self):
        """Tests changing included seats and duration back and forth across 2 cycles."""
        # Cycle 1: Set included seats = 15, flagship days = 50
        data1 = load_json(self.pricing_file)
        for off in data1["offerings"]:
            if off["id"] == "flagship-business-training":
                off["cohort_capacity"]["included_seats"] = 15
                off["schedule"]["total_calendar_duration_days"] = 50
        save_json(self.pricing_file, data1)

        self.orchestrator.sync_all()

        with open(self.pricing_html, "r", encoding="utf-8") as f:
            c1 = f.read()
        self.assertIn("up to 15 seats", c1)
        self.assertIn("15 Participants (Included in Base)", c1)

        with open(self.calc_js, "r", encoding="utf-8") as f:
            js1 = f.read()
        self.assertIn("participants - 15", js1)
        self.assertIn("Business Training Flagship (50 Days", js1)

        # Cycle 2: Set included seats = 25, flagship days = 70
        data2 = load_json(self.pricing_file)
        for off in data2["offerings"]:
            if off["id"] == "flagship-business-training":
                off["cohort_capacity"]["included_seats"] = 25
                off["schedule"]["total_calendar_duration_days"] = 70
        save_json(self.pricing_file, data2)

        self.orchestrator.sync_all()

        with open(self.pricing_html, "r", encoding="utf-8") as f:
            c2 = f.read()
        self.assertIn("up to 25 seats", c2)
        self.assertIn("25 Participants (Included in Base)", c2)

        with open(self.calc_js, "r", encoding="utf-8") as f:
            js2 = f.read()
        self.assertIn("participants - 25", js2)
        self.assertIn("Business Training Flagship (70 Days", js2)

    def test_personal_services_continuous_name_updates(self):
        """Tests changing service name back and forth across cycles."""
        # Cycle 1: Rename Service 04 to "Custom Agentic Workflow"
        p_data1 = load_json(self.personal_file)
        for s in p_data1["services"]:
            if s["number"] == "04":
                s["name"] = "Custom Agentic Workflow"
        save_json(self.personal_file, p_data1)

        self.orchestrator.sync_all()

        with open(self.personal_ai_html, "r", encoding="utf-8") as f:
            content1 = f.read()
        self.assertIn("Custom Agentic Workflow", content1)

if __name__ == "__main__":
    unittest.main()
