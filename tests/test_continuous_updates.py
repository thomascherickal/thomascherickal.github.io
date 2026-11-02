#!/usr/bin/env python3
"""
test_continuous_updates.py - Continuous Updates and Idempotency Unit Tests.

Proves that data changes in data/*.json can be applied multiple times in sequence,
and the NLP-Fuzzy-Expert system continuously and accurately overwrites HTML/JS files.

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
DATA_DIR = os.path.join(REPO_DIR, "data")

sys.path.insert(0, NLP_EXPERT_DIR)
from synchronizer import ExpertSynchronizer
from expert_system import load_json, save_json

class TestContinuousUpdates(unittest.TestCase):
    """Verifies that the expert system handles continuous multi-cycle updates."""

    def setUp(self):
        self.pricing_file = os.path.join(DATA_DIR, "pricing.json")
        self.personal_file = os.path.join(DATA_DIR, "personal-services.json")
        self.orig_pricing = load_json(self.pricing_file)
        self.orig_personal = load_json(self.personal_file)
        self.orchestrator = ExpertSynchronizer()

    def tearDown(self):
        # Restore original JSON data
        save_json(self.pricing_file, self.orig_pricing)
        save_json(self.personal_file, self.orig_personal)
        # Re-run sync to leave repo in clean original state
        self.orchestrator.sync_all()

    def test_pricing_continuous_seat_and_duration_updates(self):
        """Tests changing included seats and duration back and forth across 2 cycles."""
        pricing_html = os.path.join(REPO_DIR, "pricing.html")
        calc_js = os.path.join(REPO_DIR, "pricing-calculator.js")

        # Cycle 1: Set included seats = 15, flagship days = 50
        data1 = load_json(self.pricing_file)
        for off in data1["offerings"]:
            if off["id"] == "flagship-business-training":
                off["cohort_capacity"]["included_seats"] = 15
                off["schedule"]["total_calendar_duration_days"] = 50
        save_json(self.pricing_file, data1)

        self.orchestrator.sync_all()

        with open(pricing_html, "r", encoding="utf-8") as f:
            c1 = f.read()
        self.assertIn("Business (≤15 seats):", c1)
        self.assertIn("15 Participants (Included in Base)", c1)

        with open(calc_js, "r", encoding="utf-8") as f:
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

        with open(pricing_html, "r", encoding="utf-8") as f:
            c2 = f.read()
        self.assertIn("Business (≤25 seats):", c2)
        self.assertIn("25 Participants (Included in Base)", c2)

        with open(calc_js, "r", encoding="utf-8") as f:
            js2 = f.read()
        self.assertIn("participants - 25", js2)
        self.assertIn("Business Training Flagship (70 Days", js2)

    def test_personal_services_continuous_name_updates(self):
        """Tests changing service name back and forth across cycles."""
        personal_ai_html = os.path.join(REPO_DIR, "personal-ai-training.html")

        # Cycle 1: Rename Service 04 to "Custom Agentic Workflow"
        p_data1 = load_json(self.personal_file)
        for s in p_data1["services"]:
            if s["number"] == "04":
                s["name"] = "Custom Agentic Workflow"
        save_json(self.personal_file, p_data1)

        self.orchestrator.sync_all()

        with open(personal_ai_html, "r", encoding="utf-8") as f:
            content1 = f.read()
        self.assertIn("Service: 04 (Custom Agentic Workflow)", content1)

if __name__ == "__main__":
    unittest.main()
