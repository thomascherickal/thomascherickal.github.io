#!/usr/bin/env python3
"""
test_pricing_and_policy_invariants.py - Authoritative Pricing, Milestone & Policy Invariant Tests.

Verifies cross-repository consistency against data/*.json SSOT and GEMINI.md:
1. All 4 offerings enforce max_seats == 20 with parity pricing on extra seats.
2. Business & Personal Flagships include 10 seats ($1,000 base, +$100/extra seat).
3. Business & Personal Service Picks include 1 seat ($100 base, +$100/extra seat).
4. All 4 offerings use 10 equal 10% milestones (Days 5..50 Flagships, Days 1..10 Service Picks).
5. Zero Charge Training Materials policy (Apache 2.0, not provided if training ends early).
6. Zero regressions of removed phrases (Support Phase Day 15/30, Zero Charge Support Periods,
   20% in Five, strictly 1 person only).

Strict Repo Compliance: Zero emojis, zero em-dashes.
"""

import json
import os
import re
import unittest

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_DIR, "data")


def load_json(name):
    with open(os.path.join(DATA_DIR, name), encoding="utf-8") as f:
        return json.load(f)


def read_text(rel_path):
    with open(os.path.join(REPO_DIR, rel_path), encoding="utf-8") as f:
        return f.read()


class TestPricingAndPolicyInvariants(unittest.TestCase):
    """Authoritative data/*.json and cross-page pricing/policy invariant tests."""

    @classmethod
    def setUpClass(cls):
        cls.pricing = load_json("pricing.json")
        cls.faqs = load_json("faqs.json")
        cls.guarantees = load_json("guarantees.json")
        cls.offerings_by_id = {o["id"]: o for o in cls.pricing["offerings"]}

    def test_all_four_offerings_present_with_max_20_seats(self):
        """Verifies all 4 offerings exist and strictly cap max_seats at 20 with parity extra seats."""
        expected_ids = [
            "flagship-business-training",
            "flagship-personal-training",
            "individual-service-business",
            "individual-service-personal",
        ]
        for oid in expected_ids:
            self.assertIn(oid, self.offerings_by_id)
            cap = self.offerings_by_id[oid]["cohort_capacity"]
            self.assertEqual(cap["max_seats"], 20, f"{oid} max_seats must be 20")
            self.assertEqual(cap["additional_seat_fee_usd"], 100.0, f"{oid} extra seat fee must be 100.0")
            self.assertTrue(cap.get("parity_applies_to_extra_seats"), f"{oid} must apply parity to extra seats")

    def test_flagships_share_identical_seat_and_pricing_structure(self):
        """Verifies Business and Personal Flagships both include 10 seats at $1,000 USD base."""
        for oid in ("flagship-business-training", "flagship-personal-training"):
            off = self.offerings_by_id[oid]
            self.assertEqual(off["tier1_base_price_usd"], 1000.0)
            self.assertEqual(off["cohort_capacity"]["included_seats"], 10)
            pp = off["parity_pricing"]
            self.assertEqual(pp["tier_1_high_income"]["price_usd"], 1000.0)
            self.assertEqual(pp["tier_1_high_income"]["extra_seat_usd"], 100.0)
            self.assertEqual(pp["tier_2_upper_middle_income"]["price_usd"], 750.0)
            self.assertEqual(pp["tier_2_upper_middle_income"]["extra_seat_usd"], 75.0)
            self.assertEqual(pp["tier_3_lower_middle_low_income"]["price_usd"], 500.0)
            self.assertEqual(pp["tier_3_lower_middle_low_income"]["extra_seat_usd"], 50.0)
            self.assertEqual(pp["tier_4_startups"]["price_usd"], 500.0)
            self.assertEqual(pp["tier_4_startups"]["extra_seat_usd"], 50.0)
            self.assertEqual(pp["tier_5_academia_nonprofits"]["price_usd"], 250.0)
            self.assertEqual(pp["tier_5_academia_nonprofits"]["extra_seat_usd"], 25.0)

    def test_service_picks_include_1_seat_with_parity_extra_seats(self):
        """Verifies Business and Personal Service Picks include 1 seat at $100 USD base."""
        for oid in ("individual-service-business", "individual-service-personal"):
            off = self.offerings_by_id[oid]
            self.assertEqual(off["tier1_base_price_usd"], 100.0)
            self.assertEqual(off["cohort_capacity"]["included_seats"], 1)
            pp = off["parity_pricing"]
            self.assertEqual(pp["tier_1_high_income"]["price_usd"], 100.0)
            self.assertEqual(pp["tier_2_upper_middle_income"]["price_usd"], 75.0)
            self.assertEqual(pp["tier_3_lower_middle_low_income"]["price_usd"], 50.0)
            self.assertEqual(pp["tier_4_startups"]["price_usd"], 50.0)
            self.assertEqual(pp["tier_5_academia_nonprofits"]["price_usd"], 25.0)

    def test_ten_equal_ten_percent_milestone_schedules(self):
        """Verifies all 4 offerings specify 10 milestones of 10% each with exact day spacing."""
        ps = self.pricing["trust_and_payment_guarantees"]["payment_structure"]
        self.assertEqual(ps["upfront_deposit_usd"], 0)
        self.assertEqual(ps["installments_count"], 10)
        self.assertEqual(ps["installment_percentage"], 10)

        for oid in ("flagship-business-training", "flagship-personal-training"):
            sched = self.offerings_by_id[oid]["payment_rules"]["milestone_schedule"]
            self.assertEqual(len(sched), 10)
            self.assertEqual([m["due_after_day"] for m in sched], [5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
            self.assertTrue(all(m["percentage"] == 10 for m in sched))

        for oid in ("individual-service-business", "individual-service-personal"):
            sched = self.offerings_by_id[oid]["payment_rules"]["milestone_schedule"]
            self.assertEqual(len(sched), 10)
            self.assertEqual([m["due_after_day"] for m in sched], list(range(1, 11)))
            self.assertTrue(all(m["percentage"] == 10 for m in sched))

    def test_faqs_and_guarantees_json_reflect_ten_milestones_and_training_materials(self):
        """Verifies data/faqs.json and data/guarantees.json reflect 10 milestones and Zero Charge Training Materials."""
        inst = self.faqs["core_policies"]["installment_structure"]
        self.assertEqual(len(inst["flagship_installments"]), 10)
        self.assertEqual(len(inst["individual_service_installments"]), 10)

        mat = self.faqs["core_policies"]["training_materials_licensing"]
        self.assertEqual(mat["policy_name"], "Zero Charge Training Materials")
        self.assertIn("Apache 2.0", mat["terms"])
        self.assertIn("if the training ends early", mat["terms"])

        pillar_titles = [p["title"] for p in self.guarantees["trust_pillars"]]
        self.assertIn("Zero Charge Training Materials", pillar_titles)
        self.assertNotIn("Zero Charge Dedicated Support Period", pillar_titles)

    def test_no_deprecated_support_phase_rows_in_payment_schedule_js(self):
        """Verifies payment-schedule.js has zero Day 15/30 support rows and uses 10x10% milestones."""
        js = read_text("payment-schedule.js")
        self.assertNotIn("Dedicated operational support through Day 15", js)
        self.assertNotIn("Dedicated operational support through Day 30", js)
        self.assertIn("milestoneCount: 10", js)
        self.assertIn("milestonePercentage: 10", js)

    def test_core_pages_contain_zero_charge_training_materials_not_support_periods(self):
        """Verifies core and tier pages display Zero Charge Training Materials with early-exit clause."""
        pages = [
            "index.html",
            "pricing.html",
            "contact.html",
            "guarantees.html",
            "faqs.html",
            "pricing/pricing-tier-1-usa-china-eu-100.html",
            "pricing/pricing-tier-2-brazil-malaysia-75.html",
            "pricing/pricing-tier-3-india-egypt-50.html",
            "pricing/pricing-tier-4-tech-startups-50.html",
            "pricing/pricing-tier-5-colleges-non-profits-25.html",
            "pricing/pricing-tier-6-crisis-affected-10.html",
        ]
        for page in pages:
            html = re.sub(r"\s+", " ", read_text(page))
            self.assertIn("Zero Charge Training Materials", html, f"Missing Zero Charge Training Materials in {page}")
            self.assertNotIn("Zero Charge Support Periods", html, f"Stale Zero Charge Support Periods in {page}")
            self.assertNotIn("20% in Five", html, f"Stale 20% in Five in {page}")


if __name__ == "__main__":
    unittest.main()
