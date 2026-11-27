#!/usr/bin/env python3
"""Unit tests for python/ssot (load, format, validate, render idempotency)."""

import os
import tempfile
import unittest

from python.ssot.load import SSOTData
from python.ssot.format import (
    money,
    tier_price_list,
    seats_phrase,
    duration_phrase,
    installment_phrase,
    reply_sla_phrase,
    quarterly_cap_phrase,
    call_formatter,
)
from python.ssot.validate import validate_all
from python.ssot.render import render_file
from jinja2 import Environment, FileSystemLoader


class TestSSOTSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = SSOTData()

    def test_load_dotted_paths(self):
        self.assertEqual(self.data.get("offerings.flagship-personal-training.cohort_capacity.max_seats"), 20)
        self.assertEqual(self.data.get("offerings.individual-service-business.cohort_capacity.included_seats"), 10)
        self.assertEqual(self.data.get("tiers.1.multiplier"), 1.0)
        with self.assertRaises(KeyError):
            self.data.get("offerings.nonexistent.field")

    def test_format_money(self):
        self.assertEqual(money(1000), "$1,000")
        self.assertEqual(money(100), "$100")
        self.assertEqual(money(250.0), "$250")

    def test_format_tier_price_list(self):
        self.assertEqual(
            tier_price_list(self.data, "flagship-business-training"),
            "$1,000 / $750 / $500 / $500 / $250 / Custom",
        )
        self.assertEqual(
            tier_price_list(self.data, "individual-service-business"),
            "$100 / $75 / $50 / $50 / $25 / Custom",
        )

    def test_format_seats_and_duration_and_installments(self):
        self.assertEqual(
            seats_phrase(self.data, "flagship-personal-training"),
            "up to 10 participants (+$100 per extra seat, max 20)",
        )
        self.assertEqual(
            duration_phrase(self.data, "flagship-business-training"),
            "2 calendar months (includes weekends as holidays)",
        )
        self.assertEqual(
            installment_phrase(self.data, "flagship-business-training"),
            "10 equal installments of 10%",
        )
        self.assertEqual(reply_sla_phrase(self.data), "within one week")
        self.assertEqual(quarterly_cap_phrase(self.data), "2 flagship cohorts per quarter")

    def test_call_formatter(self):
        self.assertEqual(call_formatter(self.data, "money(1000)"), "$1,000")
        self.assertEqual(call_formatter(self.data, "reply_sla_phrase()"), "within one week")
        with self.assertRaises(KeyError):
            call_formatter(self.data, "unknown_fn()")

    def test_validate_all_passes(self):
        self.assertEqual(validate_all(), [])

    def test_render_idempotency(self):
        with tempfile.TemporaryDirectory() as tmp:
            tpl_dir = os.path.join(tmp, "templates")
            os.makedirs(tpl_dir)
            env = Environment(loader=FileSystemLoader(tpl_dir), autoescape=False)
            html_path = os.path.join(tmp, "sample.html")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(
                    "<p><!--ssot:fmt.tier_price_list(flagship-business-training)-->old<!--/ssot--></p>"
                    "<p><!--ssot:offerings.flagship-personal-training.cohort_capacity.max_seats-->0<!--/ssot--></p>"
                )
            self.assertTrue(render_file(html_path, self.data, env, check_only=False))
            self.assertFalse(render_file(html_path, self.data, env, check_only=False))
            self.assertFalse(render_file(html_path, self.data, env, check_only=True))

    def test_acceptance_invariants(self):
        for oid, off in self.data.offerings.items():
            self.assertEqual(off["cohort_capacity"]["included_seats"], 10, f"{oid}: included_seats must be 10")
            self.assertEqual(off["cohort_capacity"]["max_seats"], 20, f"{oid}: max_seats must be 20")
            self.assertEqual(len(off["payment_rules"]["milestone_schedule"]), 10, f"{oid}: expected 10 milestones")
        tier1 = self.data.tiers["1"]
        self.assertNotIn("China", tier1.get("examples", []))
        self.assertTrue(self.data.raw["tiers"].get("precedence_rule"))


if __name__ == "__main__":
    unittest.main()
