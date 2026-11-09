#!/usr/bin/env python3
"""
test_rules_and_pre_push_coverage.py - Coverage tests for rarely-hit rule
branches in nlp-fuzzy-expert/rules.py and for python/generate_pre_push_summary.py.

All tests operate on in-memory strings or temporary sandboxes; no repository
files (including data/*.json) are modified.

Strict Repo Compliance:
- Zero emojis.
- Zero em-dashes (spaced hyphens ' - ' or plain hyphens '-' used).
"""

import io
import os
import sys
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(TESTS_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "nlp-fuzzy-expert"))
sys.path.insert(0, os.path.join(REPO_DIR, "python"))

import rules as rules_mod  # noqa: E402
import generate_pre_push_summary as pps  # noqa: E402


class FakeKB:
    """Minimal knowledge base stub exposing get_fact and raw_data."""

    def __init__(self, facts=None, raw=None):
        self.facts = facts or {}
        self.raw_data = raw or {}

    def get_fact(self, domain, predicate, default=None):
        return self.facts.get((domain, predicate), default)


def get_rule(name):
    for r in rules_mod.build_rules():
        if r.name == name:
            return r
    raise AssertionError(f"Rule {name} not found")


TIERS = [
    {"tier": 1, "name": "Tier One", "multiplier": 1.0, "discountPercent": 0},
    {"tier": 2, "name": "Tier Two", "multiplier": 0.75, "discountPercent": 25},
]


class TestRulesBranches(unittest.TestCase):
    """Exercises the less common branches in rules.py."""

    def test_pricing_section5_cards(self):
        """Verifies PricingSection5CardsRule rewrites all rate box fields."""
        rule = get_rule("PricingSection5CardsRule")
        content = (
            '<div class="service-rate-box">'
            '<span class="service-rate-val svc-indiv-price" data-base="1">$1 USD</span>'
            '<div class="service-rate-tiers svc-indiv-tiers">old</div>'
            'Business (<=3 seats):'
            '<span class="service-rate-val svc-biz-price" data-base="2">$2 USD</span>'
            '<div class="service-rate-tiers svc-biz-tiers">old</div>'
            '<div class="service-coverage-label">1 Days Training + 1 Days Support</div>'
            '</div>'
        )
        kb = FakeKB()
        ok, info = rule.evaluate("x/pricing.html", content, kb, None, None)
        self.assertTrue(ok)
        out, occ = rule.action_fn("x/pricing.html", content, info, kb, None, None)
        self.assertIn('data-base="200">$200 USD', out)
        self.assertIn("T2: $150", out)
        self.assertIn("5 Days Training + 5 Days Support", out)
        self.assertEqual(len(occ), 1)

    def test_business_services_pricing_cards(self):
        """Verifies BusinessServices rule updates Section 5 category and description on pricing.html."""
        rule = None
        for r in rules_mod.build_rules():
            if r.action_fn.__name__ == "act_business_svcs":
                rule = r
        self.assertIsNotNone(rule)
        raw = {"business_services": {"services": [{
            "number": "01", "slug": "svc-a.html", "name": "New Name",
            "description": "New Desc", "role_chip": "Chip", "category": "New Cat"
        }]}}
        content = (
            "SERVICE 01 · Old Cat\n"
            '<h3><a href="business-services/svc-a.html" class="t">Old Name</a></h3>'
            "<p>Old Desc</p>"
        )
        out, _ = rule.action_fn("x/pricing.html", content, {}, FakeKB(raw=raw), None, None)
        self.assertIn("SERVICE 01 · New Cat", out)
        self.assertIn(">New Name</a>", out)
        self.assertIn("<p>New Desc</p>", out)

    def test_tiers_calculator_tier_files_and_payment_schedule(self):
        """Verifies TiersParityMatrixRule handles tier calculator and payment-schedule.js files."""
        rule = get_rule("TiersParityMatrixRule")
        kb = FakeKB(raw={"tiers": {"tiers": TIERS}})

        calc = "const TIER = { id: 2, multiplier: 0.5, discountPercent: 50 };"
        path = "/r/pricing-calculator/pricing-calculator-tier-2.js"
        ok, info = rule.evaluate(path, calc, kb, None, None)
        self.assertTrue(ok)
        out, occ = rule.action_fn(path, calc, info, kb, None, None)
        self.assertIn("multiplier: 0.75", out)
        self.assertIn("discountPercent: 25", out)
        self.assertEqual(len(occ), 1)

        sched = "1: { multiplier: 0.9, discount: 9 },\n2: { multiplier: 0.1, discount: 1 }"
        out2, occ2 = rule.action_fn("/r/payment-schedule.js", sched, {}, kb, None, None)
        self.assertIn("1: { multiplier: 1.0, discount: 0 }", out2)
        self.assertIn("2: { multiplier: 0.75, discount: 25 }", out2)
        self.assertEqual(len(occ2), 2)


SAMPLE_REPORT = """# Report

## Python Unit Test Suite Details (`tests/`)

| Test Module | Total Tests | Passed | Failed | Errors | Skipped | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `test_a` | 2 | 2 | 0 | 0 | 0 | PASSED |
| **Total (1 modules)** | **2** | **2** | **0** | **0** | **0** | **PASSED** |

### Coverage Executive Summary

| Language | Tracked Files | Total | Covered | Missed | Coverage % | Status |
|:---|---:|---:|---:|---:|---:|:---|
| Python (`x`) | 2 | 20 | 15 | 5 | **75.00%** | WARN |
| JavaScript (Root scripts) | 1 | 10 | 10 | 0 | **100.00%** | PASS |
| **Repository Combined Total** | **3** | **30** | **25** | **5** | **83.33%** | **WARN** |

### Python Test Coverage

| File | Statements | Covered | Missed | Coverage % | Missing Lines |
|:---|---:|---:|---:|---:|:---|
| `nlp-fuzzy-expert/a.py` | 10 | 10 | 0 | **100.0%** | `-` |
| `python/b.py` | 10 | 5 | 5 | **50.0%** | `1-5` |

### JavaScript Test Coverage

| File | Executable Lines | Covered | Missed | Coverage % | Missing Lines |
|:---|---:|---:|---:|---:|:---|
| `components.js` | 5 | 5 | 0 | **100.0%** | `-` |
| `pricing-calculator.js` | 5 | 5 | 0 | **100.0%** | `-` |
| `other.js` | 5 | 5 | 0 | **100.0%** | `-` |

"""


class TestPrePushSummary(unittest.TestCase):
    """Exercises generate_pre_push_summary.py in a temporary sandbox."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pps_")
        self.orig = (pps.LAST_COMMIT_REPORT, pps.JS_CACHE, pps.SYNC_MANIFEST)
        pps.LAST_COMMIT_REPORT = os.path.join(self.tmp, "report.md")
        pps.JS_CACHE = os.path.join(self.tmp, "js.json")
        pps.SYNC_MANIFEST = os.path.join(self.tmp, "manifest.json")

    def tearDown(self):
        pps.LAST_COMMIT_REPORT, pps.JS_CACHE, pps.SYNC_MANIFEST = self.orig
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = pps.main()
        self.assertEqual(rc, 0)
        return buf.getvalue()

    def test_missing_files_fallbacks(self):
        """Verifies defaults are used when report, cache, and manifest are absent."""
        self.assertIsNone(pps.parse_last_commit_report())
        out = self._run()
        self.assertIn("WEBSITE SYNCHRONIZATION SUMMARY REPORT", out)
        self.assertIn("Python Total", out)
        self.assertIn("JS Total", out)

    def test_unreadable_and_corrupt_inputs(self):
        """Verifies unreadable report and corrupt JSON inputs are handled gracefully."""
        os.makedirs(pps.LAST_COMMIT_REPORT)  # a directory cannot be opened as a file
        self.assertIsNone(pps.parse_last_commit_report())
        for p in (pps.JS_CACHE, pps.SYNC_MANIFEST):
            with open(p, "w", encoding="utf-8") as f:
                f.write("{not json")
        out = self._run()
        self.assertIn("Summary: 80 target files evaluated", out)

    def test_full_report_parsing_and_output(self):
        """Verifies parsing of all report tables and formatted output with real data."""
        with open(pps.LAST_COMMIT_REPORT, "w", encoding="utf-8") as f:
            f.write(SAMPLE_REPORT)
        with open(pps.JS_CACHE, "w", encoding="utf-8") as f:
            json.dump({"total": 7, "passed": 7, "failed": 0, "sections": {"a": {}, "b": {}}}, f)
        with open(pps.SYNC_MANIFEST, "w", encoding="utf-8") as f:
            json.dump({"target_documents": ["a", "b", "c"]}, f)

        data = pps.parse_last_commit_report()
        self.assertEqual(len(data["py_tests"]), 1)
        self.assertEqual(data["py_tests_total"]["tests"], "2")
        self.assertEqual(data["py_coverage_summary"]["percent"], "75.00%")
        self.assertEqual(data["js_coverage_summary"]["status"], "PASS")
        self.assertEqual(data["combined_coverage"]["files"], "3")
        self.assertEqual(len(data["py_coverage"]), 2)
        self.assertEqual(len(data["js_coverage"]), 3)

        out = self._run()
        self.assertIn("Summary: 3 target files evaluated", out)
        self.assertIn("7 tests", out)
        self.assertIn("2 sections", out)
        self.assertIn("50.0%", out)
        self.assertIn("WARN", out)
        for line in out.splitlines():
            self.assertLessEqual(len(line), 100)

    def test_zero_statement_groups(self):
        """Verifies zero-statement categories and zero totals do not divide by zero."""
        report = SAMPLE_REPORT.replace("| `nlp-fuzzy-expert/a.py` | 10 | 10 | 0 |", "| `nlp-fuzzy-expert/a.py` | 0 | 0 | 0 |")
        report = report.replace("| `python/b.py` | 10 | 5 | 5 |", "| `python/b.py` | 0 | 0 | 0 |")
        with open(pps.LAST_COMMIT_REPORT, "w", encoding="utf-8") as f:
            f.write(report)
        out = self._run()
        self.assertIn("100.0%", out)


if __name__ == "__main__":
    unittest.main()

