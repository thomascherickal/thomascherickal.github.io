#!/usr/bin/env python3
"""
test_nlp_engine.py - Unit Tests for Natural Language Processing Engine.

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
from nlp_engine import (
    nlp_engine,
    apply_sync_highlight,
    strip_sync_highlights
)

class TestNLPEngine(unittest.TestCase):
    """Unit tests for NLP text normalization, NER, and context analysis."""

    def test_text_normalization_and_tokenization(self):
        """Verifies HTML unescaping and whitespace normalization."""
        raw = "Personal&nbsp;Training &amp; 30d   Live"
        norm = nlp_engine.normalize_text(raw)
        self.assertIn("Personal Training & 30d Live", norm)
        tokens = nlp_engine.tokenize_words(norm)
        self.assertIn("personal", tokens)
        self.assertIn("training", tokens)

    def test_entity_extraction(self):
        """Verifies semantic entity recognition for MONEY, DURATION, and SEATS."""
        text = "Business Training costs $2,000 USD covering up to 10 participants for 60 days."
        entities = nlp_engine.find_entities(text)
        types = [e.entity_type for e in entities]

        self.assertIn("MONEY", types)
        self.assertIn("DURATION", types)
        self.assertIn("SEATS", types)

        money_e = next(e for e in entities if e.entity_type == "MONEY")
        self.assertEqual(money_e.normalized_value, 2000.0)

        seats_e = next(e for e in entities if e.entity_type == "SEATS")
        self.assertEqual(seats_e.normalized_value, 10)

        dur_e = next(e for e in entities if e.entity_type == "DURATION")
        self.assertEqual(dur_e.normalized_value, 60)

    def test_domain_classification(self):
        """Verifies context snippet classification into semantic domains."""
        ctx_biz = "Business organization workforce executive training program"
        ctx_pers = "Personal 1-on-1 private acceleration career coaching"
        ctx_calc = "Interactive calculator select options and parity discount"

        self.assertEqual(nlp_engine.classify_domain_context(ctx_biz), "business_flagship")
        self.assertEqual(nlp_engine.classify_domain_context(ctx_pers), "personal_flagship")
        self.assertEqual(nlp_engine.classify_domain_context(ctx_calc), "calculator")
        self.assertEqual(nlp_engine.classify_domain_context("pick individual service"), "individual_service")
        self.assertEqual(nlp_engine.classify_domain_context("walkaway guarantee installment"), "general_policy")

        ent = nlp_engine.find_entities("$300 USD")[0]
        self.assertIn("MONEY", repr(ent))

    def test_highlight_tagging_and_stripping(self):
        """Verifies highlight tagging and idempotent clean stripping."""
        val = "$2,000 USD"
        highlighted = apply_sync_highlight(val)
        self.assertEqual(highlighted, '<span class="sync-highlight">$2,000 USD</span>')
        stripped = strip_sync_highlights(highlighted)
        self.assertEqual(stripped, val)

    def test_nlp_edge_cases(self):
        """Verifies edge cases for empty strings, punctuation only, and missing numbers."""
        # Empty string
        self.assertEqual(nlp_engine.normalize_text(""), "")
        self.assertEqual(nlp_engine.tokenize_words(""), [])
        self.assertEqual(nlp_engine.find_entities(""), [])
        self.assertEqual(nlp_engine.classify_domain_context(""), "unknown")

        # Punctuation and symbols only
        punct = "!@#$%^&*()_+{}[]:;<>?,./"
        self.assertEqual(nlp_engine.find_entities(punct), [])

        # Multiple currencies and decimal formats
        money_text = "Standard rate is $1,500.50 USD with a deposit of $0 upfront and $200.00 extra."
        entities = nlp_engine.find_entities(money_text)
        money_ents = [e for e in entities if e.entity_type == "MONEY"]
        self.assertEqual(len(money_ents), 3)
        self.assertEqual(money_ents[0].normalized_value, 1500.50)
        self.assertEqual(money_ents[1].normalized_value, 0.0)
        self.assertEqual(money_ents[2].normalized_value, 200.0)

    def test_duration_and_seats_edge_cases(self):
        """Verifies parsing of various duration and seat expressions."""
        expr1 = "Total timeline 70 calendar days with 30 days live instruction and 30 days guidance."
        ents1 = nlp_engine.find_entities(expr1)
        dur_ents = [e for e in ents1 if e.entity_type == "DURATION"]
        dur_vals = [e.normalized_value for e in dur_ents]
        self.assertIn(70, dur_vals)
        self.assertIn(30, dur_vals)

        expr2 = "Designed for up to 25 participants and maximum 50 seats."
        ents2 = nlp_engine.find_entities(expr2)
        seat_ents = [e for e in ents2 if e.entity_type == "SEATS"]
        seat_vals = [e.normalized_value for e in seat_ents]
        self.assertIn(25, seat_vals)
        self.assertIn(50, seat_vals)

if __name__ == "__main__":
    unittest.main()

