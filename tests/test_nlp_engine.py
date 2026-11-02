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

    def test_highlight_tagging_and_stripping(self):
        """Verifies highlight tagging and idempotent clean stripping."""
        val = "$2,000 USD"
        highlighted = apply_sync_highlight(val)
        self.assertEqual(highlighted, '<span class="sync-highlight">$2,000 USD</span>')
        stripped = strip_sync_highlights(highlighted)
        self.assertEqual(stripped, val)

if __name__ == "__main__":
    unittest.main()

