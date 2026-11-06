#!/usr/bin/env python3
"""
test_fuzzy_engine.py - Unit Tests for Pure-Python Fuzzy Logic Engine.

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
from fuzzy_engine import (
    fuzzy_triangle,
    fuzzy_trapezoid,
    levenshtein_distance,
    levenshtein_ratio,
    token_sort_ratio,
    token_set_ratio,
    ngram_similarity,
    fuzzy_matcher
)

class TestFuzzyEngine(unittest.TestCase):
    """Unit tests for fuzzy logic and approximate string matching engine."""

    def test_membership_functions(self):
        """Verifies triangular and trapezoidal membership evaluations."""
        # Triangle
        self.assertEqual(fuzzy_triangle(0.0, 0.2, 0.5, 0.8), 0.0)
        self.assertEqual(fuzzy_triangle(0.5, 0.2, 0.5, 0.8), 1.0)
        self.assertAlmostEqual(fuzzy_triangle(0.35, 0.2, 0.5, 0.8), 0.5)

        # Trapezoid
        self.assertEqual(fuzzy_trapezoid(0.0, 0.2, 0.4, 0.6, 0.8), 0.0)
        self.assertEqual(fuzzy_trapezoid(0.5, 0.2, 0.4, 0.6, 0.8), 1.0)
        self.assertAlmostEqual(fuzzy_trapezoid(0.7, 0.2, 0.4, 0.6, 0.8), 0.5)

    def test_string_distance_and_ratios(self):
        """Verifies exact Levenshtein distance and normalized similarity ratios."""
        self.assertEqual(levenshtein_distance("flagship", "flagship"), 0)
        self.assertEqual(levenshtein_distance("flagship", "flagships"), 1)
        self.assertEqual(levenshtein_ratio("flagship", "flagship"), 1.0)
        self.assertGreater(levenshtein_ratio("flagship", "flagships"), 0.85)

    def test_token_sort_and_set_ratios(self):
        """Verifies token sort and token set robust fuzzy comparisons."""
        s1 = "Business AI Training Flagship"
        s2 = "Flagship AI Training Business"
        self.assertEqual(token_sort_ratio(s1, s2), 1.0)

        s3 = "Personal Training 1-on-1 private acceleration"
        s4 = "Personal Training"
        self.assertGreater(token_set_ratio(s3, s4), 0.7)

    def test_ngram_similarity(self):
        """Verifies character 3-gram Jaccard overlap similarity."""
        sim_exact = ngram_similarity("Claude Agent Teams", "Claude Agent Teams")
        self.assertEqual(sim_exact, 1.0)
        sim_partial = ngram_similarity("Claude Agent Teams", "Claude Teams")
        self.assertGreater(sim_partial, 0.4)

    def test_fuzzy_matcher_classification(self):
        """Verifies holistic composite score and linguistic term classification."""
        is_match, score, conf = fuzzy_matcher.match("Business Training Flagship", "Business Training Flagship")
        self.assertTrue(is_match)
        self.assertEqual(conf, "EXACT")
        self.assertEqual(score, 1.0)

        is_match_near, score_near, conf_near = fuzzy_matcher.match("Business Training Flagship", "Business AI Training Flagship")
        self.assertTrue(is_match_near)
        self.assertIn(conf_near, ["HIGH", "EXACT"])
        self.assertGreater(score_near, 0.75)

    def test_fuzzy_edge_cases(self):
        """Verifies edge cases: empty strings, single chars, identical whitespaces, unicode, and extreme mismatch."""
        # Empty string handling
        self.assertEqual(levenshtein_distance("", ""), 0)
        self.assertEqual(levenshtein_distance("abc", ""), 3)
        self.assertEqual(levenshtein_distance("", "xyz"), 3)
        self.assertEqual(levenshtein_ratio("", ""), 1.0)
        self.assertEqual(levenshtein_ratio("abc", ""), 0.0)
        self.assertEqual(ngram_similarity("", ""), 1.0)
        self.assertEqual(ngram_similarity("abc", ""), 0.0)
        self.assertEqual(token_sort_ratio("", ""), 1.0)
        self.assertEqual(token_sort_ratio("abc", ""), 0.0)

        # Single character comparisons
        self.assertEqual(levenshtein_distance("a", "a"), 0)
        self.assertEqual(levenshtein_distance("a", "b"), 1)
        self.assertEqual(levenshtein_ratio("a", "a"), 1.0)
        self.assertEqual(levenshtein_ratio("a", "b"), 0.0)

        # Extreme mismatch
        is_match, score, conf = fuzzy_matcher.match("Enterprise Cloud Kubernetes", "Fresh Apples And Oranges")
        self.assertFalse(is_match)
        self.assertEqual(conf, "NONE")
        self.assertLess(score, 0.4)

        # Case insensitivity and extra whitespace
        is_match_ws, score_ws, conf_ws = fuzzy_matcher.match("  CLAUDE CODE  ", "claude code")
        self.assertTrue(is_match_ws)
        self.assertEqual(conf_ws, "EXACT")
        self.assertEqual(score_ws, 1.0)

    def test_membership_boundary_cases(self):
        """Verifies boundary clamping on triangular and trapezoidal membership functions."""
        # Value far below lower bound
        self.assertEqual(fuzzy_triangle(-10.0, 0.0, 5.0, 10.0), 0.0)
        self.assertEqual(fuzzy_trapezoid(-10.0, 0.0, 2.0, 8.0, 10.0), 0.0)

        # Value far above upper bound
        self.assertEqual(fuzzy_triangle(20.0, 0.0, 5.0, 10.0), 0.0)
        self.assertEqual(fuzzy_trapezoid(20.0, 0.0, 2.0, 8.0, 10.0), 0.0)

        # Exact inflection points
        self.assertEqual(fuzzy_trapezoid(2.0, 0.0, 2.0, 8.0, 10.0), 1.0)
        self.assertEqual(fuzzy_trapezoid(8.0, 0.0, 2.0, 8.0, 10.0), 1.0)
        self.assertEqual(fuzzy_triangle(0.0, 0.0, 5.0, 10.0), 0.0)
        self.assertEqual(fuzzy_triangle(10.0, 0.0, 5.0, 10.0), 0.0)
        self.assertAlmostEqual(fuzzy_triangle(7.5, 0.0, 5.0, 10.0), 0.5)

        # Punctuation-only and medium/low classification
        self.assertEqual(token_set_ratio("same", "same"), 1.0)
        self.assertEqual(token_set_ratio("!!!", "???"), 1.0)
        self.assertEqual(token_set_ratio("abc", "???"), 0.0)
        self.assertEqual(fuzzy_matcher.compute_membership("", "abc"), 0.0)
        self.assertEqual(fuzzy_matcher.classify_confidence(0.75), "MEDIUM")
        self.assertEqual(fuzzy_matcher.classify_confidence(0.50), "LOW")

if __name__ == "__main__":
    unittest.main()
