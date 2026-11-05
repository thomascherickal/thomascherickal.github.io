#!/usr/bin/env python3
"""
fuzzy_engine.py - Pure-Python Fuzzy Logic Engine for Text & Semantic Matching.

Implements fuzzy sets, membership functions, linguistic variables,
fuzzy string distances, and defuzzification for confidence evaluation.

Strict Repo Compliance:
- Zero emojis.
- Zero em-dashes (spaced hyphens ' - ' or plain hyphens '-' used).
"""

import math
import re

def fuzzy_triangle(x, a, b, c):
    """Triangular fuzzy membership function."""
    if x <= a or x >= c:
        return 0.0
    if a < x <= b:
        return (x - a) / (b - a) if b > a else 1.0
    if b < x < c:
        return (c - x) / (c - b) if c > b else 1.0
    return 0.0

def fuzzy_trapezoid(x, a, b, c, d):
    """Trapezoidal fuzzy membership function."""
    if x <= a or x >= d:
        return 0.0
    if a < x <= b:
        return (x - a) / (b - a) if b > a else 1.0
    if b < x <= c:
        return 1.0
    if c < x < d:
        return (d - x) / (d - c) if d > c else 1.0
    return 0.0

def levenshtein_distance(s1, s2):
    """Computes exact edit distance between two strings."""
    if s1 == s2:
        return 0
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def levenshtein_ratio(s1, s2):
    """Computes normalized Levenshtein similarity in range [0.0, 1.0]."""
    if s1 == s2:
        return 1.0
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    dist = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    return max(0.0, 1.0 - (dist / max_len))

def clean_tokens(text):
    """Extracts alphanumeric lowercase tokens."""
    return re.findall(r'[a-z0-9]+', text.lower())

def token_sort_ratio(s1, s2):
    """Computes token sort ratio: sorts tokens before computing similarity."""
    if s1 == s2:
        return 1.0
    t1 = " ".join(sorted(clean_tokens(s1)))
    t2 = " ".join(sorted(clean_tokens(s2)))
    if t1 == t2:
        return 1.0
    return levenshtein_ratio(t1, t2)

def token_set_ratio(s1, s2):
    """Computes token set ratio: compares intersection and unique remainders."""
    if s1 == s2:
        return 1.0
    set1 = set(clean_tokens(s1))
    set2 = set(clean_tokens(s2))
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0

    intersection = set1.intersection(set2)
    diff1_to_2 = set1.difference(set2)
    diff2_to_1 = set2.difference(set1)

    t0 = " ".join(sorted(intersection))
    t1 = " ".join(sorted(intersection.union(diff1_to_2)))
    t2 = " ".join(sorted(intersection.union(diff2_to_1)))

    score1 = levenshtein_ratio(t0, t1)
    score2 = levenshtein_ratio(t0, t2)
    score3 = levenshtein_ratio(t1, t2)
    return max(score1, score2, score3)

def ngram_similarity(s1, s2, n=3):
    """Computes character n-gram Jaccard overlap similarity in [0.0, 1.0]."""
    s1_clean = re.sub(r'\s+', ' ', s1.lower()).strip()
    s2_clean = re.sub(r'\s+', ' ', s2.lower()).strip()
    if s1_clean == s2_clean:
        return 1.0
    if len(s1_clean) < n or len(s2_clean) < n:
        return levenshtein_ratio(s1_clean, s2_clean)

    ngrams1 = set([s1_clean[i:i+n] for i in range(len(s1_clean) - n + 1)])
    ngrams2 = set([s2_clean[i:i+n] for i in range(len(s2_clean) - n + 1)])

    union = ngrams1.union(ngrams2)
    if not union:
        return 1.0
    return len(ngrams1.intersection(ngrams2)) / len(union)

class FuzzyMatcher:
    """Evaluates fuzzy match degrees and defuzzifies confidence scores."""

    def __init__(self, high_threshold=0.85, medium_threshold=0.70):
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold

    def compute_membership(self, candidate, target):
        """Calculates combined fuzzy membership grade in [0.0, 1.0]."""
        if not candidate or not target:
            return 0.0
        c_clean = candidate.strip()
        t_clean = target.strip()
        if c_clean.lower() == t_clean.lower():
            return 1.0

        ts_ratio = token_sort_ratio(c_clean, t_clean)
        tset_ratio = token_set_ratio(c_clean, t_clean)
        ng_ratio = ngram_similarity(c_clean, t_clean, n=3)

        # Weighted fuzzy aggregation
        fuzzy_score = (ts_ratio * 0.40) + (tset_ratio * 0.40) + (ng_ratio * 0.20)
        return round(fuzzy_score, 4)

    def classify_confidence(self, score):
        """Linguistic classification of membership score."""
        if score >= 0.98:
            return "EXACT"
        if score >= self.high_threshold:
            return "HIGH"
        if score >= self.medium_threshold:
            return "MEDIUM"
        if score >= 0.40:
            return "LOW"
        return "NONE"

    def match(self, candidate, target, min_score=None):
        """Returns (is_match, score, linguistic_class)."""
        threshold = min_score if min_score is not None else self.high_threshold
        score = self.compute_membership(candidate, target)
        conf = self.classify_confidence(score)
        return (score >= threshold, score, conf)

fuzzy_matcher = FuzzyMatcher()

