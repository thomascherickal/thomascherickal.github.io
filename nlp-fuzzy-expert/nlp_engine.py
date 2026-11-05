#!/usr/bin/env python3
"""
nlp_engine.py - Natural Language Processing & Semantic Entity Extraction.

Tokenizes HTML text, extracts semantic entity spans (money, duration, seats),
and constructs context windows for semantic domain classification.

Strict Repo Compliance:
- Zero emojis.
- Zero em-dashes (spaced hyphens ' - ' or plain hyphens '-' used).
"""

import re
import html

# Canonical highlight wrappers
HIGHLIGHT_OPEN = '<span class="sync-highlight">'
HIGHLIGHT_CLOSE = '</span>'

def strip_sync_highlights(content):
    """Removes existing sync-highlight spans to ensure clean idempotent matching."""
    if "sync-highlight" not in content:
        return content
    return re.sub(r'<span class="sync-highlight"[^>]*>(.*?)</span>', r'\1', content)

def apply_sync_highlight(val_str):
    """Wraps text in sync-highlight span for CSS-driven IDE visibility."""
    return f'{HIGHLIGHT_OPEN}{val_str}{HIGHLIGHT_CLOSE}'

class NLPEntity:
    """Represents an extracted semantic entity in text."""

    def __init__(self, entity_type, raw_value, normalized_value, start_idx, end_idx, context=""):
        self.entity_type = entity_type
        self.raw_value = raw_value
        self.normalized_value = normalized_value
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.context = context

    def __repr__(self):
        return f"<NLPEntity {self.entity_type}: '{self.raw_value}' -> {self.normalized_value}>"

class NLPEngine:
    """Natural language processor for HTML content and context analysis."""

    def __init__(self):
        # Semantic regex patterns
        self.re_money = re.compile(r'\$[\d,]+(?:\.\d+)?(?:\s*USD)?', re.IGNORECASE)
        self.re_duration = re.compile(r'\b\d+\s*(?:calendar\s*)?(?:working\s*)?days?\b|\b\d+d\s*\+\s*\d+d\b', re.IGNORECASE)
        self.re_seats = re.compile(r'\b\d+\s*(?:participants?|seats?|learners?|heads?)\b|strictly\s*1\s*(?:person|learner|seat)\s*only', re.IGNORECASE)
        self.re_whitespace = re.compile(r'\s+')

    def normalize_text(self, text):
        """Unescapes HTML entities and normalizes multiple spaces."""
        unescaped = html.unescape(text)
        return self.re_whitespace.sub(' ', unescaped).strip()

    def tokenize_words(self, text):
        """Extracts words while preserving lower-case normalization."""
        return re.findall(r'[a-zA-Z0-9_\-]+', text.lower())

    def extract_context_window(self, content, match_pos, window_size=120):
        """Extracts context characters before and after match_pos."""
        start = max(0, match_pos - window_size)
        end = min(len(content), match_pos + window_size)
        return self.normalize_text(content[start:end])

    def find_entities(self, text):
        """Extracts all semantic entities with their context windows."""
        entities = []

        for m in self.re_money.finditer(text):
            val_clean = m.group(0).replace('$', '').replace('USD', '').replace(',', '').strip()
            try:
                numeric_val = float(val_clean)
            except ValueError:
                numeric_val = None
            ctx = self.extract_context_window(text, m.start())
            entities.append(NLPEntity("MONEY", m.group(0), numeric_val, m.start(), m.end(), ctx))

        for m in self.re_duration.finditer(text):
            num_match = re.search(r'\d+', m.group(0))
            numeric_val = int(num_match.group(0)) if num_match else None
            ctx = self.extract_context_window(text, m.start())
            entities.append(NLPEntity("DURATION", m.group(0), numeric_val, m.start(), m.end(), ctx))

        for m in self.re_seats.finditer(text):
            num_match = re.search(r'\d+', m.group(0))
            numeric_val = int(num_match.group(0)) if num_match else 1
            ctx = self.extract_context_window(text, m.start())
            entities.append(NLPEntity("SEATS", m.group(0), numeric_val, m.start(), m.end(), ctx))

        return entities

    def classify_domain_context(self, context_text):
        """Determines semantic domain of a context snippet."""
        ctx_lower = context_text.lower()
        scores = {
            "business_flagship": 0,
            "personal_flagship": 0,
            "individual_service": 0,
            "parity_tiers": 0,
            "calculator": 0,
            "general_policy": 0
        }

        if "business" in ctx_lower or "organization" in ctx_lower or "workforce" in ctx_lower:
            scores["business_flagship"] += 3
        if "personal" in ctx_lower or "1-on-1" in ctx_lower or "individual career" in ctx_lower:
            scores["personal_flagship"] += 3
        if "pick individual" in ctx_lower or "per service" in ctx_lower or "individual service" in ctx_lower:
            scores["individual_service"] += 4
        if "tier" in ctx_lower or "parity" in ctx_lower or "purchasing power" in ctx_lower:
            scores["parity_tiers"] += 3
        if "calc" in ctx_lower or "calculator" in ctx_lower or "select option" in ctx_lower:
            scores["calculator"] += 4
        if "walkaway" in ctx_lower or "installment" in ctx_lower or "guarantee" in ctx_lower:
            scores["general_policy"] += 3

        best_domain = max(scores, key=scores.get)
        return best_domain if scores[best_domain] > 0 else "unknown"

nlp_engine = NLPEngine()

