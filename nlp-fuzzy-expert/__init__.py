"""
nlp-fuzzy-expert package initialization.
Strict Repo Compliance: Zero emojis, zero em-dashes.
"""

from .expert_system import KnowledgeBase, InferenceEngine, Fact, Rule, load_json, save_json
from .fuzzy_engine import FuzzyMatcher, fuzzy_matcher
from .nlp_engine import NLPEngine, nlp_engine, apply_sync_highlight, strip_sync_highlights
from .report_generator import (
    DOMAIN_FILES,
    DOMAIN_DATA_SOURCE,
    classify_rule_field,
    write_reference_files,
    generate_markdown_reports,
)
from .auto_healer import (
    compute_data_fingerprint,
    verify_data_untouched,
    run_unit_tests,
    handle_test_verification_and_commit,
    generate_failed_tests_report,
)
from .synchronizer import ExpertSynchronizer

__all__ = [
    "KnowledgeBase",
    "InferenceEngine",
    "Fact",
    "Rule",
    "load_json",
    "save_json",
    "FuzzyMatcher",
    "fuzzy_matcher",
    "NLPEngine",
    "nlp_engine",
    "apply_sync_highlight",
    "strip_sync_highlights",
    "DOMAIN_FILES",
    "DOMAIN_DATA_SOURCE",
    "classify_rule_field",
    "write_reference_files",
    "generate_markdown_reports",
    "compute_data_fingerprint",
    "verify_data_untouched",
    "run_unit_tests",
    "handle_test_verification_and_commit",
    "generate_failed_tests_report",
    "ExpertSynchronizer",
]

