---
type: module
title: "Unit Test Suite & Verification Harness"
description: "Comprehensive unit tests covering synchronizer rules, pricing logic, link integrity, and tools."
tags: [tests, unittest, verification, qa]
timestamp: 2026-11-09T20:35:37Z
---

# Unit Test Suite & Verification Harness

Comprehensive unit tests covering synchronizer rules, pricing logic, link integrity, and tools.

## Directory Role & Agent Guidelines

- Comprehensive test verification suite.
- Run Python tests with `python3 python/run_tests.py` or `python3 -m unittest discover -s tests -v`.
- Pre-commit enforces 100% test pass rate across all suites.

## Subdirectories

- [`js/`](js/OKF.md): Subdirectory knowledge bundle

## File Inventory

- [`.js_test_reports_cache.json`](.js_test_reports_cache.json)
- [`.test_reports_cache.json`](.test_reports_cache.json)
- [`README.md`](README.md)
- [`__init__.py`](__init__.py)
- [`test_auto_healer.py`](test_auto_healer.py)
- [`test_check_links.py`](test_check_links.py)
- [`test_code_test_syncer.py`](test_code_test_syncer.py)
- [`test_continuous_updates.py`](test_continuous_updates.py)
- [`test_core_target_documents.py`](test_core_target_documents.py)
- [`test_coverage_report.py`](test_coverage_report.py)
- [`test_expert_system.py`](test_expert_system.py)
- [`test_fuzzy_engine.py`](test_fuzzy_engine.py)
- [`test_last_commit_report.py`](test_last_commit_report.py)
- [`test_master_orchestrator.py`](test_master_orchestrator.py)
- [`test_nlp_engine.py`](test_nlp_engine.py)
- [`test_python_tools.py`](test_python_tools.py)
- [`test_reference_scanner.py`](test_reference_scanner.py)
- [`test_report_generator.py`](test_report_generator.py)
- [`test_rules_and_pre_push_coverage.py`](test_rules_and_pre_push_coverage.py)
- [`test_sync_business_services.py`](test_sync_business_services.py)
- [`test_sync_claude_and_gemini.py`](test_sync_claude_and_gemini.py)
- [`test_sync_guarantees.py`](test_sync_guarantees.py)
- [`test_sync_personal_services.py`](test_sync_personal_services.py)
- [`test_sync_pricing.py`](test_sync_pricing.py)
- [`test_sync_pricing_tiers_and_calculators.py`](test_sync_pricing_tiers_and_calculators.py)
- [`test_sync_publications_config_faqs.py`](test_sync_publications_config_faqs.py)
- [`test_sync_tiers.py`](test_sync_tiers.py)
- [`test_unidirectional_sync.py`](test_unidirectional_sync.py)
- [`test_watcher.py`](test_watcher.py)

## Context & Cross-References

- Root Directory: [`/OKF.md`](../OKF.md)
- Architectural Specification: [`GEMINI.md`](/GEMINI.md)
- Repository Telemetry: [`METRICS.md`](/METRICS.md)
