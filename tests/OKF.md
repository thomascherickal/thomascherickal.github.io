---
type: module
title: "Unit Test Suite & Verification Harness"
description: "Comprehensive unit test suite covering expert synchronizer rules, pricing calculations, link integrity, and automation tools."
tags: [tests, unittest, verification, qa, test-suite]
timestamp: 2026-11-10T08:43:02Z
---

# Unit Test Suite & Verification Harness

Comprehensive unit test suite covering expert synchronizer rules, pricing calculations, link integrity, and automation tools.

## Directory Role & Agent Guidelines

- Execute all unit tests with `python3 python/run_tests.py` or `python3 -m unittest discover -s tests -v`.
- Pre-test and post-test cryptographic SHA-256 validation guarantees that `data/*.json` is never mutated by test runs.
- Always verify a 100% test pass rate across all test cases before committing.
- Pre-commit gate strictly blocks commits on any test failure.

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
