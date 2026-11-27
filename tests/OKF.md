---
type: module
title: "Unit Test Suite & Verification Harness"
description: "Comprehensive unit test suite covering expert synchronizer rules, pricing calculations, link integrity, and automation tools."
tags: [tests, unittest, verification, qa, test-suite]
timestamp: 2026-11-27T06:35:11Z
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

- [`.html_ui_test_reports_cache.json`](.html_ui_test_reports_cache.json)
- [`.js_test_reports_cache.json`](.js_test_reports_cache.json)
- [`.test_reports_cache.json`](.test_reports_cache.json)
- [`README.md`](README.md)
- [`__init__.py`](__init__.py)
- [`test_check_links.py`](test_check_links.py)
- [`test_coverage_report.py`](test_coverage_report.py)
- [`test_helpers.py`](test_helpers.py)
- [`test_html_ui.py`](test_html_ui.py)
- [`test_last_commit_report.py`](test_last_commit_report.py)
- [`test_pricing_and_policy_invariants.py`](test_pricing_and_policy_invariants.py)
- [`test_python_tools.py`](test_python_tools.py)
- [`test_ssot.py`](test_ssot.py)

## Context & Cross-References

- Root Directory: [`/OKF.md`](../OKF.md)
- Architectural Specification: [`GEMINI.md`](/GEMINI.md)
- Repository Telemetry: [`METRICS.md`](/METRICS.md)
