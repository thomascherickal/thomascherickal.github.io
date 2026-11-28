---
type: module
title: "Unified Test Suite & Verification Harness"
description: "Python unittest modules covering the SSOT build pipeline, pricing/policy invariants, 101 HTML UI checks across all 85+ pages, internal link integrity, build tooling, coverage reporting, and commit reporting."
tags: [tests, unittest, verification, qa, html-ui]
timestamp: 2026-11-28T05:16:04Z
---

# Unified Test Suite & Verification Harness

Python unittest modules covering the SSOT build pipeline, pricing/policy invariants, 101 HTML UI checks across all 85+ pages, internal link integrity, build tooling, coverage reporting, and commit reporting.

## Live Directory Telemetry

- **Path**: `/tests`
- **Direct Files**: 10 (9 Python, 1 Markdown)
- **Subdirectories**: 1
- **Total Size**: 164.2 KB | **Text/Code Lines**: 3,128
- **Primary Workflow**: Run `python3 python/run_tests.py` (full Python + JS + HTML UI suite), `python3 python/run_tests.py --summary-only` (category tables), or `python3 python/run_tests.py --python-only`.

## Directory Role & Agent Invariants

- Pre-test and post-test SHA-256 hashing in `python/run_tests.py` aborts with exit code 1 if any test mutates `data/*.json`.
- All test sandboxes must use temporary directories (`tempfile.mkdtemp()`), never repository files.
- Failed test reports are written exclusively to `failed-tests/` and automatically removed once passing.

## Subdirectories

| Subdirectory | Files | Role & Scope |
| :--- | :---: | :--- |
| [`js/`](js/OKF.md) | 15 | Zero-dependency Node.js test harness and 13 modular test suites (200 assertions across 18 sections) verifying pricing calculators, payment schedule... |

## Detailed File Inventory

| File | Type | Size / Lines | In-Context Details & Cross-References |
| :--- | :---: | :---: | :--- |
| [`README.md`](README.md) | Markdown | 4.0 KB / 76L | Heading: "Unit Test Suite (`tests/`)"; 4 H2 section(s), 574 words |
| [`__init__.py`](__init__.py) | Python | 219 B / 10L | tests package for NLP-Fuzzy-Expert validation suite. |
| [`test_check_links.py`](test_check_links.py) | Python | 4.4 KB / 113L | test_check_links.py - Unit Tests for Link and Anchor Verifier.; Classes: `TestCheckLinks` |
| [`test_coverage_report.py`](test_coverage_report.py) | Python | 7.8 KB / 178L | test_coverage_report.py - Unit Tests for Test Coverage Reporter.; Classes: `TestCoverageReport` |
| [`test_helpers.py`](test_helpers.py) | Python | 769 B / 26L | Shared modular helpers for the repository unit test suite.; Functions: `load_json`, `get_config_json_count`, `get_disk_data_json_files` |
| [`test_html_ui.py`](test_html_ui.py) | Python | 85.0 KB / 1454L | test_html_ui.py - Comprehensive HTML UI & DOM Structural Test Suite (BeautifulSoup4 + lxml).; Classes: `TestHTMLUISuite`; Functions: `get_html_test_category`, `discover_html_pages` |
| [`test_last_commit_report.py`](test_last_commit_report.py) | Python | 12.2 KB / 261L | test_last_commit_report.py - Unit Tests for Pre-Commit Report Generator (LAST-COMMIT-REPORT.md).; Classes: `TestLastCommitReport` |
| [`test_pricing_and_policy_invariants.py`](test_pricing_and_policy_invariants.py) | Python | 7.0 KB / 140L | test_pricing_and_policy_invariants.py - Authoritative Pricing, Milestone & Policy Invariant Tests.; Classes: `TestPricingAndPolicyInvariants`; Functions: `load_json`, `read_text` |
| [`test_python_tools.py`](test_python_tools.py) | Python | 35.6 KB / 709L | test_python_tools.py - Comprehensive Unit Tests for Python Utility Tools.; Classes: `TestPythonTools` |
| [`test_ssot.py`](test_ssot.py) | Python | 7.3 KB / 161L | Unit tests for python/ssot (load, format, validate, render idempotency).; Classes: `TestSSOTSystem` |

## Context & Cross-References

- Root Directory: [`OKF.md`](../OKF.md)
- Canonical Specification: [`AGENTS.md`](../AGENTS.md) (`CLAUDE.md` and `GEMINI.md` symlink to `AGENTS.md`)
- Repository Telemetry: [`METRICS.md`](../METRICS.md)
- Related: [`tests/js/`](../tests/js/OKF.md) - Modular JavaScript Unit & DOM Test Suite
- Related: [`failed-tests/`](../failed-tests/OKF.md) - Failed Test Reports Directory
- Related: [`python/`](../python/OKF.md) - Repository Python Automation & Build Tooling
- Related: [`python/ssot/`](../python/ssot/OKF.md) - Deterministic SSOT Build & Synchronization Pipeline
