---
type: reports
title: "Failed Test Reports Directory"
description: "Centralized holding directory where Python (`run_tests.py`), HTML UI (`run_html_tests.py`), and JavaScript (`pricing-tests.js`) runners write Markdown failure reports whenever a suite fails."
tags: [failed-tests, testing, reports, qa]
timestamp: 2026-11-28T05:16:04Z
---

# Failed Test Reports Directory

Centralized holding directory where Python (`run_tests.py`), HTML UI (`run_html_tests.py`), and JavaScript (`pricing-tests.js`) runners write Markdown failure reports whenever a suite fails.

## Live Directory Telemetry

- **Path**: `/failed-tests`
- **Direct Files**: 1 (1 Markdown)
- **Subdirectories**: 0
- **Total Size**: 569 B | **Text/Code Lines**: 9
- **Primary Workflow**: Run `python3 python/run_tests.py`; when a suite passes 100%, its failure report in `failed-tests/` is automatically deleted.

## Directory Role & Agent Invariants

- All failed-test reports (`failed-python-tests.md`, `failed-html-tests.md`, `failed-pricing-tests.md`) must be stored exclusively inside `failed-tests/` and never in the repository root.
- A clean passing repository leaves only `readme.md` and `OKF.md` inside `failed-tests/`.

## Subdirectories

Leaf directory (no child directories).

## Detailed File Inventory

| File | Type | Size / Lines | In-Context Details & Cross-References |
| :--- | :---: | :---: | :--- |
| [`readme.md`](readme.md) | Markdown | 569 B / 9L | Heading: "Failed Tests Reports Directory"; 0 H2 section(s), 71 words |

## Context & Cross-References

- Root Directory: [`OKF.md`](../OKF.md)
- Canonical Specification: [`AGENTS.md`](../AGENTS.md) (`CLAUDE.md` and `GEMINI.md` symlink to `AGENTS.md`)
- Repository Telemetry: [`METRICS.md`](../METRICS.md)
- Related: [`tests/`](../tests/OKF.md) - Unified Test Suite & Verification Harness
- Related: [`tests/js/`](../tests/js/OKF.md) - Modular JavaScript Unit & DOM Test Suite
- Related: [`python/`](../python/OKF.md) - Repository Python Automation & Build Tooling
