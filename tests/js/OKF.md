---
type: module
title: "Modular JavaScript Unit & DOM Test Suite"
description: "Zero-dependency Node.js test harness and 13 modular test suites (200 assertions across 18 sections) verifying pricing calculators, payment schedules, tier calculators, components navigation, and Mermaid link rewriting."
tags: [tests, javascript-tests, node, calculator-tests]
timestamp: 2026-11-27T06:56:49Z
---

# Modular JavaScript Unit & DOM Test Suite

Zero-dependency Node.js test harness and 13 modular test suites (200 assertions across 18 sections) verifying pricing calculators, payment schedules, tier calculators, components navigation, and Mermaid link rewriting.

## Live Directory Telemetry

- **Path**: `/tests/js`
- **Direct Files**: 15 (15 JavaScript)
- **Subdirectories**: 0
- **Total Size**: 111.7 KB | **Text/Code Lines**: 2,995
- **Primary Workflow**: Run `node pricing-tests.js` directly or via `python3 python/run_tests.py`.

## Directory Role & Agent Invariants

- Orchestrated by `tests/js/runner.js` and `tests/js/test_helpers.js`, persisting metrics to `tests/.js_test_reports_cache.json`.
- On failure, writes `failed-tests/failed-pricing-tests.md` and deletes it automatically when all 200 tests pass.

## Subdirectories

Leaf directory (no child directories).

## Detailed File Inventory

| File | Type | Size / Lines | In-Context Details & Cross-References |
| :--- | :---: | :---: | :--- |
| [`runner.js`](runner.js) | JavaScript | 3.4 KB / 89L | runner.js - Modular JavaScript Test Suite Runner; Functions: `runAllModules` |
| [`test_business_services.js`](test_business_services.js) | JavaScript | 5.7 KB / 153L | test_business_services.js - Business Pick-Services Tests (up to 10 seats); 1 section(s), 0 assertion(s) |
| [`test_components.js`](test_components.js) | JavaScript | 16.6 KB / 543L | test_components.js - Unit Tests for components.js.; 1 section(s), 0 assertion(s) |
| [`test_custom_package.js`](test_custom_package.js) | JavaScript | 2.6 KB / 89L | test_custom_package.js - Custom Amount Package Tests; 1 section(s), 0 assertion(s) |
| [`test_data_integrity.js`](test_data_integrity.js) | JavaScript | 5.8 KB / 132L | test_data_integrity.js - Data Integrity & Metadata Tests; 1 section(s), 0 assertion(s) |
| [`test_dom_controller.js`](test_dom_controller.js) | JavaScript | 13.8 KB / 280L | test_dom_controller.js - DOM Controller Simulated Live-Update Tests (pricing.html Pane); 1 section(s), 0 assertion(s) |
| [`test_edge_cases.js`](test_edge_cases.js) | JavaScript | 1.7 KB / 43L | test_edge_cases.js - Defensive Edge Cases & Malformed Inputs Tests; 1 section(s), 0 assertion(s) |
| [`test_extra_participants.js`](test_extra_participants.js) | JavaScript | 4.3 KB / 103L | test_extra_participants.js - Business Extra Participant Scaling Tests (+ $250/seat scaled by tier); 1 section(s), 0 assertion(s) |
| [`test_flagship_packages.js`](test_flagship_packages.js) | JavaScript | 8.2 KB / 186L | test_flagship_packages.js - Flagship Package Tests; 1 section(s), 0 assertion(s) |
| [`test_helpers.js`](test_helpers.js) | JavaScript | 12.3 KB / 345L | test_helpers.js - Shared test harness, mock DOM, and ASCII table renderer for JavaScript test suite.; Functions: `formatCell`, `renderAsciiTable`, `createHarness`, `setSection`, `assert` (+4) |
| [`test_individual_services.js`](test_individual_services.js) | JavaScript | 7.0 KB / 193L | test_individual_services.js - Individual Pick-Services Tests (strictly 1 person); 1 section(s), 0 assertion(s) |
| [`test_mermaid.js`](test_mermaid.js) | JavaScript | 4.4 KB / 154L | test_mermaid.js - Unit Tests for mermaid.js Markdown Link Transformations.; 1 section(s), 0 assertion(s) |
| [`test_payment_schedule.js`](test_payment_schedule.js) | JavaScript | 18.1 KB / 476L | test_payment_schedule.js - Comprehensive Unit Tests for payment-schedule.js; 6 section(s), 0 assertion(s) |
| [`test_tier_calculators.js`](test_tier_calculators.js) | JavaScript | 4.8 KB / 125L | test_tier_calculators.js - Comprehensive Tests for Tier-Specific Calculators (Tiers 1-6); 1 section(s), 0 assertion(s) |
| [`test_zero_selection.js`](test_zero_selection.js) | JavaScript | 2.9 KB / 84L | test_zero_selection.js - Nothing Selected (0 USD) Tests; 1 section(s), 0 assertion(s) |

## Context & Cross-References

- Parent Directory: [`tests/OKF.md`](../OKF.md)
- Root Directory: [`OKF.md`](../../OKF.md)
- Canonical Specification: [`AGENTS.md`](../../AGENTS.md) (`CLAUDE.md` and `GEMINI.md` symlink to `AGENTS.md`)
- Repository Telemetry: [`METRICS.md`](../../METRICS.md)
- Related: [`tests/`](../../tests/OKF.md) - Unified Test Suite & Verification Harness
- Related: [`pricing-calculator/`](../../pricing-calculator/OKF.md) - Tier-Specific Interactive Pricing Calculator Modules
- Related: [`failed-tests/`](../../failed-tests/OKF.md) - Failed Test Reports Directory
