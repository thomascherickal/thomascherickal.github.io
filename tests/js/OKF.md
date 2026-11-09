---
type: module
title: "JavaScript Unit & Integration Tests"
description: "Node.js and headless browser test suites verifying clientside calculators and scripts."
tags: [tests, javascript-tests, node]
timestamp: 2026-11-09T20:35:37Z
---

# JavaScript Unit & Integration Tests

Node.js and headless browser test suites verifying clientside calculators and scripts.

## Directory Role & Agent Guidelines

- Comprehensive test verification suite.
- Run Python tests with `python3 python/run_tests.py` or `python3 -m unittest discover -s tests -v`.
- Pre-commit enforces 100% test pass rate across all suites.

## Subdirectories

None (Leaf directory).

## File Inventory

- [`runner.js`](runner.js)
- [`test_business_services.js`](test_business_services.js)
- [`test_components.js`](test_components.js)
- [`test_custom_package.js`](test_custom_package.js)
- [`test_data_integrity.js`](test_data_integrity.js)
- [`test_dom_controller.js`](test_dom_controller.js)
- [`test_edge_cases.js`](test_edge_cases.js)
- [`test_extra_participants.js`](test_extra_participants.js)
- [`test_flagship_packages.js`](test_flagship_packages.js)
- [`test_helpers.js`](test_helpers.js)
- [`test_individual_services.js`](test_individual_services.js)
- [`test_mermaid.js`](test_mermaid.js)
- [`test_payment_schedule.js`](test_payment_schedule.js)
- [`test_zero_selection.js`](test_zero_selection.js)

## Context & Cross-References

- Parent Directory: [`../OKF.md`](../OKF.md)
- Architectural Specification: [`GEMINI.md`](/GEMINI.md)
- Repository Telemetry: [`METRICS.md`](/METRICS.md)
