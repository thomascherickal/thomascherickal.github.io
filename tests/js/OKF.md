---
type: module
title: "JavaScript Unit & Integration Tests"
description: "Node.js test suites verifying interactive clientside calculators, payment schedules, and edge-case pricing selections."
tags: [tests, javascript-tests, node, calculator-tests]
timestamp: 2026-11-10T08:43:02Z
---

# JavaScript Unit & Integration Tests

Node.js test suites verifying interactive clientside calculators, payment schedules, and edge-case pricing selections.

## Directory Role & Agent Guidelines

- Contains test suites: `test_business_services.js`, `test_payment_schedule.js`, `test_custom_package.js`, `test_extra_participants.js`, etc.
- Executed via `node pricing-tests.js`.
- Ensures frontend interactive pricing widgets remain 100% bug-free and synchronized with backend rules.

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
