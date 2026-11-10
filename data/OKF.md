---
type: dataset
title: "Authoritative Data Directory (SSOT)"
description: "Authoritative Single Source of Truth (SSOT) JSON files modeling pricing, services, publications, FAQs, and platform configurations."
tags: [data, ssot, read-only, pricing, services, authoritative]
timestamp: 2026-11-10T08:43:02Z
---

# Authoritative Data Directory (SSOT)

Authoritative Single Source of Truth (SSOT) JSON files modeling pricing, services, publications, FAQs, and platform configurations.

## Directory Role & Agent Guidelines

- CRITICAL AGENT RULE: The `data/` directory is strictly read-only against all source control operations, scripts, and test suites.
- `GEMINI.md` is the ultimate project specification; `data/*.json` models this specification for outward synchronization.
- Data flows strictly outward (`data/*.json` -> HTML / JS / Markdown / Schema files). Never mutate programmatically.
- Pre-test and post-test SHA-256 verification in `python/run_tests.py` verifies data integrity. Any automated change causes an immediate test failure.
- Only direct manual authoring by the user is permitted.

## Subdirectories

None (Leaf directory).

## File Inventory

- [`business-services.json`](business-services.json)
- [`claude.json`](claude.json)
- [`config.json`](config.json)
- [`contact.json`](contact.json)
- [`faqs.json`](faqs.json)
- [`gemini.json`](gemini.json)
- [`guarantees.json`](guarantees.json)
- [`personal-services.json`](personal-services.json)
- [`pricing.json`](pricing.json)
- [`publications.json`](publications.json)
- [`tiers.json`](tiers.json)

## Context & Cross-References

- Root Directory: [`/OKF.md`](../OKF.md)
- Architectural Specification: [`GEMINI.md`](/GEMINI.md)
- Repository Telemetry: [`METRICS.md`](/METRICS.md)
