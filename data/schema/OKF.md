---
type: dataset
title: "SSOT JSON Schemas (`data/schema/`)"
description: "JSON Schema (Draft 2020-12 / Draft-07) definitions validating the exact structure, types, and required fields of every authoritative file in `data/*.json`."
tags: [data, json-schema, ssot, validation]
timestamp: 2026-11-28T05:16:04Z
---

# SSOT JSON Schemas (`data/schema/`)

JSON Schema (Draft 2020-12 / Draft-07) definitions validating the exact structure, types, and required fields of every authoritative file in `data/*.json`.

## Live Directory Telemetry

- **Path**: `/data/schema`
- **Direct Files**: 12 (12 JSON)
- **Subdirectories**: 0
- **Total Size**: 5.2 KB | **Text/Code Lines**: 258
- **Primary Workflow**: Executed automatically by `python3 python/ssot/validate.py` on every pre-commit and pre-push check.

## Directory Role & Agent Invariants

- Every `data/<name>.json` file has a corresponding `data/schema/<name>.schema.json` contract.
- When adding a new canonical field to `data/*.json`, update its matching `.schema.json` and add an invariant check in `python/ssot/validate.py` if needed.

## Subdirectories

Leaf directory (no child directories).

## Detailed File Inventory

| File | Type | Size / Lines | In-Context Details & Cross-References |
| :--- | :---: | :---: | :--- |
| [`business-services.schema.json`](business-services.schema.json) | JSON | 84 B / 4L | JSON Schema: business-services.schema.json |
| [`claude.schema.json`](claude.schema.json) | JSON | 84 B / 4L | JSON Schema: claude.schema.json |
| [`config.schema.json`](config.schema.json) | JSON | 1.1 KB / 63L | JSON Schema: config.schema.json; Required: `author`, `email`, `domain`, `brand`, `brandStartYear` |
| [`contact.schema.json`](contact.schema.json) | JSON | 307 B / 17L | JSON Schema: contact.schema.json; Required: `contact_metadata`, `direct_communication` |
| [`faqs.schema.json`](faqs.schema.json) | JSON | 84 B / 4L | JSON Schema: faqs.schema.json |
| [`gemini.schema.json`](gemini.schema.json) | JSON | 84 B / 4L | JSON Schema: gemini.schema.json |
| [`guarantees.schema.json`](guarantees.schema.json) | JSON | 84 B / 4L | JSON Schema: guarantees.schema.json |
| [`personal-services.schema.json`](personal-services.schema.json) | JSON | 84 B / 4L | JSON Schema: personal-services.schema.json |
| [`pricing.schema.json`](pricing.schema.json) | JSON | 2.1 KB / 93L | JSON Schema: pricing.schema.json; Required: `currency`, `general_payment_rules`, `ip_rules`, `offerings`, `capacity` |
| [`publications.schema.json`](publications.schema.json) | JSON | 484 B / 16L | JSON Schema: publications.schema.json |
| [`services.schema.json`](services.schema.json) | JSON | 84 B / 4L | JSON Schema: services.schema.json |
| [`tiers.schema.json`](tiers.schema.json) | JSON | 789 B / 41L | JSON Schema: tiers.schema.json; Required: `tiers`, `precedence_rule` |

## Context & Cross-References

- Parent Directory: [`data/OKF.md`](../OKF.md)
- Root Directory: [`OKF.md`](../../OKF.md)
- Canonical Specification: [`AGENTS.md`](../../AGENTS.md) (`CLAUDE.md` and `GEMINI.md` symlink to `AGENTS.md`)
- Repository Telemetry: [`METRICS.md`](../../METRICS.md)
- Related: [`data/`](../../data/OKF.md) - Authoritative Data Directory (Single Source of Truth)
- Related: [`python/ssot/`](../../python/ssot/OKF.md) - Deterministic SSOT Build & Synchronization Pipeline
- Related: [`tests/`](../../tests/OKF.md) - Unified Test Suite & Verification Harness
