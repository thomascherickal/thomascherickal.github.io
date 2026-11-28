---
type: dataset
title: "Authoritative Data Directory (Single Source of Truth)"
description: "The sole authoritative JSON data store (`data/*.json`) mastering pricing, parity tiers, business/personal services, site config, contact SLA, guarantees, FAQs, publications, and Claude/Gemini guide registries."
tags: [data, ssot, read-only, pricing, services, authoritative]
timestamp: 2026-11-28T05:16:04Z
---

# Authoritative Data Directory (Single Source of Truth)

The sole authoritative JSON data store (`data/*.json`) mastering pricing, parity tiers, business/personal services, site config, contact SLA, guarantees, FAQs, publications, and Claude/Gemini guide registries.

## Live Directory Telemetry

- **Path**: `/data`
- **Direct Files**: 12 (12 JSON)
- **Subdirectories**: 1
- **Total Size**: 95.0 KB | **Text/Code Lines**: 2,246
- **Primary Workflow**: Edit `data/*.json` directly -> validate with `python3 python/ssot/validate.py` -> propagate with `python3 python/ssot/render.py && python3 python/ssot/export_site_data.py`.

## Directory Role & Agent Invariants

- CRITICAL RULE: `data/*.json` is tracked in Git and strictly read-only to all scripts, test suites, and automated source-control commands.
- Data flows strictly one way: `data/*.json` -> `python/ssot/render.py` & `export_site_data.py` -> HTML / `llms.txt` / `assets/data/site-data.json`.
- Every JSON file is validated against its matching schema in `data/schema/*.schema.json` plus cross-file invariants in `python/ssot/validate.py`.
- `python/run_tests.py` enforces pre-test and post-test SHA-256 immutability checks on every `data/*.json` file.

## Subdirectories

| Subdirectory | Files | Role & Scope |
| :--- | :---: | :--- |
| [`schema/`](schema/OKF.md) | 12 | JSON Schema (Draft 2020-12 / Draft-07) definitions validating the exact structure, types, and required fields of every authoritative file in `data/... |

## Detailed File Inventory

| File | Type | Size / Lines | In-Context Details & Cross-References |
| :--- | :---: | :---: | :--- |
| [`business-services.json`](business-services.json) | JSON | 7.1 KB / 172L | Keys: `section` (5 keys), `services` (16 items), `count` |
| [`claude.json`](claude.json) | JSON | 6.0 KB / 134L | Keys: `hub` (2 keys), `guides` (18 items) |
| [`config.json`](config.json) | JSON | 471 B / 13L | Keys: `author`, `email`, `domain`, `brand`, `brandStartYear`, `title`; (+5 more) |
| [`contact.json`](contact.json) | JSON | 7.3 KB / 203L | Keys: `contact_metadata` (6 keys), `config_references` (7 keys), `direct_communication` (6 keys), `engagement_standards` (4 items), `intake_workflow_steps` (4 items), `form_configuration` (4 keys) |
| [`faqs.json`](faqs.json) | JSON | 3.7 KB / 59L | Keys: `faq_metadata` (2 keys), `core_policies` (6 keys) |
| [`gemini.json`](gemini.json) | JSON | 6.4 KB / 134L | Keys: `hub` (2 keys), `guides` (18 items) |
| [`guarantees.json`](guarantees.json) | JSON | 4.6 KB / 96L | Keys: `guarantees_metadata` (6 keys), `config_references` (6 keys), `trust_pillars` (6 items), `agency_comparison` (5 items), `apache_license_rights` (4 items) |
| [`personal-services.json`](personal-services.json) | JSON | 7.1 KB / 172L | Keys: `section` (5 keys), `services` (16 items), `count` |
| [`pricing.json`](pricing.json) | JSON | 14.2 KB / 391L | Keys: `currency`, `purchasing_power_parity_framework`, `general_payment_rules` (6 keys), `ip_rules` (3 keys), `offerings` (4 items), `trust_and_payment_guarantees` (2 keys); (+1 more) |
| [`publications.json`](publications.json) | JSON | 26.3 KB / 578L | JSON array with 48 items |
| [`services.json`](services.json) | JSON | 7.9 KB / 149L | Keys: `count`, `services` (16 items) |
| [`tiers.json`](tiers.json) | JSON | 3.9 KB / 145L | Keys: `tiers` (6 items), `precedence_rule` |

## Context & Cross-References

- Root Directory: [`OKF.md`](../OKF.md)
- Canonical Specification: [`AGENTS.md`](../AGENTS.md) (`CLAUDE.md` and `GEMINI.md` symlink to `AGENTS.md`)
- Repository Telemetry: [`METRICS.md`](../METRICS.md)
- Related: [`data/schema/`](../data/schema/OKF.md) - SSOT JSON Schemas (`data/schema/`)
- Related: [`python/ssot/`](../python/ssot/OKF.md) - Deterministic SSOT Build & Synchronization Pipeline
- Related: [`assets/data/`](../assets/data/OKF.md) - Generated Client-Side Site Data Bundle
- Related: [`tests/`](../tests/OKF.md) - Unified Test Suite & Verification Harness
