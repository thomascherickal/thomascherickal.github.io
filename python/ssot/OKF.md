---
type: module
title: "Deterministic SSOT Build & Synchronization Pipeline"
description: "Build-time Single Source of Truth engine replacing post-hoc fuzzy rewriting: loads `data/*.json`, validates JSON Schemas and cross-file invariants, renders `<!--ssot:...-->` markers and Jinja2 blocks deterministically, exports `assets/data/site-data.json`,..."
tags: [ssot, validation, jinja2, markers, synchronization]
timestamp: 2026-11-27T06:56:49Z
---

# Deterministic SSOT Build & Synchronization Pipeline

Build-time Single Source of Truth engine replacing post-hoc fuzzy rewriting: loads `data/*.json`, validates JSON Schemas and cross-file invariants, renders `<!--ssot:...-->` markers and Jinja2 blocks deterministically, exports `assets/data/site-data.json`, and audits unprotected literals.

## Live Directory Telemetry

- **Path**: `/python/ssot`
- **Direct Files**: 7 (7 Python)
- **Subdirectories**: 0
- **Total Size**: 19.9 KB | **Text/Code Lines**: 570
- **Primary Workflow**: Run `python3 python/ssot/validate.py && python3 python/ssot/render.py && python3 python/ssot/export_site_data.py && python3 python/ssot/render.py --check`.

## Directory Role & Agent Invariants

- `load.py`: Loads all `data/*.json` files and resolves dotted paths (`offerings.<id>.*`, `tiers.<n>.*`).
- `format.py`: Owns every formatted sentence or phrase containing a protected fact (`money`, `tier_price_list`, `seats_phrase`, `duration_phrase`, `installment_phrase`, `reply_sla_phrase`, `quarterly_cap_phrase`).
- `validate.py`: Enforces all 12 JSON Schemas in `data/schema/` plus cross-file invariants (seat bounds, 100% milestone sums, due days, Tier 6 precedence, publication slug consistency, zero emojis/em-dashes, no LLM version numbers in prose).
- `render.py`: Deterministically replaces only text between `<!--ssot:...-->` and `<!--/ssot-->` markers; fails loudly on unknown paths; `--check` exits non-zero on drift.
- `export_site_data.py`: Generates `assets/data/site-data.json` for client-side JavaScript.
- `scan_literals.py`: Audits protected facts typed outside markers and refreshes `docs/ssot-inventory.csv`.

## Subdirectories

Leaf directory (no child directories).

## Detailed File Inventory

| File | Type | Size / Lines | In-Context Details & Cross-References |
| :--- | :---: | :---: | :--- |
| [`__init__.py`](__init__.py) | Python | 90 B / 1L | Single Source of Truth (SSOT) build, validation, formatting, and rendering package. |
| [`export_site_data.py`](export_site_data.py) | Python | 1.0 KB / 37L | export_site_data.py - Generate assets/data/site-data.json from data/*.json for JS.; Functions: `export_site_data` |
| [`format.py`](format.py) | Python | 2.9 KB / 86L | format.py - Deterministic phrase formatters for SSOT markers.; Functions: `money`, `tier_price_list`, `seats_phrase`, `duration_phrase`, `installment_phrase` (+3) |
| [`load.py`](load.py) | Python | 2.1 KB / 60L | load.py - Load and index data/*.json for SSOT rendering and formatting.; Classes: `SSOTData` |
| [`render.py`](render.py) | Python | 3.5 KB / 117L | render.py - Deterministic SSOT marker & block renderer.; Functions: `discover_targets`, `resolve_marker`, `render_file`, `main` |
| [`scan_literals.py`](scan_literals.py) | Python | 4.1 KB / 107L | scan_literals.py - Inventory hardcoded protected facts outside <!--ssot:...--> markers.; Functions: `collect_target_files`, `scan_file`, `main` |
| [`validate.py`](validate.py) | Python | 6.2 KB / 162L | validate.py - JSON Schema & Cross-File Invariant Validator for data/*.json.; Functions: `validate_all`, `main` |

## Context & Cross-References

- Parent Directory: [`python/OKF.md`](../OKF.md)
- Root Directory: [`OKF.md`](../../OKF.md)
- Canonical Specification: [`AGENTS.md`](../../AGENTS.md) (`CLAUDE.md` and `GEMINI.md` symlink to `AGENTS.md`)
- Repository Telemetry: [`METRICS.md`](../../METRICS.md)
- Related: [`data/`](../../data/OKF.md) - Authoritative Data Directory (Single Source of Truth)
- Related: [`data/schema/`](../../data/schema/OKF.md) - SSOT JSON Schemas (`data/schema/`)
- Related: [`templates/blocks/`](../../templates/blocks/OKF.md) - SSOT Reusable Block Templates (`*.html.j2`)
- Related: [`assets/data/`](../../assets/data/OKF.md) - Generated Client-Side Site Data Bundle
- Related: [`docs/`](../../docs/OKF.md) - Operational Documentation & SSOT Literal Inventory
- Related: [`tests/`](../../tests/OKF.md) - Unified Test Suite & Verification Harness
