---
type: dataset
title: "Generated Client-Side Site Data Bundle"
description: "Auto-generated JSON data exported from `data/*.json` by `python/ssot/export_site_data.py` for browser-side JavaScript consumption."
tags: [assets, data, ssot, generated, javascript]
timestamp: 2026-11-27T06:56:49Z
---

# Generated Client-Side Site Data Bundle

Auto-generated JSON data exported from `data/*.json` by `python/ssot/export_site_data.py` for browser-side JavaScript consumption.

## Live Directory Telemetry

- **Path**: `/assets/data`
- **Direct Files**: 1 (1 JSON)
- **Subdirectories**: 0
- **Total Size**: 27.4 KB | **Text/Code Lines**: 754
- **Primary Workflow**: Regenerate via `python3 python/ssot/export_site_data.py` whenever `data/pricing.json`, `data/tiers.json`, `data/contact.json`, or `data/config.json` changes.

## Directory Role & Agent Invariants

- CRITICAL RULE: Never edit `assets/data/site-data.json` by hand; it is overwritten automatically by `python3 python/ssot/export_site_data.py`.
- Contains exported snapshots of `pricing`, `tiers`, `contact`, and `config` domains from `data/*.json`.

## Subdirectories

Leaf directory (no child directories).

## Detailed File Inventory

| File | Type | Size / Lines | In-Context Details & Cross-References |
| :--- | :---: | :---: | :--- |
| [`site-data.json`](site-data.json) | JSON | 27.4 KB / 754L | Keys: `pricing` (7 keys), `tiers` (2 keys), `contact` (6 keys), `config` (11 keys) |

## Context & Cross-References

- Parent Directory: [`assets/OKF.md`](../OKF.md)
- Root Directory: [`OKF.md`](../../OKF.md)
- Canonical Specification: [`AGENTS.md`](../../AGENTS.md) (`CLAUDE.md` and `GEMINI.md` symlink to `AGENTS.md`)
- Repository Telemetry: [`METRICS.md`](../../METRICS.md)
- Related: [`data/`](../../data/OKF.md) - Authoritative Data Directory (Single Source of Truth)
- Related: [`python/ssot/`](../../python/ssot/OKF.md) - Deterministic SSOT Build & Synchronization Pipeline
- Related: [`pricing-calculator/`](../../pricing-calculator/OKF.md) - Tier-Specific Interactive Pricing Calculator Modules
