---
type: module
title: "SSOT Reusable Block Templates (`*.html.j2`)"
description: "Jinja2 partial templates (`<block_name>.html.j2`) rendered by `python/ssot/render.py` when evaluating `<!--ssot:block.<block_name>-->` markers."
tags: [templates, blocks, jinja2, ssot]
timestamp: 2026-11-27T06:56:49Z
---

# SSOT Reusable Block Templates (`*.html.j2`)

Jinja2 partial templates (`<block_name>.html.j2`) rendered by `python/ssot/render.py` when evaluating `<!--ssot:block.<block_name>-->` markers.

## Live Directory Telemetry

- **Path**: `/templates/blocks`
- **Direct Files**: 0 (Empty directory)
- **Subdirectories**: 0
- **Total Size**: 0 B
- **Primary Workflow**: Add `<name>.html.j2` here and run `python3 python/ssot/render.py && python3 python/ssot/render.py --check`.

## Directory Role & Agent Invariants

- Use for repeated multi-element structures (tier cards, pricing cards, publication lists, FAQ accordions).
- Keep templates deterministic and free of emojis or em-dashes so `render.py --check` stays 100% idempotent.

## Subdirectories

Leaf directory (no child directories).

## Detailed File Inventory

No direct files yet (ready for new files matching this directory's workflow).

## Context & Cross-References

- Parent Directory: [`templates/OKF.md`](../OKF.md)
- Root Directory: [`OKF.md`](../../OKF.md)
- Canonical Specification: [`AGENTS.md`](../../AGENTS.md) (`CLAUDE.md` and `GEMINI.md` symlink to `AGENTS.md`)
- Repository Telemetry: [`METRICS.md`](../../METRICS.md)
- Related: [`templates/`](../../templates/OKF.md) - SSOT Jinja2 Block Templates Root
- Related: [`python/ssot/`](../../python/ssot/OKF.md) - Deterministic SSOT Build & Synchronization Pipeline
- Related: [`data/`](../../data/OKF.md) - Authoritative Data Directory (Single Source of Truth)
