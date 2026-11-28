---
type: module
title: "SSOT Jinja2 Block Templates Root"
description: "Root directory for reusable Jinja2 HTML templates rendered into `<!--ssot:block.<name>-->` markers by `python/ssot/render.py`."
tags: [templates, jinja2, ssot, blocks]
timestamp: 2026-11-28T05:16:04Z
---

# SSOT Jinja2 Block Templates Root

Root directory for reusable Jinja2 HTML templates rendered into `<!--ssot:block.<name>-->` markers by `python/ssot/render.py`.

## Live Directory Telemetry

- **Path**: `/templates`
- **Direct Files**: 0 (Empty directory)
- **Subdirectories**: 1
- **Total Size**: 0 B
- **Primary Workflow**: Place reusable HTML block templates inside `templates/blocks/<name>.html.j2` and invoke via `<!--ssot:block.<name>--><!--/ssot-->`.

## Directory Role & Agent Invariants

- Templates receive `data`, `offerings`, and `tiers` context objects directly from `python/ssot/load.py`.

## Subdirectories

| Subdirectory | Files | Role & Scope |
| :--- | :---: | :--- |
| [`blocks/`](blocks/OKF.md) | 0 | Jinja2 partial templates (`<block_name>.html.j2`) rendered by `python/ssot/render.py` when evaluating `<!--ssot:block.<block_name>-->` markers. |

## Detailed File Inventory

No direct files yet (ready for new files matching this directory's workflow).

## Context & Cross-References

- Root Directory: [`OKF.md`](../OKF.md)
- Canonical Specification: [`AGENTS.md`](../AGENTS.md) (`CLAUDE.md` and `GEMINI.md` symlink to `AGENTS.md`)
- Repository Telemetry: [`METRICS.md`](../METRICS.md)
- Related: [`templates/blocks/`](../templates/blocks/OKF.md) - SSOT Reusable Block Templates (`*.html.j2`)
- Related: [`python/ssot/`](../python/ssot/OKF.md) - Deterministic SSOT Build & Synchronization Pipeline
- Related: [`data/`](../data/OKF.md) - Authoritative Data Directory (Single Source of Truth)
