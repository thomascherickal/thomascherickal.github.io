---
type: module
title: "Git & Editor Lifecycle Hooks"
description: "Pre-commit, pre-push, and IDE post-edit hooks enforcing SSOT validation, marker rendering, schema/Markdown sync, category-summary test gates, and `LAST-COMMIT-REPORT.md` generation."
tags: [python, git-hooks, pre-commit, pre-push, quality-gate]
timestamp: 2026-11-28T05:16:04Z
---

# Git & Editor Lifecycle Hooks

Pre-commit, pre-push, and IDE post-edit hooks enforcing SSOT validation, marker rendering, schema/Markdown sync, category-summary test gates, and `LAST-COMMIT-REPORT.md` generation.

## Live Directory Telemetry

- **Path**: `/python/hooks`
- **Direct Files**: 3 (2 Shell Hook, 1 Python)
- **Subdirectories**: 0
- **Total Size**: 5.6 KB | **Text/Code Lines**: 133
- **Primary Workflow**: Keep `.git/hooks/pre-commit` and `.git/hooks/pre-push` synchronized with `python/hooks/` templates. Never bypass hooks with `--no-verify`.

## Directory Role & Agent Invariants

- `pre-commit`: Runs `validate.py`, `build_components.py` (on header/footer edits), `render.py`, `export_site_data.py`, `md_html_sync.py`, `sync_schemas.py`, `render.py --check`, `run_tests.py --summary-only`, and `generate_last_commit_report.py`.
- `pre-push`: Runs `validate.py`, `render.py --check`, `sync_schemas.py`, and `generate_pre_push_summary.py`.
- `post_edit_hook.py`: Automatically runs `build_components.py` when `header.html` or `footer.html` is edited.

## Subdirectories

Leaf directory (no child directories).

## Detailed File Inventory

| File | Type | Size / Lines | In-Context Details & Cross-References |
| :--- | :---: | :---: | :--- |
| [`post_edit_hook.py`](post_edit_hook.py) | Python | 1.7 KB / 43L | post_edit_hook.py - Lifecycle hook triggered after file edits.; Functions: `main` |
| [`pre-commit`](pre-commit) | Shell Hook | 2.8 KB / 59L | Pre-commit hook (SSOT Build Pipeline):; Invokes: `python/ssot/validate.py`, `python/build_components.py`, `python/ssot/render.py`, `python/ssot/export_site_data.py`, `python/md_html_sync.py` |
| [`pre-push`](pre-push) | Shell Hook | 1.2 KB / 31L | Pre-push hook: Sync HTML JSON schemas and output pre-push summary reports to terminal; Invokes: `python/ssot/validate.py`, `python/ssot/render.py`, `python/sync_schemas.py`, `python/generate_pre_push_summary.py` |

## Context & Cross-References

- Parent Directory: [`python/OKF.md`](../OKF.md)
- Root Directory: [`OKF.md`](../../OKF.md)
- Canonical Specification: [`AGENTS.md`](../../AGENTS.md) (`CLAUDE.md` and `GEMINI.md` symlink to `AGENTS.md`)
- Repository Telemetry: [`METRICS.md`](../../METRICS.md)
- Related: [`python/`](../../python/OKF.md) - Repository Python Automation & Build Tooling
- Related: [`python/ssot/`](../../python/ssot/OKF.md) - Deterministic SSOT Build & Synchronization Pipeline
- Related: [`failed-tests/`](../../failed-tests/OKF.md) - Failed Test Reports Directory
