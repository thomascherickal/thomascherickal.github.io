---
type: module
title: "Repository Python Automation & Tooling Suite"
description: "Core automation scripts for component inlining, link verification, schema extraction, markdown mirroring, and metrics reporting."
tags: [python, build-system, automation, tooling, linters]
timestamp: 2026-11-10T08:43:02Z
---

# Repository Python Automation & Tooling Suite

Core automation scripts for component inlining, link verification, schema extraction, markdown mirroring, and metrics reporting.

## Directory Role & Agent Guidelines

- `build_components.py`: Inlines `header.html` and `footer.html` across all HTML pages.
- `check_links.py`: Audits internal links and anchor targets across all HTML files (run before committing).
- `sync_schemas.py`: Synchronizes embedded JSON-LD schemas into `schema/` (run before committing).
- `md_html_sync.py`: Synchronizes Markdown mirror files in `md-html-sync/` (run before committing).
- `generate_metrics.py`: Updates codebase telemetry in `METRICS.md`.
- `delete_page.py`: Safely purges HTML pages, schemas, markdown mirrors, and navigation references.

## Subdirectories

- [`hooks/`](hooks/OKF.md): Subdirectory knowledge bundle

## File Inventory

- [`README.md`](README.md)
- [`build.py`](build.py)
- [`build_components.py`](build_components.py)
- [`build_md_files.py`](build_md_files.py)
- [`check_links.py`](check_links.py)
- [`delete_page.py`](delete_page.py)
- [`generate_favicons.py`](generate_favicons.py)
- [`generate_last_commit_report.py`](generate_last_commit_report.py)
- [`generate_metrics.py`](generate_metrics.py)
- [`generate_og_card.py`](generate_og_card.py)
- [`generate_okf_files.py`](generate_okf_files.py)
- [`generate_pre_push_summary.py`](generate_pre_push_summary.py)
- [`generate_sitemap.py`](generate_sitemap.py)
- [`generate_test_coverage_report.py`](generate_test_coverage_report.py)
- [`md_html_sync.py`](md_html_sync.py)
- [`run_tests.py`](run_tests.py)
- [`sync_og_card.py`](sync_og_card.py)
- [`sync_pricing.py`](sync_pricing.py)
- [`sync_schemas.py`](sync_schemas.py)
- [`update_readme_commit_summary.py`](update_readme_commit_summary.py)

## Context & Cross-References

- Root Directory: [`/OKF.md`](../OKF.md)
- Architectural Specification: [`GEMINI.md`](/GEMINI.md)
- Repository Telemetry: [`METRICS.md`](/METRICS.md)
