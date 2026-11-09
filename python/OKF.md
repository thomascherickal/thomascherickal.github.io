---
type: module
title: "Repository Python Build & Maintenance Automation"
description: "Build scripts, component inliners, markdown synchronizers, link checkers, and metrics generators."
tags: [python, build-system, automation, tooling]
timestamp: 2026-11-09T20:35:37Z
---

# Repository Python Build & Maintenance Automation

Build scripts, component inliners, markdown synchronizers, link checkers, and metrics generators.

## Directory Role & Agent Guidelines

- Automation scripts for component inlining, link integrity verification, markdown mirrors, and metrics.
- Re-run `python3 python/build_components.py` whenever `header.html` or `footer.html` changes.

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
