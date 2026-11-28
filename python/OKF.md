---
type: module
title: "Repository Python Automation & Build Tooling"
description: "Core Python 3 automation suite for header/footer component inlining, SSOT validation/rendering (`python/ssot/`), link checking, JSON-LD schema extraction, Markdown mirroring, metrics, OKF generation, and unified test execution."
tags: [python, build-system, automation, tooling, ssot]
timestamp: 2026-11-28T05:16:04Z
---

# Repository Python Automation & Build Tooling

Core Python 3 automation suite for header/footer component inlining, SSOT validation/rendering (`python/ssot/`), link checking, JSON-LD schema extraction, Markdown mirroring, metrics, OKF generation, and unified test execution.

## Live Directory Telemetry

- **Path**: `/python`
- **Direct Files**: 23 (22 Python, 1 Markdown)
- **Subdirectories**: 2
- **Total Size**: 324.9 KB | **Text/Code Lines**: 7,491
- **Primary Workflow**: Run `python3 python/run_tests.py` (executes Python, JS, and HTML UI suites; `--summary-only` for category tables; `--python-only` for Python only).

## Directory Role & Agent Invariants

- `build_components.py`: Inlines `header.html`/`footer.html` across all HTML pages and chains `ssot/render.py`, `ssot/export_site_data.py`, `sync_schemas.py`, `md_html_sync.py`, `generate_metrics.py`, and `sync_og_card.py`.
- `run_tests.py`: Executes Python (`tests/`), JavaScript (`pricing-tests.js`), and HTML UI (`run_html_tests.py`) suites, verifies `data/*.json` SHA-256 immutability, and routes failure reports exclusively to `failed-tests/`.
- `delete_page.py`: Safely purges an HTML page along with its schema, Markdown mirror, `.sync-state.json` entry, sitemap entry, and navigation references.
- `check_links.py`, `sync_schemas.py`, `md_html_sync.py`: Run during commit preparation (automated in `python/hooks/pre-commit`).

## Subdirectories

| Subdirectory | Files | Role & Scope |
| :--- | :---: | :--- |
| [`hooks/`](hooks/OKF.md) | 3 | Pre-commit, pre-push, and IDE post-edit hooks enforcing SSOT validation, marker rendering, schema/Markdown sync, category-summary test gates, and `... |
| [`ssot/`](ssot/OKF.md) | 7 | Build-time Single Source of Truth engine replacing post-hoc fuzzy rewriting: loads `data/*.json`, validates JSON Schemas and cross-file invariants,... |

## Detailed File Inventory

| File | Type | Size / Lines | In-Context Details & Cross-References |
| :--- | :---: | :---: | :--- |
| [`README.md`](README.md) | Markdown | 4.0 KB / 98L | Heading: "Python Automation & Build Directory"; 2 H2 section(s), 474 words |
| [`build.py`](build.py) | Python | 464 B / 24L | build.py - Backward-compatible wrapper for build_components.py.; Functions: `main` |
| [`build_components.py`](build_components.py) | Python | 9.8 KB / 225L | build_components.py - Build-time component includer for thomascherickal.github.io.; Functions: `extract_nav`, `extract_footer`, `build`, `watch` |
| [`build_md_files.py`](build_md_files.py) | Python | 38.2 KB / 895L | build_md_files.py - Extract structured, high-fidelity Markdown from all HTML pages; Functions: `clean_text`, `convert_articles`, `convert_writing`, `convert_general_document`, `convert_pricing` (+8) |
| [`check_links.py`](check_links.py) | Python | 4.0 KB / 105L | Internal Link and Anchor Checker; Functions: `get_html_files`, `collect_file_targets`, `check_links` |
| [`delete_page.py`](delete_page.py) | Python | 13.3 KB / 365L | delete_page.py - Page Deletion and Synchronization Cleanup Utility.; Functions: `find_html_file`, `find_schema_files`, `find_markdown_files`, `purge_sync_state`, `clean_sitemap` (+5) |
| [`generate_favicons.py`](generate_favicons.py) | Python | 3.5 KB / 85L | generate_favicons.py - Generate all website favicon files from assets/site-icon.webp.; Functions: `generate_favicons` |
| [`generate_last_commit_report.py`](generate_last_commit_report.py) | Python | 25.0 KB / 569L | generate_last_commit_report.py - Automated Pre-Commit Report Generator.; Functions: `run_git_cmd`, `load_python_test_data`, `load_html_test_data`, `load_js_test_data`, `load_references_data` (+4) |
| [`generate_llms_full.py`](generate_llms_full.py) | Python | 6.3 KB / 125L | generate_llms_full.py - Builds llms-full.txt deterministically from data/*.json and live pages.; Functions: `clean_prose`, `extract_title_and_desc`, `generate_llms_full` |
| [`generate_metrics.py`](generate_metrics.py) | Python | 22.0 KB / 406L | generate_metrics.py - Comprehensive Repository Metrics Generator.; Functions: `get_git_output`, `format_bytes`, `generate_metrics` |
| [`generate_og_card.py`](generate_og_card.py) | Python | 12.3 KB / 457L | generate_og_card.py - Generates assets/og-card.png (1200x630) using Headless Google Chrome.; Functions: `build_og_html`, `generate_og_card` |
| [`generate_okf_files.py`](generate_okf_files.py) | Python | 62.9 KB / 1048L | generate_okf_files.py - Intelligent Open Knowledge Format (OKF.md) generator.; Functions: `sanitize_text`, `fmt_size`, `build_directory_metadata`, `build_reference_index`, `lookup_refs` (+4) |
| [`generate_pre_push_summary.py`](generate_pre_push_summary.py) | Python | 18.1 KB / 436L | generate_pre_push_summary.py - Formatted Terminal Reporter for Git pre-push.; Functions: `print_banner`, `parse_last_commit_report`, `print_sync_summary`, `print_python_tests_summary`, `print_js_tests_summary` (+1) |
| [`generate_rss.py`](generate_rss.py) | Python | 4.0 KB / 105L | generate_rss.py - Generates rss.xml (RSS 2.0) for thedigitalfuturist.com.; Functions: `get_git_rfc2822`, `extract_meta`, `generate_rss` |
| [`generate_sitemap.py`](generate_sitemap.py) | Python | 4.1 KB / 134L | generate_sitemap.py - Dynamically regenerates sitemap.xml from the repository file tree.; Functions: `is_noindex`, `get_git_lastmod`, `discover_dir_html`, `discover_core_html`, `generate_sitemap` |
| [`generate_test_coverage_report.py`](generate_test_coverage_report.py) | Python | 14.7 KB / 408L | generate_test_coverage_report.py - Comprehensive Test Coverage Engine & Reporter.; Functions: `get_git_info`, `run_python_coverage`, `format_line_ranges`, `run_javascript_coverage`, `generate_coverage_report` |
| [`md_html_sync.py`](md_html_sync.py) | Python | 29.0 KB / 611L | md_html_sync.py - Bidirectional Synchronization Engine for HTML and Markdown.; Functions: `get_html_path`, `get_file_hash`, `get_file_mtime`, `load_state`, `save_state` (+6) |
| [`precommit_auto_loop.py`](precommit_auto_loop.py) | Python | 12.3 KB / 290L | precommit_auto_loop.py - Repo-Scoped Autonomous Pre-Commit Test & Fix Loop.; Functions: `compute_data_hashes`, `verify_data_immutable`, `parse_failures`, `run_all_tests`, `print_category_test_summaries` (+2) |
| [`run_html_tests.py`](run_html_tests.py) | Python | 7.1 KB / 207L | run_html_tests.py - Dedicated HTML UI Test Suite Runner & Metrics Cache Generator.; Classes: `HTMLUITestResult`; Functions: `run_html_ui_tests` |
| [`run_tests.py`](run_tests.py) | Python | 14.7 KB / 363L | run_tests.py - Comprehensive Unit Test Suite Runner.; Classes: `RecordingTestResult`; Functions: `get_data_dir_hashes`, `run_all_tests`, `run_js_and_html_suites`, `run_full_test_suite` |
| [`sync_og_card.py`](sync_og_card.py) | Python | 4.3 KB / 120L | sync_og_card.py - Synchronizes assets/og-card.png with the factual stats in index.html.; Functions: `extract_index_stats`, `update_generate_script`, `regenerate_og_card`, `sync_og_card` |
| [`sync_schemas.py`](sync_schemas.py) | Python | 9.7 KB / 265L | sync_schemas.py - HTML to JSON Schema Synchronization Engine.; Functions: `get_target_schema_filename`, `extract_schema_from_html`, `sync_html_to_schemas`, `main` |
| [`update_readme_commit_summary.py`](update_readme_commit_summary.py) | Python | 5.2 KB / 150L | update_readme_commit_summary.py - Extracts the summary of updates from; Functions: `extract_section`, `extract_subsection`, `build_summary_from_report`, `update_readme_with_commit_summary` |

## Context & Cross-References

- Root Directory: [`OKF.md`](../OKF.md)
- Canonical Specification: [`AGENTS.md`](../AGENTS.md) (`CLAUDE.md` and `GEMINI.md` symlink to `AGENTS.md`)
- Repository Telemetry: [`METRICS.md`](../METRICS.md)
- Related: [`python/ssot/`](../python/ssot/OKF.md) - Deterministic SSOT Build & Synchronization Pipeline
- Related: [`python/hooks/`](../python/hooks/OKF.md) - Git & Editor Lifecycle Hooks
- Related: [`tests/`](../tests/OKF.md) - Unified Test Suite & Verification Harness
- Related: [`failed-tests/`](../failed-tests/OKF.md) - Failed Test Reports Directory
- Related: [`data/`](../data/OKF.md) - Authoritative Data Directory (Single Source of Truth)
