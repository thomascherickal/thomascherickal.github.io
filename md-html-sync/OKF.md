---
type: module
title: "Markdown HTML Synchronization Root"
description: "Synchronized Markdown mirrors of live HTML pages (plus `.sync-state.json` hash cache) optimized for coding agents, LLM ingestion, and clean text diffs."
tags: [md-html-sync, markdown-mirrors, llm-friendly, state-tracking]
timestamp: 2026-11-28T05:16:04Z
---

# Markdown HTML Synchronization Root

Synchronized Markdown mirrors of live HTML pages (plus `.sync-state.json` hash cache) optimized for coding agents, LLM ingestion, and clean text diffs.

## Live Directory Telemetry

- **Path**: `/md-html-sync`
- **Direct Files**: 12 (12 Markdown)
- **Subdirectories**: 5
- **Total Size**: 144.6 KB | **Text/Code Lines**: 2,888
- **Primary Workflow**: Run `python3 python/md_html_sync.py` only when preparing a commit (or automatically via `python/hooks/pre-commit`).

## Directory Role & Agent Invariants

- Never manually delete mirror files with `rm`; use `python3 python/delete_page.py <page>` to clean HTML, schema, mirror, and `.sync-state.json` atomically.
- Full rebuilds from HTML use `python3 python/build_md_files.py`.

## Subdirectories

| Subdirectory | Files | Role & Scope |
| :--- | :---: | :--- |
| [`business-services/`](business-services/OKF.md) | 16 | Synchronized Markdown mirrors for all 16 enterprise HTML pages in `business-services/`. |
| [`claude-training/`](claude-training/OKF.md) | 18 | Synchronized Markdown mirrors for all Anthropic Claude engineering guides in `claude-training/`. |
| [`gemini-training/`](gemini-training/OKF.md) | 20 | Synchronized Markdown mirrors for all Google Gemini and Gemma engineering guides in `gemini-training/`. |
| [`personal-services/`](personal-services/OKF.md) | 16 | Synchronized Markdown mirrors for all 16 personal AI training pages in `personal-services/`. |
| [`pricing/`](pricing/OKF.md) | 6 | Synchronized Markdown mirrors for the 6 Purchasing Power Parity regional tier HTML pages in `pricing/`. |

## Detailed File Inventory

| File | Type | Size / Lines | In-Context Details & Cross-References |
| :--- | :---: | :---: | :--- |
| [`404.md`](404.md) | Markdown | 834 B / 19L | Heading: "404 - Context Window Exhausted"; 1 H2 section(s), 120 words |
| [`all-business-services.md`](all-business-services.md) | Markdown | 6.2 KB / 145L | Heading: "Business AI"; 3 H2 section(s), 655 words |
| [`all-personal-services.md`](all-personal-services.md) | Markdown | 5.9 KB / 144L | Heading: "Personal AI"; 3 H2 section(s), 614 words |
| [`articles.md`](articles.md) | Markdown | 15.9 KB / 284L | Heading: "Thomas Cherickal - Articles & Case Studies"; 15 H2 section(s), 1,881 words |
| [`contact.md`](contact.md) | Markdown | 2.0 KB / 37L | Heading: "Thomas Cherickal - Start an AI Transformation"; 4 H2 section(s), 217 words |
| [`expertise.md`](expertise.md) | Markdown | 5.5 KB / 129L | Heading: "Thomas Cherickal - Generative AI Architecture, AI Agent Mastery & Tech Stack"; 4 H2 section(s), 779 words |
| [`faqs.md`](faqs.md) | Markdown | 75.8 KB / 1669L | Heading: "Thomas Cherickal - Frequently Asked Questions"; 2 H2 section(s), 10,741 words |
| [`guarantees.md`](guarantees.md) | Markdown | 4.9 KB / 86L | Heading: "Trust Guarantees & Payment Structure"; 5 H2 section(s), 841 words |
| [`home.md`](home.md) | Markdown | 9.0 KB / 149L | Heading: "Thomas Cherickal"; 10 H2 section(s), 930 words |
| [`parity-pricing.md`](parity-pricing.md) | Markdown | 4.2 KB / 49L | Heading: "Country-Wise Parity Pricing (195 Countries)"; 7 H2 section(s), 591 words |
| [`pricing.md`](pricing.md) | Markdown | 2.7 KB / 53L | Heading: "Thomas Cherickal - Generative AI Packages & Transparent Pricing"; 5 H2 section(s), 363 words |
| [`writing.md`](writing.md) | Markdown | 11.7 KB / 124L | Heading: "Thomas Cherickal - Publications & Long-Form Works"; 3 H2 section(s), 1,168 words |

## Context & Cross-References

- Root Directory: [`OKF.md`](../OKF.md)
- Canonical Specification: [`AGENTS.md`](../AGENTS.md) (`CLAUDE.md` and `GEMINI.md` symlink to `AGENTS.md`)
- Repository Telemetry: [`METRICS.md`](../METRICS.md)
- Related: [Repository Root](../OKF.md) - Repository Root & Web Portal Hub
- Related: [`python/`](../python/OKF.md) - Repository Python Automation & Build Tooling
- Related: [`md-html-sync/business-services/`](../md-html-sync/business-services/OKF.md) - Business Services Markdown Mirrors
- Related: [`md-html-sync/personal-services/`](../md-html-sync/personal-services/OKF.md) - Personal Services Markdown Mirrors
- Related: [`md-html-sync/claude-training/`](../md-html-sync/claude-training/OKF.md) - Claude Training Markdown Mirrors
- Related: [`md-html-sync/gemini-training/`](../md-html-sync/gemini-training/OKF.md) - Gemini Training Markdown Mirrors
- Related: [`md-html-sync/pricing/`](../md-html-sync/pricing/OKF.md) - PPP Tier Pages Markdown Mirrors
