---
type: module
title: "Markdown HTML Synchronization Root"
description: "Markdown mirror representations of live HTML pages, kept synchronized for consumption by LLM coding agents and crawlers."
tags: [md-html-sync, markdown-mirrors, llm-friendly, state-tracking]
timestamp: 2026-11-10T08:43:02Z
---

# Markdown HTML Synchronization Root

Markdown mirror representations of live HTML pages, kept synchronized for consumption by LLM coding agents and crawlers.

## Directory Role & Agent Guidelines

- Synchronized via `python3 python/md_html_sync.py` only during commit preparation, not after every single edit.
- Contains `.sync-state.json` to track modification hashes and avoid unnecessary rebuilds.
- Organized into subdirectories mirroring the site structure (`business-services/`, `personal-services/`, `claude-training/`, `google-training/`).
- Never delete markdown mirrors directly with `rm`; use `python3 python/delete_page.py` when deleting pages.

## Subdirectories

- [`business-services/`](business-services/OKF.md): Subdirectory knowledge bundle
- [`claude-training/`](claude-training/OKF.md): Subdirectory knowledge bundle
- [`google-training/`](google-training/OKF.md): Subdirectory knowledge bundle
- [`personal-services/`](personal-services/OKF.md): Subdirectory knowledge bundle

## File Inventory

- [`.sync-state.json`](.sync-state.json)
- [`404.md`](404.md)
- [`all-business-services.md`](all-business-services.md)
- [`all-personal-services.md`](all-personal-services.md)
- [`articles.md`](articles.md)
- [`business-ai-training.md`](business-ai-training.md)
- [`contact.md`](contact.md)
- [`expertise.md`](expertise.md)
- [`faqs.md`](faqs.md)
- [`guarantees.md`](guarantees.md)
- [`home.md`](home.md)
- [`personal-ai-training.md`](personal-ai-training.md)
- [`pricing.md`](pricing.md)
- [`writing.md`](writing.md)

## Context & Cross-References

- Root Directory: [`/OKF.md`](../OKF.md)
- Architectural Specification: [`GEMINI.md`](/GEMINI.md)
- Repository Telemetry: [`METRICS.md`](/METRICS.md)
