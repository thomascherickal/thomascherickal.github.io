---
type: module
title: "Article Footers Module"
description: "Reusable syndication footer templates (dark HTML, Markdown, and Word `.docx`) appended to external articles on Dev.to, Medium, Hashnode, Substack, and HackerNoon."
tags: [article-footers, templates, reusable-components, syndication]
timestamp: 2026-11-27T06:56:49Z
---

# Article Footers Module

Reusable syndication footer templates (dark HTML, Markdown, and Word `.docx`) appended to external articles on Dev.to, Medium, Hashnode, Substack, and HackerNoon.

## Live Directory Telemetry

- **Path**: `/article-footers`
- **Direct Files**: 4 (1 DOCX, 1 HTML, 1 Markdown, 1 SVG)
- **Subdirectories**: 0
- **Total Size**: 135.7 KB | **Text/Code Lines**: 1,239
- **Primary Workflow**: Keep all three formats (`article-footer.md`, `article-footer_dark.html`, `article-footer-plain.docx`) aligned whenever social links, portfolio icon, or book descriptions change.

## Directory Role & Agent Invariants

- Preserve `article-footer.md`, `article-footer_dark.html`, and `article-footer-plain.docx` in this directory; never delete or move them.
- CRITICAL RULE: Never insert an article footer into any Claude guide (`claude-training/claude-*.html` or `md-html-sync/claude-training/claude-*.md`).
- All links use canonical email `thomascherickal@gmail.com` and include `https://dev.to/thomascherickal` (Mastodon excluded).

## Subdirectories

Leaf directory (no child directories).

## Detailed File Inventory

| File | Type | Size / Lines | In-Context Details & Cross-References |
| :--- | :---: | :---: | :--- |
| [`article-footer-plain.docx`](article-footer-plain.docx) | DOCX | 10.3 KB | Plain-text Word syndication footer template |
| [`article-footer.md`](article-footer.md) | Markdown | 10.5 KB / 97L | Heading: "Thomas Cherickal"; 2 H2 section(s), 975 words |
| [`article-footer_dark.html`](article-footer_dark.html) | HTML | 53.1 KB / 1134L | Title: "Thomas Cherickal / Author & Consultant Footer (Dark Mode)" |
| [`portfolio.svg`](portfolio.svg) | SVG | 61.7 KB / 8L | viewBox `0 0 800 800`; Referenced by: unreferenced |

## Context & Cross-References

- Root Directory: [`OKF.md`](../OKF.md)
- Canonical Specification: [`AGENTS.md`](../AGENTS.md) (`CLAUDE.md` and `GEMINI.md` symlink to `AGENTS.md`)
- Repository Telemetry: [`METRICS.md`](../METRICS.md)
- Related: [Repository Root](../OKF.md) - Repository Root & Web Portal Hub
- Related: [`icons/`](../icons/OKF.md) - Vector SVG Icon & Platform Badge Repository
- Related: [`assets/`](../assets/OKF.md) - Global Site Assets, Branding & Generated Data Hub
