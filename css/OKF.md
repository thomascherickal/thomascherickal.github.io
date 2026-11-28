---
type: module
title: "Global Stylesheets & Design System"
description: "Shared design-system stylesheets defining color tokens (`--gold`, `--cyan`), uniform button components (`buttons.css`), responsive tables, code windows, and guide typography."
tags: [css, design-system, tokens, responsive, buttons]
timestamp: 2026-11-28T05:16:04Z
---

# Global Stylesheets & Design System

Shared design-system stylesheets defining color tokens (`--gold`, `--cyan`), uniform button components (`buttons.css`), responsive tables, code windows, and guide typography.

## Live Directory Telemetry

- **Path**: `/css`
- **Direct Files**: 16 (16 CSS)
- **Subdirectories**: 0
- **Total Size**: 286.4 KB | **Text/Code Lines**: 12,250
- **Primary Workflow**: After editing CSS, verify zero underlined buttons and layout compliance across all 85+ HTML pages via `python3 python/run_html_tests.py`.

## Directory Role & Agent Invariants

- `buttons.css` enforces uniform button formatting and `text-decoration: none !important;` across `:hover`, `:focus`, `:active`, and `:visited`.
- Default hyperlinks use gold (`var(--gold, #fbbf24)`) transitioning to gold-dim (`#d97706`) on hover.
- Code samples require high-contrast dark surfaces (`#040711` or `#060913`) with crisp borders and vertical separation.

## Subdirectories

Leaf directory (no child directories).

## Detailed File Inventory

| File | Type | Size / Lines | In-Context Details & Cross-References |
| :--- | :---: | :---: | :--- |
| [`article.css`](article.css) | CSS | 14.2 KB / 697L | ==========================================================================; 10 CSS custom vars, 3 `@media` rules; Linked by: `all-business-services.html`, `all-personal-services.html` (+71 more) |
| [`background.css`](background.css) | CSS | 11.6 KB / 383L | ==========================================================================; 2 CSS custom vars, 5 `@media` rules; Linked by: `404.html`, `all-business-services.html` (+87 more) |
| [`business-services.css`](business-services.css) | CSS | 6.9 KB / 265L | ==========================================================================; 17 CSS custom vars, 0 `@media` rules; Linked by: `business-services/business-service-advanced-ai-agents.html`, `business-services/business-service-agentic-assist... |
| [`buttons.css`](buttons.css) | CSS | 17.9 KB / 670L | ==========================================================================; 18 CSS custom vars, 3 `@media` rules; Linked by: `404.html`, `all-business-services.html` (+87 more) |
| [`claude.css`](claude.css) | CSS | 12.3 KB / 577L | ==========================================================================; 15 CSS custom vars, 3 `@media` rules; Linked by: `claude-training/claude-101.html`, `claude-training/claude-apps.html` (+16 more) |
| [`cta.css`](cta.css) | CSS | 3.4 KB / 160L | ==========================================================================; 10 CSS custom vars, 2 `@media` rules; Linked by: `404.html`, `all-business-services.html` (+86 more) |
| [`faq.css`](faq.css) | CSS | 19.9 KB / 757L | ==========================================================================; 19 CSS custom vars, 4 `@media` rules; Linked by: `faqs.html` |
| [`fonts.css`](fonts.css) | CSS | 1.8 KB / 78L | ==========================================================================; 5 CSS custom vars, 0 `@media` rules; Linked by: `404.html`, `all-business-services.html` (+86 more) |
| [`footer.css`](footer.css) | CSS | 12.9 KB / 615L | ==========================================================================; 10 CSS custom vars, 12 `@media` rules; Linked by: `404.html`, `all-business-services.html` (+86 more) |
| [`gemini.css`](gemini.css) | CSS | 5.7 KB / 257L | ==========================================================================; 11 CSS custom vars, 3 `@media` rules |
| [`navbar.css`](navbar.css) | CSS | 11.2 KB / 497L | ==========================================================================; 10 CSS custom vars, 2 `@media` rules; Linked by: `404.html`, `all-business-services.html` (+86 more) |
| [`personal-services.css`](personal-services.css) | CSS | 8.3 KB / 318L | ==========================================================================; 18 CSS custom vars, 0 `@media` rules; Linked by: `business-services/business-service-continuous-learning.html`, `personal-services/personal-service-advanced-ai-a... |
| [`pricing.css`](pricing.css) | CSS | 30.5 KB / 1180L | ==========================================================================; 11 CSS custom vars, 9 `@media` rules; Linked by: `index.html`, `parity-pricing.html` (+7 more) |
| [`responsive.css`](responsive.css) | CSS | 13.9 KB / 587L | ==========================================================================; 5 CSS custom vars, 5 `@media` rules; Linked by: `404.html`, `all-business-services.html` (+85 more) |
| [`service.css`](service.css) | CSS | 11.1 KB / 452L | ==========================================================================; 10 CSS custom vars, 9 `@media` rules; Linked by: `business-services/business-service-advanced-ai-agents.html`, `business-services/business-service-agentic-assist... |
| [`styles.css`](styles.css) | CSS | 104.9 KB / 4757L | ==========================================================================; 30 CSS custom vars, 23 `@media` rules; Linked by: `404.html`, `all-business-services.html` (+87 more) |

## Context & Cross-References

- Root Directory: [`OKF.md`](../OKF.md)
- Canonical Specification: [`AGENTS.md`](../AGENTS.md) (`CLAUDE.md` and `GEMINI.md` symlink to `AGENTS.md`)
- Repository Telemetry: [`METRICS.md`](../METRICS.md)
- Related: [Repository Root](../OKF.md) - Repository Root & Web Portal Hub
- Related: [`claude-training/`](../claude-training/OKF.md) - Claude Architecture & Engineering Guides
- Related: [`gemini-training/`](../gemini-training/OKF.md) - Google Gemini & Gemma Engineering Guides
- Related: [`pricing/`](../pricing/OKF.md) - Purchasing Power Parity (PPP) Regional Tier Pages
