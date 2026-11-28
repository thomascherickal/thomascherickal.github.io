---
type: module
title: "Purchasing Power Parity (PPP) Regional Tier Pages"
description: "Six dedicated regional landing pages implementing the 195-country World Bank FY2027 Purchasing Power Parity pricing model (Tiers 1 through 6)."
tags: [pricing, ppp-tiers, global-parity, world-bank-fy2027]
timestamp: 2026-11-28T05:16:04Z
---

# Purchasing Power Parity (PPP) Regional Tier Pages

Six dedicated regional landing pages implementing the 195-country World Bank FY2027 Purchasing Power Parity pricing model (Tiers 1 through 6).

## Live Directory Telemetry

- **Path**: `/pricing`
- **Direct Files**: 6 (6 HTML)
- **Subdirectories**: 0
- **Total Size**: 886.1 KB | **Text/Code Lines**: 13,188
- **Primary Workflow**: Keep rates aligned with `data/pricing.json` and `data/tiers.json`; verify tier calculators and DOM contracts via `python3 python/run_tests.py`.

## Directory Role & Agent Invariants

- Flagship PPP range: $1,000 / $750 / $500 / $500 / $250 / Custom USD; Individual Service PPP range: $100 / $75 / $50 / $50 / $25 / Custom USD.
- Tier precedence rule: when a country appears in both Tier 3 and Tier 6 (UN LDCs), Tier 6 (Relief / Custom) takes precedence.
- Startup tier (Tier 4) follows the standard 2-month flagship program timeline.
- Every tier page clearly documents $0 upfront advance, 10 equal 10% milestone installments, walkaway guarantee, and Zero Charge Training Materials (Apache 2.0).

## Subdirectories

Leaf directory (no child directories).

## Detailed File Inventory

| File | Type | Size / Lines | In-Context Details & Cross-References |
| :--- | :---: | :---: | :--- |
| [`pricing-tier-1-usa-uk-eu-100.html`](pricing-tier-1-usa-uk-eu-100.html) | HTML | 147.1 KB / 2190L | Title: "Tier 1 AI Pricing (USA, EU, UK) / Thomas Cherickal"; JSON-LD: WebPage, BreadcrumbList, ListItem, OfferCatalog |
| [`pricing-tier-2-brazil-malaysia-75.html`](pricing-tier-2-brazil-malaysia-75.html) | HTML | 151.2 KB / 2250L | Title: "Tier 2 PPP Pricing (25% Off) / Thomas Cherickal"; JSON-LD: WebPage, BreadcrumbList, ListItem, OfferCatalog |
| [`pricing-tier-3-india-egypt-50.html`](pricing-tier-3-india-egypt-50.html) | HTML | 146.9 KB / 2187L | Title: "Tier 3 PPP Pricing (India, 50% Off) / Thomas Cherickal"; JSON-LD: WebPage, BreadcrumbList, ListItem, OfferCatalog |
| [`pricing-tier-4-tech-startups-50.html`](pricing-tier-4-tech-startups-50.html) | HTML | 147.0 KB / 2186L | Title: "Tier 4 Startup AI Pricing (50% Off) / Thomas Cherickal"; JSON-LD: WebPage, BreadcrumbList, ListItem, OfferCatalog |
| [`pricing-tier-5-colleges-non-profits-25.html`](pricing-tier-5-colleges-non-profits-25.html) | HTML | 147.0 KB / 2188L | Title: "Tier 5 University &amp; Nonprofit (75% Off) / Thomas Cherickal"; JSON-LD: WebPage, BreadcrumbList, ListItem, OfferCatalog |
| [`pricing-tier-6-crisis-affected-10.html`](pricing-tier-6-crisis-affected-10.html) | HTML | 147.0 KB / 2187L | Title: "Tier 6 Relief &amp; UN LDC Parity Pricing / Thomas Cherickal"; JSON-LD: WebPage, BreadcrumbList, ListItem, OfferCatalog |

## Context & Cross-References

- Root Directory: [`OKF.md`](../OKF.md)
- Canonical Specification: [`AGENTS.md`](../AGENTS.md) (`CLAUDE.md` and `GEMINI.md` symlink to `AGENTS.md`)
- Repository Telemetry: [`METRICS.md`](../METRICS.md)
- Related: [`data/`](../data/OKF.md) - Authoritative Data Directory (Single Source of Truth)
- Related: [`pricing-calculator/`](../pricing-calculator/OKF.md) - Tier-Specific Interactive Pricing Calculator Modules
- Related: [`md-html-sync/pricing/`](../md-html-sync/pricing/OKF.md) - PPP Tier Pages Markdown Mirrors
- Related: [`schema/`](../schema/OKF.md) - Structured Data & Schema.org JSON-LD Definitions
