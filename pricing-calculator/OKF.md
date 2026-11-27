---
type: module
title: "Tier-Specific Interactive Pricing Calculator Modules"
description: "Modular JavaScript calculator implementations (`pricing-calculator-tier-1.js` through `pricing-calculator-tier-6.js`) powering the 6 PPP tier landing pages."
tags: [pricing-calculator, javascript-modules, parity-calc]
timestamp: 2026-11-27T06:56:49Z
---

# Tier-Specific Interactive Pricing Calculator Modules

Modular JavaScript calculator implementations (`pricing-calculator-tier-1.js` through `pricing-calculator-tier-6.js`) powering the 6 PPP tier landing pages.

## Live Directory Telemetry

- **Path**: `/pricing-calculator`
- **Direct Files**: 12 (12 JavaScript)
- **Subdirectories**: 0
- **Total Size**: 123.4 KB | **Text/Code Lines**: 3,138
- **Primary Workflow**: Verify all 6 tier calculators via `node pricing-tests.js` (`tests/js/test_tier_calculators.js`).

## Directory Role & Agent Invariants

- Each tier module mirrors the multiplier and seat rules in `data/tiers.json` and `data/pricing.json`.
- Tested by 36 dedicated assertions in `tests/js/test_tier_calculators.js`.

## Subdirectories

Leaf directory (no child directories).

## Detailed File Inventory

| File | Type | Size / Lines | In-Context Details & Cross-References |
| :--- | :---: | :---: | :--- |
| [`pricing-calculator-tier-1-tests.js`](pricing-calculator-tier-1-tests.js) | JavaScript | 4.2 KB / 132L | pricing-calculator-tier-1-tests.js; Functions: `test` |
| [`pricing-calculator-tier-1.js`](pricing-calculator-tier-1.js) | JavaScript | 16.3 KB / 391L | pricing-calculator-tier-1.js; Functions: `calculatePricing`, `initDOMController`, `setClientMode`, `updateAll`; Used by: `pricing/pricing-tier-1-usa-china-eu-100.html` |
| [`pricing-calculator-tier-2-tests.js`](pricing-calculator-tier-2-tests.js) | JavaScript | 4.2 KB / 132L | pricing-calculator-tier-2-tests.js; Functions: `test` |
| [`pricing-calculator-tier-2.js`](pricing-calculator-tier-2.js) | JavaScript | 16.4 KB / 391L | pricing-calculator-tier-2.js; Functions: `calculatePricing`, `initDOMController`, `setClientMode`, `updateAll`; Used by: `pricing/pricing-tier-2-brazil-malaysia-75.html` |
| [`pricing-calculator-tier-3-tests.js`](pricing-calculator-tier-3-tests.js) | JavaScript | 4.2 KB / 132L | pricing-calculator-tier-3-tests.js; Functions: `test` |
| [`pricing-calculator-tier-3.js`](pricing-calculator-tier-3.js) | JavaScript | 16.4 KB / 391L | pricing-calculator-tier-3.js; Functions: `calculatePricing`, `initDOMController`, `setClientMode`, `updateAll`; Used by: `pricing/pricing-tier-3-india-egypt-50.html` |
| [`pricing-calculator-tier-4-tests.js`](pricing-calculator-tier-4-tests.js) | JavaScript | 4.2 KB / 132L | pricing-calculator-tier-4-tests.js; Functions: `test` |
| [`pricing-calculator-tier-4.js`](pricing-calculator-tier-4.js) | JavaScript | 16.4 KB / 391L | pricing-calculator-tier-4.js; Functions: `calculatePricing`, `initDOMController`, `setClientMode`, `updateAll`; Used by: `pricing/pricing-tier-4-tech-startups-50.html` |
| [`pricing-calculator-tier-5-tests.js`](pricing-calculator-tier-5-tests.js) | JavaScript | 4.2 KB / 132L | pricing-calculator-tier-5-tests.js; Functions: `test` |
| [`pricing-calculator-tier-5.js`](pricing-calculator-tier-5.js) | JavaScript | 16.4 KB / 391L | pricing-calculator-tier-5.js; Functions: `calculatePricing`, `initDOMController`, `setClientMode`, `updateAll`; Used by: `pricing/pricing-tier-5-colleges-non-profits-25.html` |
| [`pricing-calculator-tier-6-tests.js`](pricing-calculator-tier-6-tests.js) | JavaScript | 4.2 KB / 132L | pricing-calculator-tier-6-tests.js; Functions: `test` |
| [`pricing-calculator-tier-6.js`](pricing-calculator-tier-6.js) | JavaScript | 16.5 KB / 391L | pricing-calculator-tier-6.js; Functions: `calculatePricing`, `initDOMController`, `setClientMode`, `updateAll`; Used by: `pricing/pricing-tier-6-crisis-affected-10.html` |

## Context & Cross-References

- Root Directory: [`OKF.md`](../OKF.md)
- Canonical Specification: [`AGENTS.md`](../AGENTS.md) (`CLAUDE.md` and `GEMINI.md` symlink to `AGENTS.md`)
- Repository Telemetry: [`METRICS.md`](../METRICS.md)
- Related: [`pricing/`](../pricing/OKF.md) - Purchasing Power Parity (PPP) Regional Tier Pages
- Related: [`data/`](../data/OKF.md) - Authoritative Data Directory (Single Source of Truth)
- Related: [`assets/data/`](../assets/data/OKF.md) - Generated Client-Side Site Data Bundle
- Related: [`tests/js/`](../tests/js/OKF.md) - Modular JavaScript Unit & DOM Test Suite
