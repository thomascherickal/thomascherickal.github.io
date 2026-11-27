---
type: module
title: "Self-Hosted Vendor JavaScript Libraries"
description: "Vendored self-hosted Mermaid JS runtime (`mermaid.min.js`) loaded by root `mermaid.js` to render technical architecture diagrams offline without external CDN failures."
tags: [assets, javascript, mermaid, vendored]
timestamp: 2026-11-27T06:56:49Z
---

# Self-Hosted Vendor JavaScript Libraries

Vendored self-hosted Mermaid JS runtime (`mermaid.min.js`) loaded by root `mermaid.js` to render technical architecture diagrams offline without external CDN failures.

## Live Directory Telemetry

- **Path**: `/assets/js`
- **Direct Files**: 1 (1 JS (Vendor))
- **Subdirectories**: 0
- **Total Size**: 3.41 MB | **Text/Code Lines**: 3,636
- **Primary Workflow**: Loaded dynamically by root `mermaid.js`; verify rendering behavior via `node pricing-tests.js` (`test_mermaid.js`).

## Directory Role & Agent Invariants

- Hosts the self-hosted `mermaid.min.js` bundle referenced by `mermaid.js` across training and service pages.
- Do not add third-party tracking scripts or external CDN dependencies here.

## Subdirectories

Leaf directory (no child directories).

## Detailed File Inventory

| File | Type | Size / Lines | In-Context Details & Cross-References |
| :--- | :---: | :---: | :--- |
| [`mermaid.min.js`](mermaid.min.js) | JS (Vendor) | 3.41 MB | Minified vendor bundle; Referenced by: `business-services/business-service-ai-second-brain.html`, `guarantees.html`, `mermaid.html` |

## Context & Cross-References

- Parent Directory: [`assets/OKF.md`](../OKF.md)
- Root Directory: [`OKF.md`](../../OKF.md)
- Canonical Specification: [`AGENTS.md`](../../AGENTS.md) (`CLAUDE.md` and `GEMINI.md` symlink to `AGENTS.md`)
- Repository Telemetry: [`METRICS.md`](../../METRICS.md)
- Related: [Repository Root](../../OKF.md) - Repository Root & Web Portal Hub
- Related: [`claude-training/`](../../claude-training/OKF.md) - Claude Architecture & Engineering Guides
- Related: [`gemini-training/`](../../gemini-training/OKF.md) - Google Gemini & Gemma Engineering Guides
- Related: [`tests/js/`](../../tests/js/OKF.md) - Modular JavaScript Unit & DOM Test Suite
