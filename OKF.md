---
type: module
title: "Repository Root & Web Portal Hub"
description: "Root entry point containing core portal pages, global navigation inliners, pricing calculators, and master architectural specifications."
tags: [root, portal, entry-points, architecture, ssot]
timestamp: 2026-11-10T08:43:02Z
---

# Repository Root & Web Portal Hub

Root entry point containing core portal pages, global navigation inliners, pricing calculators, and master architectural specifications.

## Directory Role & Agent Guidelines

- Houses core landing pages (`index.html`, `business-ai-training.html`, `personal-ai-training.html`, `pricing.html`, `all-business-services.html`, `all-personal-services.html`).
- Master component sources `header.html` and `footer.html` live here. Modifying either requires immediately running `python3 python/build_components.py` to inline across all HTML pages.
- `GEMINI.md` is the ultimate authoritative project specification and single source of truth (SSOT).
- `METRICS.md` tracks codebase telemetry and is automatically refreshed by `python3 python/generate_metrics.py`.
- Strictly enforce zero emojis and zero em-dashes across all files. Buttons must never be underlined.

## Subdirectories

- [`article-footers/`](article-footers/OKF.md): Subdirectory knowledge bundle
- [`assets/`](assets/OKF.md): Subdirectory knowledge bundle
- [`business-services/`](business-services/OKF.md): Subdirectory knowledge bundle
- [`claude-training/`](claude-training/OKF.md): Subdirectory knowledge bundle
- [`css/`](css/OKF.md): Subdirectory knowledge bundle
- [`data/`](data/OKF.md): Subdirectory knowledge bundle
- [`docs/`](docs/OKF.md): Subdirectory knowledge bundle
- [`google-training/`](google-training/OKF.md): Subdirectory knowledge bundle
- [`icons/`](icons/OKF.md): Subdirectory knowledge bundle
- [`images/`](images/OKF.md): Subdirectory knowledge bundle
- [`internal-use/`](internal-use/OKF.md): Subdirectory knowledge bundle
- [`md-html-sync/`](md-html-sync/OKF.md): Subdirectory knowledge bundle
- [`nlp-fuzzy-expert/`](nlp-fuzzy-expert/OKF.md): Subdirectory knowledge bundle
- [`personal-services/`](personal-services/OKF.md): Subdirectory knowledge bundle
- [`pricing/`](pricing/OKF.md): Subdirectory knowledge bundle
- [`pricing-calculator/`](pricing-calculator/OKF.md): Subdirectory knowledge bundle
- [`python/`](python/OKF.md): Subdirectory knowledge bundle
- [`schema/`](schema/OKF.md): Subdirectory knowledge bundle
- [`tests/`](tests/OKF.md): Subdirectory knowledge bundle

## File Inventory

- [`.coverage`](.coverage)
- [`.gitignore`](.gitignore)
- [`.nojekyll`](.nojekyll)
- [`404.html`](404.html)
- [`AGENTS.md`](AGENTS.md)
- [`CNAME`](CNAME)
- [`GEMINI.md`](GEMINI.md)
- [`LAST-COMMIT-REPORT.md`](LAST-COMMIT-REPORT.md)
- [`LICENSE`](LICENSE)
- [`METRICS.md`](METRICS.md)
- [`OPTIMIZATION.md`](OPTIMIZATION.md)
- [`README.md`](README.md)
- [`SYNC_REPORT.md`](SYNC_REPORT.md)
- [`TECH-STACK.md`](TECH-STACK.md)
- [`all-business-services.html`](all-business-services.html)
- [`all-personal-services.html`](all-personal-services.html)
- [`articles.html`](articles.html)
- [`business-ai-training.html`](business-ai-training.html)
- [`components.js`](components.js)
- [`contact.html`](contact.html)
- [`expertise.html`](expertise.html)
- [`faqs.html`](faqs.html)
- [`footer.html`](footer.html)
- [`generate_section.py`](generate_section.py)
- [`google43d90cf6a204dc6d.html`](google43d90cf6a204dc6d.html)
- [`guarantees.html`](guarantees.html)
- [`header.html`](header.html)
- [`home.md`](home.md)
- [`index.html`](index.html)
- [`llms.txt`](llms.txt)
- [`mermaid.html`](mermaid.html)
- [`mermaid.js`](mermaid.js)
- [`payment-schedule.js`](payment-schedule.js)
- [`personal-ai-training.html`](personal-ai-training.html)
- [`pricing-calculator.js`](pricing-calculator.js)
- [`pricing-tests.js`](pricing-tests.js)
- [`pricing.html`](pricing.html)
- [`robots.txt`](robots.txt)
- [`site.webmanifest`](site.webmanifest)
- [`sitemap.xml`](sitemap.xml)
- ... and 1 additional files.

## Context & Cross-References

- Architectural Specification: [`GEMINI.md`](/GEMINI.md)
- Repository Telemetry: [`METRICS.md`](/METRICS.md)
