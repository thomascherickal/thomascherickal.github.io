---
type: dataset
title: "Global Site Assets, Branding & Generated Data Hub"
description: "Site-wide branding assets (logos, favicons, OG cards, profile portraits), generated client-side JSON (`assets/data/`), self-hosted Mermaid JS (`assets/js/`), and 11 topic-specific visual subfolders."
tags: [assets, branding, media, favicons, opengraph]
timestamp: 2026-11-27T06:56:49Z
---

# Global Site Assets, Branding & Generated Data Hub

Site-wide branding assets (logos, favicons, OG cards, profile portraits), generated client-side JSON (`assets/data/`), self-hosted Mermaid JS (`assets/js/`), and 11 topic-specific visual subfolders.

## Live Directory Telemetry

- **Path**: `/assets`
- **Direct Files**: 15 (7 PNG, 4 WEBP, 1 ICO, 1 JPEG, 1 JPG, 1 SVG)
- **Subdirectories**: 12
- **Total Size**: 1.12 MB | **Text/Code Lines**: 4
- **Primary Workflow**: Compress every raster image strictly under 400 KB and delete larger originals. Regenerate favicons via `python3 python/generate_favicons.py` and OG card via `python3 python/sync_og_card.py`.

## Directory Role & Agent Invariants

- Every image asset in `assets/` must be strictly under 400 KB.
- Organize new article or service visuals into dedicated topic subfolders under `assets/`.
- `assets/og-card.png` stays synchronized with `index.html` headline stats via `python3 python/sync_og_card.py`.
- Distinct from `images/`: `assets/` holds branding, favicons, OG cards, generated site data, and service/guide diagrams; `images/` holds standalone WebP editorial illustrations for `articles.html`.

## Subdirectories

| Subdirectory | Files | Role & Scope |
| :--- | :---: | :--- |
| [`advanced-ai-agents/`](advanced-ai-agents/OKF.md) | 0 | Compressed visual schematics and diagrams supporting the Advanced AI Agents business and personal service pages. |
| [`agentic-ai-systems/`](agentic-ai-systems/OKF.md) | 0 | Compressed architecture visuals supporting Agentic Assistants and persistent-memory agentic systems service pages. |
| [`ai-agents-orchestration/`](ai-agents-orchestration/OKF.md) | 0 | Compressed multi-agent topology diagrams and visual assets for orchestration service pages. |
| [`ai-developer-training/`](ai-developer-training/OKF.md) | 3 | Compressed workflow diagrams and illustrations for AI Developer Training business and personal service pages. |
| [`antigravity/`](antigravity/OKF.md) | 5 | Compressed visuals and architecture illustrations for Google Antigravity IDE and Chat documentation guides. |
| [`claude-orchestration/`](claude-orchestration/OKF.md) | 0 | Compressed architecture visuals for Claude agent orchestration guides and service tracks. |
| [`data/`](data/OKF.md) | 1 | Auto-generated JSON data exported from `data/*.json` by `python/ssot/export_site_data.py` for browser-side JavaScript consumption. |
| [`generative-ai-transformation/`](generative-ai-transformation/OKF.md) | 0 | Compressed visuals illustrating enterprise Generative AI transformation architectures and capability progressions. |
| [`guarantees/`](guarantees/OKF.md) | 1 | Compressed trust-model illustrations supporting `guarantees.html` ($0 upfront, 10 equal 10% milestone installments, walkaway guarantee, Apache 2.0... |
| [`js/`](js/OKF.md) | 1 | Vendored self-hosted Mermaid JS runtime (`mermaid.min.js`) loaded by root `mermaid.js` to render technical architecture diagrams offline without ex... |
| [`local-ai-revolution/`](local-ai-revolution/OKF.md) | 0 | Compressed visuals for Local LLM cost-slashing, VRAM sizing, and private on-premise inference guides. |
| [`personal-service-local-llms/`](personal-service-local-llms/OKF.md) | 0 | Compressed workstation local-LLM setup diagrams for `personal-services/personal-service-local-llms-cost-slashing.html`. |

## Detailed File Inventory

| File | Type | Size / Lines | In-Context Details & Cross-References |
| :--- | :---: | :---: | :--- |
| [`Book-Cover.webp`](Book-Cover.webp) | WEBP | 48.9 KB | 687x1024px, Referenced by: `writing.html` |
| [`Professional Profile Square.webp`](Professional Profile Square.webp) | WEBP | 111.8 KB | 1175x1144px, Referenced by: unreferenced |
| [`android-chrome-192x192.png`](android-chrome-192x192.png) | PNG | 51.9 KB | 192x192px, Referenced by: unreferenced |
| [`android-chrome-512x512.png`](android-chrome-512x512.png) | PNG | 338.8 KB | 512x512px, Referenced by: unreferenced |
| [`apple-touch-icon.png`](apple-touch-icon.png) | PNG | 45.7 KB | 180x180px, Referenced by: `404.html`, `all-business-services.html`, `all-personal-services.html` (+82 more) |
| [`digital-futurist-logo.jpg`](digital-futurist-logo.jpg) | JPG | 14.9 KB | 256x256px, Referenced by: `404.html`, `all-business-services.html`, `all-personal-services.html` (+87 more) |
| [`favicon-16x16.png`](favicon-16x16.png) | PNG | 705 B | 16x16px, Referenced by: `404.html`, `all-business-services.html`, `all-personal-services.html` (+45 more) |
| [`favicon-32x32.png`](favicon-32x32.png) | PNG | 2.0 KB | 32x32px, Referenced by: `404.html`, `all-business-services.html`, `all-personal-services.html` (+45 more) |
| [`favicon-96x96.png`](favicon-96x96.png) | PNG | 13.8 KB | 96x96px, Referenced by: unreferenced |
| [`favicon.ico`](favicon.ico) | ICO | 6.7 KB | 48x48px, Referenced by: `404.html`, `all-business-services.html`, `all-personal-services.html` (+82 more) |
| [`favicon.svg`](favicon.svg) | SVG | 32.0 KB / 4L | viewBox `0 0 128 128`; Referenced by: `404.html`, `all-business-services.html`, `all-personal-services.html` (+82 more) |
| [`og-card.png`](og-card.png) | PNG | 149.9 KB | 1200x630px, Referenced by: unreferenced |
| [`site-icon.jpeg`](site-icon.jpeg) | JPEG | 152.4 KB | 1024x1024px, Referenced by: unreferenced |
| [`site-icon.webp`](site-icon.webp) | WEBP | 163.9 KB | 1024x1024px, Referenced by: unreferenced |
| [`thomas-avatar.webp`](thomas-avatar.webp) | WEBP | 17.7 KB | 225x225px, Referenced by: unreferenced |

## Context & Cross-References

- Root Directory: [`OKF.md`](../OKF.md)
- Canonical Specification: [`AGENTS.md`](../AGENTS.md) (`CLAUDE.md` and `GEMINI.md` symlink to `AGENTS.md`)
- Repository Telemetry: [`METRICS.md`](../METRICS.md)
- Related: [`assets/data/`](../assets/data/OKF.md) - Generated Client-Side Site Data Bundle
- Related: [`assets/js/`](../assets/js/OKF.md) - Self-Hosted Vendor JavaScript Libraries
- Related: [`images/`](../images/OKF.md) - Editorial & Article WebP Graphics
- Related: [`icons/`](../icons/OKF.md) - Vector SVG Icon & Platform Badge Repository
- Related: [`python/`](../python/OKF.md) - Repository Python Automation & Build Tooling
