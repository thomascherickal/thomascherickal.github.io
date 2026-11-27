---
type: module
title: "Repository Root & Web Portal Hub"
description: "Static site root housing core portal HTML pages, master header/footer component templates, client-side pricing calculators, and repository governance documents."
tags: [root, portal, entry-points, architecture, ssot]
timestamp: 2026-11-27T06:56:49Z
---

# Repository Root & Web Portal Hub

Static site root housing core portal HTML pages, master header/footer component templates, client-side pricing calculators, and repository governance documents.

## Live Directory Telemetry

- **Path**: `/` (Repository Root)
- **Direct Files**: 37 (16 HTML, 9 Markdown, 5 JavaScript, 3 Text, 1 Config, 1 Python, 1 WEBMANIFEST, 1 XML)
- **Subdirectories**: 20
- **Total Size**: 1.89 MB | **Text/Code Lines**: 27,204
- **Primary Workflow**: Edit facts in `data/*.json` -> run `python3 python/ssot/validate.py && python3 python/ssot/render.py && python3 python/ssot/export_site_data.py`. After editing `header.html` or `footer.html`, immediately run `python3 python/build_components.py`.

## Directory Role & Agent Invariants

- `AGENTS.md` is the canonical project specification (`CLAUDE.md` and `GEMINI.md` are symlinks to `AGENTS.md`).
- `header.html` and `footer.html` are the single centralized navigation and footer sources inlined across all HTML pages via `python3 python/build_components.py`.
- Never type protected commercial facts outside `<!--ssot:...-->` markers; source them from `data/*.json`.
- Flagship offerings: $1,000 / $750 / $500 / $500 / $250 / Custom USD (2 calendar months (includes weekends as holidays), up to 10 participants (+$100 per extra seat, max 20), 10 equal installments of 10%, $0 upfront, 2 flagship cohorts per quarter).
- Individual service picks: $100 / $75 / $50 / $50 / $25 / Custom USD (10 calendar days (includes weekends as holidays), up to 10 participants (+$100 per extra seat, max 20)). Canonical contact reply SLA: within one week.
- Maintain zero emojis, zero em-dashes, gold default hyperlinks, and non-underlined buttons across all files.

## Subdirectories

| Subdirectory | Files | Role & Scope |
| :--- | :---: | :--- |
| [`article-footers/`](article-footers/OKF.md) | 4 | Reusable syndication footer templates (dark HTML, Markdown, and Word `.docx`) appended to external articles on Dev.to, Medium, Hashnode, Substack,... |
| [`assets/`](assets/OKF.md) | 15 | Site-wide branding assets (logos, favicons, OG cards, profile portraits), generated client-side JSON (`assets/data/`), self-hosted Mermaid JS (`ass... |
| [`business-services/`](business-services/OKF.md) | 16 | Dedicated specification pages for all 16 enterprise AI training services, including architecture blueprints, hands-on curricula, deliverables, and... |
| [`claude-training/`](claude-training/OKF.md) | 18 | In-depth technical engineering guides covering the Anthropic Claude ecosystem: Claude Code, Code Mode, Plugins, Apps (MCP Apps), Desktop, MCP, Proj... |
| [`css/`](css/OKF.md) | 16 | Shared design-system stylesheets defining color tokens (`--gold`, `--cyan`), uniform button components (`buttons.css`), responsive tables, code win... |
| [`data/`](data/OKF.md) | 12 | The sole authoritative JSON data store (`data/*.json`) mastering pricing, parity tiers, business/personal services, site config, contact SLA, guara... |
| [`docs/`](docs/OKF.md) | 6 | Reference playbooks for delivery economics, 195-country World Bank PPP tiers, service catalog summaries, historical fixes, and the generated `ssot-... |
| [`failed-tests/`](failed-tests/OKF.md) | 1 | Centralized holding directory where Python (`run_tests.py`), HTML UI (`run_html_tests.py`), and JavaScript (`pricing-tests.js`) runners write Markd... |
| [`gemini-training/`](gemini-training/OKF.md) | 18 | Technical architecture guides covering Google AI Studio, Google Antigravity IDE & Chat, Gemini Deep Research, DeepMind, NotebookLM, Skills, Spark,... |
| [`icons/`](icons/OKF.md) | 42 | Scalable SVG icons for publishing platforms, coding profiles, social channels, and UI badges referenced across the site and syndicated article foot... |
| [`images/`](images/OKF.md) | 14 | Standalone WebP technical illustrations featured on `articles.html` and publication listings. |
| [`internal-use/`](internal-use/OKF.md) | 2 | Private internal trainer documentation, delivery sprint protocols, and full-catalog teaching preparation materials (excluded from public navigation... |
| [`md-html-sync/`](md-html-sync/OKF.md) | 12 | Synchronized Markdown mirrors of live HTML pages (plus `.sync-state.json` hash cache) optimized for coding agents, LLM ingestion, and clean text di... |
| [`personal-services/`](personal-services/OKF.md) | 16 | Dedicated specification pages for all 16 personal AI acceleration services, covering architecture blueprints, hands-on projects, and PPP-calibrated... |
| [`pricing/`](pricing/OKF.md) | 6 | Six dedicated regional landing pages implementing the 195-country World Bank FY2027 Purchasing Power Parity pricing model (Tiers 1 through 6). |
| [`pricing-calculator/`](pricing-calculator/OKF.md) | 12 | Modular JavaScript calculator implementations (`pricing-calculator-tier-1.js` through `pricing-calculator-tier-6.js`) powering the 6 PPP tier landi... |
| [`python/`](python/OKF.md) | 21 | Core Python 3 automation suite for header/footer component inlining, SSOT validation/rendering (`python/ssot/`), link checking, JSON-LD schema extr... |
| [`schema/`](schema/OKF.md) | 86 | Standalone JSON-LD Schema.org graphs extracted from live HTML pages to support search engines, rich snippets, and Generative Engine Optimization (G... |
| [`templates/`](templates/OKF.md) | 0 | Root directory for reusable Jinja2 HTML templates rendered into `<!--ssot:block.<name>-->` markers by `python/ssot/render.py`. |
| [`tests/`](tests/OKF.md) | 10 | Python unittest modules covering the SSOT build pipeline, pricing/policy invariants, 101 HTML UI checks across all 85+ pages, internal link integri... |

## Detailed File Inventory

| File | Type | Size / Lines | In-Context Details & Cross-References |
| :--- | :---: | :---: | :--- |
| [`404.html`](404.html) | HTML | 53.4 KB / 595L | Title: "Page Not Found / Thomas Cherickal"; JSON-LD: WebPage, BreadcrumbList, ListItem |
| [`AGENTS.md`](AGENTS.md) | Markdown | 30.9 KB / 247L | Heading: "Project Automation & Agent Rules"; 34 H2 section(s), 4,292 words |
| [`CLAUDE.md`](CLAUDE.md) | Markdown | 30.9 KB / 247L | Heading: "Project Automation & Agent Rules"; 34 H2 section(s), 4,292 words; Symlink -> `AGENTS.md` |
| [`CNAME`](CNAME) | Config | 22 B / 1L | GitHub Pages custom domain: `thedigitalfuturist.com` |
| [`GEMINI.md`](GEMINI.md) | Markdown | 30.9 KB / 247L | Heading: "Project Automation & Agent Rules"; 34 H2 section(s), 4,292 words; Symlink -> `AGENTS.md` |
| [`LAST-COMMIT-REPORT.md`](LAST-COMMIT-REPORT.md) | Markdown | 78.6 KB / 612L | Heading: "Pre-Commit Report"; 10 H2 section(s), 11,414 words |
| [`LICENSE`](LICENSE) | Text | 2.0 KB / 38L | Text file (38 lines) |
| [`METRICS.md`](METRICS.md) | Markdown | 10.9 KB / 160L | Heading: "Repository Code & Engineering Metrics"; 9 H2 section(s), 1,702 words |
| [`README.md`](README.md) | Markdown | 31.2 KB / 444L | Heading: "The Digital Futurist - Thomas Cherickal"; 10 H2 section(s), 3,817 words |
| [`TECH-STACK.md`](TECH-STACK.md) | Markdown | 20.8 KB / 379L | Heading: "Technical Architecture & Systems Engineering"; 8 H2 section(s), 2,438 words |
| [`all-business-services.html`](all-business-services.html) | HTML | 71.3 KB / 860L | Title: "All Business Services - Business AI Transformation / Thomas Cherickal"; JSON-LD: WebPage, BreadcrumbList, ListItem |
| [`all-personal-services.html`](all-personal-services.html) | HTML | 71.1 KB / 859L | Title: "All Personal Services - AI Mentorship &amp; Training / Thomas Cherickal"; JSON-LD: WebPage, BreadcrumbList, ListItem |
| [`articles.html`](articles.html) | HTML | 133.0 KB / 1788L | Title: "Featured Articles / Thomas Cherickal"; JSON-LD: CollectionPage, BreadcrumbList, ListItem, ItemList |
| [`components.js`](components.js) | JavaScript | 20.0 KB / 625L | (function () {; Functions: `syncActiveNav`, `bindNavEvents`, `closeAllDropdowns`, `initArticlesCarousel`, `getCardsPerView` (+15); Used by: `404.html`, `all-business-services.html` (+85 more) |
| [`contact.html`](contact.html) | HTML | 93.0 KB / 1332L | Title: "Contact / Thomas Cherickal"; 3 SSOT marker(s); JSON-LD: ContactPage, BreadcrumbList, ListItem, ProfessionalService |
| [`expertise.html`](expertise.html) | HTML | 78.6 KB / 1099L | Title: "Generative AI Consultant &amp; Tech Stack / Thomas Cherickal"; JSON-LD: AboutPage, BreadcrumbList, ListItem, FAQPage |
| [`faqs.html`](faqs.html) | HTML | 300.9 KB / 4380L | Title: "Frequently Asked Questions / Thomas Cherickal"; 20 SSOT marker(s); JSON-LD: FAQPage, Question, Answer |
| [`footer.html`](footer.html) | HTML | 38.9 KB / 366L | Master component template inlined by `python/build_components.py` |
| [`generate_section.py`](generate_section.py) | Python | 108 B / 4L | Python module |
| [`google43d90cf6a204dc6d.html`](google43d90cf6a204dc6d.html) | HTML | 75 B / 1L | HTML document |
| [`guarantees.html`](guarantees.html) | HTML | 74.3 KB / 996L | Title: "Trust Guarantees & Payment Structure / Thomas Cherickal"; 10 SSOT marker(s); JSON-LD: WebSite, Person, BreadcrumbList, ListItem |
| [`header.html`](header.html) | HTML | 20.9 KB / 492L | Master component template inlined by `python/build_components.py` |
| [`home.md`](home.md) | Markdown | 13.0 KB / 166L | Heading: "Thomas Cherickal"; 11 H2 section(s), 1,411 words |
| [`index.html`](index.html) | HTML | 255.9 KB / 4128L | Title: "Thomas Cherickal / Generative AI Consultant"; 5 SSOT marker(s); JSON-LD: WebSite, Organization, ImageObject, ProfilePage |
| [`llms.txt`](llms.txt) | Text | 35.4 KB / 336L | LLM-ready site index; 0 SSOT marker(s) |
| [`mermaid.html`](mermaid.html) | HTML | 53.4 KB / 631L | Title: "Article Viewer - The Digital Futurist" |
| [`mermaid.js`](mermaid.js) | JavaScript | 4.8 KB / 135L | Load Mermaid; Functions: `transformMdLinks`, `initMermaid` |
| [`parity-pricing.html`](parity-pricing.html) | HTML | 62.6 KB / 710L | Title: "Country-Wise Parity Pricing (195 Countries) / The Digital Futurist"; JSON-LD: WebPage, Person, Organization, BreadcrumbList |
| [`parity-pricing.md`](parity-pricing.md) | Markdown | 4.2 KB / 49L | Heading: "Country-Wise Parity Pricing (195 Countries)"; 7 H2 section(s), 591 words |
| [`payment-schedule.js`](payment-schedule.js) | JavaScript | 10.5 KB / 232L | payment-schedule.js; Functions: `formatCurrency`, `calculateSchedule`, `initDOMController`, `renderSchedule`; Used by: `pricing.html` |
| [`pricing-calculator.js`](pricing-calculator.js) | JavaScript | 32.2 KB / 760L | pricing-calculator.js; Functions: `calculatePricing`, `initDOMController`, `updateFromServicesSelection`, `setClientMode`, `updateIndividualServiceCards` (+2); Used by: `pricing.html` |
| [`pricing-tests.js`](pricing-tests.js) | JavaScript | 1.5 KB / 58L | pricing-tests.js; Functions: `runTestSuite`; Used by: `internal-use/trainer-guide.md` |
| [`pricing.html`](pricing.html) | HTML | 125.8 KB / 1741L | Title: "Consulting & Training Pricing / Thomas Cherickal"; JSON-LD: WebPage, BreadcrumbList, ListItem, OfferCatalog |
| [`robots.txt`](robots.txt) | Text | 126 B / 6L | Crawler directives and sitemap pointer |
| [`site.webmanifest`](site.webmanifest) | WEBMANIFEST | 538 B / 21L | Text file (21 lines) |
| [`sitemap.xml`](sitemap.xml) | XML | 11.9 KB / 346L | XML sitemap with 83 `<loc>` entries |
| [`writing.html`](writing.html) | HTML | 126.6 KB / 2113L | Title: "Publications / Thomas Cherickal"; JSON-LD: CollectionPage, BreadcrumbList, ListItem, Book |

## Context & Cross-References

- Canonical Specification: [`AGENTS.md`](AGENTS.md) (`CLAUDE.md` and `GEMINI.md` symlink to `AGENTS.md`)
- Repository Telemetry: [`METRICS.md`](METRICS.md)
- Related: [`data/`](data/OKF.md) - Authoritative Data Directory (Single Source of Truth)
- Related: [`python/ssot/`](python/ssot/OKF.md) - Deterministic SSOT Build & Synchronization Pipeline
- Related: [`python/`](python/OKF.md) - Repository Python Automation & Build Tooling
- Related: [`css/`](css/OKF.md) - Global Stylesheets & Design System
- Related: [`schema/`](schema/OKF.md) - Structured Data & Schema.org JSON-LD Definitions
- Related: [`md-html-sync/`](md-html-sync/OKF.md) - Markdown HTML Synchronization Root
- Related: [`tests/`](tests/OKF.md) - Unified Test Suite & Verification Harness
