# Project Automation & Agent Rules

## Component Inlining (`header.html` & `footer.html`)
Whenever `header.html` or `footer.html` is modified, created, or updated:
- **ALWAYS automatically run** `python3 python/build_components.py` immediately in the same turn.
- Do not wait for the user to request it.
- This ensures all 20+ standalone HTML pages stay synchronized with the latest navigation bar, footer elements, styles, and Markdown sync files (`md-html-sync/`).

## Navigation Architecture & No New Navbars
- **DO NOT create or add new navbars** (`<nav>`, sub-navbars, in-page guide/topic navigation strips, secondary navbars) when creating or updating content or pages.
- The website relies strictly and exclusively on the centralized primary navigation bar defined in `header.html` (which is inlined across all pages via `python3 python/build_components.py`).
- Navigation to new content must only be handled via:
  1. The existing global navigation dropdowns or links in `header.html`.
  2. Standard in-content hyperlinks (cards, lists, or buttons) within the page body.
- Never add bespoke `<nav>` elements, sticky topic strips, or secondary horizontal/vertical navigation bars to content pages.

## Canonical Profile Categories & Business Website
- **Business Website**: `https://thomascherickal.github.io`
- Always categorize and list `https://thomascherickal.github.io` under the **Business Website** category across all profile registries, documentation, schemas, and links.

## Canonical Site & Profile Description
- **Canonical Description**: `"description": "Thomas Cherickal — Generative AI Consultant and The Digital Futurist."`
- Always use `"Thomas Cherickal — Generative AI Consultant and The Digital Futurist."` as the primary site, author, and profile description across all HTML files, metadata, and schemas.

## Call to Action (CTA) & Free LinkedIn Consultation Rule
- **ALWAYS include the Free LinkedIn Consultation** link/button in **every Call to Action (CTA)** block, banner, or section created or updated across any page:
  `<a class="btn btn-linkedin" href="https://linkedin.com/in/thomascherickal" rel="noopener" target="_blank">Free LinkedIn Consultation ↗</a>`
- Ensure it is prominently featured alongside other primary actions (such as `collaboration.html` or `pricing.html`).

## Claude Model Version Numbers
- **DO NOT include version numbers for Claude models** (e.g., avoid Claude 3, Claude 3.5, Claude 3.7, Claude 4, Claude 4.5, Claude 4.6, Claude 4.7, Claude 5, Claude 5.5, Sonnet 3.5/4/5.5, Opus 4/4.6/4.7/5.5, Haiku 3.5/4.5/5.5, Fable 5.1, Mythos 5.1, etc.) anywhere across general website prose, documentation, or schemas.
- Refer to Claude models exclusively by their model family names without version numbers: **Claude**, **Claude Opus** (or **Opus**), **Claude Sonnet** (or **Sonnet**), **Claude Haiku** (or **Haiku**), **Claude Fable** (or **Fable**), and **Claude Mythos** (or **Mythos**).
- When describing model capabilities or architectures, use evergreen descriptions (e.g., "frontier Claude models", "Claude Opus lead with Claude Sonnet subagents", "the Claude model family").
- **STRICT EXCEPTIONS (Model version numbers are permitted ONLY in these two cases)**:
  1. **Endpoints and Model Strings in Code Examples**: Do NOT change model numbers in API endpoints, parameters, or model identifiers within code blocks, script samples, configuration snippets, or test payloads (e.g., `model="claude-opus-5-5"`, `MODEL = "claude-sonnet-5-5"`, `claude-haiku-5-5`, `anthropic/claude-sonnet-5-5`).
  2. **Pricing Mentions**: When quoting specific pricing, token rates, cost calculations, pricing tables, or break-even analyses, always include the specific model version numbers tied to those prices (e.g., "Opus 5.5 ($4 / $20 per MTok)", "Sonnet 5.5 at $2/$10", "Haiku 5.5 at $0.10/$0.50", "Fable 5.1 at $10/$50").

## Gemini Model Version Numbers
- **DO NOT include version numbers for Gemini models** (e.g., avoid Gemini 1.0, Gemini 1.5, Gemini 1.5 Pro, Gemini 1.5 Flash, Gemini 2.0, Gemini 2.5, Gemini 2.5 Pro, Gemini 2.5 Flash, Gemini 3, Gemini 3.0, Gemini 4, Gemini 4.0, Gemini 4-Argon, etc.) anywhere across general website prose, documentation, or schemas.
- Refer to Gemini models exclusively by their model family names without version numbers: **Gemini**, **Gemini Pro** (or **Pro**), **Gemini Flash** (or **Flash**), **Gemini Ultra** (or **Ultra**), **Gemini Flash-Lite** (or **Flash-Lite**), **Gemini Nano** (or **Nano**), **Gemini Deep Think**, and **Gemini Argon** (or **Argon**).
- When describing model capabilities or architectures, use evergreen descriptions (e.g., "frontier Gemini models", "Gemini Pro strategic planner with Gemini Flash subagents", "the Gemini model family").
- **STRICT EXCEPTIONS (Model version numbers are permitted ONLY in these two cases)**:
  1. **Endpoints and Model Strings in Code Examples**: Do NOT change model numbers in API endpoints, parameters, or model identifiers within code blocks, script samples, configuration snippets, or test payloads (e.g., `model="gemini-2.5-pro"`, `MODEL = "gemini-2.5-flash"`, `gemini-1.5-pro`, `gemini-4-argon`).
  2. **Pricing Mentions**: When quoting specific pricing, token rates, cost calculations, pricing tables, or break-even analyses, always include the specific model version numbers tied to those prices (e.g., "Gemini 2.5 Pro ($1.25 / $5 per MTok)", "Gemini 2.5 Flash at $0.075/$0.30").

## No Emojis Rule
- **DO NOT use emojis** in any further generation of any file in the future in this repo.
- Ensure all generated or modified files (including HTML, CSS, JavaScript, Markdown, Python, and configuration files) remain completely free of emojis.
## Canonical Email ID
- **Canonical Email ID**: `thomascherickal@gmail.com`
- **ALWAYS** save, render, and use `thomascherickal@gmail.com` as the primary contact email address across all files, footer links, mailto references, documentation, schemas, and configurations.
