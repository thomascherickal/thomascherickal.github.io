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
- **Business Website**: `https://thedigitalfuturist.com`
- Always categorize and list `https://thedigitalfuturist.com` under the **Business Website** category across all profile registries, documentation, schemas, and links.

## Canonical Site & Profile Description
- **Canonical Description**: `"description": "Thomas Cherickal - Generative AI Consultant and The Digital Futurist."`
- Always use `"Thomas Cherickal - Generative AI Consultant and The Digital Futurist."` as the primary site, author, and profile description across all HTML files, metadata, and schemas.

## Call to Action (CTA) & Free LinkedIn Consultation Rule
- **ALWAYS include the Free LinkedIn Consultation** link/button in **every Call to Action (CTA)** block, banner, or section created or updated across any page:
  `<a class="btn btn-linkedin" href="https://linkedin.com/in/thomascherickal" rel="noopener" target="_blank">Free LinkedIn Consultation ↗</a>`
- Ensure it is prominently featured alongside other primary actions (such as `pricing.html` or `contact.html`).

## Model Version Numbers Rule (Claude, Gemini, and all LLMs)
- **DO NOT include version numbers for any LLM models** (e.g., avoid Claude 3, Claude 3.5, Gemini 1.5, Gemini 2.5, Llama 3, Qwen 2.5, DeepSeek V3, etc.) anywhere across general website prose, documentation, or schemas.
- Refer to models exclusively by their model family names without version numbers (e.g. Claude, Claude Opus, Claude Sonnet, Claude Haiku, Gemini, Gemini Pro, Gemini Flash, Gemini Argon, Qwen, DeepSeek, Mistral, Gemma).
- **STRICT EXCEPTIONS (Model version numbers are permitted ONLY in these two cases)**:
  1. **Endpoints and Model Strings in Code Examples**: Do NOT change model numbers in API endpoints, parameters, or model identifiers within code blocks, script samples, configuration snippets, or test payloads.
  2. **Pricing Mentions**: When quoting specific pricing, token rates, cost calculations, pricing tables, or break-even analyses, always include the specific model version numbers tied to those prices.

## No Emojis Rule
- **DO NOT use emojis** in any further generation of any file in the future in this repo.
- Ensure all generated or modified files (including HTML, CSS, JavaScript, Markdown, Python, and configuration files) remain completely free of emojis.

## No Em-Dash Rule
- **DO NOT use em-dashes** (the em-dash character U+2014 or the &amp;mdash; / &amp;#8212; entities) anywhere in any file in this repo.
- Use a spaced hyphen ` - ` in prose, headings, and titles.
- En-dashes in numeric ranges (e.g., `Days 1-15`, `2020-2026`) must also be written as plain hyphens `-`.

## Platform Rules
- Dev.to is an allowed platform. Include `https://dev.to/thomascherickal` in platform lists, `sameAs` arrays, and footers wherever the other publishing platforms appear.
- Never remove existing Dev.to links.
- Mastodon is excluded.

## Article Footers
- Keep `article-footer.md`, `article-footer_dark.html`, and `article-footer-plain.docx` in the repo; do not delete or move them.
- Never insert an article footer into any Claude article (`claude-*.html` or `md-html-sync/claude-*.md`).

## Canonical Email ID
- **Canonical Email ID**: `thomascherickal@gmail.com`
- **ALWAYS** save, render, and use `thomascherickal@gmail.com` as the primary contact email address across all files, footer links, mailto references, documentation, schemas, and configurations.

## HTML to JSON Schema Synchronization (`schema/`)
- **Run `python3 python/sync_schemas.py` only when committing**, not after every change.
- Live HTML files are the definitive source of truth for all structured data and schema markup.
- `python/sync_schemas.py` extracts embedded JSON-LD schemas from all HTML pages and synchronizes the copy into the corresponding JSON file under the `schema/` directory (e.g., `index.html` -> `schema/home-schema.json`, `service-*.html` -> `schema/service-*.json`, and other pages -> `schema/<page>-schema.json`).
- Ensure all updated schema files in `schema/` are staged and included in the commit.

## Markdown Synchronization (`md-html-sync/`)
- **Run `python3 python/md_html_sync.py` only when committing**, not after every change.
- Keeps Markdown files in `md-html-sync/` synchronized with their corresponding HTML pages.
- When full Markdown generation is required across all pages, run `python3 python/build_md_files.py`.

## Link Verification (`python/check_links.py`)
- **Run `python3 python/check_links.py` only before every commit**, not after every individual edit.
- If any broken links or missing anchor targets are detected, fix them immediately before finalizing and committing.

