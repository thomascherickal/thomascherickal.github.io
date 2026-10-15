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

