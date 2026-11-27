#!/usr/bin/env python3
"""
generate_okf_files.py - Intelligent Open Knowledge Format (OKF.md) generator.

Introspects every real repository directory and file (HTML, Python AST, JS, JSON,
JSON-LD, JSON Schema, Markdown, CSS, SVG, raster images, CSV, XML, shell hooks)
plus cross-file asset/page references to emit high-context OKF.md knowledge bundles
with zero filler text, zero emojis, and zero em-dashes.
"""

import ast
import csv
import json
import os
import re
import sys
from datetime import datetime, timezone

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from python.ssot.load import SSOTData
from python.ssot.format import tier_price_list, seats_phrase, duration_phrase, installment_phrase, reply_sla_phrase, quarterly_cap_phrase

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".ico"}
TEXT_EXTS = {
    ".html", ".htm", ".py", ".js", ".mjs", ".json", ".md", ".css",
    ".txt", ".xml", ".csv", ".j2", ".svg", ".yml", ".yaml", ".sh",
}
SKIP_DIRS = {".git", ".vscode", ".venv", "__pycache__", "node_modules", ".pytest_cache"}


def sanitize_text(text: str, max_len: int = 180) -> str:
    """Normalizes whitespace, removes em-dashes/en-dashes/pipes/newlines for clean Markdown."""
    if not text:
        return ""
    s = str(text)
    s = s.replace("\u2014", " - ").replace("\u2013", "-")
    s = s.replace("\r", " ").replace("\n", " ").replace("|", "/")
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_len:
        s = s[: max_len - 3].rstrip() + "..."
    return s


def fmt_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    kb = num_bytes / 1024.0
    if kb < 1024:
        return f"{kb:.1f} KB"
    return f"{kb / 1024.0:.2f} MB"


def build_directory_metadata(ssot: SSOTData) -> dict:
    """Builds authoritative, dynamically populated directory metadata from SSOT data."""
    flag_prices = tier_price_list(ssot, "flagship-business-training")
    pick_prices = tier_price_list(ssot, "individual-service-business")
    flag_seats = seats_phrase(ssot, "flagship-business-training")
    pick_seats = seats_phrase(ssot, "individual-service-business")
    flag_dur = duration_phrase(ssot, "flagship-business-training")
    pick_dur = duration_phrase(ssot, "individual-service-business")
    flag_inst = installment_phrase(ssot, "flagship-business-training")
    reply_sla = reply_sla_phrase(ssot)
    q_cap = quarterly_cap_phrase(ssot)

    biz_count = len(ssot.raw.get("business-services", {}).get("services", []))
    pers_count = len(ssot.raw.get("personal-services", {}).get("services", []))

    return {
        ".": {
            "title": "Repository Root & Web Portal Hub",
            "type": "module",
            "description": "Static site root housing core portal HTML pages, master header/footer component templates, client-side pricing calculators, and repository governance documents.",
            "tags": ["root", "portal", "entry-points", "architecture", "ssot"],
            "workflow": "Edit facts in `data/*.json` -> run `python3 python/ssot/validate.py && python3 python/ssot/render.py && python3 python/ssot/export_site_data.py`. After editing `header.html` or `footer.html`, immediately run `python3 python/build_components.py`.",
            "related": ["data", "python/ssot", "python", "css", "schema", "md-html-sync", "tests"],
            "guidelines": [
                "`AGENTS.md` is the canonical project specification (`CLAUDE.md` and `GEMINI.md` are symlinks to `AGENTS.md`).",
                "`header.html` and `footer.html` are the single centralized navigation and footer sources inlined across all HTML pages via `python3 python/build_components.py`.",
                "Never type protected commercial facts outside `<!--ssot:...-->` markers; source them from `data/*.json`.",
                "Flagship offerings: " + flag_prices + " USD (" + flag_dur + ", " + flag_seats + ", " + flag_inst + ", $0 upfront, " + q_cap + ").",
                "Individual service picks: " + pick_prices + " USD (" + pick_dur + ", " + pick_seats + "). Canonical contact reply SLA: " + reply_sla + ".",
                "Maintain zero emojis, zero em-dashes, gold default hyperlinks, and non-underlined buttons across all files.",
            ],
        },
        "article-footers": {
            "title": "Article Footers Module",
            "type": "module",
            "description": "Reusable syndication footer templates (dark HTML, Markdown, and Word `.docx`) appended to external articles on Dev.to, Medium, Hashnode, Substack, and HackerNoon.",
            "tags": ["article-footers", "templates", "reusable-components", "syndication"],
            "workflow": "Keep all three formats (`article-footer.md`, `article-footer_dark.html`, `article-footer-plain.docx`) aligned whenever social links, portfolio icon, or book descriptions change.",
            "related": [".", "icons", "assets"],
            "guidelines": [
                "Preserve `article-footer.md`, `article-footer_dark.html`, and `article-footer-plain.docx` in this directory; never delete or move them.",
                "CRITICAL RULE: Never insert an article footer into any Claude guide (`claude-training/claude-*.html` or `md-html-sync/claude-training/claude-*.md`).",
                "All links use canonical email `thomascherickal@gmail.com` and include `https://dev.to/thomascherickal` (Mastodon excluded).",
            ],
        },
        "assets": {
            "title": "Global Site Assets, Branding & Generated Data Hub",
            "type": "dataset",
            "description": "Site-wide branding assets (logos, favicons, OG cards, profile portraits), generated client-side JSON (`assets/data/`), self-hosted Mermaid JS (`assets/js/`), and 11 topic-specific visual subfolders.",
            "tags": ["assets", "branding", "media", "favicons", "opengraph"],
            "workflow": "Compress every raster image strictly under 400 KB and delete larger originals. Regenerate favicons via `python3 python/generate_favicons.py` and OG card via `python3 python/sync_og_card.py`.",
            "related": ["assets/data", "assets/js", "images", "icons", "python"],
            "guidelines": [
                "Every image asset in `assets/` must be strictly under 400 KB.",
                "Organize new article or service visuals into dedicated topic subfolders under `assets/`.",
                "`assets/og-card.png` stays synchronized with `index.html` headline stats via `python3 python/sync_og_card.py`.",
                "Distinct from `images/`: `assets/` holds branding, favicons, OG cards, generated site data, and service/guide diagrams; `images/` holds standalone WebP editorial illustrations for `articles.html`.",
            ],
        },
        "assets/data": {
            "title": "Generated Client-Side Site Data Bundle",
            "type": "dataset",
            "description": "Auto-generated JSON data exported from `data/*.json` by `python/ssot/export_site_data.py` for browser-side JavaScript consumption.",
            "tags": ["assets", "data", "ssot", "generated", "javascript"],
            "workflow": "Regenerate via `python3 python/ssot/export_site_data.py` whenever `data/pricing.json`, `data/tiers.json`, `data/contact.json`, or `data/config.json` changes.",
            "related": ["data", "python/ssot", "pricing-calculator"],
            "guidelines": [
                "CRITICAL RULE: Never edit `assets/data/site-data.json` by hand; it is overwritten automatically by `python3 python/ssot/export_site_data.py`.",
                "Contains exported snapshots of `pricing`, `tiers`, `contact`, and `config` domains from `data/*.json`.",
            ],
        },
        "assets/js": {
            "title": "Self-Hosted Vendor JavaScript Libraries",
            "type": "module",
            "description": "Vendored self-hosted Mermaid JS runtime (`mermaid.min.js`) loaded by root `mermaid.js` to render technical architecture diagrams offline without external CDN failures.",
            "tags": ["assets", "javascript", "mermaid", "vendored"],
            "workflow": "Loaded dynamically by root `mermaid.js`; verify rendering behavior via `node pricing-tests.js` (`test_mermaid.js`).",
            "related": [".", "claude-training", "gemini-training", "tests/js"],
            "guidelines": [
                "Hosts the self-hosted `mermaid.min.js` bundle referenced by `mermaid.js` across training and service pages.",
                "Do not add third-party tracking scripts or external CDN dependencies here.",
            ],
        },
        "assets/advanced-ai-agents": {
            "title": "Advanced AI Agents Visual Assets",
            "type": "dataset",
            "description": "Compressed visual schematics and diagrams supporting the Advanced AI Agents business and personal service pages.",
            "tags": ["assets", "advanced-agents", "diagrams", "swarms"],
            "workflow": "Keep every image under 400 KB and update image links in `business-services/business-service-advanced-ai-agents.html` and `personal-services/personal-service-advanced-ai-agents.html`.",
            "related": ["business-services", "personal-services", "assets"],
            "guidelines": [
                "Serves hero and architecture visuals for the Advanced AI Agents training tracks.",
                "Verify every referenced image exists and has descriptive `alt` text via `python3 python/run_html_tests.py`.",
            ],
        },
        "assets/agentic-ai-systems": {
            "title": "Agentic AI Systems Visual Assets",
            "type": "dataset",
            "description": "Compressed architecture visuals supporting Agentic Assistants and persistent-memory agentic systems service pages.",
            "tags": ["assets", "agentic-systems", "diagrams"],
            "workflow": "Keep every image under 400 KB and verify references across service pages and Markdown mirrors.",
            "related": ["business-services", "personal-services", "assets"],
            "guidelines": [
                "Serves visual assets for Agentic Assistants service tracks.",
                "All images must remain under 400 KB with descriptive `alt` attributes.",
            ],
        },
        "assets/ai-agents-orchestration": {
            "title": "AI Agent Orchestration Visual Assets",
            "type": "dataset",
            "description": "Compressed multi-agent topology diagrams and visual assets for orchestration service pages.",
            "tags": ["assets", "orchestration", "swarms", "mcp"],
            "workflow": "Keep every image under 400 KB and verify references via `python3 python/run_html_tests.py`.",
            "related": ["business-services", "personal-services", "assets"],
            "guidelines": [
                "Supports `business-service-claude-agents-orchestration.html`, `personal-service-claude-agents-orchestration.html`, and orchestration overviews.",
            ],
        },
        "assets/ai-developer-training": {
            "title": "AI Developer Training Visual Assets",
            "type": "dataset",
            "description": "Compressed workflow diagrams and illustrations for AI Developer Training business and personal service pages.",
            "tags": ["assets", "developer-training", "tdd", "coding-agents"],
            "workflow": "Keep every image under 400 KB and maintain matching references in HTML and `md-html-sync/`.",
            "related": ["business-services", "personal-services", "assets"],
            "guidelines": [
                "Supports `business-services/business-service-ai-developer-training.html` and `personal-services/personal-service-ai-developer-training.html`.",
            ],
        },
        "assets/antigravity": {
            "title": "Google Antigravity Visual Assets",
            "type": "dataset",
            "description": "Compressed visuals and architecture illustrations for Google Antigravity IDE and Chat documentation guides.",
            "tags": ["assets", "antigravity", "google-ide", "agentic-workflows"],
            "workflow": "Keep every image under 400 KB and verify references in `gemini-training/google-antigravity-*.html`.",
            "related": ["gemini-training", "assets"],
            "guidelines": [
                "Supports `gemini-training/google-antigravity-ide.html` and `gemini-training/google-antigravity-chat.html`.",
            ],
        },
        "assets/claude-orchestration": {
            "title": "Claude Orchestration Visual Assets",
            "type": "dataset",
            "description": "Compressed architecture visuals for Claude agent orchestration guides and service tracks.",
            "tags": ["assets", "claude", "orchestration", "blueprints"],
            "workflow": "Keep every image under 400 KB and verify links in `claude-training/claude-orchestration.html`.",
            "related": ["claude-training", "assets"],
            "guidelines": [
                "Supports `claude-training/claude-orchestration.html` and related orchestration service pages.",
            ],
        },
        "assets/generative-ai-transformation": {
            "title": "Generative AI Transformation Visual Assets",
            "type": "dataset",
            "description": "Compressed visuals illustrating enterprise Generative AI transformation architectures and capability progressions.",
            "tags": ["assets", "transformation", "enterprise-ai"],
            "workflow": "Keep every image under 400 KB and verify references in `business-service-generative-ai-transformation.html` and `personal-service-generative-ai-transformation.html`.",
            "related": ["business-services", "personal-services", "assets"],
            "guidelines": [
                "Supports both Business and Personal Generative AI Transformation service pages.",
            ],
        },
        "assets/guarantees": {
            "title": "Service Guarantees Visual Assets",
            "type": "dataset",
            "description": "Compressed trust-model illustrations supporting `guarantees.html` ($0 upfront, 10 equal 10% milestone installments, walkaway guarantee, Apache 2.0 IP ownership).",
            "tags": ["assets", "guarantees", "trust-model"],
            "workflow": "Keep every image under 400 KB and verify references in `guarantees.html`.",
            "related": [".", "data", "assets"],
            "guidelines": [
                "Supports `guarantees.html` and related trust sections.",
            ],
        },
        "assets/local-ai-revolution": {
            "title": "Local AI Revolution Visual Assets",
            "type": "dataset",
            "description": "Compressed visuals for Local LLM cost-slashing, VRAM sizing, and private on-premise inference guides.",
            "tags": ["assets", "local-llms", "vram", "private-ai"],
            "workflow": "Keep every image under 400 KB and verify references in `business-services/` and `claude-training/claude-local-llms.html`.",
            "related": ["business-services", "claude-training", "assets"],
            "guidelines": [
                "Supports local LLM business services and `claude-training/claude-local-llms.html`.",
            ],
        },
        "assets/personal-service-local-llms": {
            "title": "Personal Local LLMs Visual Assets",
            "type": "dataset",
            "description": "Compressed workstation local-LLM setup diagrams for `personal-services/personal-service-local-llms-cost-slashing.html`.",
            "tags": ["assets", "personal-services", "local-llms", "workstation"],
            "workflow": "Keep every image under 400 KB and verify references in `personal-services/personal-service-local-llms-cost-slashing.html`.",
            "related": ["personal-services", "assets"],
            "guidelines": [
                "Supports the Personal Local LLMs Cost Slashing service track.",
            ],
        },
        "business-services": {
            "title": f"Enterprise & Business AI Services Catalog ({biz_count} Services)",
            "type": "module",
            "description": f"Dedicated specification pages for all {biz_count} enterprise AI training services, including architecture blueprints, hands-on curricula, deliverables, and PPP pricing links.",
            "tags": ["business-services", "enterprise-ai", "service-catalog", "syllabi"],
            "workflow": "After editing pages here, run `python3 python/ssot/render.py --check`, `python3 python/run_html_tests.py`, and before commit run `python3 python/md_html_sync.py` and `python3 python/sync_schemas.py`.",
            "related": ["data", "personal-services", "md-html-sync/business-services", "schema", "pricing"],
            "guidelines": [
                f"Contains {biz_count} enterprise service pages aligned with `data/business-services.json` and indexed by `all-business-services.html`.",
                f"Individual business service picks: {pick_prices} USD ({pick_dur}, {pick_seats}, {flag_inst}, $0 upfront).",
                "Every CTA banner must include the Free LinkedIn Consultation link (`btn-linkedin`) with uniform button sizing (`css/buttons.css`) and `text-decoration: none !important;`.",
                "Use `python3 python/delete_page.py <file>` if ever removing a service page so schemas, mirrors, and navigation stay clean.",
            ],
        },
        "claude-training": {
            "title": "Claude Architecture & Engineering Guides",
            "type": "module",
            "description": "In-depth technical engineering guides covering the Anthropic Claude ecosystem: Claude Code, Code Mode, Plugins, Apps (MCP Apps), Desktop, MCP, Projects, Artifacts, Orchestration, Local LLMs, and domain workflows.",
            "tags": ["claude-training", "claude-code", "mcp", "plugins", "orchestration"],
            "workflow": "Verify guide structure, unnumbered content headings, vertical Mermaid diagrams, and standard CTAs with `python3 python/run_html_tests.py`.",
            "related": ["data", "gemini-training", "md-html-sync/claude-training", "schema"],
            "guidelines": [
                "CRITICAL RULE: Never insert an article footer (`article-footers/`) into any `claude-training/claude-*.html` page.",
                "Do not create new offers, new services, or custom CTAs; use the standard `section.cta-banner` linking to `../pricing.html`, `../contact.html`, and Free LinkedIn Consultation.",
                "Refer to LLM families without version numbers in prose (Claude, Claude Opus, Claude Sonnet, Claude Haiku); version strings are allowed only in code examples or pricing tables.",
                "Keep all content headings (`<h2>`, `<h3>`) unnumbered (numbering belongs in ToC only), code blocks with sharp dark borders + Copy button, and Mermaid diagrams vertical (`TD`/`TB`).",
            ],
        },
        "css": {
            "title": "Global Stylesheets & Design System",
            "type": "module",
            "description": "Shared design-system stylesheets defining color tokens (`--gold`, `--cyan`), uniform button components (`buttons.css`), responsive tables, code windows, and guide typography.",
            "tags": ["css", "design-system", "tokens", "responsive", "buttons"],
            "workflow": "After editing CSS, verify zero underlined buttons and layout compliance across all 85+ HTML pages via `python3 python/run_html_tests.py`.",
            "related": [".", "claude-training", "gemini-training", "pricing"],
            "guidelines": [
                "`buttons.css` enforces uniform button formatting and `text-decoration: none !important;` across `:hover`, `:focus`, `:active`, and `:visited`.",
                "Default hyperlinks use gold (`var(--gold, #fbbf24)`) transitioning to gold-dim (`#d97706`) on hover.",
                "Code samples require high-contrast dark surfaces (`#040711` or `#060913`) with crisp borders and vertical separation.",
            ],
        },
        "data": {
            "title": "Authoritative Data Directory (Single Source of Truth)",
            "type": "dataset",
            "description": "The sole authoritative JSON data store (`data/*.json`) mastering pricing, parity tiers, business/personal services, site config, contact SLA, guarantees, FAQs, publications, and Claude/Gemini guide registries.",
            "tags": ["data", "ssot", "read-only", "pricing", "services", "authoritative"],
            "workflow": "Edit `data/*.json` directly -> validate with `python3 python/ssot/validate.py` -> propagate with `python3 python/ssot/render.py && python3 python/ssot/export_site_data.py`.",
            "related": ["data/schema", "python/ssot", "assets/data", "tests"],
            "guidelines": [
                "CRITICAL RULE: `data/*.json` is tracked in Git and strictly read-only to all scripts, test suites, and automated source-control commands.",
                "Data flows strictly one way: `data/*.json` -> `python/ssot/render.py` & `export_site_data.py` -> HTML / `llms.txt` / `assets/data/site-data.json`.",
                "Every JSON file is validated against its matching schema in `data/schema/*.schema.json` plus cross-file invariants in `python/ssot/validate.py`.",
                "`python/run_tests.py` enforces pre-test and post-test SHA-256 immutability checks on every `data/*.json` file.",
            ],
        },
        "data/schema": {
            "title": "SSOT JSON Schemas (`data/schema/`)",
            "type": "dataset",
            "description": "JSON Schema (Draft 2020-12 / Draft-07) definitions validating the exact structure, types, and required fields of every authoritative file in `data/*.json`.",
            "tags": ["data", "json-schema", "ssot", "validation"],
            "workflow": "Executed automatically by `python3 python/ssot/validate.py` on every pre-commit and pre-push check.",
            "related": ["data", "python/ssot", "tests"],
            "guidelines": [
                "Every `data/<name>.json` file has a corresponding `data/schema/<name>.schema.json` contract.",
                "When adding a new canonical field to `data/*.json`, update its matching `.schema.json` and add an invariant check in `python/ssot/validate.py` if needed.",
            ],
        },
        "docs": {
            "title": "Operational Documentation & SSOT Literal Inventory",
            "type": "module",
            "description": "Reference playbooks for delivery economics, 195-country World Bank PPP tiers, service catalog summaries, historical fixes, and the generated `ssot-inventory.csv` literal-fact audit.",
            "tags": ["docs", "playbooks", "pricing-reference", "ssot-inventory"],
            "workflow": "Refresh `docs/ssot-inventory.csv` via `python3 python/ssot/scan_literals.py`. Keep all prose free of emojis, em-dashes, and LLM version numbers.",
            "related": ["data", "python/ssot", "pricing"],
            "guidelines": [
                "`ssot-inventory.csv` is generated by `python3 python/ssot/scan_literals.py` to track remaining unprotected literal facts across HTML, JS, Markdown, and schemas.",
                "All pricing figures documented here must remain aligned with `AGENTS.md` and `data/pricing.json`.",
            ],
        },
        "failed-tests": {
            "title": "Failed Test Reports Directory",
            "type": "reports",
            "description": "Centralized holding directory where Python (`run_tests.py`), HTML UI (`run_html_tests.py`), and JavaScript (`pricing-tests.js`) runners write Markdown failure reports whenever a suite fails.",
            "tags": ["failed-tests", "testing", "reports", "qa"],
            "workflow": "Run `python3 python/run_tests.py`; when a suite passes 100%, its failure report in `failed-tests/` is automatically deleted.",
            "related": ["tests", "tests/js", "python"],
            "guidelines": [
                "All failed-test reports (`failed-python-tests.md`, `failed-html-tests.md`, `failed-pricing-tests.md`) must be stored exclusively inside `failed-tests/` and never in the repository root.",
                "A clean passing repository leaves only `readme.md` and `OKF.md` inside `failed-tests/`.",
            ],
        },
        "gemini-training": {
            "title": "Google Gemini & Gemma Engineering Guides",
            "type": "module",
            "description": "Technical architecture guides covering Google AI Studio, Google Antigravity IDE & Chat, Gemini Deep Research, DeepMind, NotebookLM, Skills, Spark, Multimodality, Gemma local models, and Gemini model tracks.",
            "tags": ["gemini-training", "gemini", "gemma", "antigravity-ide", "multimodal"],
            "workflow": "Verify guide structure, unnumbered headings, Copy buttons, vertical Mermaid diagrams, and standard CTAs with `python3 python/run_html_tests.py`.",
            "related": ["data", "claude-training", "md-html-sync/gemini-training", "schema"],
            "guidelines": [
                "Do not create new offers, new services, or custom CTAs; use the standard `section.cta-banner` linking to `../pricing.html`, `../contact.html`, and Free LinkedIn Consultation.",
                "Keep model family names version-free in prose (Gemini, Gemini Pro, Gemini Flash, Gemini Argon, Gemma), except inside code snippets or explicit pricing tables.",
                "Every code sample must have sharp dark boundaries and a working Copy button; Mermaid diagrams must be vertical (`TD`/`TB`) or square.",
            ],
        },
        "icons": {
            "title": "Vector SVG Icon & Platform Badge Repository",
            "type": "dataset",
            "description": "Scalable SVG icons for publishing platforms, coding profiles, social channels, and UI badges referenced across the site and syndicated article footers.",
            "tags": ["icons", "svg", "vectors", "publishing-platforms"],
            "workflow": "Verify SVG references before removing any icon; Dev.to (`dev.svg`) is an approved platform and must never be removed.",
            "related": [".", "article-footers", "assets"],
            "guidelines": [
                "Keep SVG markup clean and lightweight without embedded raster bitmaps.",
                "Dev.to is an approved publishing platform; Mastodon is excluded.",
            ],
        },
        "images": {
            "title": "Editorial & Article WebP Graphics",
            "type": "dataset",
            "description": "Standalone WebP technical illustrations featured on `articles.html` and publication listings.",
            "tags": ["images", "webp", "editorial", "articles"],
            "workflow": "Ensure every image stays strictly under 400 KB and has descriptive `alt` text on referencing pages.",
            "related": [".", "assets"],
            "guidelines": [
                "Distinct from `assets/`: stores editorial article cover illustrations in WebP format.",
                "Every image must be compressed under 400 KB.",
            ],
        },
        "internal-use": {
            "title": "Internal Trainer Mastery & Operational Runbooks",
            "type": "module",
            "description": "Private internal trainer documentation, delivery sprint protocols, and full-catalog teaching preparation materials (excluded from public navigation and sitemaps).",
            "tags": ["internal-use", "trainer-guide", "playbooks", "delivery"],
            "workflow": "Keep excluded from `header.html` and `sitemap.xml`; preserve zero emojis and zero em-dashes.",
            "related": ["docs", "data"],
            "guidelines": [
                "Strictly internal operational reference; do not link from public HTML navigation or `sitemap.xml`.",
            ],
        },
        "md-html-sync": {
            "title": "Markdown HTML Synchronization Root",
            "type": "module",
            "description": "Synchronized Markdown mirrors of live HTML pages (plus `.sync-state.json` hash cache) optimized for coding agents, LLM ingestion, and clean text diffs.",
            "tags": ["md-html-sync", "markdown-mirrors", "llm-friendly", "state-tracking"],
            "workflow": "Run `python3 python/md_html_sync.py` only when preparing a commit (or automatically via `python/hooks/pre-commit`).",
            "related": [".", "python", "md-html-sync/business-services", "md-html-sync/personal-services", "md-html-sync/claude-training", "md-html-sync/gemini-training", "md-html-sync/pricing"],
            "guidelines": [
                "Never manually delete mirror files with `rm`; use `python3 python/delete_page.py <page>` to clean HTML, schema, mirror, and `.sync-state.json` atomically.",
                "Full rebuilds from HTML use `python3 python/build_md_files.py`.",
            ],
        },
        "md-html-sync/business-services": {
            "title": "Business Services Markdown Mirrors",
            "type": "module",
            "description": f"Synchronized Markdown mirrors for all {biz_count} enterprise HTML pages in `business-services/`.",
            "tags": ["md-html-sync", "business-services", "markdown"],
            "workflow": "Synchronized automatically from `business-services/*.html` via `python3 python/md_html_sync.py`.",
            "related": ["business-services", "md-html-sync"],
            "guidelines": [
                "Mirrors `business-services/*.html` 1-to-1; edit the live HTML page (or `data/*.json` for SSOT facts) and let `md_html_sync.py` refresh these mirrors.",
            ],
        },
        "md-html-sync/claude-training": {
            "title": "Claude Training Markdown Mirrors",
            "type": "module",
            "description": "Synchronized Markdown mirrors for all Anthropic Claude engineering guides in `claude-training/`.",
            "tags": ["md-html-sync", "claude-training", "markdown"],
            "workflow": "Synchronized automatically from `claude-training/*.html` via `python3 python/md_html_sync.py`.",
            "related": ["claude-training", "md-html-sync"],
            "guidelines": [
                "Mirrors `claude-training/*.html` 1-to-1 and strictly excludes article footers.",
            ],
        },
        "md-html-sync/gemini-training": {
            "title": "Gemini Training Markdown Mirrors",
            "type": "module",
            "description": "Synchronized Markdown mirrors for all Google Gemini and Gemma engineering guides in `gemini-training/`.",
            "tags": ["md-html-sync", "gemini-training", "markdown"],
            "workflow": "Synchronized automatically from `gemini-training/*.html` via `python3 python/md_html_sync.py`.",
            "related": ["gemini-training", "md-html-sync"],
            "guidelines": [
                "Mirrors `gemini-training/*.html` 1-to-1 with preserved code fences and reference links.",
            ],
        },
        "md-html-sync/personal-services": {
            "title": "Personal Services Markdown Mirrors",
            "type": "module",
            "description": f"Synchronized Markdown mirrors for all {pers_count} personal AI training pages in `personal-services/`.",
            "tags": ["md-html-sync", "personal-services", "markdown"],
            "workflow": "Synchronized automatically from `personal-services/*.html` via `python3 python/md_html_sync.py`.",
            "related": ["personal-services", "md-html-sync"],
            "guidelines": [
                "Mirrors `personal-services/*.html` 1-to-1; keep pricing and seat figures aligned with `data/pricing.json`.",
            ],
        },
        "md-html-sync/pricing": {
            "title": "PPP Tier Pages Markdown Mirrors",
            "type": "module",
            "description": "Synchronized Markdown mirrors for the 6 Purchasing Power Parity regional tier HTML pages in `pricing/`.",
            "tags": ["md-html-sync", "pricing", "ppp-tiers", "markdown"],
            "workflow": "Synchronized automatically from `pricing/*.html` via `python3 python/md_html_sync.py`.",
            "related": ["pricing", "md-html-sync", "data"],
            "guidelines": [
                "Mirrors `pricing/pricing-tier-*.html` 1-to-1 for LLM and agent retrieval.",
            ],
        },
        "personal-services": {
            "title": f"Personal AI Training Services Catalog ({pers_count} Services)",
            "type": "module",
            "description": f"Dedicated specification pages for all {pers_count} personal AI acceleration services, covering architecture blueprints, hands-on projects, and PPP-calibrated pricing.",
            "tags": ["personal-services", "personal-training", "service-catalog", "syllabi"],
            "workflow": "After editing pages here, run `python3 python/ssot/render.py --check`, `python3 python/run_html_tests.py`, and before commit run `python3 python/md_html_sync.py` and `python3 python/sync_schemas.py`.",
            "related": ["data", "business-services", "md-html-sync/personal-services", "schema", "pricing"],
            "guidelines": [
                f"Contains {pers_count} personal training service pages aligned with `data/personal-services.json` and indexed by `all-personal-services.html`.",
                f"Individual personal service picks: {pick_prices} USD ({pick_dur}, {pick_seats}, {flag_inst}, $0 upfront).",
                "Every CTA banner must include the Free LinkedIn Consultation link (`btn-linkedin`) with uniform button styling.",
            ],
        },
        "pricing": {
            "title": "Purchasing Power Parity (PPP) Regional Tier Pages",
            "type": "module",
            "description": "Six dedicated regional landing pages implementing the 195-country World Bank FY2027 Purchasing Power Parity pricing model (Tiers 1 through 6).",
            "tags": ["pricing", "ppp-tiers", "global-parity", "world-bank-fy2027"],
            "workflow": "Keep rates aligned with `data/pricing.json` and `data/tiers.json`; verify tier calculators and DOM contracts via `python3 python/run_tests.py`.",
            "related": ["data", "pricing-calculator", "md-html-sync/pricing", "schema"],
            "guidelines": [
                f"Flagship PPP range: {flag_prices} USD; Individual Service PPP range: {pick_prices} USD.",
                "Tier precedence rule: when a country appears in both Tier 3 and Tier 6 (UN LDCs), Tier 6 (Relief / Custom) takes precedence.",
                "Startup tier (Tier 4) follows the standard 2-month flagship program timeline.",
                "Every tier page clearly documents $0 upfront advance, 10 equal 10% milestone installments, walkaway guarantee, and Zero Charge Training Materials (Apache 2.0).",
            ],
        },
        "pricing-calculator": {
            "title": "Tier-Specific Interactive Pricing Calculator Modules",
            "type": "module",
            "description": "Modular JavaScript calculator implementations (`pricing-calculator-tier-1.js` through `pricing-calculator-tier-6.js`) powering the 6 PPP tier landing pages.",
            "tags": ["pricing-calculator", "javascript-modules", "parity-calc"],
            "workflow": "Verify all 6 tier calculators via `node pricing-tests.js` (`tests/js/test_tier_calculators.js`).",
            "related": ["pricing", "data", "assets/data", "tests/js"],
            "guidelines": [
                "Each tier module mirrors the multiplier and seat rules in `data/tiers.json` and `data/pricing.json`.",
                "Tested by 36 dedicated assertions in `tests/js/test_tier_calculators.js`.",
            ],
        },
        "python": {
            "title": "Repository Python Automation & Build Tooling",
            "type": "module",
            "description": "Core Python 3 automation suite for header/footer component inlining, SSOT validation/rendering (`python/ssot/`), link checking, JSON-LD schema extraction, Markdown mirroring, metrics, OKF generation, and unified test execution.",
            "tags": ["python", "build-system", "automation", "tooling", "ssot"],
            "workflow": "Run `python3 python/run_tests.py` (executes Python, JS, and HTML UI suites; `--summary-only` for category tables; `--python-only` for Python only).",
            "related": ["python/ssot", "python/hooks", "tests", "failed-tests", "data"],
            "guidelines": [
                "`build_components.py`: Inlines `header.html`/`footer.html` across all HTML pages and chains `ssot/render.py`, `ssot/export_site_data.py`, `sync_schemas.py`, `md_html_sync.py`, `generate_metrics.py`, and `sync_og_card.py`.",
                "`run_tests.py`: Executes Python (`tests/`), JavaScript (`pricing-tests.js`), and HTML UI (`run_html_tests.py`) suites, verifies `data/*.json` SHA-256 immutability, and routes failure reports exclusively to `failed-tests/`.",
                "`delete_page.py`: Safely purges an HTML page along with its schema, Markdown mirror, `.sync-state.json` entry, sitemap entry, and navigation references.",
                "`check_links.py`, `sync_schemas.py`, `md_html_sync.py`: Run during commit preparation (automated in `python/hooks/pre-commit`).",
            ],
        },
        "python/hooks": {
            "title": "Git & Editor Lifecycle Hooks",
            "type": "module",
            "description": "Pre-commit, pre-push, and IDE post-edit hooks enforcing SSOT validation, marker rendering, schema/Markdown sync, category-summary test gates, and `LAST-COMMIT-REPORT.md` generation.",
            "tags": ["python", "git-hooks", "pre-commit", "pre-push", "quality-gate"],
            "workflow": "Keep `.git/hooks/pre-commit` and `.git/hooks/pre-push` synchronized with `python/hooks/` templates. Never bypass hooks with `--no-verify`.",
            "related": ["python", "python/ssot", "failed-tests"],
            "guidelines": [
                "`pre-commit`: Runs `validate.py`, `build_components.py` (on header/footer edits), `render.py`, `export_site_data.py`, `md_html_sync.py`, `sync_schemas.py`, `render.py --check`, `run_tests.py --summary-only`, and `generate_last_commit_report.py`.",
                "`pre-push`: Runs `validate.py`, `render.py --check`, `sync_schemas.py`, and `generate_pre_push_summary.py`.",
                "`post_edit_hook.py`: Automatically runs `build_components.py` when `header.html` or `footer.html` is edited.",
            ],
        },
        "python/ssot": {
            "title": "Deterministic SSOT Build & Synchronization Pipeline",
            "type": "module",
            "description": "Build-time Single Source of Truth engine replacing post-hoc fuzzy rewriting: loads `data/*.json`, validates JSON Schemas and cross-file invariants, renders `<!--ssot:...-->` markers and Jinja2 blocks deterministically, exports `assets/data/site-data.json`, and audits unprotected literals.",
            "tags": ["ssot", "validation", "jinja2", "markers", "synchronization"],
            "workflow": "Run `python3 python/ssot/validate.py && python3 python/ssot/render.py && python3 python/ssot/export_site_data.py && python3 python/ssot/render.py --check`.",
            "related": ["data", "data/schema", "templates/blocks", "assets/data", "docs", "tests"],
            "guidelines": [
                "`load.py`: Loads all `data/*.json` files and resolves dotted paths (`offerings.<id>.*`, `tiers.<n>.*`).",
                "`format.py`: Owns every formatted sentence or phrase containing a protected fact (`money`, `tier_price_list`, `seats_phrase`, `duration_phrase`, `installment_phrase`, `reply_sla_phrase`, `quarterly_cap_phrase`).",
                "`validate.py`: Enforces all 12 JSON Schemas in `data/schema/` plus cross-file invariants (seat bounds, 100% milestone sums, due days, Tier 6 precedence, publication slug consistency, zero emojis/em-dashes, no LLM version numbers in prose).",
                "`render.py`: Deterministically replaces only text between `<!--ssot:...-->` and `<!--/ssot-->` markers; fails loudly on unknown paths; `--check` exits non-zero on drift.",
                "`export_site_data.py`: Generates `assets/data/site-data.json` for client-side JavaScript.",
                "`scan_literals.py`: Audits protected facts typed outside markers and refreshes `docs/ssot-inventory.csv`.",
            ],
        },
        "schema": {
            "title": "Structured Data & Schema.org JSON-LD Definitions",
            "type": "dataset",
            "description": "Standalone JSON-LD Schema.org graphs extracted from live HTML pages to support search engines, rich snippets, and Generative Engine Optimization (GEO/AEO).",
            "tags": ["schema", "json-ld", "structured-data", "schema-org", "geo-aeo"],
            "workflow": "Live HTML `<script type=\"application/ld+json\">` blocks are the source of truth; synchronize into `schema/` before committing via `python3 python/sync_schemas.py`.",
            "related": [".", "business-services", "personal-services", "claude-training", "gemini-training", "python"],
            "guidelines": [
                "Every HTML page maps deterministically to a JSON-LD file in `schema/` (`index.html` -> `home-schema.json`, `service-*` -> `service-*.json`, guides -> `<page>-schema.json`).",
                "Uses canonical author description (`Thomas Cherickal - Generative AI Consultant and The Digital Futurist.`), canonical email `thomascherickal@gmail.com`, and includes `https://dev.to/thomascherickal` in `sameAs`.",
            ],
        },
        "templates": {
            "title": "SSOT Jinja2 Block Templates Root",
            "type": "module",
            "description": "Root directory for reusable Jinja2 HTML templates rendered into `<!--ssot:block.<name>-->` markers by `python/ssot/render.py`.",
            "tags": ["templates", "jinja2", "ssot", "blocks"],
            "workflow": "Place reusable HTML block templates inside `templates/blocks/<name>.html.j2` and invoke via `<!--ssot:block.<name>--><!--/ssot-->`.",
            "related": ["templates/blocks", "python/ssot", "data"],
            "guidelines": [
                "Templates receive `data`, `offerings`, and `tiers` context objects directly from `python/ssot/load.py`.",
            ],
        },
        "templates/blocks": {
            "title": "SSOT Reusable Block Templates (`*.html.j2`)",
            "type": "module",
            "description": "Jinja2 partial templates (`<block_name>.html.j2`) rendered by `python/ssot/render.py` when evaluating `<!--ssot:block.<block_name>-->` markers.",
            "tags": ["templates", "blocks", "jinja2", "ssot"],
            "workflow": "Add `<name>.html.j2` here and run `python3 python/ssot/render.py && python3 python/ssot/render.py --check`.",
            "related": ["templates", "python/ssot", "data"],
            "guidelines": [
                "Use for repeated multi-element structures (tier cards, pricing cards, publication lists, FAQ accordions).",
                "Keep templates deterministic and free of emojis or em-dashes so `render.py --check` stays 100% idempotent.",
            ],
        },
        "tests": {
            "title": "Unified Test Suite & Verification Harness",
            "type": "module",
            "description": "Python unittest modules covering the SSOT build pipeline, pricing/policy invariants, 101 HTML UI checks across all 85+ pages, internal link integrity, build tooling, coverage reporting, and commit reporting.",
            "tags": ["tests", "unittest", "verification", "qa", "html-ui"],
            "workflow": "Run `python3 python/run_tests.py` (full Python + JS + HTML UI suite), `python3 python/run_tests.py --summary-only` (category tables), or `python3 python/run_tests.py --python-only`.",
            "related": ["tests/js", "failed-tests", "python", "python/ssot"],
            "guidelines": [
                "Pre-test and post-test SHA-256 hashing in `python/run_tests.py` aborts with exit code 1 if any test mutates `data/*.json`.",
                "All test sandboxes must use temporary directories (`tempfile.mkdtemp()`), never repository files.",
                "Failed test reports are written exclusively to `failed-tests/` and automatically removed once passing.",
            ],
        },
        "tests/js": {
            "title": "Modular JavaScript Unit & DOM Test Suite",
            "type": "module",
            "description": "Zero-dependency Node.js test harness and 13 modular test suites (200 assertions across 18 sections) verifying pricing calculators, payment schedules, tier calculators, components navigation, and Mermaid link rewriting.",
            "tags": ["tests", "javascript-tests", "node", "calculator-tests"],
            "workflow": "Run `node pricing-tests.js` directly or via `python3 python/run_tests.py`.",
            "related": ["tests", "pricing-calculator", "failed-tests"],
            "guidelines": [
                "Orchestrated by `tests/js/runner.js` and `tests/js/test_helpers.js`, persisting metrics to `tests/.js_test_reports_cache.json`.",
                "On failure, writes `failed-tests/failed-pricing-tests.md` and deletes it automatically when all 200 tests pass.",
            ],
        },
    }


def build_reference_index() -> dict:
    """Scans HTML, CSS, JS, and MD files once to map asset/file paths to referencing files."""
    ref_map = {}  # basename or relpath -> set of referencing relpaths
    scan_exts = {".html", ".css", ".js", ".md"}
    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        rel_root = os.path.relpath(root, REPO_ROOT)
        if rel_root.startswith("md-html-sync"):
            continue
        for fn in files:
            if fn == "OKF.md" or fn == "LAST-COMMIT-REPORT.md" or fn == "METRICS.md":
                continue
            ext = os.path.splitext(fn)[1].lower()
            if ext not in scan_exts:
                continue
            abs_p = os.path.join(root, fn)
            rel_p = os.path.relpath(abs_p, REPO_ROOT).replace("\\", "/")
            try:
                with open(abs_p, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except OSError:
                continue
            # Match quoted paths or markdown links containing extensions of interest
            for m in re.finditer(
                r"""(?:src|href|url|content)=["']([^"'#?]+)["']|\]\(([^)#?]+)\)""",
                text,
            ):
                target = m.group(1) or m.group(2) or ""
                if not target or target.startswith(("http://", "https://", "mailto:", "data:", "tel:")):
                    continue
                norm = target.lstrip("./")
                while norm.startswith("../"):
                    norm = norm[3:]
                if norm:
                    ref_map.setdefault(norm, set()).add(rel_p)
                    base = os.path.basename(norm)
                    if base:
                        ref_map.setdefault(base, set()).add(rel_p)
    return ref_map


def lookup_refs(ref_map: dict, rel_file: str) -> list:
    """Returns sorted list of files referencing rel_file."""
    rel_norm = rel_file.replace("\\", "/")
    refs = set(ref_map.get(rel_norm, set()))
    if not refs:
        refs = set(ref_map.get(os.path.basename(rel_norm), set()))
    refs.discard(rel_norm)
    return sorted(refs)


def format_refs(refs: list, limit: int = 3) -> str:
    if not refs:
        return "unreferenced"
    shown = ", ".join(f"`{r}`" for r in refs[:limit])
    if len(refs) > limit:
        shown += f" (+{len(refs) - limit} more)"
    return shown


def inspect_file(abs_path: str, rel_file: str, ref_map: dict) -> tuple:
    """Returns (file_type, size_lines_str, details_str, line_count)."""
    size_bytes = os.path.getsize(abs_path)
    ext = os.path.splitext(abs_path)[1].lower()
    fname = os.path.basename(abs_path)

    # Binary raster images
    if ext in IMAGE_EXTS:
        dims = ""
        if Image is not None:
            try:
                with Image.open(abs_path) as im:
                    dims = f"{im.width}x{im.height}px"
            except Exception:
                dims = ""
        refs = lookup_refs(ref_map, rel_file)
        kb = size_bytes / 1024.0
        size_note = f"{dims}, " if dims else ""
        limit_flag = " [OVER 400 KB LIMIT]" if kb >= 400.0 else ""
        details = f"{size_note}Referenced by: {format_refs(refs)}{limit_flag}"
        return (ext.lstrip(".").upper(), fmt_size(size_bytes), sanitize_text(details, 220), 0)

    # SVG files
    if ext == ".svg":
        try:
            with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                raw = f.read()
            lines_n = raw.count("\n") + 1
            vb = re.search(r'viewBox=["\']([^"\']+)["\']', raw)
            vb_str = f"viewBox `{vb.group(1)}`; " if vb else ""
            refs = lookup_refs(ref_map, rel_file)
            details = f"{vb_str}Referenced by: {format_refs(refs)}"
            return ("SVG", f"{fmt_size(size_bytes)} / {lines_n}L", sanitize_text(details, 220), lines_n)
        except OSError:
            return ("SVG", fmt_size(size_bytes), "Scalable vector graphic", 0)

    # Word documents or other binaries
    if ext == ".docx":
        return ("DOCX", fmt_size(size_bytes), "Plain-text Word syndication footer template", 0)

    # Text-based files
    try:
        with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
    except OSError:
        return (ext.lstrip(".").upper() or "FILE", fmt_size(size_bytes), "Unreadable file", 0)

    lines_list = raw.splitlines()
    line_count = len(lines_list)
    size_lines = f"{fmt_size(size_bytes)} / {line_count}L"

    # HTML
    if ext in (".html", ".htm"):
        title_m = re.search(r"<title[^>]*>(.*?)</title>", raw, re.IGNORECASE | re.DOTALL)
        h1_m = re.search(r"<h1[^>]*>(.*?)</h1>", raw, re.IGNORECASE | re.DOTALL)
        title = sanitize_text(re.sub(r"<[^>]+>", "", title_m.group(1)), 90) if title_m else ""
        h1 = sanitize_text(re.sub(r"<[^>]+>", "", h1_m.group(1)), 80) if h1_m else ""
        markers = len(re.findall(r"<!--ssot:", raw))
        ld_types = re.findall(r'"@type"\s*:\s*"([^"]+)"', raw)
        unique_ld = list(dict.fromkeys(ld_types))[:4]
        parts = []
        if title:
            parts.append(f"Title: \"{title}\"")
        elif h1:
            parts.append(f"H1: \"{h1}\"")
        if markers:
            parts.append(f"{markers} SSOT marker(s)")
        if unique_ld:
            parts.append("JSON-LD: " + ", ".join(unique_ld))
        if fname in ("header.html", "footer.html"):
            parts.append("Master component template inlined by `python/build_components.py`")
        if not parts:
            parts.append("HTML document")
        return ("HTML", size_lines, sanitize_text("; ".join(parts), 240), line_count)

    # Python
    if ext == ".py":
        doc = ""
        classes = []
        funcs = []
        try:
            tree = ast.parse(raw)
            raw_doc = ast.get_docstring(tree) or ""
            if raw_doc:
                doc = sanitize_text(raw_doc.splitlines()[0], 110)
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                elif isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                    funcs.append(node.name)
        except SyntaxError:
            doc = "Python script"
        parts = []
        if doc:
            parts.append(doc)
        if classes:
            parts.append("Classes: " + ", ".join(f"`{c}`" for c in classes[:4]))
        if funcs:
            shown_f = ", ".join(f"`{fn}`" for fn in funcs[:5])
            if len(funcs) > 5:
                shown_f += f" (+{len(funcs) - 5})"
            parts.append("Functions: " + shown_f)
        if not parts:
            parts.append("Python module")
        return ("Python", size_lines, sanitize_text("; ".join(parts), 240), line_count)

    # JavaScript
    if ext in (".js", ".mjs"):
        if size_bytes > 500_000:
            refs = lookup_refs(ref_map, rel_file)
            return ("JS (Vendor)", fmt_size(size_bytes), sanitize_text(f"Minified vendor bundle; Referenced by: {format_refs(refs)}", 220), line_count)
        summary = ""
        for ln in lines_list[:12]:
            stripped = ln.strip().lstrip("/*").rstrip("*/").strip()
            if stripped and not stripped.startswith(("Strict Repo", "@", "use strict")):
                summary = sanitize_text(stripped, 110)
                break
        sections = re.findall(r"setSection\(['\"]([^'\"]+)['\"]\)", raw)
        asserts = len(re.findall(r"\bh\.assert\(|harness\.assert\(", raw))
        funcs = re.findall(r"function\s+([A-Za-z0-9_]+)\s*\(", raw)
        parts = []
        if summary:
            parts.append(summary)
        if sections:
            parts.append(f"{len(sections)} section(s), {asserts} assertion(s)")
        elif funcs:
            shown = ", ".join(f"`{fn}`" for fn in funcs[:5])
            if len(funcs) > 5:
                shown += f" (+{len(funcs) - 5})"
            parts.append("Functions: " + shown)
        refs = lookup_refs(ref_map, rel_file)
        if refs and not rel_file.startswith("tests/"):
            parts.append("Used by: " + format_refs(refs, 2))
        return ("JavaScript", size_lines, sanitize_text("; ".join(parts) or "JavaScript module", 240), line_count)

    # JSON / JSON-LD / JSON Schema
    if ext == ".json":
        try:
            obj = json.loads(raw)
        except Exception:
            return ("JSON", size_lines, "JSON file", line_count)
        parts = []
        if isinstance(obj, dict):
            if "$schema" in obj:
                title = obj.get("title") or os.path.basename(abs_path)
                req = obj.get("required", [])
                parts.append(f"JSON Schema: {title}")
                if req:
                    parts.append("Required: " + ", ".join(f"`{k}`" for k in req[:5]))
            elif "@graph" in obj or "@context" in obj:
                graph = obj.get("@graph", [obj])
                types = [g.get("@type") for g in graph if isinstance(g, dict) and g.get("@type")]
                flat_types = []
                for t in types:
                    if isinstance(t, list):
                        flat_types.extend(t)
                    else:
                        flat_types.append(str(t))
                unique_t = list(dict.fromkeys(flat_types))[:5]
                names = [g.get("name") or g.get("headline") for g in graph if isinstance(g, dict) and (g.get("name") or g.get("headline"))]
                if unique_t:
                    parts.append("JSON-LD `@type`: " + ", ".join(unique_t))
                if names:
                    parts.append(f"Entity: \"{sanitize_text(names[0], 70)}\"")
            else:
                top_keys = list(obj.keys())
                counts = []
                for k, v in obj.items():
                    if isinstance(v, list):
                        counts.append(f"`{k}` ({len(v)} items)")
                    elif isinstance(v, dict):
                        counts.append(f"`{k}` ({len(v)} keys)")
                    else:
                        counts.append(f"`{k}`")
                parts.append("Keys: " + ", ".join(counts[:6]))
                if len(top_keys) > 6:
                    parts.append(f"(+{len(top_keys) - 6} more)")
        elif isinstance(obj, list):
            parts.append(f"JSON array with {len(obj)} items")
        return ("JSON", size_lines, sanitize_text("; ".join(parts) or "JSON data", 240), line_count)

    # Markdown
    if ext == ".md":
        h1_m = re.search(r"^#\s+(.+)$", raw, re.MULTILINE)
        h2_count = len(re.findall(r"^##\s+", raw, re.MULTILINE))
        words = len(raw.split())
        parts = []
        if h1_m:
            parts.append(f"Heading: \"{sanitize_text(h1_m.group(1), 95)}\"")
        parts.append(f"{h2_count} H2 section(s), {words:,} words")
        if os.path.islink(abs_path):
            parts.append(f"Symlink -> `{os.readlink(abs_path)}`")
        return ("Markdown", size_lines, sanitize_text("; ".join(parts), 240), line_count)

    # CSS
    if ext == ".css":
        comment_m = re.search(r"/\*\s*(.*?)\s*\*/", raw, re.DOTALL)
        vars_n = len(set(re.findall(r"--[a-zA-Z0-9_-]+", raw)))
        media_n = len(re.findall(r"@media\b", raw))
        refs = lookup_refs(ref_map, rel_file)
        parts = []
        if comment_m:
            first_line = sanitize_text(comment_m.group(1).splitlines()[0], 90)
            if first_line:
                parts.append(first_line)
        parts.append(f"{vars_n} CSS custom vars, {media_n} `@media` rules")
        if refs:
            parts.append("Linked by: " + format_refs(refs, 2))
        return ("CSS", size_lines, sanitize_text("; ".join(parts), 240), line_count)

    # CSV
    if ext == ".csv":
        rows = list(csv.reader(lines_list))
        hdr = ", ".join(rows[0][:6]) if rows else ""
        data_rows = max(0, len(rows) - 1)
        return ("CSV", size_lines, sanitize_text(f"{data_rows:,} data rows; Columns: `{hdr}`", 220), line_count)

    # XML / sitemap
    if ext == ".xml":
        urls = len(re.findall(r"<loc>", raw))
        return ("XML", size_lines, f"XML sitemap with {urls} `<loc>` entries" if urls else "XML document", line_count)

    # Shell hooks / extensionless scripts / txt
    first_comment = ""
    for ln in lines_list[:6]:
        s = ln.strip()
        if s.startswith("#") and not s.startswith("#!"):
            first_comment = sanitize_text(s.lstrip("#").strip(), 110)
            break
    if fname == "llms.txt":
        markers = len(re.findall(r"<!--ssot:", raw))
        return ("Text", size_lines, f"LLM-ready site index; {markers} SSOT marker(s)", line_count)
    if fname == "robots.txt":
        return ("Text", size_lines, "Crawler directives and sitemap pointer", line_count)
    if fname == "CNAME":
        return ("Config", size_lines, f"GitHub Pages custom domain: `{raw.strip()}`", line_count)
    if fname in ("pre-commit", "pre-push"):
        invoked = re.findall(r"python3\s+[\"']?\$REPO_ROOT/([^\s\"']+)", raw)
        inv_str = ", ".join(f"`{x}`" for x in invoked[:5])
        details = f"{first_comment}; Invokes: {inv_str}" if inv_str else (first_comment or "Git hook script")
        return ("Shell Hook", size_lines, sanitize_text(details, 240), line_count)

    return (ext.lstrip(".").upper() or "Text", size_lines, first_comment or f"Text file ({line_count} lines)", line_count)


def build_okf_content(rel_dir: str, meta: dict, all_meta: dict, ref_map: dict, now_iso: str) -> str:
    abs_dir = os.path.join(REPO_ROOT, rel_dir) if rel_dir != "." else REPO_ROOT
    items = sorted(os.listdir(abs_dir))
    files = [
        f for f in items
        if (os.path.isfile(os.path.join(abs_dir, f)) or os.path.islink(os.path.join(abs_dir, f)))
        and f != "OKF.md"
        and not f.startswith(".")
    ]
    subdirs = [
        s for s in items
        if os.path.isdir(os.path.join(abs_dir, s))
        and not s.startswith(".")
        and s not in SKIP_DIRS
    ]

    # Inspect all files in this directory
    inspected = []
    total_bytes = 0
    total_lines = 0
    ext_counts = {}
    for f in files:
        abs_f = os.path.join(abs_dir, f)
        rel_f = f if rel_dir == "." else os.path.join(rel_dir, f).replace("\\", "/")
        ftype, size_lines, details, lcnt = inspect_file(abs_f, rel_f, ref_map)
        total_bytes += os.path.getsize(abs_f)
        total_lines += lcnt
        ext_counts[ftype] = ext_counts.get(ftype, 0) + 1
        inspected.append((f, ftype, size_lines, details))

    ext_summary = ", ".join(f"{cnt} {k}" for k, cnt in sorted(ext_counts.items(), key=lambda x: (-x[1], x[0]))) or "Empty directory"
    tags_str = "[" + ", ".join(meta["tags"]) + "]"

    lines = [
        "---",
        f'type: {meta["type"]}',
        f'title: "{meta["title"]}"',
        f'description: "{sanitize_text(meta["description"], 260)}"',
        f"tags: {tags_str}",
        f"timestamp: {now_iso}",
        "---",
        "",
        f'# {meta["title"]}',
        "",
        meta["description"],
        "",
        "## Live Directory Telemetry",
        "",
        f"- **Path**: `/{rel_dir}`" if rel_dir != "." else "- **Path**: `/` (Repository Root)",
        f"- **Direct Files**: {len(files)} ({ext_summary})",
        f"- **Subdirectories**: {len(subdirs)}",
        f"- **Total Size**: {fmt_size(total_bytes)}" + (f" | **Text/Code Lines**: {total_lines:,}" if total_lines else ""),
        f'- **Primary Workflow**: {meta["workflow"]}',
        "",
        "## Directory Role & Agent Invariants",
        "",
    ]

    for g in meta["guidelines"]:
        lines.append(f"- {g}")
    lines.append("")

    lines.append("## Subdirectories")
    lines.append("")
    if subdirs:
        lines.append("| Subdirectory | Files | Role & Scope |")
        lines.append("| :--- | :---: | :--- |")
        for s in subdirs:
            child_rel = s if rel_dir == "." else f"{rel_dir}/{s}"
            child_abs = os.path.join(abs_dir, s)
            child_files = [
                cf for cf in os.listdir(child_abs)
                if os.path.isfile(os.path.join(child_abs, cf)) and cf != "OKF.md" and not cf.startswith(".")
            ]
            cmeta = all_meta.get(child_rel)
            role = sanitize_text(cmeta["description"], 150) if cmeta else f"Contains {len(child_files)} files."
            lines.append(f"| [`{s}/`]({s}/OKF.md) | {len(child_files)} | {role} |")
    else:
        lines.append("Leaf directory (no child directories).")
    lines.append("")

    lines.append("## Detailed File Inventory")
    lines.append("")
    if inspected:
        lines.append("| File | Type | Size / Lines | In-Context Details & Cross-References |")
        lines.append("| :--- | :---: | :---: | :--- |")
        for f, ftype, size_lines, details in inspected:
            lines.append(f"| [`{f}`]({f}) | {ftype} | {size_lines} | {details} |")
    else:
        lines.append("No direct files yet (ready for new files matching this directory's workflow).")
    lines.append("")

    lines.append("## Context & Cross-References")
    lines.append("")
    depth = 0 if rel_dir == "." else rel_dir.count("/") + 1
    root_prefix = "../" * depth
    if rel_dir != ".":
        parent_dir = os.path.dirname(rel_dir)
        if parent_dir:
            lines.append(f"- Parent Directory: [`{parent_dir}/OKF.md`](../OKF.md)")
        lines.append(f"- Root Directory: [`OKF.md`]({root_prefix}OKF.md)")
    lines.append(f"- Canonical Specification: [`AGENTS.md`]({root_prefix}AGENTS.md) (`CLAUDE.md` and `GEMINI.md` symlink to `AGENTS.md`)")
    lines.append(f"- Repository Telemetry: [`METRICS.md`]({root_prefix}METRICS.md)")
    for rel_target in meta.get("related", []):
        if rel_target == rel_dir:
            continue
        target_label = "Repository Root" if rel_target == "." else f"`{rel_target}/`"
        target_href = f"{root_prefix}OKF.md" if rel_target == "." else f"{root_prefix}{rel_target}/OKF.md"
        tmeta = all_meta.get(rel_target)
        t_desc = f" - {tmeta['title']}" if tmeta else ""
        lines.append(f"- Related: [{target_label}]({target_href}){t_desc}")
    lines.append("")

    return "\n".join(lines)


def main():
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ssot = SSOTData()
    all_meta = build_directory_metadata(ssot)
    ref_map = build_reference_index()

    count = 0
    for rel_dir, meta in sorted(all_meta.items()):
        abs_dir = os.path.join(REPO_ROOT, rel_dir) if rel_dir != "." else REPO_ROOT
        if not os.path.isdir(abs_dir):
            continue
        okf_path = os.path.join(abs_dir, "OKF.md")
        content = build_okf_content(rel_dir, meta, all_meta, ref_map, now_iso)
        with open(okf_path, "w", encoding="utf-8") as f:
            f.write(content)
        count += 1
    print(f"Successfully generated {count} in-context OKF.md files.")


if __name__ == "__main__":
    main()
