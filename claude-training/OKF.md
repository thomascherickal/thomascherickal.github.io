---
type: module
title: "Claude Architecture & Engineering Guides"
description: "In-depth technical engineering guides covering the Anthropic Claude ecosystem: Claude Code, Code Mode, Plugins, Apps (MCP Apps), Desktop, MCP, Projects, Artifacts, Orchestration, Local LLMs, and domain workflows."
tags: [claude-training, claude-code, mcp, plugins, orchestration]
timestamp: 2026-11-28T05:16:04Z
---

# Claude Architecture & Engineering Guides

In-depth technical engineering guides covering the Anthropic Claude ecosystem: Claude Code, Code Mode, Plugins, Apps (MCP Apps), Desktop, MCP, Projects, Artifacts, Orchestration, Local LLMs, and domain workflows.

## Live Directory Telemetry

- **Path**: `/claude-training`
- **Direct Files**: 18 (18 HTML)
- **Subdirectories**: 0
- **Total Size**: 1.49 MB | **Text/Code Lines**: 17,052
- **Primary Workflow**: Verify guide structure, unnumbered content headings, vertical Mermaid diagrams, and standard CTAs with `python3 python/run_html_tests.py`.

## Directory Role & Agent Invariants

- CRITICAL RULE: Never insert an article footer (`article-footers/`) into any `claude-training/claude-*.html` page.
- Do not create new offers, new services, or custom CTAs; use the standard `section.cta-banner` linking to `../pricing.html`, `../contact.html`, and Free LinkedIn Consultation.
- Refer to LLM families without version numbers in prose (Claude, Claude Opus, Claude Sonnet, Claude Haiku); version strings are allowed only in code examples or pricing tables.
- Keep all content headings (`<h2>`, `<h3>`) unnumbered (numbering belongs in ToC only), code blocks with sharp dark borders + Copy button, and Mermaid diagrams vertical (`TD`/`TB`).

## Subdirectories

Leaf directory (no child directories).

## Detailed File Inventory

| File | Type | Size / Lines | In-Context Details & Cross-References |
| :--- | :---: | :---: | :--- |
| [`claude-101.html`](claude-101.html) | HTML | 84.4 KB / 941L | Title: "Claude 101 Architecture Guide / Thomas Cherickal"; JSON-LD: WebPage, Person, Organization, ImageObject |
| [`claude-apps.html`](claude-apps.html) | HTML | 83.6 KB / 926L | Title: "Claude Apps &amp; MCP UI Guide / Thomas Cherickal"; JSON-LD: WebPage, Person, Organization |
| [`claude-artifacts.html`](claude-artifacts.html) | HTML | 86.5 KB / 958L | Title: "Claude Artifacts Engineering Guide / Thomas Cherickal"; JSON-LD: WebPage, Person, Organization, ImageObject |
| [`claude-code-mode.html`](claude-code-mode.html) | HTML | 82.0 KB / 901L | Title: "Claude Code Mode Sandboxing Guide / Thomas Cherickal"; JSON-LD: WebPage, Person, Organization |
| [`claude-code.html`](claude-code.html) | HTML | 84.6 KB / 940L | Title: "Claude Code CLI Engineering Guide / Thomas Cherickal"; JSON-LD: WebPage, Person, Organization, ImageObject |
| [`claude-dashboards.html`](claude-dashboards.html) | HTML | 83.9 KB / 934L | Title: "Claude Dashboards Engineering Guide / Thomas Cherickal"; JSON-LD: WebPage, Person, Organization, ImageObject |
| [`claude-design.html`](claude-design.html) | HTML | 83.3 KB / 925L | Title: "Claude Design Systems Guide / Thomas Cherickal"; JSON-LD: WebPage, Person, Organization, ImageObject |
| [`claude-desktop.html`](claude-desktop.html) | HTML | 84.0 KB / 927L | Title: "Claude Desktop Workflows Guide / Thomas Cherickal"; JSON-LD: WebPage, Person, Organization, ImageObject |
| [`claude-fable.html`](claude-fable.html) | HTML | 85.6 KB / 939L | Title: "Claude Fable Creative Engine Guide / Thomas Cherickal"; JSON-LD: WebPage, Person, Organization, ImageObject |
| [`claude-local-llms.html`](claude-local-llms.html) | HTML | 83.8 KB / 943L | Title: "Claude with Local LLMs Guide / Thomas Cherickal"; JSON-LD: WebPage, Person, Organization, ImageObject |
| [`claude-mcp.html`](claude-mcp.html) | HTML | 86.2 KB / 959L | Title: "Claude MCP Server Architecture / Thomas Cherickal"; JSON-LD: WebPage, Person, Organization, ImageObject |
| [`claude-mods.html`](claude-mods.html) | HTML | 83.4 KB / 945L | Title: "Claude Mods Extension Guide / Thomas Cherickal"; JSON-LD: WebPage, Person, Organization, ImageObject |
| [`claude-office.html`](claude-office.html) | HTML | 84.4 KB / 939L | Title: "Claude for Microsoft Office Guide / Thomas Cherickal"; JSON-LD: WebPage, Person, Organization, ImageObject |
| [`claude-orchestration.html`](claude-orchestration.html) | HTML | 85.9 KB / 956L | Title: "Claude Agent Orchestration Guide / Thomas Cherickal"; JSON-LD: WebPage, Person, Organization, ImageObject |
| [`claude-plugins.html`](claude-plugins.html) | HTML | 87.4 KB / 973L | Title: "Claude Plugins (Skills &amp; Connectors) / Thomas Cherickal"; JSON-LD: WebPage, Person, Organization, ImageObject |
| [`claude-projects.html`](claude-projects.html) | HTML | 85.3 KB / 993L | Title: "Claude Projects Context Guide / Thomas Cherickal"; JSON-LD: WebPage, Person, Organization, ImageObject |
| [`claude-science.html`](claude-science.html) | HTML | 83.9 KB / 929L | Title: "Claude for Scientific Research / Thomas Cherickal"; JSON-LD: WebPage, Person, Organization, ImageObject |
| [`claude-whiteboards.html`](claude-whiteboards.html) | HTML | 87.5 KB / 1024L | Title: "Claude Whiteboards Architecture / Thomas Cherickal"; JSON-LD: WebPage, Person, Organization, ImageObject |

## Context & Cross-References

- Root Directory: [`OKF.md`](../OKF.md)
- Canonical Specification: [`AGENTS.md`](../AGENTS.md) (`CLAUDE.md` and `GEMINI.md` symlink to `AGENTS.md`)
- Repository Telemetry: [`METRICS.md`](../METRICS.md)
- Related: [`data/`](../data/OKF.md) - Authoritative Data Directory (Single Source of Truth)
- Related: [`gemini-training/`](../gemini-training/OKF.md) - Google Gemini & Gemma Engineering Guides
- Related: [`md-html-sync/claude-training/`](../md-html-sync/claude-training/OKF.md) - Claude Training Markdown Mirrors
- Related: [`schema/`](../schema/OKF.md) - Structured Data & Schema.org JSON-LD Definitions
