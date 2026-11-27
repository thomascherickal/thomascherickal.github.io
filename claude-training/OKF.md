---
type: module
title: "Claude Architecture & Engineering Guides"
description: "In-depth technical engineering guides covering the Anthropic Claude ecosystem: Claude Code, Code Mode, Plugins, Apps (MCP Apps), Desktop, MCP, Projects, Artifacts, Orchestration, Local LLMs, and domain workflows."
tags: [claude-training, claude-code, mcp, plugins, orchestration]
timestamp: 2026-11-27T06:56:49Z
---

# Claude Architecture & Engineering Guides

In-depth technical engineering guides covering the Anthropic Claude ecosystem: Claude Code, Code Mode, Plugins, Apps (MCP Apps), Desktop, MCP, Projects, Artifacts, Orchestration, Local LLMs, and domain workflows.

## Live Directory Telemetry

- **Path**: `/claude-training`
- **Direct Files**: 18 (18 HTML)
- **Subdirectories**: 0
- **Total Size**: 1.46 MB | **Text/Code Lines**: 16,735
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
| [`claude-101.html`](claude-101.html) | HTML | 82.8 KB / 925L | Title: "Claude 101: Architecture, Features & Agentic Engineering / Thomas Cherickal"; JSON-LD: WebPage, Person, Organization, ImageObject |
| [`claude-apps.html`](claude-apps.html) | HTML | 81.7 KB / 897L | Title: "Claude Apps: Interactive MCP Apps, Live Workspace UI & Desktop Export / Thomas Cherickal"; JSON-LD: WebPage, Person, Organization |
| [`claude-artifacts.html`](claude-artifacts.html) | HTML | 85.0 KB / 943L | Title: "Claude Artifacts: Interactive Code, Diagrams & Live Previews / Thomas Cherickal"; JSON-LD: WebPage, Person, Organization, ImageObject |
| [`claude-code-mode.html`](claude-code-mode.html) | HTML | 79.9 KB / 871L | Title: "Claude Code Mode: Programmatic Tool Orchestration & Sandboxed Execution / Thomas Cherickal"; JSON-LD: WebPage, Person, Organization |
| [`claude-code.html`](claude-code.html) | HTML | 83.2 KB / 925L | Title: "Claude Code: Terminal CLI, Agent Workflows & IDE Integration / Thomas Cherickal"; JSON-LD: WebPage, Person, Organization, ImageObject |
| [`claude-dashboards.html`](claude-dashboards.html) | HTML | 82.3 KB / 917L | Title: "Claude Dashboards: Executive Analytics & Live Data Visualization / Thomas Cherickal"; JSON-LD: WebPage, Person, Organization, ImageObject |
| [`claude-design.html`](claude-design.html) | HTML | 81.5 KB / 907L | Title: "Claude Design: UI/UX Prototyping & Design Systems / Thomas Cherickal"; JSON-LD: WebPage, Person, Organization, ImageObject |
| [`claude-desktop.html`](claude-desktop.html) | HTML | 82.6 KB / 912L | Title: "Claude Desktop: Native Workstation & MCP Workflows / Thomas Cherickal"; JSON-LD: WebPage, Person, Organization, ImageObject |
| [`claude-fable.html`](claude-fable.html) | HTML | 84.3 KB / 924L | Title: "Claude Fable: Frontier Long-Horizon Reasoning & Agentic Engineering / Thomas Cherickal"; JSON-LD: WebPage, Person, Organization, ImageObject |
| [`claude-local-llms.html`](claude-local-llms.html) | HTML | 81.9 KB / 925L | Title: "Claude Integration with Sovereign Local LLMs / Thomas Cherickal"; JSON-LD: WebPage, Person, Organization, ImageObject |
| [`claude-mcp.html`](claude-mcp.html) | HTML | 84.4 KB / 941L | Title: "Claude MCP: Model Context Protocol & Custom Tool Servers / Thomas Cherickal"; JSON-LD: WebPage, Person, Organization, ImageObject |
| [`claude-mods.html`](claude-mods.html) | HTML | 82.0 KB / 930L | Title: "Claude Mods: Custom Extensions, Hooks & Behavior Tuning / Thomas Cherickal"; JSON-LD: WebPage, Person, Organization, ImageObject |
| [`claude-office.html`](claude-office.html) | HTML | 83.0 KB / 924L | Title: "Claude Office: Enterprise Spreadsheets, Decks & Document Automation / Thomas Cherickal"; JSON-LD: WebPage, Person, Organization, ImageObject |
| [`claude-orchestration.html`](claude-orchestration.html) | HTML | 84.2 KB / 940L | Title: "Claude Multi-Agent Swarm Orchestration / Thomas Cherickal"; JSON-LD: WebPage, Person, Organization, ImageObject |
| [`claude-plugins.html`](claude-plugins.html) | HTML | 85.7 KB / 956L | Title: "Claude Plugins: Unified Skills, Connectors & Production Ecosystems / Thomas Cherickal"; JSON-LD: WebPage, Person, Organization, ImageObject |
| [`claude-projects.html`](claude-projects.html) | HTML | 83.7 KB / 977L | Title: "Claude Projects: Persistent Context & Knowledge Workspaces / Thomas Cherickal"; JSON-LD: WebPage, Person, Organization, ImageObject |
| [`claude-science.html`](claude-science.html) | HTML | 82.2 KB / 912L | Title: "Claude for Scientific Discovery & Quantitative Research / Thomas Cherickal"; JSON-LD: WebPage, Person, Organization, ImageObject |
| [`claude-whiteboards.html`](claude-whiteboards.html) | HTML | 85.9 KB / 1009L | Title: "Claude Whiteboards: Visual Architecture & Collaborative Diagramming / Thomas Cherickal"; JSON-LD: WebPage, Person, Organization, ImageObject |

## Context & Cross-References

- Root Directory: [`OKF.md`](../OKF.md)
- Canonical Specification: [`AGENTS.md`](../AGENTS.md) (`CLAUDE.md` and `GEMINI.md` symlink to `AGENTS.md`)
- Repository Telemetry: [`METRICS.md`](../METRICS.md)
- Related: [`data/`](../data/OKF.md) - Authoritative Data Directory (Single Source of Truth)
- Related: [`gemini-training/`](../gemini-training/OKF.md) - Google Gemini & Gemma Engineering Guides
- Related: [`md-html-sync/claude-training/`](../md-html-sync/claude-training/OKF.md) - Claude Training Markdown Mirrors
- Related: [`schema/`](../schema/OKF.md) - Structured Data & Schema.org JSON-LD Definitions
