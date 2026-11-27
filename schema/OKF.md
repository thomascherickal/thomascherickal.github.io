---
type: dataset
title: "Structured Data & Schema.org JSON-LD Definitions"
description: "Standalone JSON-LD Schema.org graphs extracted from live HTML pages to support search engines, rich snippets, and Generative Engine Optimization (GEO/AEO)."
tags: [schema, json-ld, structured-data, schema-org, geo-aeo]
timestamp: 2026-11-27T06:56:49Z
---

# Structured Data & Schema.org JSON-LD Definitions

Standalone JSON-LD Schema.org graphs extracted from live HTML pages to support search engines, rich snippets, and Generative Engine Optimization (GEO/AEO).

## Live Directory Telemetry

- **Path**: `/schema`
- **Direct Files**: 86 (86 JSON)
- **Subdirectories**: 0
- **Total Size**: 362.5 KB | **Text/Code Lines**: 8,574
- **Primary Workflow**: Live HTML `<script type="application/ld+json">` blocks are the source of truth; synchronize into `schema/` before committing via `python3 python/sync_schemas.py`.

## Directory Role & Agent Invariants

- Every HTML page maps deterministically to a JSON-LD file in `schema/` (`index.html` -> `home-schema.json`, `service-*` -> `service-*.json`, guides -> `<page>-schema.json`).
- Uses canonical author description (`Thomas Cherickal - Generative AI Consultant and The Digital Futurist.`), canonical email `thomascherickal@gmail.com`, and includes `https://dev.to/thomascherickal` in `sameAs`.

## Subdirectories

Leaf directory (no child directories).

## Detailed File Inventory

| File | Type | Size / Lines | In-Context Details & Cross-References |
| :--- | :---: | :---: | :--- |
| [`404-schema.json`](404-schema.json) | JSON | 1.1 KB / 37L | JSON-LD `@type`: WebPage, BreadcrumbList; Entity: "404 Page Not Found - Thomas Cherickal" |
| [`all-business-services-schema.json`](all-business-services-schema.json) | JSON | 1.2 KB / 37L | JSON-LD `@type`: WebPage, BreadcrumbList; Entity: "All Business Services - Business AI Transformation - Thomas Cherickal" |
| [`all-personal-services-schema.json`](all-personal-services-schema.json) | JSON | 1.2 KB / 37L | JSON-LD `@type`: WebPage, BreadcrumbList; Entity: "All Personal Services - AI Mentorship & Training - Thomas Cherickal" |
| [`all-services-schema.json`](all-services-schema.json) | JSON | 1.1 KB / 37L | JSON-LD `@type`: WebPage, BreadcrumbList; Entity: "All Services - Business & Personal AI Transformation - Thomas Cheri..." |
| [`articles-schema.json`](articles-schema.json) | JSON | 17.5 KB / 293L | JSON-LD `@type`: CollectionPage, BreadcrumbList, ItemList; Entity: "Featured Articles / Thomas Cherickal" |
| [`business-service-advanced-ai-agents.json`](business-service-advanced-ai-agents.json) | JSON | 2.5 KB / 58L | JSON-LD `@type`: WebPage, BreadcrumbList, Service; Entity: "Advanced AI Agents: Multi-Agent Swarms, Self-Correction & Autonomou..." |
| [`business-service-agentic-assistants.json`](business-service-agentic-assistants.json) | JSON | 3.1 KB / 74L | JSON-LD `@type`: WebPage, BreadcrumbList, Service; Entity: "Agentic Assistants Training / Thomas Cherickal" |
| [`business-service-ai-assisted-leads-magnet.json`](business-service-ai-assisted-leads-magnet.json) | JSON | 3.1 KB / 74L | JSON-LD `@type`: WebPage, BreadcrumbList, Service; Entity: "AI Leads Magnet & Conversion Systems / Thomas Cherickal" |
| [`business-service-ai-budget-cost-control.json`](business-service-ai-budget-cost-control.json) | JSON | 3.1 KB / 74L | JSON-LD `@type`: WebPage, BreadcrumbList, Service; Entity: "AI Budget & Cost Control / Thomas Cherickal" |
| [`business-service-ai-developer-training.json`](business-service-ai-developer-training.json) | JSON | 3.1 KB / 74L | JSON-LD `@type`: WebPage, BreadcrumbList, Service; Entity: "AI Developer Training / Thomas Cherickal" |
| [`business-service-ai-second-brain.json`](business-service-ai-second-brain.json) | JSON | 3.0 KB / 74L | JSON-LD `@type`: WebPage, BreadcrumbList, Service; Entity: "AI Second Brain / Thomas Cherickal" |
| [`business-service-claude-agents-orchestration.json`](business-service-claude-agents-orchestration.json) | JSON | 2.7 KB / 59L | JSON-LD `@type`: WebPage, BreadcrumbList, Service; Entity: "Guide to AI Agent Orchestration with Claude Desktop / Thomas Cherickal" |
| [`business-service-claude-research-training.json`](business-service-claude-research-training.json) | JSON | 3.2 KB / 74L | JSON-LD `@type`: WebPage, BreadcrumbList, Service; Entity: "Claude Research Training: Scientific Breakthroughs & Corporate R&D..." |
| [`business-service-competitor-research.json`](business-service-competitor-research.json) | JSON | 2.6 KB / 58L | JSON-LD `@type`: WebPage, BreadcrumbList, Service; Entity: "Competitor Research & Intelligence - Service Specification / Thomas..." |
| [`business-service-continuous-learning.json`](business-service-continuous-learning.json) | JSON | 2.6 KB / 58L | JSON-LD `@type`: WebPage, BreadcrumbList, Service; Entity: "Continuous Learning & Organizational Research - Service Specificati..." |
| [`business-service-gemini-research-training.json`](business-service-gemini-research-training.json) | JSON | 3.1 KB / 74L | JSON-LD `@type`: WebPage, BreadcrumbList, Service; Entity: "Gemini Research Training: Scientific Breakthroughs & Corporate R&D..." |
| [`business-service-generative-ai-transformation.json`](business-service-generative-ai-transformation.json) | JSON | 2.6 KB / 58L | JSON-LD `@type`: WebPage, BreadcrumbList, Service; Entity: "Generative AI Transformation - Service Specification / Thomas Cheri..." |
| [`business-service-knowledge-graphs.json`](business-service-knowledge-graphs.json) | JSON | 3.1 KB / 74L | JSON-LD `@type`: WebPage, BreadcrumbList, Service; Entity: "Knowledge Graphs & GraphRAG / Thomas Cherickal" |
| [`business-service-local-ai-for-sensitive-data.json`](business-service-local-ai-for-sensitive-data.json) | JSON | 2.6 KB / 58L | JSON-LD `@type`: WebPage, BreadcrumbList, Service; Entity: "Local AI for Sensitive Data - Service Specification / Thomas Cherickal" |
| [`business-service-local-llms-cost-slashing.json`](business-service-local-llms-cost-slashing.json) | JSON | 3.1 KB / 74L | JSON-LD `@type`: WebPage, BreadcrumbList, Service; Entity: "Local LLMs & Cost Slashing / Thomas Cherickal" |
| [`business-service-workflow-automations.json`](business-service-workflow-automations.json) | JSON | 2.6 KB / 58L | JSON-LD `@type`: WebPage, BreadcrumbList, Service; Entity: "Workflow Automations - Service Specification / Thomas Cherickal" |
| [`claude-101-schema.json`](claude-101-schema.json) | JSON | 3.0 KB / 88L | JSON-LD `@type`: TechArticle, Article, BreadcrumbList; Entity: "Claude 101: Architecture, Features & Agentic Engineering" |
| [`claude-apps-schema.json`](claude-apps-schema.json) | JSON | 2.0 KB / 53L | JSON-LD `@type`: TechArticle, Article; Entity: "Claude Apps (MCP Apps) Interactive UI & Workflows" |
| [`claude-artifacts-schema.json`](claude-artifacts-schema.json) | JSON | 3.1 KB / 89L | JSON-LD `@type`: TechArticle, Article, BreadcrumbList; Entity: "Claude Artifacts: Interactive Code, Diagrams & Live Previews" |
| [`claude-code-mode-schema.json`](claude-code-mode-schema.json) | JSON | 2.1 KB / 53L | JSON-LD `@type`: TechArticle, Article; Entity: "Claude Code Mode: Programmatic Tool Orchestration" |
| [`claude-code-schema.json`](claude-code-schema.json) | JSON | 3.1 KB / 89L | JSON-LD `@type`: TechArticle, Article, BreadcrumbList; Entity: "Claude Code: Terminal CLI, Agent Workflows & IDE Integration" |
| [`claude-cowork-schema.json`](claude-cowork-schema.json) | JSON | 3.1 KB / 89L | JSON-LD `@type`: TechArticle, Article, BreadcrumbList; Entity: "Claude and Microsoft 365: Office Work, Delegated" |
| [`claude-custom-mcp-schema.json`](claude-custom-mcp-schema.json) | JSON | 3.2 KB / 90L | JSON-LD `@type`: TechArticle, Article, BreadcrumbList; Entity: "Custom MCP Servers With Claude: Connecting AI to Systems Across the..." |
| [`claude-dashboards-schema.json`](claude-dashboards-schema.json) | JSON | 3.2 KB / 89L | JSON-LD `@type`: TechArticle, Article, BreadcrumbList; Entity: "Claude Dashboards: Executive Analytics & Live Data Visualization" |
| [`claude-design-schema.json`](claude-design-schema.json) | JSON | 3.1 KB / 88L | JSON-LD `@type`: TechArticle, Article, BreadcrumbList; Entity: "Claude Design: UI/UX Prototyping & Design Systems" |
| [`claude-desktop-schema.json`](claude-desktop-schema.json) | JSON | 3.1 KB / 89L | JSON-LD `@type`: TechArticle, Article, BreadcrumbList; Entity: "Claude Desktop: Native Workstation & MCP Workflows" |
| [`claude-fable-schema.json`](claude-fable-schema.json) | JSON | 3.1 KB / 89L | JSON-LD `@type`: TechArticle, Article, BreadcrumbList; Entity: "Claude Fable: Frontier Models & Long-Horizon Reasoning" |
| [`claude-features-schema.json`](claude-features-schema.json) | JSON | 3.1 KB / 88L | JSON-LD `@type`: TechArticle, Article, BreadcrumbList; Entity: "Claude Features: Projects, Artifacts, Deep Research & Extensibility" |
| [`claude-local-llms-schema.json`](claude-local-llms-schema.json) | JSON | 3.1 KB / 88L | JSON-LD `@type`: TechArticle, Article, BreadcrumbList; Entity: "Claude Integration with Sovereign Local LLMs" |
| [`claude-mcp-schema.json`](claude-mcp-schema.json) | JSON | 3.1 KB / 88L | JSON-LD `@type`: TechArticle, Article, BreadcrumbList; Entity: "Claude MCP: Model Context Protocol & Custom Tool Servers" |
| [`claude-mods-schema.json`](claude-mods-schema.json) | JSON | 3.1 KB / 89L | JSON-LD `@type`: TechArticle, Article, BreadcrumbList; Entity: "Claude Mods: Custom Extensions, Hooks & Behavior Tuning" |
| [`claude-office-schema.json`](claude-office-schema.json) | JSON | 3.1 KB / 89L | JSON-LD `@type`: TechArticle, Article, BreadcrumbList; Entity: "Claude Office: Enterprise Spreadsheets, Decks & Document Automation" |
| [`claude-orchestration-schema.json`](claude-orchestration-schema.json) | JSON | 3.1 KB / 89L | JSON-LD `@type`: TechArticle, Article, BreadcrumbList; Entity: "Claude Multi-Agent Swarm Orchestration" |
| [`claude-plugins-schema.json`](claude-plugins-schema.json) | JSON | 3.1 KB / 88L | JSON-LD `@type`: TechArticle, Article, BreadcrumbList; Entity: "Claude Plugins Architecture & Production Ecosystems" |
| [`claude-projects-schema.json`](claude-projects-schema.json) | JSON | 4.7 KB / 119L | JSON-LD `@type`: TechArticle, Article, BreadcrumbList, FAQPage; Entity: "Claude Projects: Persistent Context & Knowledge Workspaces" |
| [`claude-science-schema.json`](claude-science-schema.json) | JSON | 3.1 KB / 88L | JSON-LD `@type`: TechArticle, Article, BreadcrumbList; Entity: "Claude for Scientific Discovery & Quantitative Research" |
| [`claude-whiteboards-schema.json`](claude-whiteboards-schema.json) | JSON | 5.4 KB / 127L | JSON-LD `@type`: TechArticle, Article, BreadcrumbList, FAQPage; Entity: "Claude Whiteboards: Visual Architecture & Collaborative Diagramming" |
| [`contact-schema.json`](contact-schema.json) | JSON | 4.9 KB / 131L | JSON-LD `@type`: ContactPage, BreadcrumbList, ProfessionalService; Entity: "Contact & Project Intake - Thomas Cherickal / The Digital Futurist" |
| [`expertise-schema.json`](expertise-schema.json) | JSON | 10.0 KB / 227L | JSON-LD `@type`: AboutPage, BreadcrumbList, FAQPage, ItemList; Entity: "Capabilities & Tech Stack - Thomas Cherickal" |
| [`faqs-schema.json`](faqs-schema.json) | JSON | 41.4 KB / 534L | JSON-LD `@type`: FAQPage |
| [`google-ai-studio-schema.json`](google-ai-studio-schema.json) | JSON | 3.2 KB / 90L | JSON-LD `@type`: TechArticle, Article, BreadcrumbList; Entity: "Google AI Studio: Rapid Prototyping and System Instructions" |
| [`google-antigravity-chat-schema.json`](google-antigravity-chat-schema.json) | JSON | 3.3 KB / 91L | JSON-LD `@type`: TechArticle, Article, BreadcrumbList; Entity: "Google Antigravity Chat: Agentic Workspace Conversations and Tool O..." |
| [`google-antigravity-ide-schema.json`](google-antigravity-ide-schema.json) | JSON | 3.3 KB / 90L | JSON-LD `@type`: TechArticle, Article, BreadcrumbList; Entity: "Google Antigravity IDE: Autonomous Multi-Agent Development" |
| [`google-code-wiki-schema.json`](google-code-wiki-schema.json) | JSON | 3.3 KB / 91L | JSON-LD `@type`: TechArticle, Article, BreadcrumbList; Entity: "Google Code Wiki: Automated Repository Documentation & Architecture..." |
| [`google-deep-research-schema.json`](google-deep-research-schema.json) | JSON | 3.2 KB / 90L | JSON-LD `@type`: TechArticle, Article, BreadcrumbList; Entity: "Google Deep Research: Autonomous Multi-Step Web and Corpus Synthesis" |
| [`google-deepmind-schema.json`](google-deepmind-schema.json) | JSON | 3.2 KB / 90L | JSON-LD `@type`: TechArticle, Article, BreadcrumbList; Entity: "Google DeepMind: Frontier AI Research, Element Models and Scientifi..." |
| [`google-gemini-101-schema.json`](google-gemini-101-schema.json) | JSON | 3.2 KB / 90L | JSON-LD `@type`: TechArticle, Article, BreadcrumbList; Entity: "Google Gemini 101: Frontier Multimodal Architecture" |
| [`google-gemini-4-argon-schema.json`](google-gemini-4-argon-schema.json) | JSON | 3.2 KB / 90L | JSON-LD `@type`: TechArticle, Article, BreadcrumbList; Entity: "Gemini Argon: Frontier Reasoning and Long-Horizon Agents" |
| [`google-gemini-4-carbon-schema.json`](google-gemini-4-carbon-schema.json) | JSON | 3.2 KB / 90L | JSON-LD `@type`: TechArticle, Article, BreadcrumbList; Entity: "Gemini Carbon: Deep Thinking and Long-Horizon Autonomous Agents" |
| [`google-gemini-4-helium-schema.json`](google-gemini-4-helium-schema.json) | JSON | 3.2 KB / 90L | JSON-LD `@type`: TechArticle, Article, BreadcrumbList; Entity: "Gemini Helium: The Next-Generation Flash-Lite Architecture Preview" |
| [`google-gemini-4-helix-schema.json`](google-gemini-4-helix-schema.json) | JSON | 3.4 KB / 93L | JSON-LD `@type`: TechArticle, Article, BreadcrumbList; Entity: "Gemini 4 Helix: The New Flash-Lite Architecture" |
| [`google-gemini-4-hydrogen-schema.json`](google-gemini-4-hydrogen-schema.json) | JSON | 3.3 KB / 90L | JSON-LD `@type`: TechArticle, Article, BreadcrumbList; Entity: "Gemini Hydrogen: Fast Flash Foundation and High-Throughput Swarms" |
| [`google-gemini-4-krypton-schema.json`](google-gemini-4-krypton-schema.json) | JSON | 3.3 KB / 90L | JSON-LD `@type`: TechArticle, Article, BreadcrumbList; Entity: "Gemini Krypton: Next-Generation Frontier Architecture Preview" |
| [`google-gemini-4-neon-schema.json`](google-gemini-4-neon-schema.json) | JSON | 3.2 KB / 91L | JSON-LD `@type`: TechArticle, Article, BreadcrumbList; Entity: "Gemini Neon: Reasoning, Coding, and 24/7 Agentic Assistants" |
| [`google-gemini-desktop-schema.json`](google-gemini-desktop-schema.json) | JSON | 3.2 KB / 90L | JSON-LD `@type`: TechArticle, Article, BreadcrumbList; Entity: "Gemini Desktop: Computer Use and Operating System Automation" |
| [`google-gemini-notebook-schema.json`](google-gemini-notebook-schema.json) | JSON | 3.2 KB / 90L | JSON-LD `@type`: TechArticle, Article, BreadcrumbList; Entity: "Gemini Notebook and Deep Research Grounding" |
| [`google-gemini-skills-schema.json`](google-gemini-skills-schema.json) | JSON | 3.2 KB / 90L | JSON-LD `@type`: TechArticle, Article, BreadcrumbList; Entity: "Gemini Skills: Reusable Agentic Capabilities and Workflow Modules" |
| [`google-gemini-spark-schema.json`](google-gemini-spark-schema.json) | JSON | 3.2 KB / 90L | JSON-LD `@type`: TechArticle, Article, BreadcrumbList; Entity: "Gemini Spark: Lightweight High-Throughput Inference and Always-On A..." |
| [`google-gemma-local-models-schema.json`](google-gemma-local-models-schema.json) | JSON | 3.2 KB / 91L | JSON-LD `@type`: TechArticle, Article, BreadcrumbList; Entity: "Google Gemma Local Models: A Practical Guide" |
| [`google-multimodality-schema.json`](google-multimodality-schema.json) | JSON | 3.2 KB / 90L | JSON-LD `@type`: TechArticle, Article, BreadcrumbList; Entity: "Google Multimodality: Creative Generative Ecosystem" |
| [`guarantees-schema.json`](guarantees-schema.json) | JSON | 1.7 KB / 48L | JSON-LD `@type`: WebPage, AboutPage, BreadcrumbList; Entity: "Trust Guarantees & Payment Structure / Thomas Cherickal" |
| [`home-schema.json`](home-schema.json) | JSON | 10.7 KB / 276L | JSON-LD `@type`: WebSite, Organization, ProfilePage, ImageObject, Person; Entity: "Thomas Cherickal - The Digital Futurist" |
| [`parity-pricing-schema.json`](parity-pricing-schema.json) | JSON | 4.3 KB / 114L | JSON-LD `@type`: WebPage, BreadcrumbList, OfferCatalog; Entity: "Country-Wise Parity Pricing (195 Countries) / The Digital Futurist" |
| [`personal-service-advanced-ai-agents.json`](personal-service-advanced-ai-agents.json) | JSON | 3.1 KB / 69L | JSON-LD `@type`: WebPage, BreadcrumbList, Service; Entity: "Advanced AI Agents - Personal Service / Thomas Cherickal" |
| [`personal-service-agentic-assistants.json`](personal-service-agentic-assistants.json) | JSON | 3.2 KB / 69L | JSON-LD `@type`: WebPage, BreadcrumbList, Service; Entity: "Agentic Assistant Training - Personal Service / Thomas Cherickal" |
| [`personal-service-ai-assisted-leads-magnet.json`](personal-service-ai-assisted-leads-magnet.json) | JSON | 3.2 KB / 69L | JSON-LD `@type`: WebPage, BreadcrumbList, Service; Entity: "AI-Assisted Leads Magnet - Personal Service / Thomas Cherickal" |
| [`personal-service-ai-budget-cost-control.json`](personal-service-ai-budget-cost-control.json) | JSON | 3.2 KB / 69L | JSON-LD `@type`: WebPage, BreadcrumbList, Service; Entity: "AI Budget & Cost Control - Personal Service / Thomas Cherickal" |
| [`personal-service-ai-developer-training.json`](personal-service-ai-developer-training.json) | JSON | 3.2 KB / 69L | JSON-LD `@type`: WebPage, BreadcrumbList, Service; Entity: "AI Developer Training - Personal Service / Thomas Cherickal" |
| [`personal-service-ai-second-brain.json`](personal-service-ai-second-brain.json) | JSON | 3.1 KB / 69L | JSON-LD `@type`: WebPage, BreadcrumbList, Service; Entity: "AI Second Brain - Personal Service / Thomas Cherickal" |
| [`personal-service-claude-agents-orchestration.json`](personal-service-claude-agents-orchestration.json) | JSON | 3.2 KB / 69L | JSON-LD `@type`: WebPage, BreadcrumbList, Service; Entity: "Claude Agent Orchestration - Personal Service / Thomas Cherickal" |
| [`personal-service-claude-research-training.json`](personal-service-claude-research-training.json) | JSON | 3.3 KB / 69L | JSON-LD `@type`: WebPage, BreadcrumbList, Service; Entity: "Claude Research Training - Personal Service / Thomas Cherickal" |
| [`personal-service-competitor-research.json`](personal-service-competitor-research.json) | JSON | 3.2 KB / 69L | JSON-LD `@type`: WebPage, BreadcrumbList, Service; Entity: "Competitor Research - Personal Service / Thomas Cherickal" |
| [`personal-service-continuous-learning.json`](personal-service-continuous-learning.json) | JSON | 3.2 KB / 69L | JSON-LD `@type`: WebPage, BreadcrumbList, Service; Entity: "Continuous Learning & Synthesis - Personal Service / Thomas Cherickal" |
| [`personal-service-gemini-research-training.json`](personal-service-gemini-research-training.json) | JSON | 3.3 KB / 69L | JSON-LD `@type`: WebPage, BreadcrumbList, Service; Entity: "Gemini Research Training - Personal Service / Thomas Cherickal" |
| [`personal-service-generative-ai-transformation.json`](personal-service-generative-ai-transformation.json) | JSON | 3.2 KB / 69L | JSON-LD `@type`: WebPage, BreadcrumbList, Service; Entity: "Generative AI Transformation - Personal Service / Thomas Cherickal" |
| [`personal-service-knowledge-graphs.json`](personal-service-knowledge-graphs.json) | JSON | 3.1 KB / 69L | JSON-LD `@type`: WebPage, BreadcrumbList, Service; Entity: "Knowledge Graphs - Personal Service / Thomas Cherickal" |
| [`personal-service-local-ai-for-sensitive-data.json`](personal-service-local-ai-for-sensitive-data.json) | JSON | 3.3 KB / 69L | JSON-LD `@type`: WebPage, BreadcrumbList, Service; Entity: "Local AI for Sensitive Data - Personal Service / Thomas Cherickal" |
| [`personal-service-local-llms-cost-slashing.json`](personal-service-local-llms-cost-slashing.json) | JSON | 3.2 KB / 69L | JSON-LD `@type`: WebPage, BreadcrumbList, Service; Entity: "Local LLMs & Cost Slashing - Personal Service / Thomas Cherickal" |
| [`personal-service-workflow-automations.json`](personal-service-workflow-automations.json) | JSON | 3.1 KB / 69L | JSON-LD `@type`: WebPage, BreadcrumbList, Service; Entity: "Workflow Automations - Personal Service / Thomas Cherickal" |
| [`pricing-schema.json`](pricing-schema.json) | JSON | 10.9 KB / 212L | JSON-LD `@type`: WebPage, BreadcrumbList, OfferCatalog; Entity: "Services Pricing & Engagement Packages - Thomas Cherickal" |
| [`writing-schema.json`](writing-schema.json) | JSON | 27.6 KB / 748L | JSON-LD `@type`: CollectionPage, BreadcrumbList, Book, ItemList; Entity: "Publications - Thomas Cherickal" |

## Context & Cross-References

- Root Directory: [`OKF.md`](../OKF.md)
- Canonical Specification: [`AGENTS.md`](../AGENTS.md) (`CLAUDE.md` and `GEMINI.md` symlink to `AGENTS.md`)
- Repository Telemetry: [`METRICS.md`](../METRICS.md)
- Related: [Repository Root](../OKF.md) - Repository Root & Web Portal Hub
- Related: [`business-services/`](../business-services/OKF.md) - Enterprise & Business AI Services Catalog (16 Services)
- Related: [`personal-services/`](../personal-services/OKF.md) - Personal AI Training Services Catalog (16 Services)
- Related: [`claude-training/`](../claude-training/OKF.md) - Claude Architecture & Engineering Guides
- Related: [`gemini-training/`](../gemini-training/OKF.md) - Google Gemini & Gemma Engineering Guides
- Related: [`python/`](../python/OKF.md) - Repository Python Automation & Build Tooling
