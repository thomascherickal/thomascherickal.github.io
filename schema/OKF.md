---
type: dataset
title: "Structured Data & Schema.org JSON-LD Definitions"
description: "Comprehensive Schema.org JSON-LD definitions extracted from live HTML pages to empower search engine indexing and generative search engines (GEO/AEO)."
tags: [schema, json-ld, structured-data, schema-org, geo-aeo]
timestamp: 2026-11-27T06:35:11Z
---

# Structured Data & Schema.org JSON-LD Definitions

Comprehensive Schema.org JSON-LD definitions extracted from live HTML pages to empower search engine indexing and generative search engines (GEO/AEO).

## Directory Role & Agent Guidelines

- Contains 71+ JSON-LD files mirroring schemas embedded in live HTML pages.
- Live HTML files are the definitive source of truth for schemas; `python/sync_schemas.py` extracts schemas into `schema/`.
- Synchronized before committing via `python3 python/sync_schemas.py`.
- Ensures rich snippets for Organization, Person, Service, FAQPage, and TechnicalArticle.

## Subdirectories

None (Leaf directory).

## File Inventory

- [`404-schema.json`](404-schema.json)
- [`all-business-services-schema.json`](all-business-services-schema.json)
- [`all-personal-services-schema.json`](all-personal-services-schema.json)
- [`all-services-schema.json`](all-services-schema.json)
- [`articles-schema.json`](articles-schema.json)
- [`business-service-advanced-ai-agents.json`](business-service-advanced-ai-agents.json)
- [`business-service-agentic-assistants.json`](business-service-agentic-assistants.json)
- [`business-service-ai-assisted-leads-magnet.json`](business-service-ai-assisted-leads-magnet.json)
- [`business-service-ai-budget-cost-control.json`](business-service-ai-budget-cost-control.json)
- [`business-service-ai-developer-training.json`](business-service-ai-developer-training.json)
- [`business-service-ai-second-brain.json`](business-service-ai-second-brain.json)
- [`business-service-claude-agents-orchestration.json`](business-service-claude-agents-orchestration.json)
- [`business-service-claude-research-training.json`](business-service-claude-research-training.json)
- [`business-service-competitor-research.json`](business-service-competitor-research.json)
- [`business-service-continuous-learning.json`](business-service-continuous-learning.json)
- [`business-service-gemini-research-training.json`](business-service-gemini-research-training.json)
- [`business-service-generative-ai-transformation.json`](business-service-generative-ai-transformation.json)
- [`business-service-knowledge-graphs.json`](business-service-knowledge-graphs.json)
- [`business-service-local-ai-for-sensitive-data.json`](business-service-local-ai-for-sensitive-data.json)
- [`business-service-local-llms-cost-slashing.json`](business-service-local-llms-cost-slashing.json)
- [`business-service-workflow-automations.json`](business-service-workflow-automations.json)
- [`claude-101-schema.json`](claude-101-schema.json)
- [`claude-apps-schema.json`](claude-apps-schema.json)
- [`claude-artifacts-schema.json`](claude-artifacts-schema.json)
- [`claude-code-mode-schema.json`](claude-code-mode-schema.json)
- [`claude-code-schema.json`](claude-code-schema.json)
- [`claude-cowork-schema.json`](claude-cowork-schema.json)
- [`claude-custom-mcp-schema.json`](claude-custom-mcp-schema.json)
- [`claude-dashboards-schema.json`](claude-dashboards-schema.json)
- [`claude-design-schema.json`](claude-design-schema.json)
- [`claude-desktop-schema.json`](claude-desktop-schema.json)
- [`claude-fable-schema.json`](claude-fable-schema.json)
- [`claude-features-schema.json`](claude-features-schema.json)
- [`claude-local-llms-schema.json`](claude-local-llms-schema.json)
- [`claude-mcp-schema.json`](claude-mcp-schema.json)
- [`claude-mods-schema.json`](claude-mods-schema.json)
- [`claude-office-schema.json`](claude-office-schema.json)
- [`claude-orchestration-schema.json`](claude-orchestration-schema.json)
- [`claude-plugins-schema.json`](claude-plugins-schema.json)
- [`claude-projects-schema.json`](claude-projects-schema.json)
- ... and 46 additional files.

## Context & Cross-References

- Root Directory: [`/OKF.md`](../OKF.md)
- Architectural Specification: [`GEMINI.md`](/GEMINI.md)
- Repository Telemetry: [`METRICS.md`](/METRICS.md)
