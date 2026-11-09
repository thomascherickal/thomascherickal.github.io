---
type: dataset
title: "Authoritative Knowledge Data Hub"
description: "Single Source of Truth (SSOT) JSON data files modeling pricing, tiers, services, and configs."
tags: [data, ssot, authoritative, json]
timestamp: 2026-11-09T20:35:37Z
---

# Authoritative Knowledge Data Hub

Single Source of Truth (SSOT) JSON data files modeling pricing, tiers, services, and configs.

## Directory Role & Agent Guidelines

- **CRITICAL AGENT RULE**: The `data/` directory is strictly read-only to source control and code scripts.
- All JSON files here are the authoritative Single Source of Truth (SSOT) specified by `GEMINI.md`.
- Data flows strictly outward (`data/*.json` -> HTML / JS / Markdown / Schemas). Never edit or mutate programmatically.

## Subdirectories

None (Leaf directory).

## File Inventory

- [`business-services.json`](business-services.json)
- [`claude.json`](claude.json)
- [`config.json`](config.json)
- [`contact.json`](contact.json)
- [`faqs.json`](faqs.json)
- [`gemini.json`](gemini.json)
- [`guarantees.json`](guarantees.json)
- [`personal-services.json`](personal-services.json)
- [`pricing.json`](pricing.json)
- [`publications.json`](publications.json)
- [`tiers.json`](tiers.json)

## Context & Cross-References

- Root Directory: [`/OKF.md`](../OKF.md)
- Architectural Specification: [`GEMINI.md`](/GEMINI.md)
- Repository Telemetry: [`METRICS.md`](/METRICS.md)
