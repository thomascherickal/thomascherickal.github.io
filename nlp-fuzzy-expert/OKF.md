---
type: module
title: "NLP Fuzzy Expert Knowledge Synchronization Engine"
description: "Forward-chaining inference engine, fuzzy matcher, continuous file watcher daemon, and auto-healing subsystem synchronizing data/*.json outward."
tags: [nlp-fuzzy-expert, inference-engine, synchronization, fuzzy-matcher, watcher]
timestamp: 2026-11-10T08:43:02Z
---

# NLP Fuzzy Expert Knowledge Synchronization Engine

Forward-chaining inference engine, fuzzy matcher, continuous file watcher daemon, and auto-healing subsystem synchronizing data/*.json outward.

## Directory Role & Agent Guidelines

- Core engine: `expert_system.py`, `synchronizer.py`, `watcher.py`, and `auto_healer.py`.
- Run `python3 nlp-fuzzy-expert/synchronizer.py` to synchronize facts from `data/*.json` outward to HTML and JS files.
- Prints working steps 1-6 with 1-based changed line numbers and an ASCII summary table.
- Optimization: Summaries are printed to terminal; any failed checks are recorded in `FAILED-CHECKS.md`. If all pass, it records `Everything passed.`
- `save_json()` strictly blocks writing to any path inside `data/` via `PermissionError`.

## Subdirectories

None (Leaf directory).

## File Inventory

- [`.scanner_cache.json`](.scanner_cache.json)
- [`FAILED-CHECKS.md`](FAILED-CHECKS.md)
- [`README.md`](README.md)
- [`__init__.py`](__init__.py)
- [`auto_healer.py`](auto_healer.py)
- [`business-services.md`](business-services.md)
- [`business_services_references.json`](business_services_references.json)
- [`claude.md`](claude.md)
- [`claude_references.json`](claude_references.json)
- [`code-sync.md`](code-sync.md)
- [`code_references.json`](code_references.json)
- [`code_sync_config.json`](code_sync_config.json)
- [`code_test_syncer.py`](code_test_syncer.py)
- [`config.md`](config.md)
- [`config_references.json`](config_references.json)
- [`contact.md`](contact.md)
- [`contact_references.json`](contact_references.json)
- [`expert_system.py`](expert_system.py)
- [`faqs.md`](faqs.md)
- [`faqs_references.json`](faqs_references.json)
- [`fuzzy_engine.py`](fuzzy_engine.py)
- [`gemini.md`](gemini.md)
- [`gemini_references.json`](gemini_references.json)
- [`guarantees.md`](guarantees.md)
- [`guarantees_references.json`](guarantees_references.json)
- [`nlp_engine.py`](nlp_engine.py)
- [`personal-services.md`](personal-services.md)
- [`personal_services_references.json`](personal_services_references.json)
- [`pricing.md`](pricing.md)
- [`pricing_references.json`](pricing_references.json)
- [`publications.md`](publications.md)
- [`publications_references.json`](publications_references.json)
- [`reference_scanner.py`](reference_scanner.py)
- [`references_dictionary.json`](references_dictionary.json)
- [`references_manifest.json`](references_manifest.json)
- [`report_generator.py`](report_generator.py)
- [`rules.py`](rules.py)
- [`sync_manifest.json`](sync_manifest.json)
- [`synchronizer.py`](synchronizer.py)
- [`tiers.md`](tiers.md)
- ... and 2 additional files.

## Context & Cross-References

- Root Directory: [`/OKF.md`](../OKF.md)
- Architectural Specification: [`GEMINI.md`](/GEMINI.md)
- Repository Telemetry: [`METRICS.md`](/METRICS.md)
