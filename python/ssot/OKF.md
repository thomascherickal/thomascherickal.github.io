---
type: module
title: "Deterministic SSOT Build & Synchronization Pipeline"
description: "Schema validation, deterministic marker/block rendering, literal auditing, and JS data export from data/*.json."
tags: [ssot, validation, jinja2, markers, synchronization]
timestamp: 2026-11-27T06:35:11Z
---

# Deterministic SSOT Build & Synchronization Pipeline

Schema validation, deterministic marker/block rendering, literal auditing, and JS data export from data/*.json.

## Directory Role & Agent Guidelines

- Core pipeline: `load.py`, `format.py`, `validate.py`, `render.py`, `export_site_data.py`, and `scan_literals.py`.
- Run `python3 python/ssot/validate.py` to verify all `data/*.json` schemas and cross-file invariants.
- Run `python3 python/ssot/render.py` to render `<!--ssot:...-->` markers deterministically across HTML and `llms.txt`.
- Use `python3 python/ssot/render.py --check` in hooks and CI to fail loudly on unrendered drift.

## Subdirectories

None (Leaf directory).

## File Inventory

- [`__init__.py`](__init__.py)
- [`export_site_data.py`](export_site_data.py)
- [`format.py`](format.py)
- [`load.py`](load.py)
- [`render.py`](render.py)
- [`scan_literals.py`](scan_literals.py)
- [`validate.py`](validate.py)

## Context & Cross-References

- Parent Directory: [`../OKF.md`](../OKF.md)
- Architectural Specification: [`GEMINI.md`](/GEMINI.md)
- Repository Telemetry: [`METRICS.md`](/METRICS.md)
