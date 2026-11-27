---
type: reports
title: "Failed Test Reports Directory"
description: "Centralized directory where Python, JavaScript, and HTML UI test runners store failure reports when tests fail."
tags: [failed-tests, testing, reports]
timestamp: 2026-11-27T06:35:11Z
---

# Failed Test Reports Directory

Centralized directory where Python, JavaScript, and HTML UI test runners store failure reports when tests fail.

## Directory Role & Agent Guidelines

- All failed-test reports must be written exclusively inside `failed-tests/`.
- Reports are automatically deleted by each runner when its suite passes 100%.

## Subdirectories

None (Leaf directory).

## File Inventory

- [`readme.md`](readme.md)

## Context & Cross-References

- Root Directory: [`/OKF.md`](../OKF.md)
- Architectural Specification: [`GEMINI.md`](/GEMINI.md)
- Repository Telemetry: [`METRICS.md`](/METRICS.md)
