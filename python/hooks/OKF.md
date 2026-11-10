---
type: module
title: "Git Repository Lifecycle Hooks"
description: "Pre-commit, pre-push, and editor lifecycle hooks enforcing quality gates, test pass rates, and schema synchronization."
tags: [python, git-hooks, pre-commit, pre-push, quality-gate]
timestamp: 2026-11-10T08:43:02Z
---

# Git Repository Lifecycle Hooks

Pre-commit, pre-push, and editor lifecycle hooks enforcing quality gates, test pass rates, and schema synchronization.

## Directory Role & Agent Guidelines

- `pre-commit`: Automatically executes `python3 python/run_tests.py` and `node pricing-tests.js`. Aborts commit if any test fails.
- `pre-push`: Performs final link and schema verification before pushing to origin.
- `post_edit_hook.py`: IDE lifecycle trigger for component inlining.
- Never bypass git hooks with `--no-verify`.

## Subdirectories

None (Leaf directory).

## File Inventory

- [`post_edit_hook.py`](post_edit_hook.py)
- [`pre-commit`](pre-commit)
- [`pre-push`](pre-push)

## Context & Cross-References

- Parent Directory: [`../OKF.md`](../OKF.md)
- Architectural Specification: [`GEMINI.md`](/GEMINI.md)
- Repository Telemetry: [`METRICS.md`](/METRICS.md)
