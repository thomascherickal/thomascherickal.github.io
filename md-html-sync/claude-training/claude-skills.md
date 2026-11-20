# Claude Skills: Custom Skill Engineering & Repositories
## AI Agent Architecture & Orchestration - Thomas Cherickal

> **// PROVIDER DEEP DIVE**  
> **Learn how to author modular SKILL.md capability packs with YAML metadata, clear step-by-step protocols, and bundled helper scripts. Covers progressive disclosure, skill composition, testing conventions, and clean repository organization so Claude executes specialized team workflows reliably on demand.**

---

## Table of Contents

If you find yourself pasting the same 40-line checklist into Claude every week, it belongs in a Skill. A Claude Skill is a small, self-contained folder centered on a SKILL.md file that teaches Claude a repeatable workflow - complete with trigger metadata, step-by-step instructions, and optional helper scripts - loaded only when the task actually needs it.

1. What a Claude Skill Is and How Progressive Disclosure Works
2. Anatomy of a Clean SKILL.md Pack
3. Composing Skills and Organizing a Team Skill Repository
4. Testing Skills Before Sharing Them with Your Team
5. Complete Ready-to-Use SKILL.md Example
6. Authoritative References

---

## What a Claude Skill Is and How Progressive Disclosure Works

Loading every company handbook into every chat wastes tokens and confuses the model. Claude Skills use progressive disclosure: Claude first scans only the short YAML title and description of each available skill. Only when your request matches a skill does Claude read the full SKILL.md instructions and run any bundled helper scripts.

---

## Anatomy of a Clean SKILL.md Pack

Following the Agent Skills open specification, every skill folder stays simple and readable:

- SKILL.md - Required main file with YAML frontmatter (name and description) and clear numbered workflow steps.
- scripts/ - Optional Python or Bash helpers for deterministic checks (like validating CSVs or linting output).
- references/ - Optional longer templates or policy tables read only when needed.

---

## Composing Skills and Organizing a Team Skill Repository

Small, single-purpose skills compose far better than giant do-everything manuals. Build one skill for weekly-release-notes, another for sql-safety-check, and a third for brand-copy-audit. Store them under a clean folder hierarchy in Git (for example, skills/<skill-name>/SKILL.md) with clear kebab-case names so teammates can discover and contribute improvements through normal pull requests.

---

## Testing Skills Before Sharing Them with Your Team

Before rolling a skill out to colleagues, run three quick checks: test that the YAML description triggers reliably on real user phrasing (and does not trigger by accident on unrelated prompts), run any bundled helper scripts against sample edge-case inputs, and confirm the final output matches your team's quality checklist on at least three realistic examples.

---

## Complete Ready-to-Use SKILL.md Example

Save this file as skills/release-notes/SKILL.md to give any teammate a consistent, automated release-note workflow.

#### `skills/release-notes/SKILL.md`
```python
---
name: release-notes
description: Turn merged pull requests or git commit logs into plain-English customer release notes. Use when preparing a product release or weekly changelog.
---

# Release Notes Protocol

1. Read the provided commit log or PR list carefully.
2. Group changes into three plain-English buckets:
   - New Features
   - Improvements
   - Bug Fixes
3. Rewrite internal developer jargon into clear customer benefits.
4. Flag any breaking change in a bold callout at the top.
5. Never invent features that do not appear in the provided log.
```

---

## Authoritative References

1. Anthropic Agent Skills Documentation - Official guide to authoring SKILL.md packs, progressive disclosure, and helper scripts.
2. Anthropic Official Skills Repository - Reference examples of production SKILL.md packs and repository layout conventions.
3. Agent Skills Open Standard - Portable specification for reusable YAML + Markdown skill directories.
4. Claude Code Skills Reference - How Claude Code discovers and executes workspace and personal skills.

---

### Ready to Orchestrate Production Claude AI Agent Teams?
- [Schedule Strategy Consultation via Email →](contact.html)
- [View Advisory Packages & Rates](pricing.html)
- [Explore Claude Agents Suite](claude-101.html)
