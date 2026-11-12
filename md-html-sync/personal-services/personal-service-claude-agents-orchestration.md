# Claude Agent Orchestration
## Business AI Transformation Service - Thomas Cherickal

> **// SERVICE SPECIFICATION 03**  
> **The principles behind Claude AI agent orchestration and how to run agent teams at scale. Build multi-agent
 workflows that automate complex operations while keeping quality and cost under control, even on a small budget.**

---

## Service
 Overview & Scope

- Claude Agent Teams: Coordinating several Claude agents on one goal, with an orchestrator
 on Opus or Fable at high or xhigh effort and Sonnet or Haiku workers at lower effort.
- Five Core Architectures: Orchestrator-worker, hierarchical, sequential pipeline, parallel
 fan-out and fan-in, and router, with guidance on when each one fits.
- Model Context Protocol (MCP) Integration: Connecting agents to internal APIs, databases
 and files through MCP servers with defined input schemas.
- Verification Before Commit: Reviewer subagents and test suites check worker output before
 it changes anything.
- Spend Limits at Every Level: Budgets per session, day, month and year, iteration caps,
 and a kill switch that stops concurrent cloud agents.

---

## Standard
 Architecture & Tools

- Orchestration Layer: Claude Code subagents, background agents in their own git worktrees,
 and the Claude API with Python asyncio for concurrent calls.
- Model Tier Allocation: Claude Opus or Fable (planning and final synthesis), Claude Sonnet
 (coding, tool use and analysis), and Claude Haiku (search, classification and summarization).
- Shared State: Task and result state kept in files or a SQL database, so interrupted runs
 can resume.
- Cost and Safety Controls: Prompt caching, Batch API fan-out, a restricted tool list for
 each subagent, Docker sandboxes, and run logs that record tokens per agent.

---

## Guarantees
 & Commitments

- No Runaway Loops: Iteration counters, timeouts and token caps on every agent.
- Full IP & Code Ownership: All orchestration code, prompts, skills and MCP
 configurations transfer to you.
- Enterprise Data Boundaries: Agents act with each user's own permissions, and write
 actions need approval until you choose to relax them.
- 30-Day Production SLA: Support channel for adjustments, new tools and debugging.

---

## Executive Summary

- Claude Agent Teams: Opus or Fable planning paired with Sonnet
 and Haiku workers.
- Model Context Protocol Tools: MCP servers connecting agents to
 SQL, internal APIs and file systems.
- Low-Budget Operation: Prompt caching, batch processing,
 cheaper models for narrow jobs and hard spend limits.
- Complete Code & IP Transfer: Full ownership, containerized
 runtimes and 30-day support.

---

### Ready to Deploy?
- [Commission this Service via Email →](contact.html)
- [Book Strategy Session on Topmate ↗](https://topmate.io/thomascherickal)
