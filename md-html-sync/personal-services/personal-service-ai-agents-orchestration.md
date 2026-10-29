# Claude AI Agent Orchestration 101
## Enterprise AI Transformation Service - Thomas Cherickal

> **// SERVICE SPECIFICATION 03**  
> **The principles behind Claude AI agent orchestration and how to run agent teams at scale. Build multi-agent
 workflows that automate complex operations while keeping quality and cost under control, even on a small budget.**

---

## 1. Service
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

## 2. Standard
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

## 3.
 Deliverable Examples & 4-Week Schedule

- Orchestration Repository: Documented Python codebase with the orchestrator, worker
 definitions and routing logic, plus its tests.
- Custom MCP Server Suite: MCP servers in Python or Rust connecting your agents to SQL
 databases, ticket queues and internal APIs.
- Checkpoint & Resume Logic: Saved state that lets a multi-hour agent run pick up where
 it stopped.
- Cost & Run Report: Per-agent breakdown of tokens, cost, tool calls and failures for
 each run, with a worked budget for your workload.

| Timeline | Focus & Architecture | Core Deliverables & Artifacts | Verification & Sign-off Criteria |
| :--- | :--- | :--- | :--- |
| Week 1 | Architecture Selection & Agent Topology Design | Architecture specification selecting from the five core patterns (orchestrator-worker, hierarchical, sequential pipeline, parallel fan-out/fan-in, router), model tier assignments (Opus or Fable lead, Sonnet or Haiku workers), and effort level tuning | Stakeholder sign-off on agent team topology, effort budgets, and git worktree isolation model |
| Week 2 | MCP Tooling & Async Orchestration Engine | Custom Python or Rust MCP servers with defined input schemas for SQL, ticket queues, and internal APIs, Python asyncio orchestration engine, and Docker sandbox setup | Verified schema-enforced tool execution between subagents and MCP servers within restricted container boundaries |
| Week 3 | Verification Graphs, State Checkpoints & Spend Controls | Reviewer subagent verification logic, automated test suites before commit, SQL or file state checkpoint & resume engine, prompt caching, and per-session/daily token budgets | Verified state recovery after simulated run interruption and reviewer subagents catching errors before commit |
| Week 4 | Batch API Fan-out, Cost Reporting & Handover | Batch API processing pipeline, per-agent token and cost telemetry reporting, hard kill-switch breakers, complete codebase transfer, and 30-day SLA onboarding | Multi-agent stress run completed with zero runaway loops, confirmed budget ceiling enforcement, full IP transfer, and client sign-off |

---

## 4. Guarantees
 & Commitments

- No Runaway Loops: Iteration counters, timeouts and token caps on every agent.
- Full IP & Code Ownership: All orchestration code, prompts, skills and MCP
 configurations transfer to you.
- Enterprise Data Boundaries: Agents act with each user's own permissions, and write
 actions need approval until you choose to relax them.
- 30-Day Production SLA: Support channel for adjustments, new tools and debugging.

---

## 5. Investment
 & Pricing Quote

- Modular Service Selection & Flexible Engagement: Clients can freely pick and choose their services, either individually or bundled under our comprehensive transformation programs:
 - Individual Service Pick (Personal / 1-on-1): $100 USD per service for 5 days of live training + 5 days of dedicated support (10 days total coverage per service). Strictly 1 person only.
 - Individual Service Pick (Business / Teams): $200 USD per service for 5 days of live training + 5 days of dedicated support (10 days total coverage per service). Covers up to 20 participants (additional seats at $100 USD per seat).
 - Comprehensive Business Training: $2,000 USD for 40 days (20 days live training + 20 days guidance, Q&A and written guides). Includes cohorts up to 20 participants. Extra seats at $100 USD per seat).
 - Comprehensive Personal Training: $1,000 USD for 40 days (20 days live training + 20 days guidance, Q&A and written guides). Strictly for 1 person only (private acceleration).
- Global Purchasing Power Parity (PPP) Available for All Service Pricing:
 - Tier 1 (Standard / 100%): Individual $100 USD | Business $200 USD | Business 40d $2,000 USD | Personal 40d $1,000 USD (USA, UK, EU, Canada, Australia)
 - Tier 2 (Upper-Middle-Income / 25% Off): Individual $75 USD | Business $150 USD | Business 40d $1,500 USD | Personal 40d $750 USD: e.g. Brazil, Mexico, Malaysia, Thailand, Türkiye
 - Tier 3 & 4 (Lower-Middle/Low-Income & Startups / 50% Off): Individual $50 USD | Business $100 USD | Business 40d $1,000 USD | Personal 40d $500 USD
 - Tier 5 (Academia & Non-Profits / 75% Off): Individual $25 USD | Business $50 USD | Business 40d $500 USD | Personal 40d $250 USD
 - Tier 6 (Crisis-Affected & Under-Resourced Regions): Individual $10 USD | Business $20 USD | Custom Subsidized Relief / Micro-Rates
 - Calculate and lock in your exact tier rate live on the Pricing page.
- Payment Schedule:
 - Zero upfront advance; billed in 5 equal milestone installments due only after verified instruction and runtime delivery.

---

## 6. Summary

- Claude Agent Teams: Opus or Fable planning paired with Sonnet
 and Haiku workers.
- Strict 1-Month Delivery: 4-week progression from workflow
 breakdown to production handover.
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
