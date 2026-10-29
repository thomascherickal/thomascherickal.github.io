# Generative AI Transformation
## Enterprise AI Transformation Service - Thomas Cherickal

> **// SERVICE SPECIFICATION 01**  
> **Move your business from one-off chatbot prompts to teams of Claude agents that carry real work. Learn how Claude
 agent orchestration works, where agents hit their limits, and how coordinated agents grounded in basic domain
 knowledge can take on tasks that used to need several people.**

---

## 1. Service
 Overview & Scope

- From Chat Windows to Agent Teams: Moving day-to-day operations out of isolated chat
 sessions and into coordinated teams of Claude agents with defined roles and handoffs.
- Orchestrator and Subagent Design: An orchestrator on Claude Opus or Fable at high effort
 keeps the plan and checks results, while Sonnet and Haiku subagents search, run code and read documents in
 parallel.
- Grounding in Your Systems: Custom Model Context Protocol (MCP) servers in Python or Rust,
 plus connectors that follow each user's permissions, so agents work from your actual data and stay inside
 set boundaries.
- Verification and Spend Limits: Agent output is checked by pytest and cargo test suites
 before it lands, and token budgets are set per session, per day and per month.
- Production Handover: Containerized deployment on Linux with Docker, written runbooks and
 a working session with your team before sign-off.

---

## 2. Standard
 Architecture & Tools

- Models and Orchestration: Claude Opus and Fable for planning, Claude Sonnet for execution
 and Claude Haiku for high-volume routing, run as Claude Agent Teams, Claude Code subagents and background
 agents in separate git worktrees.
- Tool Integration: MCP servers written in Python (with FastAPI where an HTTP service is
 needed) or Rust, Claude connectors, and SQL access kept read-only wherever possible.
- Runtime Isolation: Docker containers running as non-root with dropped capabilities
 (`CAP_DROP=ALL`), Claude Code permission modes, and connector write tools set to Needs approval.
- Cost Controls: Prompt caching, effort levels matched to each agent's role, the Batch API
 for non-urgent work, and hard spend limits with a kill switch.

---

## 3.
 Deliverable Examples & 4-Week Schedule

- Agent Team Blueprint: Written architecture covering each agent's role, model and effort
 level, the handoffs between agents, failure fallbacks and where state is kept.
- Custom MCP Servers: Tested MCP servers connecting your Claude agents to your databases,
 internal APIs and file stores.
- Automated Verification Suite: pytest and cargo test suites wired into CI/CD that check
 agent output against known-good cases on every change.
- Deployment Package: Docker Compose files, configuration, runbooks and recorded
 walkthroughs your team can operate after handover.

| Timeline | Focus & Architecture | Core Deliverables & Artifacts | Verification & Sign-off Criteria |
| :--- | :--- | :--- | :--- |
| Week 1 | Workflow Mapping & Agent Team Blueprint | Agent team blueprint, task allocation matrix, orchestrator-subagent role definitions (Claude Opus or Fable planner, Claude Sonnet and Haiku workers), and effort budget allocations | Stakeholder approval of agent team architecture, data boundary specifications, and handoff protocols |
| Week 2 | Custom MCP Server Engineering & Grounding | Custom Model Context Protocol (MCP) servers in Python (FastAPI) and Rust, read-only SQL connections, internal API connectors, and user-permission boundary configuration | Successful end-to-end execution of agent tool calls against internal data sources with enforced input schemas and write approvals |
| Week 3 | Automated Verification Suite & Docker Sandboxing | Automated pytest and cargo test suites in CI/CD verifying agent output against known-good benchmarks, Docker container isolation (CAP_DROP=ALL, non-root), and separate git worktrees | Agent reasoning loops pass verification test suites with zero unhandled exceptions and verified runtime isolation |
| Week 4 | Spend Circuit Breakers, Rollout & Production Handover | Prompt caching setup, Batch API integration, per-session/daily spend limits with kill switch, Docker Compose deployment manifests, operational runbooks, and 30-day SLA onboarding | Stress test completed with bounded iteration limits and budget ceiling enforcement, complete IP handover, and client sign-off |

---

## 4. Guarantees
 & Commitments

- Bounded Execution: Every agent loop has an iteration cap, a wall-clock timeout and a
 token budget, so no run continues unchecked.
- 100% Commercial IP Ownership: All agent code, prompts, skills, MCP server implementations
 and architecture diagrams transfer to you on completion.
- Data Boundaries: Credentials stay in your environment, containers run with restricted
 network access, and agents see only what each user's permissions allow.
- 30-Day Production SLA & Support: Direct engineering access for tuning, bug fixes and
 updates when Claude models or APIs change.

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

- Claude Agent Teams: Moving from single chat windows to
 coordinated teams of Claude agents with clear roles.
- Strict 1-Month Delivery: 4-week progression from workflow
 mapping to production handover.
- Model Context Protocol (MCP): Agents grounded in your
 databases and internal APIs through custom Python and Rust servers.
- Verification and Cost Control: Test-gated output, prompt
 caching and hard spend limits on every run.
- 100% Commercial IP Ownership: Complete code transfer, Docker
 manifests, runbooks and 30-day support.

---

### Ready to Deploy?
- [Commission this Service via Email →](contact.html)
- [Book Strategy Session on Topmate ↗](https://topmate.io/thomascherickal)
