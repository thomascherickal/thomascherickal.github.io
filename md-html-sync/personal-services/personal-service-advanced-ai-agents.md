# Advanced AI Agents
## Business AI Transformation Service - Thomas Cherickal

> **// SERVICE SPECIFICATION 10**  
> **Building production autonomous agent systems: hierarchical teams of Claude agents, self-correcting loops driven
 by tests, persistent state, and sandboxed execution for business-critical workloads.**

---

## Service
 Overview & Scope

- Multi-Agent Teams: Hierarchical Claude agent teams that carry multi-day, multi-person
 workflows, with human checkpoints at the steps that matter.
- Self-Correction Loops: Agents read compiler errors, test failures and lint output, fix
 the code, and run the checks again until they pass or reach a retry limit.
- Sandboxed Tool Execution: Agent shell commands, code execution and database queries run
 inside Docker containers with restricted permissions and network access.
- Spend and Loop Limits: Iteration caps, token budgets and kill switches on every loop.
- Run Logging: Records of each agent's steps, tool calls, token use and failures for review
 and debugging.

---

## Standard
 Architecture & Tools

- Agent Frameworks: Claude Agent Teams, Claude Code subagents and background agents in
 separate git worktrees, and the Claude API with Python asyncio or Rust Tokio for custom orchestration.
- Model Layer: Claude Opus or Fable (lead planner and synthesis), Claude Sonnet (code and
 tool use), and Claude Haiku (fast classification and search).
- Execution Sandboxing: Docker containers with dropped capabilities (`CAP_DROP=ALL`),
 non-root execution and egress restrictions.
- State & Persistence: Checkpoints in a SQL database or files, so long runs can resume
 after an interruption.

---

## Guarantees
 & Commitments

- Guaranteed Termination: Hard cycle limits, timeouts and fallback paths end every run.
- Complete Commercial IP Ownership: Full code, copyright and documentation transfer at
 sign-off.
- 30-Day Operational SLA & Support: Engineering channel for fixes, tuning and Claude
 model or API updates.
- Data Containment: Disposable containers, with outbound traffic limited to approved
 destinations.

---

## Executive Summary

- Claude Agent Teams: Production orchestrator-worker and
 hierarchical architectures.
- Test-Driven Self-Correction: Test, lint and type-check
 feedback loops that work toward passing code.
- Sandboxed Runtimes: Docker isolation with dropped capabilities
 and restricted network access.
- Enterprise IP & Handover: Full IP transfer, runbooks and
 30-day support.

---

### Ready to Deploy?
- [Commission this Service via Email →](contact.html)
- [Book Strategy Session on Topmate ↗](https://topmate.io/thomascherickal)
