# Agentic AI Assistants - Hermes Agent
## Enterprise AI Transformation Service — Thomas Cherickal

> **// SERVICE SPECIFICATION 04**  
> **Deploying Hermes Agent, the open-source agent from Nous Research, as a self-hosted assistant that remembers
 across sessions, builds its own skills and runs scheduled tasks on your own infrastructure.**

---

## 1. Service
 Overview & Scope

- Self-Hosted Hermes Agent: Installing and configuring Hermes Agent (MIT-licensed, released
 by Nous Research in February 2026) as an always-on service on your Linux server or workstation.
- Persistent Memory: Setting up session history and memory so the agent keeps project
 context, preferences and past decisions across conversations.
- Skills From Experience: Configuring how the agent saves solved procedures as reusable
 skills, and reviewing those skills before your team relies on them.
- Model Choice: Connecting Hermes Agent to local models through Ollama, llama.cpp, vLLM or
 LM Studio, to Claude, or to OpenRouter, depending on your cost and privacy needs.
- Tools, Schedules and Access: Connecting MCP servers and internal tools, setting up
 scheduled tasks, and reaching the agent from the terminal or your team chat, all inside Docker with limited
 permissions.

---

## 2. Standard
 Architecture & Tools

- Agent Framework: Hermes Agent with its memory store, skills library and task scheduler.
- Model Backends: Local open-weight models (Qwen, Gemma, Llama, Mistral) via Ollama,
 llama.cpp, vLLM or LM Studio, plus Claude or OpenRouter through an OpenAI-compatible endpoint.
- Tools and Integrations: MCP servers in Python or Rust, Python tool scripts, and messaging
 channels such as Slack, Discord or Telegram.
- Isolation: Docker containers running as non-root, with limited network access and file
 permissions.

---

## 3.
 Deliverable Examples & 4-Week Schedule

- Containerized Hermes Agent Deployment: Docker Compose setup with configuration, secrets
 handling, persistent storage for memory and skills, and backups.
- Custom Tool & MCP Integrations: Python or Rust tools that let the agent query your
 databases, read internal documents and call internal APIs.
- Starter Skills Library: A reviewed set of skills for your team's recurring tasks, written
 and tested before handover.
- Evaluation Suite: pytest checks covering tool selection, argument correctness, memory
 recall and recovery from failed steps.

| Timeline | Focus & Architecture | Core Deliverables & Artifacts | Verification & Sign-off Criteria |
| :--- | :--- | :--- | :--- |
| Week 1 | Hermes Agent Installation & Model Backend Configuration | Docker Compose deployment of Hermes Agent on Linux server or workstation, persistent storage volumes, and model backend connection (Ollama, llama.cpp, vLLM, LM Studio, Claude, or OpenRouter) | Always-on agent service responding reliably with chosen local or API model backend and baseline latency verified |
| Week 2 | Persistent Memory & Task Scheduler Setup | Session history configuration, cross-conversation memory store, scheduled task routines, and terminal/team chat messaging integration (Slack, Discord, or Telegram) | Agent retains project context and preferences across distinct sessions and executes scheduled background tasks on time |
| Week 3 | Custom Tools, MCP Servers & Skills Library | Python and Rust MCP servers, internal database and document tools, schema-checked tool validation, and starter library of reviewed, tested skills for recurring procedures | Agent executes multi-step tool calls with schema validation and successfully saves and reuses reviewed procedure skills |
| Week 4 | Evaluation Suite, Docker Isolation & Handover | pytest test suite covering tool selection, argument correctness, and memory recall, non-root Docker security lockdown, complete configuration transfer, and 30-day support onboarding | All pytest evaluation checks pass, backup routines verified, complete configuration and IP handed over with client sign-off |

---

## 4. Guarantees
 & Commitments

- Private Operation Option: When paired with local models, the agent runs without calls to
 commercial LLM APIs.
- Checked Tool Calls: Tool inputs are validated against schemas, and failed calls are
 caught, retried or escalated.
- 100% Client Code & Configuration Ownership: All configuration, tools, skills and
 documentation transfer to you.
- 30-Day Engineering Support: Help with model swaps, runtime tuning and new tool
 integrations.

---

## 5. Investment
 & Pricing Quote

- Engagement Duration: Strictly 1 month (4 weeks of structured milestone delivery and
 runtime verification).
- Base Investment Rate (Tier 1 Standard): $2,000 USD (flat base rate for
 complete 1-month delivery).
- Global Purchasing Power Parity (PPP) & Subsidized Rates:
  - Tier 1 (Standard / 100%): $2,000 USD (USA, UK, EU, Canada, Australia)
  - Tier 2 (Developing Nations / 25% Off): $1,500 USD (India, Brazil, Mexico, UAE,
 Poland)
  - Tier 3 & 4 (Least Developed & Startups / 50% Off): $1,000 USD
  - Tier 5 (Academia & Non-Profits / 75% Off): $500 USD
  - Tier 6 (Backward & Underprivileged Nations): Custom Subsidized Relief /
 Micro-Rates
  - Calculate and lock in your exact tier rate live on the Pricing page.
- Payment Milestones:
  - 50% upfront milestone initialization / 50% upon completed Week 4 delivery, runtime verification, and
 client sign-off.

---

## 6. Summary

- Hermes Agent, Self-Hosted: Nous Research's open-source agent
 running on your infrastructure.
- Strict 1-Month Delivery: 4-week program from installation to
 production tools and handover.
- Memory and Skills: An agent that keeps context across sessions
 and reuses reviewed skills.
- Flexible Model Backends: Local models for privacy and cost,
 Claude or OpenRouter for harder work.
- 100% IP & Code Transfer: Full ownership, deployment files
 and 30-day support.

---

### Ready to Deploy?
- [Commission this Service via Email →](contact.html)
- [Book Strategy Session on Topmate ↗](https://topmate.io/thomascherickal)
