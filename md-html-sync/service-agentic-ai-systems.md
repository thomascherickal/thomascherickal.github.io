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

- Modular Service Selection & Flexible Engagement: Clients can freely pick and choose their services, either individually or bundled under our comprehensive transformation programs:
  - Individual Service Pick (Personal / 1-on-1): $1,000 USD per service for 5 days of live training + 5 days of dedicated support (10 days total coverage per service). Strictly 1 person only.
  - Individual Service Pick (Business / Teams): $2,000 USD per service for 5 days of live training + 5 days of dedicated support (10 days total coverage per service). Covers up to 10 participants (additional seats at $500 USD per seat).
  - Comprehensive Business Training: $5,000 USD for 60 days (30 days live training + 30 days guidance and detailed guides and materials). Includes cohorts up to 10 participants. Extra seats at $500 USD per seat.
  - Comprehensive Personal Training: $2,500 USD for 30 days (15 days live training + 15 days guidance and detailed guides and materials). Strictly for 1 person only (private acceleration).
- Global Purchasing Power Parity (PPP) Available for All Service Pricing:
  - Tier 1 (Standard / 100%): Individual $1,000 USD | Business $2,000 USD | Business 60d $5,000 USD | Personal 30d $2,500 USD (USA, UK, EU, Canada, Australia)
  - Tier 2 (Developing Nations / 25% Off): Individual $750 USD | Business $1,500 USD | Business 60d $3,750 USD | Personal 30d $1,875 USD (India, Brazil, Mexico, UAE, Poland)
  - Tier 3 & 4 (Lower-Middle/Low-Income & Startups / 50% Off): Individual $500 USD | Business $1,000 USD | Business 60d $2,500 USD | Personal 30d $1,250 USD
  - Tier 5 (Academia & Non-Profits / 75% Off): Individual $250 USD | Business $500 USD | Business 60d $1,250 USD | Personal 30d $625 USD
  - Tier 6 (Crisis-Affected & Under-Resourced Regions): Individual $100 USD | Business $200 USD | Custom Subsidized Relief / Micro-Rates
  - Calculate and lock in your exact tier rate live on the Pricing page.
- Payment Schedule:
  - Zero upfront advance; billed in 5 equal milestone installments due only after verified instruction and runtime delivery.

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
