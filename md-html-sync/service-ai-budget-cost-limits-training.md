# AI Budget & Cost Limits Training
## Enterprise AI Transformation Service — Thomas Cherickal

> **// SERVICE SPECIFICATION 09**  
> **Training staff at every level to control AI spending: hard spending caps, token budgets, prompt caching, effort
 settings, and sending minor tasks to free and local models.**

---

## 1. Service
 Overview & Scope

- Hard Spending Caps: Setting limits per session, day, month and year so no person, team or
 agent can exceed its budget.
- User & Team Budgets: Allocating token budgets across developers, departments and
 production services, with workspace-level limits on the Claude platform.
- Prompt Caching & Context Trimming: Caching stable prompt prefixes for up to 90% lower
 cost on cached input, keeping context lean, and clearing it between task phases.
- Model & Effort Matching: Using Haiku at low effort for simple jobs, Sonnet for most
 work, and Opus or Fable at higher effort only when a task needs it.
- Free and Local Models: Moving minor tasks to local models (Gemma, Qwen, Llama and Mistral
 through Ollama or LM Studio) and to free models on OpenRouter.

---

## 2. Standard
 Architecture & Tools

- Platform Controls: Workspace spend limits and usage reports on the Claude platform,
 per-key limits, and /usage in Claude Code.
- Cost Tracking Scripts: Python scripts that read token usage from API responses and log
 cost per user, team and agent.
- Token Reduction: Prompt caching, effort settings, the Batch API for non-urgent work, and
 smaller context windows.
- Free Model Routing: Local models through Ollama or LM Studio, and free OpenRouter models
 for low-stakes tasks.

---

## 3.
 Deliverable Examples & 4-Week Schedule

- Budget Policy & Limits Setup: Written spending policy plus configured limits for each
 user, team and workspace.
- Prompt Caching & Context Playbook: Guide and Python code samples for adding caching
 and trimming context in your existing applications.
- Cost Tracking Dashboard: Dashboard showing spend per team, tokens used, and which models
 handled which work.
- Agent Kill Switch: Code that stops an agent run as soon as it reaches its token or cost
 budget.

| Timeline | Focus & Architecture | Core Deliverables & Artifacts | Verification & Sign-off Criteria |
| :--- | :--- | :--- | :--- |
| Week 1 | Spend Audit & Workspace Budget Policy Setup | Historical API spend audit, written AI spending policy, workspace-level limits on the Claude platform, per-key quotas, and /usage tracking in Claude Code | Baseline cost mapping approved with hard spend ceilings and user/team quota limits active in workspace configuration |
| Week 2 | Prompt Caching & Context Trimming Optimization | Prompt caching integration for up to 90% input cost reduction, context trimming and phase-clearing guide, and Python context-optimization code samples | Verified prompt cache hit rates on repeated system prompts and measured reduction in average prompt token consumption |
| Week 3 | Model/Effort Matching & Free/Local Model Routing | Model tier routing guidelines (Haiku at low effort, Sonnet for standard work, Opus or Fable for high reasoning), Ollama and LM Studio local model setup (Gemma, Qwen, Llama, Mistral), and free OpenRouter routing for low-stakes tasks | Demonstration of routine and low-stakes tasks routed to free and local models with zero degradation on critical tasks |
| Week 4 | Cost Tracking Dashboard, Agent Kill Switches & Handover | Python cost-tracking scripts, live dashboard displaying spend per team and model usage, automated agent budget kill-switch code, Batch API setup for non-urgent tasks, and 30-day support onboarding | Agent kill-switch stops runs reaching token limits, spend dashboard operational, 30%-50% cost savings measured, and client sign-off |

---

## 4. Guarantees
 & Commitments

- No Surprise Bills: Hard caps stop spending at the agreed limit.
- Measurable 30%-50% Cost Savings: Lower token spend from caching, model matching and local
 routing, measured against your bills before the engagement.
- 100% Client Ownership: All scripts, configurations, dashboards and training materials
 belong to you.
- 30-Day Support: Help adjusting budgets, limits and model choices.

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

- Hard Spending Caps: User, team and workspace limits enforced
 on the platform and in code.
- Strict 1-Month Delivery: 4-week program from spend audit to
 live cost tracking.
- Prompt Caching: Cutting recurring token spend through caching
 and leaner context.
- Free and Local Models: Minor tasks moved off paid APIs.
- 100% IP & Governance Transfer: Configurations, workshop
 materials and 30-day support.

---

### Ready to Deploy?
- [Commission this Service via Email →](contact.html)
- [Book Strategy Session on Topmate ↗](https://topmate.io/thomascherickal)
