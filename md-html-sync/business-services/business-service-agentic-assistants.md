# Agentic Assistants Training
## Business AI Transformation Service - Thomas Cherickal

> **// SERVICE SPECIFICATION 04**  
> **Deploying Hermes Agent, the open-source agent from Nous Research, as a self-hosted assistant that remembers
 across sessions, builds its own skills and runs scheduled tasks on your own infrastructure.**

---

## Service
 Overview & Scope

- Self-Hosted Hermes Agent: Installing and configuring Hermes Agent (MIT-licensed, released
 by Nous Research) as an always-on service on your Linux server or workstation.
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

## Standard
 Architecture & Tools

- Agent Framework: Hermes Agent with its memory store, skills library and task scheduler.
- Model Backends: Local open-weight models (Qwen, Gemma, Llama, Mistral) via Ollama,
 llama.cpp, vLLM or LM Studio, plus Claude or OpenRouter through an OpenAI-compatible endpoint.
- Tools and Integrations: MCP servers in Python or Rust, Python tool scripts, and messaging
 channels such as Slack, Discord or Telegram.
- Isolation: Docker containers running as non-root, with limited network access and file
 permissions.

---

## Guarantees
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

## Executive Summary

- Hermes Agent, Self-Hosted: Nous Research's open-source agent
 running on your infrastructure.
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
