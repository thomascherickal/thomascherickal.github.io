# AI Second Brain
## Business AI Transformation Service - Thomas Cherickal

> **// SERVICE SPECIFICATION - 10 DAYS**  
> **Build a private knowledge vault that captures, organizes and answers from your own notes.**

---

## Service Overview & Scope

A focused 10-day engagement: 5 days of live training (4 hours per day, Monday to Friday) followed by 5 days of dedicated support. An AI second brain is a personal vault of plain Markdown notes that an AI assistant helps you capture, organize, search and review. Everything stays in files you own.

- Markdown Vault: Design and build a personal vault in Obsidian with a clear folder structure, note templates and Git-based backup.
- Capture Pipeline: Set up one inbox for web clippings, PDFs, emails and quick notes, converted to clean Markdown automatically.
- AI Organization: Use a local model through Ollama to summarize, tag and title notes, with every suggestion reviewed before it is saved.
- Ask Your Notes: Build a small retrieval pipeline (embeddings plus a vector index) that answers questions with citations to your own notes.
- Weekly Review Automation: Schedule a weekly digest of new notes, open tasks and suggested links.
- Privacy and Access: Keep sensitive notes local, and set clear rules for what may be sent to any cloud model.

---

## Standard Architecture & Tools

- Notes Layer: Obsidian vault of Markdown files with YAML frontmatter, tracked in Git.
- Models: Ollama with one small local chat model and one embedding model; an optional Claude or OpenRouter key for harder questions.
- Index: Chroma (a local vector database) built from heading-based chunks of your notes.
- Automation: Python scripts and cron for capture conversion, tagging and the weekly digest.

---

## Deliverables & 5-Day Training Schedule

- Working Vault: A structured personal vault with templates and a Git backup.
- Scripts: Capture conversion, auto-tagging, indexing and query scripts, with comments.
- Weekly Digest: One scheduled digest of new and unfinished notes.
- Handover Checklist: A one-page guide for daily capture, weekly review, backup and re-indexing.

---

## Guarantees & Commitments

- Your Files, Your Format: Notes stay as plain Markdown that any editor can open; there is no lock-in.
- Private Operation Option: With local models, no note content is sent to a commercial API.
- Clear Scope: The agreed scope is one vault, one capture inbox, one index and one weekly digest. Larger builds are a separate engagement.
- 100% Client Code & Configuration Ownership: All scripts, templates and documentation transfer to you.
- 5-Day Support: Help for 5 days after training.

---

## Executive Summary

- One Vault, Owned by You: Plain Markdown notes in Obsidian with Git backup.
- Strict 10-Day Delivery: 5 days of training (4 hours per day) plus 5 days of support.
- Capture to Answer: Automated capture, local-model tagging and cited question answering.
- Private by Default: Local models for sensitive notes, cloud only by your rule.
- 100% IP & Code Transfer: Full ownership of scripts, templates and documentation.

---

### Ready to Deploy?
- [Commission this Service via Email →](contact.html)
- [Book Strategy Session on Topmate ↗](https://topmate.io/thomascherickal)
