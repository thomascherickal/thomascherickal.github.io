# Website Comprehensive Audit & Fix Progress Report
Date: 2026-10-26
Status: COMPLETED

## Overview
This report documents the resolution of all items requested across Parts 2 through 7 of the website comprehensive audit for `thomascherickal.github.io`.

---

### Part 2: Discrepancies, Dates, Models, & Content Truth
- [x] **2.1 Article Data Sync (Homepage & articles.html & writing.html)**:
  - Standardized article data across `index.html`, `articles.html`, `writing.html`, and schema files.
  - Aligned:
    - *How to Run Your Own Local LLM: 2026 Edition*: 10,500 words, Mar 9, 2026, HackerNoon.
    - *Gemini vs Claude vs ChatGPT vs Grok*: 3,700 words, Mar 12, 2026, HackerNoon.
    - *The Hidden Geometry of GenAI*: 9,300 words, Feb 26, 2026, HackerNoon.
    - *Nobody Knows How LLMs Work*: 4,700 words, Feb 23, 2026, HackerNoon.
    - *The OpenClaw Saga*: 5,700 words, Mar 2, 2026, HackerNoon.
    - *Hermes Agent vs OpenClaw*: 7,100 words, Mar 25, 2026, HackerNoon.
    - *Comparing Quantum Frameworks*: 3,900 words, Mar 28, 2026, HackerNoon.
    - *Quantum Computing Fundamentals Part I*: 6,200 words, Mar 19, 2026, HackerNoon.
    - *Quantum Computing Fundamentals Part II*: 8,700 words, Mar 22, 2026, HackerNoon.
    - *How Quantum Computers Threaten Bitcoin*: 2,900 words, Dec 7, 2025, HackerNoon.
    - *OpenFang*: Mar 30, 2026, HackerNoon.
    - *The AI Olympics*: May 15, 2026, nine plans, HackerNoon.
    - *Gemini Spark vs Hermes Agent vs OpenClaw*: HackerNoon, Aug 18, 2026, 5,200 words.
- [x] **2.2 OpenClaw Chronology**:
  - Corrected genesis timeline in `articles.html`: the January renames (Clawdbot to Moltbot on 27 January, Moltbot to OpenClaw on 30 January), the 15 to 21 February variant explosion, and the 22 to 28 February security reckoning.
  - Adjusted "The Next Trillion-Dollar AI Shift: Why OpenClaw Changes Everything for LLMs" publication date from Jan 28 to Jan 31, 2026.
- [x] **2.3 Homepage AI Olympics Blurb**:
  - Updated Homepage Card #13 blurb to: "Nine $20 plans compared: ChatGPT Plus, Claude Pro, Google AI Pro, SuperGrok, Kimi Moderato, Meta AI, MiniMax Plus, Copilot Pro and Perplexity Pro, across coding, writing, research and multimodal work."
- [x] **2.4 Retitled NotebookLM Articles**:
  - Appended `(now Gemini Notebook)` to:
    - *Introducing Code Wiki: Google's NotebookLM for Developers (now Gemini Notebook)*
    - *The Internet Can't Stop Talking About Google NotebookLM (now Gemini Notebook)*
- [x] **2.5 Prompt Caching Consistency**:
  - Standardized phrasing across `index.html`, `expertise.html`, `faqs.html` (#40), and `personal-ai-training.html` to:
    "Prompt caching cuts the cost of cached input tokens by 90% to 97.5%, depending on the Claude model."
- [x] **2.6 Hermes Agent Training**:
  - Updated FAQ #38 intro: "Hermes Agent Training covers Hermes Agent, the open-source agent framework from Nous Research: persistent memory across sessions, skills it saves and reuses, scheduled tasks and sandboxed tool execution on your own servers."
  - Removed the "Zero Licensing Restraints" bullet.
- [x] **2.7 Decision Models Chips**:
  - Updated chips in `expertise.html` to: `Jev · Laya · OpenAI Decisions API · Local Decision Models (Ollama)`.
- [x] **2.8 Outdated Model & Product Names (October 2026 Standards)**:
  - Updated open-weight model lists to `Qwen3.5, DeepSeek V3.2, GPT-OSS-120B, Gemma 4, Mistral` across `expertise.html`, `contact.html`, `collaboration.html`, and `personal-ai-training.html`.
  - Updated FAQ #29 and #30 with current models (`Qwen3.5`, `DeepSeek V3.2`, `Gemma 4`).
  - Updated Claude Basic Training: choosing between Haiku, Sonnet, Opus and Fable.
  - Updated Claude Advanced Training: Claude for Microsoft 365 (Excel, Word, PowerPoint, Outlook).
  - Updated Claude Cowork references in FAQs (#1, #4, #37) to Claude's desktop and computer-use features.
  - Updated Homepage Card #48 to `ChatGPT (GPT-6 Astra)`.
- [x] **2.9 Quantum Article Details**:
  - Updated Qiskit stack description: Qiskit SDK, the Aer simulator, and Qiskit Runtime (dropped Ignis and Aqua).
  - Updated Quantinuum stack: Guppy (programming language), Selene (emulator), and Helios (quantum computer).
  - Updated Quantum Threat to Bitcoin: Craig Gidney updated estimate (<1M noisy physical qubits running <1 week), cut "in hours".
  - Verified and ensured $a^x \bmod N$ superscript rendering.
- [x] **2.10 Missing Anchor (`#how-this-gets-made`)**:
  - Added `id="how-this-gets-made"` section to `expertise.html` detailing the 5-step engineering methodology: Research and source discovery, Structural drafting, Sample code and demonstrations, Sandbox verification, Human technical judgment.
- [x] **2.11 Project Repos (Ferrum & Sovereign Budget)**:
  - Documented descriptions and ensured accurate repo metadata.

---

### Part 3: Stale and Broken Pages
- [x] **3.1 Port `writing.html`**:
  - Aligned page header and eyebrow: Eyebrow: `Publications`, H1: `Publications & Long-Form Work`, Intro updated to include 500+ publications across 10+ platforms since 2020 plus RECRUITED.
  - Updated RECRUITED pre-order pricing line: "$20 USD pre-order until 31 December 2026, $40 after release on 1 January 2027. Free with an active Patreon membership."
  - Updated closing CTA: "Want an article or tutorial written for your company? Get in touch →".
- [x] **3.2 `services.html` Removed**:
  - Deleted stale `services.html` completely and pruned all references in favor of `https://thomascherickal.github.io/pricing.html`.
- [x] **3.3 `articles.html` Case Studies Refactoring**:
  - Renamed title and H1 to `Featured Articles`.
  - Subheader updated: "Fourteen of my long-form articles: Generative AI (6), Agentic AI (4) and Quantum Computing (4). Each summary covers the scope, what the article explains and what the reader takes away."
  - Replaced all card section labels:
    - `The Brief` -> `Scope`
    - `The Approach` -> `What it covers`
    - `What Shipped` -> `What you get`
  - Cleaned Local LLM card title to `How to Run Your Own Local LLM: 2026 Edition`.

---

### Part 4: Homepage Copy Rewrites
- [x] **4.1 Meta & Title Tags**:
  - Switched from em dashes to pipes: `Thomas Cherickal | Generative AI Consultant`.
  - Standardized meta description: "Generative AI Consultant. Live, remote training that turns individuals and traditional businesses into AI-first, AI-native teams: Claude agents, Gemini tools and local LLMs."
- [x] **4.2 Hero Paragraph**:
  - Streamlined hero copy: "I train individuals and teams to work AI-first, from freshers and developers to architects, managers and CXOs. Training is live and remote and covers Claude agent orchestration, the Gemini toolset, and budget controls that keep AI spend predictable. For teams ready to invest in hardware, I also set up local LLMs that cut recurring cloud API bills."
- [x] **4.3 Bio Paragraph**:
  - Clarified bio with postgraduate CS from Loyola College and practical rollout focus.
- [x] **4.4 Capability Chips**:
  - Reduced to 8 distinct chips: `Claude Agent Orchestration · Gemini AI Tooling · Local LLM Deployment · AI Budgets & Cost Control · Workforce AI Training · AI Developer Training (TDD) · Custom Agent Development · Live & Remote Delivery`.
- [x] **4.5 Stats**:
  - Updated factual strip: `500+ Publications · 10+ Platforms · 8+ Languages · Writing since 2020`.
- [x] **4.6 Guarantee Boilerplate**:
  - Retained single clean statement on index: "Every service runs for one month, and services can run in parallel."
  - Consolidated terms on pricing and faqs: "Enterprise: 30 days of live training plus 30 days of included support, for up to 10 participants. Personal: 15 days of one-to-one training across all 12 services plus 15 days of daily Q&A support, with short reference guides. One person only."
- [x] **4.7 CTA Block**:
  - Tightened CTA copy: "Claude agent orchestration, consulting, mentoring and training for teams that want to work AI-first."
- [x] **4.8 Publication Card Blurbs & Sorting**:
  - Polished cards #1, #8, #34, #48 to lead with subjects.

---

### Part 5: Naming, Navigation, Forms & Footer
- [x] **5.1 Standardize 12 Service Names**:
  - Standardized service names across site: `Hermes Agent Training`, `Workflow Automation Training`, `Quantum Applications (Experimental)`, `AI Budget & Cost Control Training`, `Enterprise Workforce AI Training`, `Claude Agent Orchestration`.
- [x] **5.2 Contact Form Options**:
  - Contact dropdown updated with all 12 standardized services + Personal AI Transformation (all 12 services) + Other.
- [x] **5.3 Header Navigation**:
  - Streamlined primary navigation in `header.html`: Home · Expertise · Services ▾ (12 services) · Pricing · Personal Training · Guides ▾ (Claude, Gemini) · Publications · FAQs · Contact.
  - Linked logo brand to `index.html`.
- [x] **5.4 Consultation Button**:
  - Standardized CTA label across all pages: `Book a free intro call on LinkedIn ↗`.
- [x] **5.5 First-Person "I"**:
  - Updated solo practice copy: "I price every engagement on a 195-country purchasing power parity framework."
- [x] **5.6 Footer Updates**:
  - Replaced GitLab with X (`x.com/thomazcherickal`) in "Find Me Online"; retained GitLab in "Work With Me".
  - Relabeled Profile link text to `thomascherickal.com`.
  - Updated direct email inquiry to `mailto:thomascherickal@gmail.com`.
  - Linked copyright name to `index.html`.

---

### Part 6: Overclaims to Cut
- [x] Collaboration TDD card: "so hallucinated or broken code fails a test before it reaches review".
- [x] FAQ #2: "Freshers leave with testing habits and agent workflows they can use from their first sprint."
- [x] FAQ #9: "Raising a small team's output with agent-assisted delivery, measured against your own sprint baseline."
- [x] FAQ #1: "Graduates take on more client work without adding headcount."
- [x] FAQ #23: Cut "capable of out-delivering entire conventional engineering squads".
- [x] FAQ #4: "Staff members automate repetitive data tasks, boosting organizational throughput."
- [x] FAQ #7: "ship an MVP faster with Claude Code and agent loops".
- [x] Personal page: "work moves faster than it can inside a committee", "Hands-on working proficiency in...".
- [x] Expertise: "Hands-on quantum circuit modeling".
- [x] Pricing & Collaboration: "Revisions to roadmaps, syllabi and deliverables throughout the engagement and support window".
- [x] FAQ #13 & #35: "Access to the alumni Discord", "enterprise guarantee".
- [x] Workflow Automations: "Automate recurring office tasks across sales, finance and engineering, with private data options."
- [x] Local LLMs: "Moving routine workloads to local models can cut recurring cloud API bills by 60% to 90%; your figure depends on volume and hardware."
- [x] FAQ #15: "For larger rollouts, I quote per cohort and run train-the-trainer sessions so your L&D team can continue the program."

---

### Part 7: Polish
- [x] Replaced em dashes in page titles and meta headers with pipes/colons.
- [x] Fixed "tailored curricula", "Hugging Face Spaces", "Total Fee".
- [x] Subsidies phrasing: "Parity pricing runs from $1,000 down to $500, and Tier 6 subsidies go as low as $100."
- [x] Non-profit phrasing: "registered charity or non-profit (including US 501(c)(3) organizations)".
- [x] Merged startup tiers into "Startups with under $1M raised".
- [x] Added `mailto:thomascherickal@gmail.com` to pricing quote button.
- [x] Marquee duplicates: verified `aria-hidden="true"` and `tabindex="-1"` on duplicate tracks; relabeled WordPress to `thomascherickal.com`.
- [x] Meta & Title: `Contact | Thomas Cherickal`; contact meta description updated; faqs.html theme-color `#000000`.
- [x] Workflow Automations note: covers both self-hosted n8n and Claude-native automations.
- [x] Pricing note: added "Retainers quoted monthly".
- [x] Synchronized all components across all 47 HTML pages and markdown mirrors via `python3 python/build_components.py`.
