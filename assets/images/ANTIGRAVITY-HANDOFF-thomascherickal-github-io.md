# Handoff Brief — thomascherickal.github.io Repositioning & Technical Rebuild

**Prepared for:** Google Antigravity (agentic execution)
**Repo:** `thomascherickal/thomascherickal.github.io`
**Site:** https://thomascherickal.github.io
**Owner:** Thomas Cherickal — The Digital Futurist (2020–present), Chennai, India
**Version:** 2.0 (August 2026) — supersedes v1.0
**Scope:** Full repositioning + technical/SEO remediation of a static site, expanding from 4 pages to 5

---

## 0. How to use this document

This is a complete, self-contained work order. Read all of it before editing anything.

Execute in this order:

1. **Section 3** — Non-negotiable brand constraints (read first, violate none)
2. **Section 4** — The repositioning (drives all copy changes)
3. **Sections 5–9** — Page-by-page copy and structure changes
4. **Section 10** — JSON-LD blocks (drop-in, highest priority technical item)
5. **Section 11** — Favicon, book cover, and image assets (files supplied)
6. **Section 12** — Technical / performance / accessibility fixes
7. **Section 13** — Build `portfolio.html` (new fifth page — do this LAST, after everything else passes)
8. **Section 14** — Acceptance criteria (verify every item before declaring done)

Existing stack: static HTML, single external `style.css`, no build step, no framework, GitHub Pages. **Do not introduce a build system, framework, bundler, or npm dependency.** Keep it hand-editable static HTML.

### Supplied asset files

These accompany this brief and are production-ready. Copy them into the repo at the paths given in **Section 11**.

```
favicon.ico                     multi-res 16/32/48
favicon.svg                     vector cross, sharpest at small sizes
favicon-16x16.png
favicon-32x32.png
favicon-96x96.png
apple-touch-icon.png            180x180
android-chrome-192x192.png
android-chrome-512x512.png
assets/images/site-icon.webp    512x512
recruited-cover.webp            700x1000, 58 KB — USE THIS as the book cover
recruited-cover.jpg             JPEG fallback
recruited-cover-full.webp       1049x1500 archival
og-card-book-BASE.png           1200x630 starting point for the social card
```

Current repo file set (before changes):

```
index.html   about.html   writing.html   services.html
style.css    sitemap.xml  robots.txt
recruited-cover.jpg          <- REPLACE with supplied file
assets/images/thomas-avatar.webp
```

---

## 1. Executive summary of what is wrong today

The site is technically sound but **strategically generic and structurally under-marked-up**.

**Strategic problem:** it sells "Generative AI Consultant" — a category with tens of thousands of occupants, sold by the hour, requiring onsite delivery. It does not match what the owner is best at, nor what he wants his working life to look like. It also completely omits his quantum computing track record, which is one of his two genuine differentiators.

**Structural problems (verified by live audit):**

- JSON-LD exists on `index.html` only. Three of four pages have zero structured data.
- The `Person` node has **no `sameAs` array**, despite 22 verified profile URLs sitting in the footer of every page.
- No `Book` schema for RECRUITED. No `Service`/`offerCatalog`. No `Organization`. No `BreadcrumbList`.
- Homepage claims **500+ articles**; the writing page and its meta description claim **20+**. A 25× internal contradiction on the owner's strongest credential.
- Zero `aria-label` attributes across all four pages. No skip-link. 22 emoji-only footer links unreadable to screen readers.
- No `width`/`height` or `loading="lazy"` on any image. `recruited-cover.jpg` is 365 KB unoptimised.
- `og:image` is a small square avatar while `twitter:card` is `summary_large_image` → cropped social previews.
- **No favicon at all.**
- The identical "Start a Conversation" contact block is duplicated on all four pages.
- Email address exposed in plaintext eight times.
- Substack linked inconsistently: `thomascherickal.substack.com` in body, `thesingularitypoint.substack.com` in footer.
- Quantum computing — a documented content pillar since 2020 with 4+ published deep dives — appears nowhere in any headline, title, meta description, or capability card.

**What is already correct — do not break it:** all 61 outbound links resolve; canonical tag on every page; valid `sitemap.xml` with current `lastmod`; valid `robots.txt`; clean heading hierarchy (single `<h1>`, no skipped levels); single external stylesheet; no render-blocking JS.

---

## 2. Background reasoning (context, not instructions)

Three positioning options were evaluated. Selected: **Technical Content Engineering / Developer Education**, with a dual AI + Quantum specialisation.

| Option | Verdict |
|---|---|
| Rust + AI infrastructure | **Rejected.** Owner cannot reach proficiency in reasonable time. Multi-year rebuild against decade-deep engineers. Does not use his strongest existing asset. |
| Broad Generative AI consulting/training | **Rejected as the core.** Overcrowded; hours-billed; onsite delivery incompatible with a work-from-home goal; competes against free vendor enablement from Microsoft, Google, OpenAI and Anthropic; tool-usage training decays with every model release. Retained as a secondary remote-only offer. |
| **Technical Content Engineering & Developer Education for AI and Quantum companies** | **Selected.** |

Rationale:

- **Uses the existing credential rather than building a new one.** 500+ published long-form technical articles across 10+ platforms is the hardest-to-fake asset in this market and is currently unmonetised.
- **Favourable market economics.** API and developer documentation sits in the top rate band of technical writing; writers with API/developer-docs experience earn materially more than general technology writers; expert specialists command four figures per long-form piece and more for whitepapers and research.
- **Currency arbitrage.** Clients predominantly US/EU paying USD to an operator based in Chennai.
- **Async and remote by nature.** No live calendar, no travel, no onsite delivery. Protects the owner's music commitments (violin, choral) — a stated primary life constraint.
- **The quantum axis is the moat.** The population that can write credibly about *both* frontier AI *and* quantum computing is extremely small. Quantum companies (IBM, IonQ, Quantinuum, Rigetti, PsiQuantum, Xanadu, Pasqal, QuEra, Q-CTRL, Classiq, Multiverse and the SDK ecosystem around them) are staffed with physicists who cannot write for developers, and they all ship developer-facing SDKs that need documentation, tutorials and learning paths. Narrow market, very few credible suppliers — the inverse of the GenAI consulting problem.
- **This is not repositioning churn.** Quantum computing has been a documented content pillar since 2020, with published Qiskit, Q#, Quantinuum and post-quantum-cryptography deep dives. Adding it back is *restoring* an existing asset that the previous site build dropped, not inventing a new identity.

**Standing commitment recorded by the owner: no further repositioning for 24 months.**

---

## 3. NON-NEGOTIABLE BRAND CONSTRAINTS

Violating any of these is a failed build.

### 3.1 Excluded entities — never appear anywhere in any file

Never reference: Augmentron Consultancy, augmentron.com, augmentron.io, augmentronconsultancy.com, Mary Cynthia, Arockia Jenitha, study abroad consulting, MSME/GST details, partner college counts. If any such URL is encountered, replace with `https://thomascherickal.com`.

### 3.2 Canonical URLs — exact strings, no variants

| Property | Exact value |
|---|---|
| Primary site | `https://thomascherickal.com` |
| GitHub Pages | `https://thomascherickal.github.io` |
| LinkedIn | `https://linkedin.com/in/thomascherickal` (no hyphen) |
| Patreon | `https://patreon.com/thomascherickal` (never any other variant) |
| Topmate | `https://topmate.io/thomascherickal` |
| Gumroad | `https://thomascherickal.gumroad.com` |
| Newsletter | `https://thomascherickal.kit.com` |
| GitHub | `https://github.com/thomascherickal` |
| Substack | `https://thesingularitypoint.substack.com` — **use this everywhere; remove the `thomascherickal.substack.com` variant from `writing.html`** |

### 3.3 Hard rules

- **Dual URL rule.** Wherever `thomascherickal.com` appears, `thomascherickal.github.io` must also appear, and vice versa.
- **Brand start date.** Always "The Digital Futurist (2020–present)" or "since 2020". Never 2018 or any other year.
- **Exclude Dev.to** from every publishing-platform list.
- **No speaking engagements.** Remove all speaking offers, keynote offers, webinar offers, and the phrase "speaking opportunity" from every page including contact form copy. Deliberate business decision.
- **RECRUITED pricing.** $5.00 USD pre-order, free with active Patreon subscription, `priceValidUntil` 2026-12-31, offer URL `https://patreon.com/thomascherickal`. Price must be **visible on-page**, not schema-only.
- **Books listed:** RECRUITED only.
- **No spoken-language claims anywhere.** Do not add `knowsLanguage` to any schema block and do not state language proficiency in copy. The only language reference permitted is `"availableLanguage": "English"` on the service node and `inLanguage: "en-US"` on page nodes.

### 3.4 Visual aesthetic — permanent default, do not redesign

Cyberpunk Digital Futurist:

- Background: pure black `#000000`
- Palette: toned-down neon — muted cyan, gold, magenta, violet, lime, orange, pink
- Headings: Times New Roman, italic, bold, in cyan or gold. **No rainbow-gradient titles.**
- Body: Open Sans
- Display/brand font: Orbitron
- Subtle grid overlay
- Per-category card accent colours
- Rainbow gradients only on dividers, borders, and pills

**Quantum accent colour:** use muted violet for all quantum-specific cards, chips, and badges. AI-specific elements keep cyan. This gives the dual specialisation a visual language without adding new palette entries.

Preserve the existing `style.css` design language. Additive changes only; do not restyle wholesale.

### 3.5 Copy voice

All prose must match the Thomas Cherickal authorial fingerprint:

- Em-dashes rendered as space-hyphen-space ( - ), not `—`, in body copy
- Short-sentence punches after dense technical paragraphs
- Rhetorical question stacks used sparingly for emphasis
- Technically precise, named tools, no vague hand-waving
- Optimistic but honest about risk
- Never corporate-generic, never LinkedIn-influencer register

---

## 4. THE REPOSITIONING

### 4.1 Positioning statement (canonical — use verbatim where a full statement is needed)

> **Technical Content Engineer & Developer Educator for AI and Quantum companies.**
> I write the documentation, deep dives, tutorials and courses that make complex AI and quantum systems understandable to the engineers who have to use them.

**Short form (for meta titles, ≤60 chars):**
`Technical Content Engineer — AI & Quantum`

**Medium form (for meta descriptions):**
`Technical Content Engineer & Developer Educator for AI and Quantum companies. Documentation, deep dives, and courses on LLMs, agents, Qiskit, PennyLane and quantum machine learning. 500+ published articles since 2020.`

### 4.2 Quantum credentials — the exact claims permitted

Use these. Do not inflate beyond them, and do not soften them either.

| Claim | Where to use |
|---|---|
| **IBM Qiskit** — hands-on | Chips, capability cards, `knowsAbout` |
| **PennyLane** — hands-on | Chips, capability cards, `knowsAbout` |
| **Quantum Machine Learning (QML)** | Capability cards, `knowsAbout`, service descriptions |
| **Quantum Technologies** | Domain chips, `knowsAbout` |
| **Microsoft Q#**, **Quantinuum stack** | Chips only (published comparative work exists) |
| **Post-quantum cryptography risk** | Service and capability copy (published work exists) |
| **Quantum algorithms** (Grover, Shor, VQE, QAOA) | Capability card detail |

**Wording note for the agent — do not "correct" this.** Buyer-facing copy should lead with what the buyer needs (*"I make quantum computing legible to working developers"*) and support it with the concrete stack (*Qiskit, PennyLane, QML*). Quantum companies employ physics PhDs; what they lack is people who can explain their SDK. Translator positioning outsells researcher positioning here and is fully defensible under technical scrutiny. `Quantum Technologies` remains present as a domain term in chips and `knowsAbout`.

### 4.3 The offer stack (priority order)

| # | Offer | Delivery |
|---|---|---|
| 1 | **Developer documentation & API/SDK guides** — AI *and* quantum SDKs | Async, remote |
| 2 | **Technical deep dives & explainers** — LLMs, agents, quantum computing, QML | Async, remote |
| 3 | **Developer education** — courses, learning paths, tutorial series | Async, remote |
| 4 | **Quantum developer content** — Qiskit/PennyLane tutorials, QML explainers, post-quantum crypto briefings | Async, remote |
| 5 | **Launch & migration content** — model/SDK launch explainers, migration guides | Async, remote |
| 6 | **Books & digital products** — RECRUITED, Gumroad | Async |
| 7 | **Remote team training** — GenAI and quantum-readiness for engineering teams | Live remote only |
| 8 | **1:1 mentoring via Topmate** | Live remote |

### 4.4 Target buyer

Primary A: **AI infrastructure, LLM, agent-framework and developer-tooling companies** — seed to Series C, US/EU, remote-first.
Primary B: **Quantum computing companies and quantum SDK vendors** — hardware providers, cloud quantum platforms, and the QML/quantum-software ecosystem.
Secondary: enterprise engineering teams needing remote GenAI or quantum-readiness enablement.

**Not a segment:** students and individual founders. Top-of-funnel audience served by free articles and the book — never addressed as customers in site copy.

### 4.5 Sitewide terminology changes

| Remove / replace | Use instead |
|---|---|
| "Generative AI Consultant" (as primary identity) | "Technical Content Engineer & Developer Educator" |
| "Emerging Technologies Expert" | "AI & Quantum Specialist" |
| "Rust Systems Engineer" role card | **Delete** |
| "Go Cloud Engineer" role card | **Delete** |
| "HTML/CSS/JavaScript" role card | **Delete** |
| "Deployment" role card | **Delete** |
| "Speaking & Workshops" collab card | **Delete** |
| "Advisory Roles" collab card | Replace with "Developer Education" |
| "onsite" / "on site" anywhere | **Remove** — all delivery is remote |
| "8+ Years in Tech" | "6+ Years Technical Writing" |
| "20+ Top Writer Ranks" | "24 Featured Deep Dives" |

**Note on Rust and Go:** deleted as *headline roles*, retained as line items in the "Languages I Write About" chip group. Writing credibly about a language is a different and honest claim from selling engineering hours in it.

---

## 5. `index.html` — Homepage

### 5.1 Hero

- **`<h1>`:** `Thomas Cherickal` (unchanged)
- **Tagline (replaces current):**
  `Technical Content Engineer & Developer Educator - I make complex AI and quantum systems understandable to the engineers who use them.`
- **Pill/chip row (replaces current four — five chips):**
  `Developer Documentation` · `Technical Deep Dives` · `Developer Education` · `AI & LLM Systems` · `Quantum Computing`
  Apply the muted-violet accent to the Quantum chip.
- **Hero paragraph (replaces current).** Must contain, in the owner's voice: 500+ published long-form technical articles across 10+ platforms since 2020; writes documentation, deep dives, tutorials and courses for AI *and* quantum companies; hands-on with LLMs, agent frameworks and local inference (Claude Code, Google Antigravity, Ollama, LM Studio, llama.cpp) **and** with quantum SDKs (IBM Qiskit, PennyLane, quantum machine learning); fully remote and async.
- **Kicker above `<h1>`:** keep `// The Digital Futurist · est. 2020 · chennai, india`

### 5.2 CTA row

1. `📚 Read the Portfolio` → `portfolio.html`
2. `📝 Commission Content` → `services.html#contact`
3. `📧 Newsletter` → `https://thomascherickal.kit.com`
4. `🐙 GitHub` → `https://github.com/thomascherickal`

Remove the Topmate button from the hero — it is now a secondary offer and belongs on `services.html` only.

### 5.3 Stat block — FIX THE CONTRADICTION

| Position | Value | Label |
|---|---|---|
| 1 | `500+` | `Articles Published` |
| 2 | `10+` | `Platforms` |
| 3 | `6+` | `Years Technical Writing` |
| 4 | `2` | `Deep Domains — AI & Quantum` |

Delete "8+ Years in Tech" and "20+ Top Writer Ranks".

### 5.4 Destination cards — expand from 3 to 4

1. **Portfolio & Case Studies** → `portfolio.html` — "Selected content projects, with the brief, the approach, and what shipped."
2. **Publications** → `writing.html` — "500+ published articles across 10+ platforms. Deep dives on GenAI, agents, local inference, and quantum computing."
3. **Capabilities & Tech Stack** → `about.html` — "What I write about across AI and quantum, what I can run and verify, and the tools I work in daily."
4. **Services & Commissions** → `services.html` — "Documentation, deep dives, courses, quantum developer content, and remote team training."

**Every card description must be numerically consistent with the stat block. Never write "20+" where the site claims "500+".**

### 5.5 Mid-page value section

Replace "Bridging Frontier AI to Enterprise Reality" with a section titled **"Why Technical Content Fails"** — short argument that most AI and quantum developer content is written either by engineers and physicists who cannot write, or by writers who cannot run the code, and that the owner sits in the intersection. Name the quantum case explicitly: quantum companies have PhDs and no developer-facing explainers.

Two CTAs: `See the Work` → `portfolio.html`, `Commission a Piece` → `services.html#contact`.

### 5.6 Contact section

**Remove the full contact form from `index.html`.** Replace with a single-line CTA band linking to `services.html#contact`. (See §7.4 — the form lives on `services.html` only.)

---

## 6. `about.html` — Capabilities & Tech Stack

### 6.1 Title and H1

- `<title>`: `Capabilities & Tech Stack — Thomas Cherickal | AI & Quantum Technical Content`
- `<h1>`: `Capabilities, Coverage & Tech Stack`
- Subtitle: what he writes about and can verify by running, across AI and quantum, for AI and quantum companies.

### 6.2 Role cards — 7 cards (was 10)

**Delete:** Rust Systems Engineer, Go Cloud Engineer, HTML/CSS/JavaScript, Deployment.

1. **📝 Developer Documentation** — API references, SDK guides, quickstarts, integration and migration guides. For AI APIs and quantum SDKs alike. Content engineers can actually follow.
2. **🔬 Technical Deep Dives** — Long-form explainers on LLM internals, agent architectures, quantisation, fine-tuning, inference, and quantum algorithms. Researched, run, verified.
3. **🎓 Developer Education** — Courses, tutorial series, and structured learning paths that take a developer from zero to shipping.
4. **⚛️ Quantum Computing & QML** *(violet accent)* — Hands-on IBM Qiskit and PennyLane. Quantum machine learning, variational circuits, quantum algorithms (Grover, Shor, VQE, QAOA), and quantum technologies explained for working developers. Published comparative work on Qiskit, Microsoft Q# and the Quantinuum stack.
5. **🔐 Post-Quantum & Quantum Risk** *(violet accent)* — Post-quantum cryptography risk, the threat model against current encryption, and quantum-readiness briefings for engineering and security teams.
6. **🧠 LLM & Agent Systems** — RAG pipelines, agentic workflows, prompt frameworks, evaluation. Written from hands-on use, not press releases.
7. **⚡ Local & Private AI** — Ollama, LM Studio, llama.cpp, GGUF, SLM fine-tuning and quantisation. Deployment content for teams that cannot send data to a cloud API.
8. **🐍 Python for AI/ML** — Code gets written, run, and verified before it gets documented. PyTorch, HuggingFace, FastAPI, LangChain.

### 6.3 Tech stack chip groups

- **⚙️ Languages I Write About** — Python, Rust, Go, JavaScript, Bash, Git/GitHub/GitLab
- **🤖 AI & ML** — Generative AI, LLMs, SLMs, AI Agents, Agentic Frameworks, RAG, Prompt Engineering, Evaluation
- **⚛️ Quantum Stack** *(violet accent — NEW)* — IBM Qiskit, PennyLane, Quantum Machine Learning, Variational Circuits, Microsoft Q#, Quantinuum, Quantum Algorithms, Post-Quantum Cryptography, Quantum Technologies
- **🛠️ Local AI Stack** — Ollama, LM Studio, llama.cpp, GGUF, Unsloth, Qwen, Gemma, Phi, HuggingFace Hub
- **🧰 Daily Toolchain** — Claude Code, Google Antigravity, Google AI Studio, NotebookLM, Microsoft Copilot, OpenAI API, Code Wiki
- **📝 Content & Docs** *(NEW)* — Markdown, MDX, Docusaurus, Mintlify, Jupyter, OpenAPI, Diátaxis framework, Technical SEO, JSON-LD
- **📚 Domains** — Generative AI, Agentic AI, Quantum Computing, Quantum Machine Learning, Quantum Technologies, Local Inference, Cybersecurity, Open Source

**Quantum Stack** and **Content & Docs** are new groups. Both are credibility signals for the new positioning and do not currently exist.

### 6.4 Contact

Remove the full form. Single-line CTA band → `services.html#contact`.

---

## 7. `services.html` — Services & Commissions

### 7.1 Title and H1

- `<title>`: `Services & Commissions — Thomas Cherickal | AI & Quantum Content`
- `<h1>`: `Services, Commissions & Collaboration`
- Subtitle: "Documentation, deep dives, developer education, and quantum developer content. All delivery is remote and asynchronous."

### 7.2 Replace the four training tiers with six service blocks

The current four tiers are all live-delivery, hours-billed, and two imply onsite. Replace entirely.

**Service 1 — Developer Documentation**
API references, SDK guides, quickstarts, integration and migration guides, for AI APIs and quantum SDKs.
Bullets: written against the live API or SDK · every code sample run and verified · delivered in Markdown/MDX ready to publish · one revision round included.
CTA: `Discuss a Docs Project →` `#contact`

**Service 2 — Technical Deep Dives & Explainers**
Commissioned long-form, 2,000–8,000 words, across AI and quantum.
Bullets: original research and benchmarking · published-chart embeds with attribution · engineer-grade accuracy, general-reader clarity · full SEO/AEO structuring included.
CTA: `Commission a Deep Dive →` `#contact`

**Service 3 — Quantum Developer Content** *(violet accent)*
Qiskit and PennyLane tutorials, quantum machine learning explainers, quantum algorithm walkthroughs, SDK documentation, and post-quantum cryptography briefings.
Bullets: hands-on Qiskit and PennyLane · notebooks that run · quantum concepts written for developers, not physicists · comparative framework coverage (Qiskit, Q#, Quantinuum).
CTA: `Discuss Quantum Content →` `#contact`

**Service 4 — Developer Education & Courses**
Tutorial series, structured learning paths, workshop curricula, certification material.
Bullets: Diátaxis-structured · working repo or notebook per module · progressive difficulty · async, self-serve by design.
CTA: `Scope a Course →` `#contact`

**Service 5 — Launch & Migration Content**
Model, API and SDK launch explainers, changelog deep dives, competitor comparisons, migration guides.
Bullets: fast turnaround for launch windows · verified against live product · comparison tables with sourced figures.
CTA: `Discuss Launch Content →` `#contact`

**Service 6 — Remote Team Training (secondary)**
Live remote enablement for engineering teams. 2 hours, remote only. Two tracks: Generative AI, or Quantum Readiness & Post-Quantum Risk.
Bullets: agentic AI development and multi-agent patterns · local LLM deployment (Ollama, LM Studio, llama.cpp) · Claude Code terminal integration and TDD workflows · quantum-readiness and post-quantum crypto risk track available · ends in a deployed working artefact per participant.
CTA: `Book Remote Training →` `#contact`
**Must state "Live Remote" explicitly. No onsite. No travel.**

Retain a small **1:1 Mentoring via Topmate** card below the six, visually de-emphasised, linking to `https://topmate.io/thomascherickal`.

### 7.3 Collaboration section — 4 cards

Delete "Speaking & Workshops" and "Advisory Roles". Final four:

1. **✍️ Technical Writing** — Commissioned articles, documentation, and deep dives
2. **📚 Developer Education** — Courses, tutorial series, and learning paths
3. **🤝 Content Partnerships** — Ongoing retainer content for AI and quantum companies
4. **🎯 Content Strategy** — Developer content audits, docs architecture, and roadmaps

Link row below (unchanged targets): Topmate · GitHub · LinkedIn.

### 7.4 Contact section — the ONLY full form on the site

Keep the full `formsubmit.co` form here, with `id="contact"`.

- Remove `"speaking opportunity"` from the intro. New copy: "Have a documentation project, a deep dive to commission, a course to build, quantum content to write, or a team to train? Send a message or reach out directly."
- Add a **Project type** `<select>`: Documentation · Deep Dive / Article · Quantum Content · Course / Tutorial Series · Launch Content · Remote Team Training · Other
- Anti-spam: hidden honeypot input (`_honey`), `_captcha` enabled, `_subject` set to `Website Enquiry — thomascherickal.github.io`
- **Obfuscate the email address.** No plaintext `thomascherickal@gmail.com`. Render via inline JS join or HTML entity encoding, with `mailto:` built at runtime. Entity-encode the `formsubmit.co` action URL, or use formsubmit's hashed-address endpoint.
- Add: "Typical response within 24 hours. All work is remote and asynchronous."

---

## 8. `writing.html` — Publications

### 8.1 Fix the 500+ / 20+ contradiction — CRITICAL

- `<title>`: `Publications — Thomas Cherickal | 500+ AI & Quantum Articles`
- Meta description: **rewrite**, must not say "20+". Use: "Selected technical deep dives from 500+ published articles across 10+ platforms — Generative AI, agents, local inference, quantum computing and QML — plus the book RECRUITED."
- On-page intro: `Selected work - 24 pieces from 500+ published articles across 10+ platforms since 2020.`

Search all HTML files for `20+` and `20 technical` and eliminate every instance referring to article count.

### 8.2 Platform diversity

All 24 currently listed articles are HackerNoon while the page claims five platforms. Either add at least 4 non-HackerNoon pieces with a platform badge on each card, or retitle to "Selected HackerNoon Deep Dives" plus an "Also published on" row linking to Medium, Substack and Hashnode profiles. Add a platform badge to every article item either way. **Do not list Dev.to.**

### 8.3 Category order — quantum moves UP

1. 🤖 Generative AI, LLMs & Agents
2. ⚛️ Quantum Computing & QML *(violet accent — promoted to position 2)*
3. 🦾 Agentic AI Frameworks
4. 🛠️ AI Tools & Developer Productivity
5. 🔮 Futurism & AGI
6. 🔗 Blockchain *(move last, or collapse behind a "More" disclosure)*

The four existing quantum articles stay. Retitle the category from "Quantum Computing & QAI" to "Quantum Computing & QML".

### 8.4 RECRUITED block

- **Replace the cover image** with the supplied `recruited-cover.webp` (§11.2)
- **Correct the subtitle.** The site currently says *"The AI-Powered Career Playbook for Professionals Who Refuse to Be Left Behind"*. The actual cover art reads **"The Inbound Recruiter Blueprint: How to Make Recruiters Chase You"**. Use the cover's subtitle everywhere — on-page, in `<meta>`, and in the `Book` schema. The two must match.
- **Add the visible price:** `$5.00 USD — pre-order · Free with an active Patreon subscription`
- CTA `🎨 Pre-Order on Patreon` → `https://patreon.com/thomascherickal`
- Add `Book` JSON-LD (§10.4)
- Explicit `width="700" height="1000"`, `loading="lazy"`, `decoding="async"`, meaningful `alt`

### 8.5 Add a "Work With Me" band

Below books: "Commissioning technical content? See services and rates." → `services.html#contact`.

### 8.6 Contact

Remove the full form. Single-line CTA band → `services.html#contact`.

---

## 9. Navigation — all pages

Nav becomes five items in this order:

`Home` · `Portfolio` · `Publications` · `Capabilities` · `Services`

→ `index.html` · `portfolio.html` · `writing.html` · `about.html` · `services.html`

Mark the active item with `aria-current="page"`. Verify the nav does not overflow on a 360 px viewport; if it does, collapse to a hamburger at that breakpoint using CSS only (a checkbox toggle — no JS framework).

---

## 10. JSON-LD — DROP-IN BLOCKS

**Highest-value technical work in this brief.** Place each block immediately before `</head>`.

### 10.1 `index.html` — replace the existing block entirely

```html
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@graph": [
    {
      "@type": "WebSite",
      "@id": "https://thomascherickal.github.io/#website",
      "url": "https://thomascherickal.github.io/",
      "name": "Thomas Cherickal — The Digital Futurist",
      "description": "Technical Content Engineer and Developer Educator for AI and Quantum companies.",
      "publisher": { "@id": "https://thomascherickal.github.io/#organization" },
      "inLanguage": "en-US"
    },
    {
      "@type": "Organization",
      "@id": "https://thomascherickal.github.io/#organization",
      "name": "The Digital Futurist",
      "alternateName": "Thomas Cherickal — The Digital Futurist",
      "url": "https://thomascherickal.com",
      "foundingDate": "2020",
      "founder": { "@id": "https://thomascherickal.github.io/#person" },
      "logo": {
        "@type": "ImageObject",
        "url": "https://thomascherickal.github.io/assets/images/site-icon.webp",
        "width": 512,
        "height": 512
      },
      "address": {
        "@type": "PostalAddress",
        "addressLocality": "Chennai",
        "addressRegion": "Tamil Nadu",
        "postalCode": "600015",
        "addressCountry": "IN"
      }
    },
    {
      "@type": "ProfilePage",
      "@id": "https://thomascherickal.github.io/#profilepage",
      "url": "https://thomascherickal.github.io/",
      "name": "Thomas Cherickal — Technical Content Engineer & Developer Educator",
      "isPartOf": { "@id": "https://thomascherickal.github.io/#website" },
      "primaryImageOfPage": { "@id": "https://thomascherickal.github.io/#primaryimage" },
      "mainEntity": { "@id": "https://thomascherickal.github.io/#person" },
      "inLanguage": "en-US"
    },
    {
      "@type": "ImageObject",
      "@id": "https://thomascherickal.github.io/#primaryimage",
      "url": "https://thomascherickal.github.io/assets/images/thomas-avatar.webp",
      "caption": "Thomas Cherickal",
      "width": 512,
      "height": 512
    },
    {
      "@type": "Person",
      "@id": "https://thomascherickal.github.io/#person",
      "name": "Thomas Cherickal",
      "alternateName": ["Thomas Mathew Cherickal", "The Digital Futurist"],
      "url": "https://thomascherickal.com",
      "mainEntityOfPage": { "@id": "https://thomascherickal.github.io/#profilepage" },
      "image": { "@id": "https://thomascherickal.github.io/#primaryimage" },
      "jobTitle": "Technical Content Engineer & Developer Educator",
      "description": "Technical Content Engineer and Developer Educator for AI and Quantum companies. Documentation, deep dives, and courses on LLMs, agents, IBM Qiskit, PennyLane and quantum machine learning. 500+ published technical articles across 10+ platforms since 2020.",
      "email": "thomascherickal@gmail.com",
      "nationality": { "@type": "Country", "name": "India" },
      "worksFor": { "@id": "https://thomascherickal.github.io/#organization" },
      "address": {
        "@type": "PostalAddress",
        "addressLocality": "Chennai",
        "addressRegion": "Tamil Nadu",
        "postalCode": "600015",
        "addressCountry": "IN"
      },
      "knowsAbout": [
        "Technical Writing",
        "Developer Documentation",
        "API Documentation",
        "Developer Education",
        "Generative AI",
        "Large Language Models",
        "Small Language Models",
        "Agentic AI",
        "Retrieval Augmented Generation",
        "Local LLM Deployment",
        "Model Quantisation",
        "Python",
        "Quantum Computing",
        "Quantum Machine Learning",
        "Quantum Technologies",
        "Quantum Algorithms",
        "IBM Qiskit",
        "PennyLane",
        "Post-Quantum Cryptography",
        "Technical SEO"
      ],
      "sameAs": [
        "https://thomascherickal.com",
        "https://thomascherickal.github.io",
        "https://github.com/thomascherickal",
        "https://linkedin.com/in/thomascherickal",
        "https://gitlab.com/thomascherickal",
        "https://hackernoon.com/u/thomascherickal",
        "https://thomascherickal.medium.com",
        "https://thomascherickal.hashnode.dev",
        "https://thesingularitypoint.substack.com",
        "https://thomascherickal.carrd.co/",
        "https://thomascherickal.quora.com",
        "https://reddit.com/user/thomascherickal",
        "https://www.kaggle.com/thomascherickal",
        "https://profile.codersrank.io/user/thomascherickal/",
        "https://www.geeksforgeeks.org/profile/thomascherickal",
        "https://hubpages.com/@thomascherickal",
        "https://www.deep-ml.com/profile/thomascherickal",
        "https://hackerrank.com/profile/thomascherickal",
        "https://leetcode.com/u/thomascherickal",
        "https://linktr.ee/thomascherickal",
        "https://patreon.com/thomascherickal",
        "https://thomascherickal.gumroad.com",
        "https://topmate.io/thomascherickal"
      ]
    }
  ]
}
</script>
```

**There is deliberately no `knowsLanguage` property. Do not add one.**

Verify every `sameAs` URL returns HTTP 200 before shipping. All 22 were confirmed live at audit time.

### 10.2 `about.html`

```html
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@graph": [
    {
      "@type": "AboutPage",
      "@id": "https://thomascherickal.github.io/about.html#webpage",
      "url": "https://thomascherickal.github.io/about.html",
      "name": "Capabilities & Tech Stack — Thomas Cherickal",
      "description": "Developer documentation, technical deep dives, developer education, quantum computing and QML, LLM and agent systems, local AI, and Python for AI/ML.",
      "isPartOf": { "@id": "https://thomascherickal.github.io/#website" },
      "about": { "@id": "https://thomascherickal.github.io/#person" },
      "inLanguage": "en-US",
      "breadcrumb": { "@id": "https://thomascherickal.github.io/about.html#breadcrumb" }
    },
    {
      "@type": "BreadcrumbList",
      "@id": "https://thomascherickal.github.io/about.html#breadcrumb",
      "itemListElement": [
        { "@type": "ListItem", "position": 1, "name": "Home", "item": "https://thomascherickal.github.io/" },
        { "@type": "ListItem", "position": 2, "name": "Capabilities & Tech Stack" }
      ]
    }
  ]
}
</script>
```

### 10.3 `services.html`

```html
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@graph": [
    {
      "@type": "WebPage",
      "@id": "https://thomascherickal.github.io/services.html#webpage",
      "url": "https://thomascherickal.github.io/services.html",
      "name": "Services & Commissions — Thomas Cherickal",
      "isPartOf": { "@id": "https://thomascherickal.github.io/#website" },
      "inLanguage": "en-US",
      "breadcrumb": { "@id": "https://thomascherickal.github.io/services.html#breadcrumb" }
    },
    {
      "@type": "BreadcrumbList",
      "@id": "https://thomascherickal.github.io/services.html#breadcrumb",
      "itemListElement": [
        { "@type": "ListItem", "position": 1, "name": "Home", "item": "https://thomascherickal.github.io/" },
        { "@type": "ListItem", "position": 2, "name": "Services & Commissions" }
      ]
    },
    {
      "@type": "ProfessionalService",
      "@id": "https://thomascherickal.github.io/services.html#service",
      "name": "The Digital Futurist — Technical Content Engineering",
      "url": "https://thomascherickal.github.io/services.html",
      "description": "Developer documentation, technical deep dives, quantum developer content, developer education, launch content, and remote engineering-team training for AI and quantum companies.",
      "provider": { "@id": "https://thomascherickal.github.io/#person" },
      "areaServed": { "@type": "Place", "name": "Worldwide (remote)" },
      "availableLanguage": "English",
      "hasOfferCatalog": {
        "@type": "OfferCatalog",
        "name": "Technical Content & Developer Education Services",
        "itemListElement": [
          {
            "@type": "Offer",
            "itemOffered": {
              "@type": "Service",
              "name": "Developer Documentation",
              "description": "API references, SDK guides, quickstarts, integration and migration guides for AI APIs and quantum SDKs. Every code sample run and verified. Delivered in Markdown or MDX.",
              "serviceType": "Technical Documentation"
            }
          },
          {
            "@type": "Offer",
            "itemOffered": {
              "@type": "Service",
              "name": "Technical Deep Dives & Explainers",
              "description": "Commissioned long-form technical articles from 2,000 to 8,000 words across AI and quantum computing, with original research and verified benchmarks.",
              "serviceType": "Technical Writing"
            }
          },
          {
            "@type": "Offer",
            "itemOffered": {
              "@type": "Service",
              "name": "Quantum Developer Content",
              "description": "IBM Qiskit and PennyLane tutorials, quantum machine learning explainers, quantum algorithm walkthroughs, quantum SDK documentation, and post-quantum cryptography briefings written for working developers.",
              "serviceType": "Technical Documentation"
            }
          },
          {
            "@type": "Offer",
            "itemOffered": {
              "@type": "Service",
              "name": "Developer Education & Courses",
              "description": "Tutorial series, structured learning paths, and workshop curricula with a working repository or notebook per module.",
              "serviceType": "Developer Education"
            }
          },
          {
            "@type": "Offer",
            "itemOffered": {
              "@type": "Service",
              "name": "Launch & Migration Content",
              "description": "Model, API and SDK launch explainers, changelog deep dives, comparison pieces, and migration guides on launch-window turnaround.",
              "serviceType": "Technical Content"
            }
          },
          {
            "@type": "Offer",
            "itemOffered": {
              "@type": "Service",
              "name": "Remote Engineering Team Training",
              "description": "Live remote enablement for engineering teams in Generative AI or Quantum Readiness. Agentic AI development, local LLM deployment, Claude Code workflows, and post-quantum cryptography risk. Two hours, remote only.",
              "serviceType": "Technical Training"
            }
          }
        ]
      }
    }
  ]
}
</script>
```

### 10.4 `writing.html`

```html
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@graph": [
    {
      "@type": "CollectionPage",
      "@id": "https://thomascherickal.github.io/writing.html#webpage",
      "url": "https://thomascherickal.github.io/writing.html",
      "name": "Publications — Thomas Cherickal",
      "description": "Selected technical deep dives from 500+ published articles across 10+ platforms, covering Generative AI, agents, local inference, quantum computing and QML, plus the book RECRUITED.",
      "isPartOf": { "@id": "https://thomascherickal.github.io/#website" },
      "author": { "@id": "https://thomascherickal.github.io/#person" },
      "inLanguage": "en-US",
      "breadcrumb": { "@id": "https://thomascherickal.github.io/writing.html#breadcrumb" }
    },
    {
      "@type": "BreadcrumbList",
      "@id": "https://thomascherickal.github.io/writing.html#breadcrumb",
      "itemListElement": [
        { "@type": "ListItem", "position": 1, "name": "Home", "item": "https://thomascherickal.github.io/" },
        { "@type": "ListItem", "position": 2, "name": "Publications" }
      ]
    },
    {
      "@type": "Book",
      "@id": "https://thomascherickal.github.io/writing.html#recruited",
      "name": "RECRUITED",
      "alternativeHeadline": "The Inbound Recruiter Blueprint: How to Make Recruiters Chase You",
      "author": { "@id": "https://thomascherickal.github.io/#person" },
      "publisher": { "@id": "https://thomascherickal.github.io/#organization" },
      "inLanguage": "en",
      "bookFormat": "https://schema.org/EBook",
      "image": "https://thomascherickal.github.io/recruited-cover.webp",
      "description": "How to make recruiters come to you. A system for using frontier AI tools - ChatGPT, Claude, Gemini, NotebookLM, Perplexity, Google Antigravity - to rebuild your professional presence across GitHub and LinkedIn so that inbound offers find you.",
      "url": "https://thomascherickal.com/recruited",
      "offers": {
        "@type": "Offer",
        "price": "5.00",
        "priceCurrency": "USD",
        "availability": "https://schema.org/PreOrder",
        "priceValidUntil": "2026-12-31",
        "url": "https://patreon.com/thomascherickal",
        "seller": { "@id": "https://thomascherickal.github.io/#organization" }
      }
    }
  ]
}
</script>
```

### 10.5 `portfolio.html`

See §13.5.

---

## 11. FAVICON, BOOK COVER & IMAGE ASSETS

### 11.1 Favicon — files supplied, do not regenerate

The favicon derives from the owner's supplied artwork: an illuminated cross against a starfield. A vector version in the site's gold palette is supplied alongside the raster set because a photographic image degrades badly at 16 px.

**Copy to repo root:**

```
favicon.ico
favicon.svg
favicon-16x16.png
favicon-32x32.png
favicon-96x96.png
apple-touch-icon.png
android-chrome-192x192.png
android-chrome-512x512.png
```

**Copy to `assets/images/`:**

```
site-icon.webp
```

**Add to `<head>` on ALL FIVE pages, in this order.** Modern browsers prefer the SVG and fall back to the raster set automatically.

```html
<link rel="icon" href="/favicon.ico" sizes="any" />
<link rel="icon" href="/favicon.svg" type="image/svg+xml" />
<link rel="icon" type="image/png" sizes="32x32" href="/favicon-32x32.png" />
<link rel="icon" type="image/png" sizes="16x16" href="/favicon-16x16.png" />
<link rel="apple-touch-icon" sizes="180x180" href="/apple-touch-icon.png" />
<link rel="manifest" href="/site.webmanifest" />
<meta name="theme-color" content="#000000" />
```

**`site.webmanifest`** at repo root:

```json
{
  "name": "Thomas Cherickal — The Digital Futurist",
  "short_name": "T. Cherickal",
  "description": "Technical Content Engineer & Developer Educator for AI and Quantum companies.",
  "icons": [
    { "src": "/android-chrome-192x192.png", "sizes": "192x192", "type": "image/png" },
    { "src": "/android-chrome-512x512.png", "sizes": "512x512", "type": "image/png" }
  ],
  "theme_color": "#000000",
  "background_color": "#000000",
  "display": "standalone",
  "start_url": "/"
}
```

**Owner's note to the agent:** the favicon choice is deliberate and personal. Do not substitute a monogram, initial, or generic tech glyph. Two variants are supplied — `favicon.svg` (vector, sharpest) and the raster set (photographic). If both are wired in as above, browsers pick the best available automatically. **Do not remove either.**

### 11.2 Book cover — file supplied

**Delete** the existing `recruited-cover.jpg` (365 KB) and replace with the supplied files:

```
recruited-cover.webp        700x1000, 58 KB   <- primary, use in <picture>
recruited-cover.jpg         JPEG fallback
recruited-cover-full.webp   1049x1500 archival, not referenced in HTML
```

Markup on `writing.html`:

```html
<picture>
  <source srcset="recruited-cover.webp" type="image/webp">
  <img src="recruited-cover.jpg"
       width="700" height="1000"
       loading="lazy" decoding="async"
       alt="RECRUITED by Thomas Cherickal — The Inbound Recruiter Blueprint: How to Make Recruiters Chase You">
</picture>
```

**Subtitle correction is mandatory.** The cover art reads *"The Inbound Recruiter Blueprint: How to Make Recruiters Chase You"*. The site currently displays a different subtitle. Update on-page copy, meta tags, and `Book` schema to match the cover. Any surviving instance of *"The AI-Powered Career Playbook for Professionals Who Refuse to Be Left Behind"* is a failed build.

The cover depicts GitHub, LinkedIn, Perplexity, Claude, Google Antigravity and NotebookLM. Book description copy should name those six tools, since the cover promises them.

### 11.3 Social card

`og-card-book-BASE.png` (1200×630) is supplied as a starting point for the book/social card — cover art on black with space at right for text.

Create the primary site card at `assets/images/og-card.png`, **exactly 1200×630**, black background, in the site's Orbitron/cyan-gold treatment, containing:

- `Thomas Cherickal`
- `Technical Content Engineer & Developer Educator`
- `AI · Quantum · Developer Documentation`
- `500+ articles · 10+ platforms · since 2020`

Point all `og:image` and `twitter:image` tags at it on all five pages, except `writing.html`, which may use the book-derived card.

### 11.4 Remaining image rules

- Explicit `width` and `height` on **every** `<img>` sitewide (currently zero have them) to eliminate CLS.
- `loading="lazy"` and `decoding="async"` on every image **except** the hero avatar on `index.html`, which gets `loading="eager"` and `fetchpriority="high"`.
- Every `<img>` needs a meaningful `alt`. Decorative images get `alt=""`.

---

## 12. TECHNICAL, PERFORMANCE & ACCESSIBILITY FIXES

### 12.1 Meta tags — all five pages

```html
<meta property="og:site_name" content="Thomas Cherickal — The Digital Futurist" />
<meta property="og:locale" content="en_US" />
<meta property="og:image:width" content="1200" />
<meta property="og:image:height" content="630" />
<meta property="og:image:alt" content="Thomas Cherickal — Technical Content Engineer & Developer Educator for AI and Quantum companies" />
<meta name="twitter:site" content="@thomascherickal" />
<meta name="twitter:creator" content="@thomascherickal" />
<meta name="author" content="Thomas Cherickal" />
```

Remove `<meta name="keywords">` from all pages — ignored by every major engine, signals dated SEO.

Rewrite every `<title>` and `<meta name="description">` to the new positioning. **No title may still say "Generative AI Consultant" as the primary identity, and every title and description must reference both AI and Quantum.**

### 12.2 Accessibility — currently zero `aria-label` attributes sitewide

1. Skip-link as the first focusable element on every page:
   ```html
   <a href="#main" class="skip-link">Skip to content</a>
   ```
   Wrap primary content in `<main id="main">`. Style `.skip-link` off-screen until `:focus`.
2. `aria-label` on **all 22 footer social links**, e.g. `aria-label="Thomas Cherickal on GitHub"`.
3. `aria-label="Back to top"` on the `↑` control; `aria-label="Main navigation"` on `<nav>`.
4. `aria-current="page"` on the active nav item.
5. All form inputs need associated `<label>` elements, not placeholder-only.
6. Verify text contrast against pure black meets WCAG AA (4.5:1). **The muted violet used for quantum accents is the highest risk — check it first and lighten if it fails.**
7. Verify `<html lang="en">` on all five pages.

### 12.3 Duplication

After this rebuild the full contact form appears **once**, on `services.html`. The other four pages get a one-line CTA band.

### 12.4 Email protection

Eight plaintext instances currently exist. After the rebuild: **zero**. Entity encoding or runtime JS assembly. Address must remain functional for humans.

### 12.5 New root files

**`llms.txt`:**

```
# Thomas Cherickal — The Digital Futurist

> Technical Content Engineer and Developer Educator for AI and Quantum companies.
> 500+ published technical articles across 10+ platforms since 2020.
> Based in Chennai, India. All work remote and asynchronous.

## What I do
- Developer documentation: API references, SDK guides, quickstarts, migration guides
- Technical deep dives on AI, LLMs, agents, local inference, and quantum computing
- Quantum developer content: IBM Qiskit, PennyLane, quantum machine learning,
  quantum algorithms, post-quantum cryptography
- Developer education: courses, tutorial series, structured learning paths
- Launch and migration content for AI and quantum products
- Remote Generative AI and quantum-readiness training for engineering teams

## Pages
- Home: https://thomascherickal.github.io/
- Portfolio & Case Studies: https://thomascherickal.github.io/portfolio.html
- Publications: https://thomascherickal.github.io/writing.html
- Capabilities & Tech Stack: https://thomascherickal.github.io/about.html
- Services & Commissions: https://thomascherickal.github.io/services.html

## Elsewhere
- Primary site: https://thomascherickal.com
- GitHub: https://github.com/thomascherickal
- LinkedIn: https://linkedin.com/in/thomascherickal
- HackerNoon: https://hackernoon.com/u/thomascherickal
- Newsletter: https://thomascherickal.kit.com

## Contact
https://thomascherickal.github.io/services.html#contact
```

Add to `robots.txt`:
```
# LLM/AI crawler summary
Sitemap: https://thomascherickal.github.io/llms.txt
```

**`404.html`** — GitHub Pages serves this automatically. Match the site aesthetic, link to all five pages.

**`site.webmanifest`** — see §11.1.

### 12.6 Sitemap

Update `lastmod` on all URLs to the deploy date and **add the new `portfolio.html` entry**:

```xml
<url>
  <loc>https://thomascherickal.github.io/portfolio.html</loc>
  <lastmod>DEPLOY_DATE</lastmod>
  <changefreq>monthly</changefreq>
  <priority>1.0</priority>
</url>
```

Raise `writing.html` priority to `1.0` — it is now a primary sales asset.

---

## 13. BUILD `portfolio.html` — NEW FIFTH PAGE

**Do this LAST.** Complete Sections 5–12 and verify them before starting this page. It depends on the finished nav, footer, styles, and schema graph.

### 13.1 Why this page exists

`writing.html` proves *volume* — 500+ articles. It does not prove *fitness for commissioned work*. A buyer at a quantum SDK company does not want a list of links; they want evidence that this person can take a brief, understand a system, and ship publishable technical content on a deadline.

`portfolio.html` is the closing asset. It is the page linked in outbound pitches. Structurally it is a **case-study page**, not a link list.

### 13.2 Page shell

Clone the exact shell from `services.html` after that page is finished: same `<head>` pattern, nav, footer, skip-link, `<main id="main">`, back-to-top. Change only page-specific meta and content.

- `<title>`: `Portfolio & Case Studies — Thomas Cherickal | AI & Quantum Technical Content`
- Canonical: `https://thomascherickal.github.io/portfolio.html`
- Meta description: `Selected technical content case studies across AI and quantum computing - the brief, the approach, and what shipped. Documentation, deep dives, courses and developer education by Thomas Cherickal.`
- `<h1>`: `Portfolio & Case Studies`
- Kicker above H1: `Selected Work`
- Subtitle: `Not a link list. The brief, the approach, and what shipped - for each project.`

### 13.3 Case study card — repeating component

Build **six** case study blocks. Each is a card using the existing card styling, with a coloured left border in the domain accent (cyan for AI, violet for quantum, gold for education/other).

Each card contains, in this exact order:

1. **Domain badge** — pill: `AI`, `Quantum`, or `Developer Education` (accent-coloured)
2. **Format badge** — pill: `Deep Dive`, `Documentation`, `Tutorial Series`, `Comparative Analysis`, `Course`
3. **`<h3>` Project title**
4. **Meta row** — `Published: [platform] · [length] words · [year]`
5. **The Brief** — 1–2 sentences on what the piece had to accomplish and for whom
6. **The Approach** — 3–4 bullets on method: what was researched, what was run and verified, what was benchmarked, how it was structured
7. **What Shipped** — 1–2 sentences on the delivered artefact and any measurable outcome
8. **Stack row** — small chips naming actual tools used (e.g. `Qiskit` `PennyLane` `Jupyter`, or `Ollama` `llama.cpp` `GGUF`)
9. **CTA** — `Read the piece →` linking to the live published URL

### 13.4 The six case studies to build

Source all content from existing published work listed on `writing.html`. **Do not invent projects, clients, metrics, or outcomes.** If a specific figure is not known, describe the approach rather than claiming a result.

| # | Domain | Source article | Angle |
|---|---|---|---|
| 1 | Quantum | *Comparing Quantum Frameworks: IBM Qiskit, Microsoft Q#, and Quantinuum's New Stack* | Comparative technical analysis across three SDKs — the flagship quantum credential. **Place this first.** |
| 2 | Quantum | *Quantum Computing Fundamentals Part I & II: 10 Easy Pieces / 10 Not-So Easy Pieces* | A two-part progressive learning path — proof of course/curriculum design |
| 3 | AI | *How to Run Your Own Local LLM — 2026 Edition* | Hands-on deployment documentation, verified on real hardware |
| 4 | AI | *Google Gemini vs Anthropic Claude vs OpenAI ChatGPT vs xAI Grok: The Ultimate Comparison* | Multi-vendor comparative analysis with sourced figures |
| 5 | AI | *The OpenClaw Saga* / *Hermes Agent vs OpenClaw* | Fast-turnaround launch and ecosystem coverage |
| 6 | Quantum | *How Quantum Computers Threaten Bitcoin and the Entire Internet* | Post-quantum cryptography risk explained for a non-specialist technical audience |

**Ratio is deliberate: three quantum, three AI.** The quantum work is the differentiator and must not be outnumbered on the page that closes deals.

### 13.5 JSON-LD for `portfolio.html`

```html
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@graph": [
    {
      "@type": "CollectionPage",
      "@id": "https://thomascherickal.github.io/portfolio.html#webpage",
      "url": "https://thomascherickal.github.io/portfolio.html",
      "name": "Portfolio & Case Studies — Thomas Cherickal",
      "description": "Selected technical content case studies across AI and quantum computing - the brief, the approach, and what shipped.",
      "isPartOf": { "@id": "https://thomascherickal.github.io/#website" },
      "author": { "@id": "https://thomascherickal.github.io/#person" },
      "about": { "@id": "https://thomascherickal.github.io/#person" },
      "inLanguage": "en-US",
      "breadcrumb": { "@id": "https://thomascherickal.github.io/portfolio.html#breadcrumb" },
      "mainEntity": { "@id": "https://thomascherickal.github.io/portfolio.html#worklist" }
    },
    {
      "@type": "BreadcrumbList",
      "@id": "https://thomascherickal.github.io/portfolio.html#breadcrumb",
      "itemListElement": [
        { "@type": "ListItem", "position": 1, "name": "Home", "item": "https://thomascherickal.github.io/" },
        { "@type": "ListItem", "position": 2, "name": "Portfolio & Case Studies" }
      ]
    },
    {
      "@type": "ItemList",
      "@id": "https://thomascherickal.github.io/portfolio.html#worklist",
      "name": "Selected Technical Content Case Studies",
      "itemListOrder": "https://schema.org/ItemListOrderAscending",
      "numberOfItems": 6,
      "itemListElement": [
        {
          "@type": "ListItem",
          "position": 1,
          "item": {
            "@type": "TechArticle",
            "headline": "Comparing Quantum Programming Frameworks: IBM Qiskit, Microsoft Q#, and Quantinuum's New Stack",
            "url": "https://hackernoon.com/comparing-quantum-programming-frameworks-ibm-qiskit-microsoft-q-and-quantinuums-new-stack",
            "author": { "@id": "https://thomascherickal.github.io/#person" },
            "about": "Quantum Computing",
            "keywords": "IBM Qiskit, Microsoft Q#, Quantinuum, quantum programming frameworks"
          }
        }
      ]
    }
  ]
}
</script>
```

**Extend `itemListElement` to all six** following the same `ListItem` → `TechArticle` pattern, incrementing `position`, with the correct `headline`, `url`, `about` (`"Quantum Computing"` or `"Artificial Intelligence"`) and `keywords` for each. Set `numberOfItems` to the final count.

### 13.6 Closing band

End the page above the footer with a CTA band:

> **Need something like this written?**
> Documentation, deep dives, courses, and quantum developer content. Remote and asynchronous.
> `Start a Conversation →` → `services.html#contact`

### 13.7 After building

- Add `portfolio.html` to the nav on all five pages (§9)
- Add to `sitemap.xml` with priority `1.0` (§12.6)
- Add to `llms.txt` (already listed in §12.5)
- Update the homepage `📚 Read the Portfolio` CTA and destination card to point here (§5.2, §5.4)

---

## 14. ACCEPTANCE CRITERIA

Do not report completion until every box is verifiable.

**Strategic**
- [ ] No page presents "Generative AI Consultant" as the primary identity
- [ ] Every page's `<h1>`, `<title>`, and meta description reflect Technical Content Engineer / Developer Educator
- [ ] **Every page title and meta description references both AI and Quantum**
- [ ] Quantum appears in: homepage hero chips, homepage stat block, ≥2 `about.html` capability cards, a dedicated `services.html` service block, a promoted `writing.html` category, ≥3 `portfolio.html` case studies, and `knowsAbout`
- [ ] IBM Qiskit, PennyLane, and Quantum Machine Learning each appear on at least two pages
- [ ] The word "onsite" (and "on site") appears nowhere in the repo
- [ ] No speaking, keynote, or webinar offers remain anywhere, including form copy
- [ ] Rust Systems Engineer, Go Cloud Engineer, HTML/CSS/JavaScript, Deployment role cards deleted
- [ ] Students and individual founders are not addressed as customer segments

**Consistency**
- [ ] `grep -r "20+" *.html` returns no result referring to article count
- [ ] "500+" is the only article-count figure on the site
- [ ] `grep -ri "AI-Powered Career Playbook" .` returns nothing
- [ ] The book subtitle everywhere is "The Inbound Recruiter Blueprint: How to Make Recruiters Chase You"
- [ ] Only `thesingularitypoint.substack.com` appears
- [ ] `patreon.com/thomascherickal` is the only Patreon URL
- [ ] `linkedin.com/in/thomascherickal` (no hyphen) is the only LinkedIn URL
- [ ] Dev.to appears nowhere
- [ ] "2020" is the only brand start year
- [ ] Wherever `thomascherickal.com` appears, `thomascherickal.github.io` also appears
- [ ] No excluded entity (§3.1) appears in any file

**Structured data**
- [ ] All five pages have a JSON-LD block
- [ ] `Person` has a 22-entry `sameAs` array and a `knowsAbout` array including all quantum terms
- [ ] **`knowsLanguage` does not appear in any file**
- [ ] `Organization`, `Book` (with $5.00 Offer and correct subtitle), `ProfessionalService` (6-item `offerCatalog` including Quantum Developer Content), `ItemList` on portfolio, and `BreadcrumbList` on all four inner pages present
- [ ] Every block validates in Google Rich Results Test **and** Schema.org validator with zero errors
- [ ] Every `sameAs` URL returns HTTP 200

**Assets**
- [ ] All eight favicon files at repo root; `site-icon.webp` in `assets/images/`
- [ ] Favicon `<link>` block present on all five pages
- [ ] `site.webmanifest` present and valid
- [ ] Favicon visibly renders in a browser tab
- [ ] Old `recruited-cover.jpg` (365 KB) replaced by supplied files
- [ ] `recruited-cover.webp` under 60 KB and used via `<picture>`
- [ ] `assets/images/og-card.png` exists at exactly 1200×630, referenced by all pages
- [ ] Every `<img>` has `width`, `height`, and meaningful `alt`
- [ ] `loading="lazy"` on all images except the hero avatar

**Technical**
- [ ] `<meta name="keywords">` removed from all pages
- [ ] Zero plaintext instances of the email address
- [ ] Full contact form on `services.html` only
- [ ] `llms.txt`, `404.html`, `site.webmanifest` exist
- [ ] `sitemap.xml` includes `portfolio.html` and updated `lastmod`
- [ ] Nav shows five items on all five pages with `aria-current="page"`

**Accessibility**
- [ ] Skip-link present and functional on all five pages
- [ ] `<main id="main">` wraps primary content on all five pages
- [ ] All 22 footer links have `aria-label`
- [ ] All form inputs have associated `<label>` elements
- [ ] Lighthouse Accessibility ≥ 95 on all five pages
- [ ] All text passes WCAG AA contrast against `#000000` — **violet quantum accent verified specifically**

**Regression — must remain true**
- [ ] All outbound links resolve (re-run a full link check; baseline was 61/61)
- [ ] Canonical tag on every page
- [ ] Single `<h1>` per page, no skipped heading levels
- [ ] Single external stylesheet, no build step, no framework, no npm
- [ ] Cyberpunk aesthetic preserved exactly (§3.4)
- [ ] Lighthouse Performance ≥ 95, SEO 100 on all five pages

---

## 15. Out of scope

- Any change to `thomascherickal.com` (separate WordPress property)
- Adding a blog engine, CMS, or build pipeline
- Public rate cards or pricing for services (deliberate — quoted per project)
- Any redesign of the visual language
- Inventing case study clients, metrics, or outcomes not present in published work

---

**End of brief.**
