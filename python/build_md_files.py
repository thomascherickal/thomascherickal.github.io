#!/usr/bin/env python3
"""
build_md_files.py — Extract structured, high-fidelity Markdown from all HTML pages
in thomascherickal.github.io and write them to md-html-sync/.
"""

import os
import re
import glob
from bs4 import BeautifulSoup, NavigableString, Tag

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
SYNC_DIR = os.path.join(REPO_DIR, "md-html-sync")
os.makedirs(SYNC_DIR, exist_ok=True)

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

def convert_portfolio() -> str:
    html_file = os.path.join(REPO_DIR, "portfolio.html")
    with open(html_file, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    
    lines = [
        "# Thomas Cherickal — Portfolio & Case Studies",
        "## Generative AI Consultant",
        "",
        "> **// Verified Case Studies**  ",
        "> **The brief, the approach, and what shipped — 14 verified, human-directed case studies across Generative AI (6), Agentic AI (4), and Quantum Computing (4), backed by runtime-verified code, deep analysis, and equally deep insights.**",
        "",
        "---",
        ""
    ]
    
    cards = soup.find_all("div", class_="case-study-card")
    for i, card in enumerate(cards, 1):
        title_el = card.find(["h2", "h3"])
        title = clean_text(title_el.get_text()) if title_el else f"Case Study #{i}"
        
        badges = [clean_text(b.get_text()) for b in card.find_all("span", class_="role-chip")]
        badge_str = " · ".join(f"`{b}`" for b in badges) if badges else ""
        
        cta_el = card.find("a", class_="case-study-cta-left")
        cta_url = cta_el.get("href", "") if cta_el else ""
        
        img_el = card.find("img", class_="case-study-cover-img")
        img_src = img_el.get("src", "") if img_el else ""
        img_alt = img_el.get("alt", title) if img_el else title
        
        meta_el = card.find("div", class_="case-study-meta")
        meta_text = clean_text(meta_el.get_text()) if meta_el else ""
        
        summary_el = card.find("div", class_="case-study-summary")
        summary_text = clean_text(summary_el.get_text()) if summary_el else ""
        if summary_text.lower().startswith("executive summary:"):
            summary_text = summary_text[len("executive summary:"):].strip()
        
        brief = ""
        approach_bullets = []
        shipped = ""
        
        sections = card.find_all("div", class_="case-study-section-title")
        for s in sections:
            s_title = clean_text(s.get_text()).lower()
            nxt = s.find_next_sibling()
            if not nxt:
                continue
            if "brief" in s_title:
                brief = clean_text(nxt.get_text())
            elif "approach" in s_title:
                if nxt.name == "ul":
                    approach_bullets = [clean_text(li.get_text()) for li in nxt.find_all("li")]
                else:
                    approach_bullets = [clean_text(nxt.get_text())]
            elif "shipped" in s_title:
                shipped = clean_text(nxt.get_text())
        
        tech_tags = [clean_text(t.get_text()) for t in card.find_all("span", class_="tech-tag")]
        
        lines.append(f"## {i}. {title}")
        if badge_str:
            lines.append(f"**Track**: {badge_str}  ")
        if meta_text:
            lines.append(f"**Details**: {meta_text}  ")
        if cta_url:
            lines.append(f"**Article Link**: [Read the piece →]({cta_url})")
        lines.append("")
        
        if img_src:
            lines.append(f"![{img_alt}]({img_src})")
            lines.append("")
        
        if summary_text:
            lines.append(f"### Executive Summary")
            lines.append(summary_text)
            lines.append("")
        
        if brief:
            lines.append(f"### The Brief")
            lines.append(brief)
            lines.append("")
            
        if approach_bullets:
            lines.append(f"### The Approach")
            for bullet in approach_bullets:
                lines.append(f"- {bullet}")
            lines.append("")
            
        if shipped:
            lines.append(f"### What Shipped")
            lines.append(shipped)
            lines.append("")
            
        if tech_tags:
            tag_str = ", ".join(f"`{t}`" for t in tech_tags)
            lines.append(f"**Technologies & Keywords**: {tag_str}")
            lines.append("")
            
        lines.append("---")
        lines.append("")
        
    return "\n".join(lines).strip() + "\n"

def convert_writing() -> str:
    html_file = os.path.join(REPO_DIR, "writing.html")
    with open(html_file, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    
    lines = [
        "# Thomas Cherickal — Publications & Long-Form Works",
        "## Generative AI Consultant",
        "",
        "> **// 500+ Publications Across 10+ Platforms Since 2020**  ",
        "> **Selected long-form technical deep dives across Generative AI, AI Agents, Quantum Computing, LLM Architecture, Rust Systems, and Post-Quantum Cryptography.**",
        "",
        "---",
        "",
        "## Featured Book: RECRUITED",
        "### RECRUITED — The Inbound Recruiter Blueprint: How to Make Recruiters Chase You",
        "*(Pre-Order Status — $20.00 USD pre-release until December 31, 2026 ($40.00 USD after release) · Free with an active Patreon subscription)*",
        "",
        "![RECRUITED Book Cover](assets/recruited-cover.jpg)",
        "",
        "A comprehensive transformation system showing professionals how to use frontier AI tools — GitHub, LinkedIn, Perplexity, Claude, Google Antigravity, and Gemini Notebook — to rebuild their professional presence so that inbound recruiter offers find them.",
        "",
        "- [🎗 Pre-Order on Patreon](https://patreon.com/thomascherickal)",
        "",
        "---",
        "",
        "## Featured Technical Deep Dives (48 Selected Pieces Across 12 Categories)",
        ""
    ]
    
    cats = soup.find_all("h3", class_="writing-category-title")
    for cat in cats:
        cat_title = clean_text(cat.get_text())
        lines.append(f"### {cat_title}")
        lines.append("")
        
        parent = cat.parent
        cards = parent.find_all("a", class_="article-card")
        for c in cards:
            href = c.get("href", "")
            title_el = c.find("div", class_="article-title")
            title = clean_text(title_el.get_text()) if title_el else "Article"
            
            pills = [clean_text(p.get_text()) for p in c.find_all("span", class_="meta-pill")]
            pill_str = " · ".join(pills) if pills else ""
            
            if pill_str:
                lines.append(f"- [{title}]({href}) — *{pill_str}*")
            else:
                lines.append(f"- [{title}]({href})")
        lines.append("")
        
    lines.append("---")
    lines.append("*Explore more publications at [thomascherickal.com](https://thomascherickal.com) and [HackerNoon](https://hackernoon.com/u/thomascherickal).*")
    return "\n".join(lines).strip() + "\n"

def convert_services() -> str:
    html_file = os.path.join(REPO_DIR, "services.html")
    with open(html_file, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    
    lines = [
        "# Thomas Cherickal — Generative AI Services & Offerings",
        "## AI Agent Mastery, Corporate AI Training & Autonomous Systems",
        "",
        "> **// GENERATIVE AI CONSULTANT & QUANTUM SYSTEMS EXPLORER**  ",
        "> **AI Agent Mastery: create teams of AI agents with Claude, OpenAI, Gemini, or Grok. Learn the in-depth principles that make AI agent orchestration successful, understand deep agent constraints, and perform the tasks of multiple people with an agent team — grounded in essential task domain knowledge. Comprehensive live and remote workforce training (Employees, Developers, Staff, CXOs), Local LLM cost optimization, agent orchestration, AI-assisted TDD, and early quantum systems exploration.**",
        "",
        "**Quick Actions**:",
        "- [✉️ Book a Strategy Consultation](contact.html)",
        "- [🌐 View Pricing & Country Parity Index](pricing.html)",
        "",
        "---",
        "",
        "## Frontier AI Tools & Stack Expertise",
        "",
        "Hands-on mastery, configuration, and developer pair-programming across industry-standard AI development environments and foundation APIs:",
        ""
    ]
    
    chips = [clean_text(c.get_text()) for c in soup.find_all("span", class_="role-chip")]
    seen = set()
    uniq_chips = []
    for c in chips:
        if c not in seen:
            seen.add(c)
            uniq_chips.append(c)
    
    lines.append(" · ".join(f"`{c}`" for c in uniq_chips))
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Core Capabilities & Advisory Offerings")
    lines.append("")
    
    roles = soup.find_all("div", class_="role-card")
    for i, r in enumerate(roles, 1):
        icon = clean_text(r.find("span", class_="role-card-icon").get_text()) if r.find("span", class_="role-card-icon") else "⚡"
        title_el = r.find("h3", class_="role-card-title")
        title = clean_text(title_el.get_text()) if title_el else f"Service #{i}"
        
        desc_el = r.find("p", class_="role-card-desc")
        desc = clean_text(desc_el.get_text()) if desc_el else ""
        
        lines.append(f"### {i}. {icon} {title}")
        if desc:
            lines.append(desc)
            lines.append("")
        
        ul = r.find("ul")
        if ul:
            for li in ul.find_all("li"):
                lines.append(f"- {clean_text(li.get_text())}")
            lines.append("")
            
        hl_div = r.find("div", style=lambda s: s and "border-top" in s)
        if hl_div:
            hl_text = clean_text(hl_div.get_text())
            if hl_text:
                lines.append(f"> **Focus**: {hl_text}")
                lines.append("")
                
        btn = r.find("a", class_="btn")
        if btn and btn.get("href"):
            btn_text = clean_text(btn.get_text())
            lines.append(f"**Action**: [{btn_text}]({btn.get('href')})")
            lines.append("")
        
        lines.append("---")
        lines.append("")
        
    lines.append("## Dedicated Enterprise AI Transformation Service Pages")
    lines.append("")
    lines.append("1. [🧠 Generative AI Transformation](service-generative-ai-transformation.html) — Agentic workflows, metrics, safeguards.")
    lines.append("2. [📦 Local LLMs & Cost Slashing](service-local-llms-cost-slashing.html) — 60–90% inference savings via Ollama, vLLM, llama.cpp.")
    lines.append("3. [🤖 AI Agent Orchestration Fundamentals](service-ai-agents-orchestration.html) — Scalable swarms & low budgets.")
    lines.append("4. [🏎️ Agentic AI Assistants (Hermes Agent)](service-agentic-ai-systems.html) — Persistent memory & autonomous loops.")
    lines.append("5. [🤗 Enterprise Coding Model Optimization](service-enterprise-coding-model-optimization.html) — Claude Code, Antigravity, Codex.")
    lines.append("6. [🌐 Training for WorkFlows Automations](service-training-for-workflows-automations.html) — Self-hosted n8n automations.")
    lines.append("7. [⚛️ Quantum Applications (Experimental)](service-quantum-applications.html) — Enterprise quantum readiness.")
    lines.append("8. [⚙️ Low Code and No Code Enterprise Automation Training](service-low-code-no-code-automation-training.html) — 50% daily work automated.")
    lines.append("9. [💲 AI Budget & Cost Limits Training](service-ai-budget-cost-limits-training.html) — Token governance & free models.")
    lines.append("10. [🛡️ Enterprise Workforce AI Training](service-enterprise-workforce-ai-training.html) — Freshers, Devs & CXOs.")
    lines.append("")
    
    return "\n".join(lines).strip() + "\n"

def convert_collaboration() -> str:
    html_file = os.path.join(REPO_DIR, "collaboration.html")
    with open(html_file, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
        
    lines = [
        "# Thomas Cherickal — Ways to Collaborate",
        "## Generative AI Consulting, Corporate Training & Dedicated Retainers",
        "",
        "> **// CLIENT COLLABORATION DIRECTORY**  ",
        "> **Structured, high-integrity models for AI Agent Mastery & team orchestration, remote & live workforce training, local LLM infrastructure, autonomous agent orchestration, and quantum exploration.**",
        "",
        "---",
        "",
        "## 9 Ways We Can Work Together",
        ""
    ]
    
    collab_cards = soup.find_all("div", class_="collab-card")
    for i, c in enumerate(collab_cards, 1):
        title_el = c.find(["h3", "h4"])
        title = clean_text(title_el.get_text()) if title_el else f"Option #{i}"
        
        desc_el = c.find("p", class_="collab-desc") or c.find("p")
        desc = clean_text(desc_el.get_text()) if desc_el else ""
        
        lines.append(f"### {title}")
        if desc:
            lines.append(desc)
            lines.append("")
            
        ul = c.find("ul")
        if ul:
            for li in ul.find_all("li"):
                lines.append(f"- {clean_text(li.get_text())}")
            lines.append("")
            
        tags = [clean_text(t.get_text()) for t in c.find_all("span", class_="tech-tag")]
        if tags:
            lines.append("**Keywords**: " + ", ".join(f"`{t}`" for t in tags))
            lines.append("")
            
        lines.append("---")
        lines.append("")
        
    lines.append("## How We Collaborate: 8-Step Lifecycle")
    lines.append("")
    step_cards = soup.find_all("div", class_="step-card")
    for i, sc in enumerate(step_cards, 1):
        num = clean_text(sc.find("div", class_="step-num").get_text()) if sc.find("div", class_="step-num") else f"0{i}"
        h3 = clean_text(sc.find("h3").get_text()) if sc.find("h3") else f"Step {i}"
        desc = clean_text(sc.find("p").get_text()) if sc.find("p") else ""
        lines.append(f"### {num}. {h3}")
        if desc:
            lines.append(desc)
        lines.append("")
        
    lines.append("---")
    lines.append("")
    lines.append("## Who I Collaborate With (Audience & Parity Tiers)")
    lines.append("")
    
    aud_cards = soup.find_all("div", class_="audience-card")
    for ac in aud_cards:
        h3 = clean_text(ac.find("h3").get_text()) if ac.find("h3") else "Audience"
        desc = clean_text(ac.find("p").get_text()) if ac.find("p") else ""
        lines.append(f"### {h3}")
        if desc:
            lines.append(desc)
        lines.append("")
        
    lines.append("---")
    lines.append("")
    lines.append("## Collaboration FAQs")
    lines.append("")
    faq_cards = soup.find_all("div", class_="faq-card")
    for i, fc in enumerate(faq_cards, 1):
        q = clean_text(fc.find("h3").get_text()) if fc.find("h3") else f"Question #{i}"
        ans = clean_text(fc.find("p").get_text()) if fc.find("p") else ""
        lines.append(f"### {i}. {q}")
        if ans:
            lines.append(ans)
        lines.append("")
        
    return "\n".join(lines).strip() + "\n"

def convert_pricing() -> str:
    html_file = os.path.join(REPO_DIR, "pricing.html")
    with open(html_file, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
        
    lines = [
        "# Thomas Cherickal — Generative AI Packages & Transparent Pricing",
        "## Transparent Retainer Packages, Enterprise Migrations & Global Parity Index",
        "",
        "> **// TRANSPARENT MILESTONE RATES & EQUAL-WEIGHT DELIVERABLES**  ",
        "> **Clear milestone pricing, runtime-verified code, infinite revisions until satisfied, and global equity with an interactive 198-country Purchasing Power Parity (PPP) rate calculator.**",
        "",
        "---",
        "",
        "## Generative AI Engagement Packages",
        ""
    ]
    
    packages = soup.find_all("div", class_="package-card")
    for i, pkg in enumerate(packages, 1):
        title_el = pkg.find(["h2", "h3"])
        title = clean_text(title_el.get_text()) if title_el else f"Package #{i}"
        
        # Exact price
        price_val = ""
        for div in pkg.find_all("div"):
            txt = clean_text(div.get_text())
            if ("$" in txt or "% OFF" in txt or "OFF" in txt) and len(txt) < 30 and ("–" in txt or "-" in txt or "/mo" in txt or "OFF" in txt):
                price_val = txt
                break
        if not price_val:
            price_val = "Custom Quote"
            
        badge = clean_text(pkg.find("div", style=lambda s: s and "border-radius:9999px" in s).get_text()) if pkg.find("div", style=lambda s: s and "border-radius:9999px" in s) else ""
        
        lines.append(f"### {i}. {title}")
        if badge:
            lines.append(f"*{badge}*  ")
        if "$" in price_val:
            lines.append(f"**Investment**: `{price_val} USD`")
        else:
            lines.append(f"**Investment**: `{price_val}`")
        lines.append("")
        
        ul = pkg.find("ul")
        if ul:
            lines.append("**Deliverables & Scope**:")
            for li in ul.find_all("li"):
                lines.append(f"- {clean_text(li.get_text())}")
            lines.append("")
            
        btn = pkg.find("a", class_="btn")
        if btn and btn.get("href"):
            lines.append(f"**Commission**: [{clean_text(btn.get_text())}]({btn.get('href')})")
            lines.append("")
            
        lines.append("---")
        lines.append("")
        
    lines.append("## Foundational Guarantees on Every Engagement")
    lines.append("")
    lines.append("1. **Sandbox-Verified Demonstrations**: Every sample code guide, prompt pipeline, and agent orchestration demonstration is tested and verified in isolated sandboxes.")
    lines.append("2. **Infinite Revisions**: Continuous iteration until your technical leads and management are 100% satisfied.")
    lines.append("3. **Equal-Weight Milestones**: Work is structured into transparent, equal milestone phases with clear deliverables.")
    lines.append("4. **Complete Training & Guide IP Transfer**: Full commercial ownership of all training curriculum assets, architecture roadmaps, sample code guides, and demonstration sandboxes transfers upon final sign-off.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Individual Services & Base Rates (A La Carte)")
    lines.append("")
    
    rate_cards = soup.find_all("div", class_="price-card-faq")
    for rc in rate_cards:
        t_el = rc.find("div", class_="price-card-title")
        t = clean_text(t_el.get_text()) if t_el else "Service"
        p_el = rc.find("div", class_="price-amount")
        p = clean_text(p_el.get_text()) if p_el else ""
        desc_el = rc.find("div", style=lambda s: s and "color:var(--text-secondary)" in s)
        desc = clean_text(desc_el.get_text()) if desc_el else ""
        lines.append(f"### {t}")
        if p:
            lines.append(f"**Base Rate**: `{p}`  ")
        if desc:
            lines.append(desc)
        lines.append("")
        
    lines.append("---")
    lines.append("")
    lines.append("## Global Equity: Purchasing Power Parity (PPP) Tiers")
    lines.append("Rates are automatically indexed across 198 countries to ensure fair pricing worldwide across 6 Parity Tiers:")
    lines.append("- **Tier 1 (Base, 100%)**: US, UK, EU, Canada, Australia, Singapore, etc.")
    lines.append("- **Tier 2 (75%)**: Japan, South Korea, UAE, Israel, New Zealand, etc.")
    lines.append("- **Tier 3 (50%)**: Eastern Europe, Latin America, South Africa, Malaysia, etc.")
    lines.append("- **Tier 4 (50% Off / Startups)**: Pre-Seed, Seed-stage, and bootstrapped startups globally.")
    lines.append("- **Tier 5 (75% Off / Non-Profits & Academia)**: Non-profits, educational institutions, universities, and academia.")
    lines.append("- **Tier 6 (Custom Quote / Backward Nations)**: Customized subsidized quotes for underprivileged and backward nations.")
    lines.append("")
    lines.append("- [Check your country on the interactive calculator](pricing.html)")
    lines.append("- [Start intake conversation](contact.html)")
    
    return "\n".join(lines).strip() + "\n"

def convert_expertise() -> str:
    html_file = os.path.join(REPO_DIR, "expertise.html")
    with open(html_file, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
        
    lines = [
        "# Thomas Cherickal — Generative AI Architecture, AI Agent Mastery & Tech Stack",
        "## Generative AI Consultant",
        "",
        "> **// TECHNICAL ROLES & CAPABILITIES DIRECTORY**  ",
        "> **10 specialized capability areas, 10 curated tech-stack chip groups, and the Python, Rust, and quantum execution tooling behind every enterprise delivery.**",
        "",
        "---",
        "",
        "## Specialized Technical Roles & Offerings",
        ""
    ]
    
    roles = soup.find_all("div", class_="role-card")
    for i, r in enumerate(roles, 1):
        icon_el = r.find("span", class_="role-card-icon")
        icon = clean_text(icon_el.get_text()) if icon_el else "⚡"
        
        title_el = r.find(["h3", "h4"])
        title = clean_text(title_el.get_text()) if title_el else f"Role #{i}"
        
        desc_el = r.find("p", class_="role-card-desc") or r.find("p")
        desc = clean_text(desc_el.get_text()) if desc_el else ""
        
        lines.append(f"### {i}. {icon} {title}")
        if desc:
            lines.append(desc)
            lines.append("")
            
        ul = r.find("ul")
        if ul:
            for li in ul.find_all("li"):
                lines.append(f"- {clean_text(li.get_text())}")
            lines.append("")
            
        tags = [clean_text(t.get_text()) for t in r.find_all("span", class_="tech-tag")]
        if tags:
            lines.append("**Keywords**: " + ", ".join(f"`{t}`" for t in tags))
            lines.append("")
            
        lines.append("---")
        lines.append("")
        
    lines.append("## Tech-Stack Expertise (10 Curated Chip Groups)")
    lines.append("")
    
    tech_cats = soup.find_all("div", class_="tech-category")
    for tc in tech_cats:
        tc_title_el = tc.find(["h3", "h4"])
        tc_title = clean_text(tc_title_el.get_text()) if tc_title_el else "Tech Stack Category"
        lines.append(f"### {tc_title}")
        
        tags = [clean_text(t.get_text()) for t in tc.find_all("span", class_="tech-tag")]
        if tags:
            lines.append(" · ".join(f"`{t}`" for t in tags))
        lines.append("")
        
    lines.append("---")
    lines.append("")
    lines.append("## Workflow & Methodology: Research → Build → Run → Verify → Explain")
    lines.append("**AI accelerates the workflow. Human verification owns the result.**")
    lines.append("1. **Research & Source Discovery**: Primary arXiv papers, documentation archives, hardware specifications.")
    lines.append("2. **Structural Drafting**: Pedagogical structure, outline stress-testing, modular architecture.")
    lines.append("3. **Sample Code Guides & Demonstrations**: Executable sample projects, technical explainers, and guide artifacts in Python, Rust, and Qiskit.")
    lines.append("4. **Sandbox Verification & Demonstrations**: Live sandbox execution, test suites (`pytest`, `cargo test`), and quantum simulators verifying demonstrator integrity.")
    lines.append("5. **Human Technical Judgment**: Single-point intellectual accountability, domain precision, authoritative code review.")
    
    return "\n".join(lines).strip() + "\n"

def convert_contact() -> str:
    return """# Thomas Cherickal — Start an AI Transformation
## Generative AI Consultant

> **// PROJECT INTAKE & DIRECT COMMUNICATION**  
> **Commission AI Agent Mastery, live/remote organizational training, local LLM cost optimization, agent orchestration, or dedicated monthly retainers directly.**

---

## Direct Communication Channels

- **Primary Email**: [thomascherickal@gmail.com](mailto:thomascherickal@gmail.com)
- **LinkedIn Consultation**: [linkedin.com/in/thomascherickal](https://linkedin.com/in/thomascherickal)
- **Book a 1:1 Video Consult**: [topmate.io/thomascherickal](https://topmate.io/thomascherickal)
- **GitHub Profile & Repos**: [github.com/thomascherickal](https://github.com/thomascherickal)
- **Newsletter (Kit)**: [thomascherickal.kit.com](https://thomascherickal.kit.com)
- **Patreon (RECRUITED Pre-Order)**: [patreon.com/thomascherickal](https://patreon.com/thomascherickal)

---

## Engagement Standards

- **Response Time**: All verified enterprise queries receive a detailed response within 24 business hours.
- **Non-Disclosure Agreements (NDAs)**: Mutual NDAs executed prior to reviewing proprietary codebase or confidential data.
- **Initial Audit**: Every engagement begins with an objective scoping call to identify ROI bottlenecks.
- **Purchasing Power Parity**: 6 PPP tiers apply across 198 countries to ensure accessible global rates.

---

## What Happens Next?

1. **Intake & Scope Audit**: Review your current tech stack, pain points, inference bills, or workforce training needs.
2. **Milestone Proposal**: Transparent quote with equal-weight milestones, delivery dates, and acceptance criteria.
3. **Execution & Daily Syncs**: Live iterative training development with continuous guide verification and sandbox demos.
4. **Knowledge & Assets Handover**: Comprehensive training modules, sample code guides, verified sandbox demos, and 100% complete IP transfer.

---
*Ready to discuss? Email directly at [thomascherickal@gmail.com](mailto:thomascherickal@gmail.com).*
"""

def convert_faqs() -> str:
    html_file = os.path.join(REPO_DIR, "faqs.html")
    with open(html_file, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
        
    lines = [
        "# Thomas Cherickal — Frequently Asked Questions",
        "## Generative AI Consulting, Training, Pricing & Engagement Policies",
        "",
        "> **// CLIENT MANDATES & POLICIES**  ",
        "> **Turnaround timelines, code verification standards, AI workflows, Purchasing Power Parity (PPP), revision policies, and collaboration details.**",
        "",
        "---",
        "",
        "## Frequently Asked Questions",
        ""
    ]
    
    faq_cards = soup.find_all("div", class_="faq-card")
    for i, c in enumerate(faq_cards, 1):
        num_el = c.find("div", class_="faq-num")
        num = clean_text(num_el.get_text()) if num_el else f"FAQ #{i:02d}"
        
        badge_el = c.find("span", class_="faq-category-badge")
        badge = clean_text(badge_el.get_text()) if badge_el else ""
        
        q_el = c.find(["h2", "h3"])
        q = clean_text(q_el.get_text()) if q_el else f"Question #{i}"
        
        ans_el = c.find("p", class_="faq-answer") or c.find("p")
        ans = clean_text(ans_el.get_text()) if ans_el else ""
        
        lines.append(f"### {i}. {q}")
        if badge or num:
            meta_parts = [p for p in [badge, num] if p]
            lines.append(f"*{' · '.join(meta_parts)}*  ")
        lines.append(ans)
        lines.append("")
        lines.append("---")
        lines.append("")
        
    return "\n".join(lines).strip() + "\n"

def convert_404() -> str:
    return """# 404 — Quantum State Collapse
## Page Not Found

> **// 404 — QUANTUM STATE COLLAPSE**  
> **The requested URL was not found on this server. The measurement collapsed the wave function into an undefined state.**

---

### Navigation Options
- [Return to Homepage →](index.html)
- [Read Portfolio & Case Studies →](portfolio.html)
- [Explore Publications & Deep Dives →](writing.html)
- [View Services & Capabilities →](services.html)
- [Contact via Email →](contact.html)

---
*© 2026 Thomas Cherickal · The Digital Futurist*
"""

def convert_service_page(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
        
    h1_el = soup.find("h1")
    h1 = clean_text(h1_el.get_text()) if h1_el else os.path.basename(file_path).replace(".html", "").replace("-", " ").title()
    
    eyebrow_el = soup.find("p", class_="service-hero-badge") or soup.find("p", class_="section-label")
    eyebrow = clean_text(eyebrow_el.get_text()) if eyebrow_el else "// ENTERPRISE AI TRANSFORMATION SERVICE"
    
    sub_el = soup.find("p", class_="service-hero-sub") or soup.find("p", class_="section-sub")
    sub = clean_text(sub_el.get_text()) if sub_el else ""
    
    lines = [
        f"# {h1}",
        f"## Enterprise AI Transformation Service — Thomas Cherickal",
        "",
        f"> **{eyebrow}**  ",
        f"> **{sub}**" if sub else "> **Enterprise transformation, verification, and implementation.**",
        "",
        "---",
        ""
    ]
    
    main = soup.find("main") or soup
    for sec in main.find_all(["section", "div"]):
        h2 = sec.find("h2")
        if not h2:
            continue
        h2_text = clean_text(h2.get_text())
        if not any(k in h2_text for k in ["Overview", "Architecture", "Deliverable", "Guarantee", "Investment"]):
            continue
            
        lines.append(f"## {h2_text}")
        lines.append("")
        
        for p in sec.find_all("p", recursive=False):
            p_text = clean_text(p.get_text())
            if p_text:
                lines.append(p_text)
                lines.append("")
                
        cards = sec.find_all("div", class_=lambda c: c and any(k in c for k in ["card", "box", "step", "spec", "grid"]))
        for c in cards:
            h3 = c.find(["h3", "h4"])
            if h3:
                lines.append(f"### {clean_text(h3.get_text())}")
                c_desc = clean_text(c.find("p").get_text()) if c.find("p") else ""
                if c_desc:
                    lines.append(c_desc)
                lines.append("")
                
        ul = sec.find("ul")
        if ul:
            for li in ul.find_all("li"):
                lines.append(f"- {clean_text(li.get_text())}")
            lines.append("")
            
        chips = [clean_text(chip.get_text()) for chip in sec.find_all("span", class_=lambda cl: cl and ("chip" in cl or "tag" in cl))]
        if chips:
            lines.append("**Tech Stack & Tools**: " + " · ".join(f"`{c}`" for c in chips))
            lines.append("")
            
        lines.append("---")
        lines.append("")
        
    lines.append("### Ready to Deploy?")
    lines.append("- [Commission this Service via Email →](contact.html)")
    lines.append("- [Book Strategy Session on Topmate ↗](https://topmate.io/thomascherickal)")
    lines.append("- [Return to All Services →](services.html)")
    lines.append("")
    
    return "\n".join(lines).strip() + "\n"

def main():
    print(f"Generating Markdown files in: {SYNC_DIR}")
    
    converters = {
        "portfolio.html": ("portfolio.md", convert_portfolio),
        "writing.html": ("writing.md", convert_writing),
        "services.html": ("services.md", convert_services),
        "collaboration.html": ("collaboration.md", convert_collaboration),
        "pricing.html": ("pricing.md", convert_pricing),
        "expertise.html": ("expertise.md", convert_expertise),
        "contact.html": ("contact.md", convert_contact),
        "faqs.html": ("faqs.md", convert_faqs),
        "404.html": ("404.md", convert_404)
    }
    
    for html_file, (md_file, converter) in converters.items():
        dest = os.path.join(SYNC_DIR, md_file)
        print(f"Building {md_file} from {html_file}...")
        content = converter()
        with open(dest, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  -> Wrote {len(content.splitlines())} lines to {md_file}")
        
    service_files = sorted(glob.glob(os.path.join(REPO_DIR, "service-*.html")))
    for s_path in service_files:
        base = os.path.basename(s_path)
        md_name = base.replace(".html", ".md")
        dest = os.path.join(SYNC_DIR, md_name)
        print(f"Building {md_name} from {base}...")
        content = convert_service_page(s_path)
        with open(dest, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  -> Wrote {len(content.splitlines())} lines to {md_name}")
        
    print("\nInitial Markdown generation completed successfully!")

if __name__ == "__main__":
    main()
