#!/usr/bin/env python3
"""
build_md_files.py - Extract structured, high-fidelity Markdown from all HTML pages
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

def convert_articles() -> str:
    html_file = os.path.join(REPO_DIR, "articles.html")
    with open(html_file, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    
    lines = [
        "# Thomas Cherickal - Articles & Case Studies",
        "## Generative AI Consultant",
        "",
        "> **// Verified Case Studies**  ",
        "> **The brief, the approach, and what shipped - 14 verified, human-directed case studies across Generative AI (6), Agentic AI (4), and Quantum Computing (4), backed by runtime-verified code, deep analysis, and equally deep insights.**",
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
        "# Thomas Cherickal - Publications & Long-Form Works",
        "## Generative AI Consultant",
        "",
        "> **// 500+ Publications Across 10+ Platforms Since 2020**  ",
        "> **Selected long-form technical deep dives across Generative AI, AI Agents, Quantum Computing, LLM Architecture, Rust Systems, and Post-Quantum Cryptography.**",
        "",
        "---",
        "",
        "## Featured Book: RECRUITED",
        "### RECRUITED - The Inbound Recruiter Blueprint: How to Make Recruiters Chase You",
        "*(Pre-Order Status - $20.00 USD pre-release until December 31, 2026 ($40.00 USD after release) · Free with an active Patreon subscription)*",
        "",
        "![RECRUITED Book Cover](assets/Book%20Cover%20LQ.jpeg)",
        "",
        "A comprehensive transformation system showing professionals how to use frontier AI tools - GitHub, LinkedIn, Perplexity, Claude, Google Antigravity, and Gemini Notebook - to rebuild their professional presence so that inbound recruiter offers find them.",
        "",
        "- [Pre-Order on Patreon](https://patreon.com/thomascherickal)",
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
                lines.append(f"- [{title}]({href}) - *{pill_str}*")
            else:
                lines.append(f"- [{title}]({href})")
        lines.append("")
        
    lines.append("---")
    lines.append("*Explore more publications at [thomascherickal.com](https://thomascherickal.com) and [HackerNoon](https://hackernoon.com/u/thomascherickal).*")
    return "\n".join(lines).strip() + "\n"

def convert_collaboration() -> str:
    html_file = os.path.join(REPO_DIR, "collaboration.html")
    with open(html_file, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
        
    lines = [
        "# Thomas Cherickal - Ways to Collaborate",
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
        "# Thomas Cherickal - Generative AI Packages & Transparent Pricing",
        "## Transparent Retainer Packages, Enterprise Migrations & Global Parity Index",
        "",
        "> **// TRANSPARENT MILESTONE RATES & EQUAL-WEIGHT DELIVERABLES**  ",
        "> **Clear milestone pricing, runtime-verified code, infinite revisions until satisfied, and global equity with an interactive 195-country Purchasing Power Parity (PPP) rate calculator.**",
        "",
        "---",
        "",
        "## Enterprise Services",
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
            if ("$" in txt or "% OFF" in txt or "OFF" in txt) and len(txt) < 30 and ("–" in txt or "-" in txt or "/mo" in txt or "OFF" in txt or "USD" in txt or txt.strip().startswith("$")):
                price_val = txt
                break
        if not price_val:
            price_val = "Custom Quote"
            
        badge = clean_text(pkg.find("div", style=lambda s: s and "border-radius:9999px" in s).get_text()) if pkg.find("div", style=lambda s: s and "border-radius:9999px" in s) else ""
        
        lines.append(f"### {i}. {title}")
        if badge:
            lines.append(f"*{badge}*  ")
        if "$" in price_val:
            inv = price_val if "USD" in price_val else f"{price_val} USD"
            lines.append(f"**Investment**: `{inv}`")
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
    lines.append("## Global Equity: Purchasing Power Parity (PPP) Six Tiers")
    lines.append("Rates are automatically indexed across 195 countries to ensure fair pricing worldwide across 6 Parity Tiers:")
    lines.append("")
    parity_cards = soup.find_all("div", class_="parity-tier-card")
    if parity_cards:
        for pc in parity_cards:
            p_badge = clean_text(pc.find(class_="parity-badge").get_text()) if pc.find(class_="parity-badge") else ""
            p_title = clean_text(pc.find(class_="parity-title").get_text()) if pc.find(class_="parity-title") else ""
            p_sub = clean_text(pc.find(class_="parity-subtitle").get_text()) if pc.find(class_="parity-subtitle") else ""
            p_rate = clean_text(pc.find(class_="parity-rate-val").get_text()) if pc.find(class_="parity-rate-val") else ""
            p_ratesub = clean_text(pc.find(class_="parity-rate-sub").get_text()) if pc.find(class_="parity-rate-sub") else ""
            p_desc = clean_text(pc.find(class_="parity-desc").get_text()) if pc.find(class_="parity-desc") else ""
            
            lines.append(f"### {p_title}")
            if p_badge:
                lines.append(f"*{p_badge}* - {p_sub}  ")
            lines.append(f"**Rate**: `{p_rate}` ({p_ratesub})  ")
            if p_desc:
                lines.append(p_desc)
            lines.append("")
    else:
        lines.append("- **Tier 1 (Base, 100%)**: US, UK, EU, Canada, Australia, Singapore, etc.")
        lines.append("- **Tier 2 (75%)**: India, Brazil, Mexico, South Africa, Malaysia, Poland, Turkey, etc.")
        lines.append("- **Tier 3 (50%)**: Bangladesh, Kenya, Nigeria, Pakistan, Vietnam, Egypt, etc.")
        lines.append("- **Tier 4 (50% Off / Startups)**: Pre-Seed, Seed-stage, and bootstrapped startups globally (<$1M raised).")
        lines.append("- **Tier 5 (75% Off / Non-Profits & Academia)**: Non-profits, educational institutions, universities, and academia.")
        lines.append("- **Tier 6 (Custom Quote / Underprivileged)**: Customized subsidized quotes and micro-rates.")
        lines.append("")
    lines.append("- [Check your country on the interactive calculator](pricing.html)")
    lines.append("- [Start intake conversation](contact.html)")
    
    return "\n".join(lines).strip() + "\n"

def convert_expertise() -> str:
    html_file = os.path.join(REPO_DIR, "expertise.html")
    with open(html_file, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
        
    lines = [
        "# Thomas Cherickal - Generative AI Architecture, AI Agent Mastery & Tech Stack",
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
        title_el = r.find(["h3", "h4"])
        title = clean_text(title_el.get_text()) if title_el else f"Role #{i}"
        
        desc_el = r.find("p", class_="role-card-desc") or r.find("p")
        desc = clean_text(desc_el.get_text()) if desc_el else ""
        
        lines.append(f"### {i}. {title}")
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
    return """# Thomas Cherickal - Start an AI Transformation
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
- **Purchasing Power Parity**: 6 PPP tiers apply across 195 countries to ensure accessible global rates.

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
        "# Thomas Cherickal - Frequently Asked Questions",
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
    
    faq_cards = soup.find_all(["div", "article"], class_="faq-card")
    for i, c in enumerate(faq_cards, 1):
        num_el = c.find("div", class_="faq-num") or c.find("span", class_="faq-num-pill")
        num = clean_text(num_el.get_text()) if num_el else f"FAQ #{i:02d}"
        
        badge_el = c.find("span", class_="faq-category-badge")
        badge = clean_text(badge_el.get_text()) if badge_el else ""
        
        q_el = c.find(["h2", "h3"])
        q = clean_text(q_el.get_text()) if q_el else f"Question #{i}"
        
        ans_el = c.find(class_="faq-answer") or c.find("p")
        if ans_el:
            # Format paragraphs and lists nicely
            ans_parts = []
            for child in ans_el.children:
                if child.name in ["p", "div"]:
                    t = clean_text(child.get_text())
                    if t:
                        ans_parts.append(t)
                elif child.name == "ul":
                    for li in child.find_all("li"):
                        t = clean_text(li.get_text())
                        if t:
                            ans_parts.append(f"- {t}")
            if not ans_parts:
                ans = clean_text(ans_el.get_text())
            else:
                ans = "\n\n".join(ans_parts)
        else:
            ans = ""
        
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
    return """# 404 - Context Window Exhausted
## Page Not Found

> **// 404 - CONTEXT WINDOW EXHAUSTED**  
> **High-Temperature Hallucination Detected**  
> **Even with a 2-million-token context window, our autonomous multi-agent swarm couldn’t ground this route. Either your prompt drifted into unmapped latent space at temperature 2.0, our RAG retrieval scored zero cosine similarity, or an overconfident LLM hallucinated a URL that was never trained into the weights.**
> 
> *Don’t waste your token budget wandering the void. Re-ground your attention weights below:*

---

### Navigation Options
- [Return to Home →](index.html)
- [Transparent Pricing & Parity Tiers →](pricing.html)
- [Explore Publications & Deep Dives →](writing.html)
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
        f"## Enterprise AI Transformation Service - Thomas Cherickal",
        "",
        f"> **{eyebrow}**  ",
        f"> **{sub}**" if sub else "> **Enterprise transformation, verification, and implementation.**",
        "",
        "---",
        ""
    ]
    
    main = soup.find("main") or soup
    h2_elements = main.find_all("h2")
    for h2 in h2_elements:
        h2_text = clean_text(h2.get_text())
        if not any(k in h2_text for k in ["Overview", "Architecture", "Deliverable", "Guarantee", "Investment", "Summary"]):
            continue
            
        lines.append(f"## {h2_text}")
        lines.append("")
        
        curr = h2.next_sibling
        while curr:
            if curr.name == "h2":
                break
            if curr.name == "p":
                p_text = clean_text(curr.get_text())
                if p_text:
                    lines.append(p_text)
                    lines.append("")
            elif curr.name in ["ul", "ol"]:
                for li in curr.find_all("li", recursive=False):
                    nested_ul = li.find("ul")
                    nested_ol = li.find("ol")
                    if nested_ul:
                        nested_ul.extract()
                    if nested_ol:
                        nested_ol.extract()
                        
                    li_text = clean_text(li.get_text())
                    if li_text:
                        lines.append(f"- {li_text}")
                    
                    if nested_ul:
                        for n_li in nested_ul.find_all("li"):
                            n_text = clean_text(n_li.get_text())
                            if n_text:
                                lines.append(f" - {n_text}")
                    if nested_ol:
                        for idx, n_li in enumerate(nested_ol.find_all("li"), 1):
                            n_text = clean_text(n_li.get_text())
                            if n_text:
                                lines.append(f"  {idx}. {n_text}")
                lines.append("")
            elif isinstance(curr, Tag):
                table = curr if curr.name == "table" else curr.find("table")
                if table and "data-table" in table.get("class", []):
                    md_table = html_table_to_md(table)
                    if md_table:
                        lines.append(md_table)
                        lines.append("")
            curr = curr.next_sibling
            
        lines.append("---")
        lines.append("")
    lines.append("### Ready to Deploy?")
    lines.append("- [Commission this Service via Email →](contact.html)")
    lines.append("- [Book Strategy Session on Topmate ↗](https://topmate.io/thomascherickal)")
    lines.append("")
    
    return "\n".join(lines).strip() + "\n"

def html_table_to_md(table) -> str:
    rows = table.find_all("tr")
    if not rows:
        return ""
    md_rows = []
    headers = []
    th_cells = rows[0].find_all(["th", "td"])
    for th in th_cells:
        headers.append(clean_text(th.get_text()))
    if headers:
        md_rows.append("| " + " | ".join(headers) + " |")
        md_rows.append("| " + " | ".join([":---"] * len(headers)) + " |")
    
    start_idx = 1 if headers and rows[0].find("th") else 0
    for r in rows[start_idx:]:
        cells = r.find_all(["td", "th"])
        if not cells:
            continue
        row_vals = [clean_text(c.get_text()) for c in cells]
        md_rows.append("| " + " | ".join(row_vals) + " |")
    return "\n".join(md_rows)

def convert_agents_all_providers(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    h1_el = soup.find("h1")
    h1 = clean_text(h1_el.get_text()) if h1_el else "All Agentic Providers"
    badge_el = soup.find("p", class_="hero-badge")
    badge = clean_text(badge_el.get_text()) if badge_el else "// ORCHESTRATION ARCHITECTURE PLAYBOOK"
    
    hero_p = soup.find("header", class_="hero")
    sub = ""
    if hero_p:
        for p in hero_p.find_all("p"):
            if "hero-badge" not in p.get("class", []):
                sub = clean_text(p.get_text())
                break

    lines = [
        f"# {h1}",
        f"## Enterprise Agentic AI Playbook - Thomas Cherickal",
        "",
        f"> **{badge}**  ",
        f"> **{sub}**" if sub else "> **Architectural comparison across 8 frontier agent platforms.**",
        "",
        "---",
        "",
        "## Quick Provider Navigation",
        ""
    ]

    nav_cards = soup.find_all("a", class_="provider-nav-card")
    for nc in nav_cards:
        href = nc.get("href", "#")
        name = clean_text(nc.find("span", class_="nav-card-name").get_text()) if nc.find("span", class_="nav-card-name") else "Provider"
        price = clean_text(nc.find("span", class_="nav-card-price").get_text()) if nc.find("span", class_="nav-card-price") else ""
        lines.append(f"- [{name}]({href}) - *{price}*")
    lines.append("")
    lines.append("---")
    lines.append("")

    matrix_table = soup.find("table", class_="provider-matrix-table")
    if matrix_table:
        lines.append("## Master Provider Comparison Matrix")
        lines.append("")
        lines.append(html_table_to_md(matrix_table))
        lines.append("")
        lines.append("---")
        lines.append("")

    arch_cards = soup.find_all("article", class_="arch-card")
    if arch_cards:
        lines.append("## Architectural Paradigms & System Design")
        lines.append("")
        for ac in arch_cards:
            h3 = ac.find("h3")
            if h3:
                lines.append(f"### {clean_text(h3.get_text())}")
                lines.append("")
            for p in ac.find_all("p"):
                lines.append(clean_text(p.get_text()))
                lines.append("")
            ul = ac.find("ul")
            if ul:
                for li in ul.find_all("li"):
                    lines.append(f"- {clean_text(li.get_text())}")
                lines.append("")
        lines.append("---")
        lines.append("")

    deep_cards = soup.find_all("article", class_="provider-deep-card")
    if deep_cards:
        lines.append("## Detailed Provider Architectural Deep Dives")
        lines.append("")
        for dc in deep_cards:
            h3 = dc.find("h3", class_="deep-card-title")
            if h3:
                lines.append(f"### {clean_text(h3.get_text())}")
                lines.append("")
            pill = dc.find("span", class_="pricing-pill")
            if pill:
                lines.append(f"**Minimum Access**: `{clean_text(pill.get_text())}`  ")
            
            for p in dc.find_all("p"):
                lines.append(clean_text(p.get_text()))
                lines.append("")
            
            meta_items = dc.find_all("li", class_="deep-meta-item")
            if meta_items:
                lines.append("**Key Metadata & Links**:")
                for mi in meta_items:
                    lines.append(f"- {clean_text(mi.get_text())}")
                lines.append("")
            lines.append("---")
            lines.append("")

    code_win = soup.find("div", class_="code-window")
    if code_win:
        bar = code_win.find("div", class_="code-bar")
        filename = clean_text(bar.find("span").get_text()) if bar and bar.find("span") else "multi_agent_router.py"
        code_tag = code_win.find("code")
        if code_tag:
            lines.append("## Production Multi-Agent Orchestration Script")
            lines.append(f"#### `{filename}`")
            lines.append("```python")
            lines.append(code_tag.get_text().strip())
            lines.append("```")
            lines.append("")
            lines.append("---")
            lines.append("")

    roadmap_cards = soup.find_all("div", class_="roadmap-card")
    if roadmap_cards:
        lines.append("## Enterprise Implementation Roadmap")
        lines.append("")
        for rc in roadmap_cards:
            step = clean_text(rc.find("div", class_="roadmap-step").get_text()) if rc.find("div", class_="roadmap-step") else ""
            h4 = clean_text(rc.find("h4").get_text()) if rc.find("h4") else ""
            lines.append(f"### Step {step}: {h4}")
            for p in rc.find_all("p"):
                lines.append(clean_text(p.get_text()))
            lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("### Ready to Deploy Autonomous AI Agent Teams?")
    lines.append("- [Schedule Strategy Consultation via Email →](collaboration.html)")
    lines.append("- [View Advisory Packages & Rates](pricing.html)")
    lines.append("- [Explore Contact Page](contact.html)")
    lines.append("")

    return "\n".join(lines).strip() + "\n"

def convert_agent_deepdive(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
        
    h1_el = soup.find("h1")
    h1 = clean_text(h1_el.get_text()) if h1_el else "Claude AI Agent Orchestration"
    
    badge_el = soup.find("p", class_="hero-badge")
    badge = clean_text(badge_el.get_text()) if badge_el else "// PROVIDER DEEP DIVE"
    
    hero_p = soup.find("header", class_="hero")
    sub = ""
    if hero_p:
        for p in hero_p.find_all("p"):
            if "hero-badge" not in p.get("class", []):
                sub = clean_text(p.get_text())
                break
                
    lines = [
        f"# {h1}",
        f"## AI Agent Architecture & Orchestration - Thomas Cherickal",
        "",
        f"> **{badge}**  ",
        f"> **{sub}**" if sub else "> **Production architecture, protocols, and implementation.**",
        "",
        "---",
        ""
    ]
    
    pricing = soup.find("div", class_="pricing-hero-card")
    if pricing:
        val_el = pricing.find("span", class_="pricing-val")
        val = clean_text(val_el.get_text()) if val_el else ""
        desc_spans = pricing.find_all("span")
        notes = []
        for s in desc_spans:
            t = clean_text(s.get_text())
            if t and t != val and "Minimum" not in t:
                notes.append(t)
        lines.append("## Minimum Subscription & Access Pricing")
        lines.append(f"**Starting Price**: `{val}`  ")
        if notes:
            lines.append(f"*{' · '.join(notes)}*")
        lines.append("")
        lines.append("---")
        lines.append("")

    main = soup.find("main") or soup
    cards = main.find_all(["article", "div"], class_="content-card")
    for card in cards:
        h2 = card.find("h2")
        if not h2:
            continue
        h2_text = clean_text(h2.get_text())
        lines.append(f"## {h2_text}")
        lines.append("")
        
        for elem in card.children:
            if not getattr(elem, 'name', None):
                continue
            if elem.name == "h2":
                continue
            elif elem.name == "h3":
                lines.append(f"### {clean_text(elem.get_text())}")
                lines.append("")
            elif elem.name == "p":
                p_text = clean_text(elem.get_text())
                if p_text:
                    lines.append(p_text)
                    lines.append("")
            elif elem.name in ["ul", "ol"]:
                is_num = (elem.name == "ol")
                for i, li in enumerate(elem.find_all("li", recursive=False), 1):
                    prefix = f"{i}. " if is_num else "- "
                    lines.append(f"{prefix}{clean_text(li.get_text())}")
                lines.append("")
            elif elem.name == "div" and "code-window" in elem.get("class", []):
                bar = elem.find("div", class_="code-bar")
                filename = clean_text(bar.find("span").get_text()) if bar and bar.find("span") else "script.py"
                code_tag = elem.find("code")
                code_text = code_tag.get_text() if code_tag else ""
                lines.append(f"#### `{filename}`")
                lines.append("```python")
                lines.append(code_text.strip())
                lines.append("```")
                lines.append("")
            elif elem.name == "div" and any(c in elem.get("class", []) for c in ["table-container", "pricing-table-wrapper", "table-wrap"]):
                tbl = elem.find("table")
                if tbl:
                    lines.append(html_table_to_md(tbl))
                    lines.append("")
            elif elem.name == "div" and "step-grid" in elem.get("class", []):
                for card_el in elem.find_all("div", class_="step-card"):
                    num_el = card_el.find("div", class_="step-num")
                    title_el = card_el.find("div", class_="step-title")
                    desc_el = card_el.find("div", class_="step-desc")
                    s_num = clean_text(num_el.get_text()) if num_el else ""
                    s_title = clean_text(title_el.get_text()) if title_el else ""
                    s_desc = clean_text(desc_el.get_text()) if desc_el else ""
                    heading_parts = [p for p in [s_num, s_title] if p]
                    if heading_parts:
                        lines.append(f"#### {' - '.join(heading_parts)}")
                    if s_desc:
                        lines.append(s_desc)
                    lines.append("")
            elif elem.name == "table":
                lines.append(html_table_to_md(elem))
                lines.append("")

        lines.append("---")
        lines.append("")
        
    lines.append("### Ready to Orchestrate Production Claude AI Agent Teams?")
    lines.append("- [Schedule Strategy Consultation via Email →](collaboration.html)")
    lines.append("- [View Advisory Packages & Rates](pricing.html)")
    lines.append("- [Explore Claude Agents Suite](claude-101.html)")
    lines.append("")
    return "\n".join(lines).strip() + "\n"
    
# Backward-compatibility alias
convert_portfolio = convert_articles

def main():
    print(f"Generating Markdown files in: {SYNC_DIR}")
    
    converters = {
        "articles.html": ("articles.md", convert_articles),
        "writing.html": ("writing.md", convert_writing),
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

    agent_files = sorted(
        glob.glob(os.path.join(REPO_DIR, "agents-*.html")) +
        glob.glob(os.path.join(REPO_DIR, "claude-*.html")) +
        [g for g in glob.glob(os.path.join(REPO_DIR, "google-*.html")) if not os.path.basename(g).startswith("google43d")]
    )
    for a_path in agent_files:
        base = os.path.basename(a_path)
        md_name = base.replace(".html", ".md")
        dest = os.path.join(SYNC_DIR, md_name)
        print(f"Building {md_name} from {base}...")
        if base == "agents-all-agentic-providers.html":
            content = convert_agents_all_providers(a_path)
        else:
            content = convert_agent_deepdive(a_path)
        with open(dest, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  -> Wrote {len(content.splitlines())} lines to {md_name}")
        
    print("\nInitial Markdown generation completed successfully!")

if __name__ == "__main__":
    main()
