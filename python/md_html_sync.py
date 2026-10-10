#!/usr/bin/env python3
"""
md_html_sync.py — Bidirectional Synchronization Engine for HTML and Markdown.

Synchronizes:
- index.html <---> md-html-sync/home.md
- <name>.html <---> md-html-sync/<name>.md (for all site pages)

Supports:
- python3 md_html_sync.py          (Bidirectional sync based on modification time / hashes)
- python3 md_html_sync.py --watch  (Continuous live watcher for real-time auto-sync)
- python3 md_html_sync.py --html-to-md  (Force sync HTML to Markdown)
- python3 md_html_sync.py --md-to-html  (Force sync Markdown to HTML)
"""

import os
import sys
import time
import json
import hashlib
import re
import glob
from bs4 import BeautifulSoup, NavigableString, Tag

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
SYNC_DIR = os.path.join(REPO_DIR, "md-html-sync")
STATE_FILE = os.path.join(SYNC_DIR, ".sync-state.json")

os.makedirs(SYNC_DIR, exist_ok=True)

# Map HTML files to Markdown files in md-html-sync/
PAGE_MAPPINGS = {
    "index.html": "home.md",
    "portfolio.html": "portfolio.md",
    "writing.html": "writing.md",
    "services.html": "services.md",
    "collaboration.html": "collaboration.md",
    "pricing.html": "pricing.md",
    "expertise.html": "expertise.md",
    "contact.html": "contact.md",
    "faqs.html": "faqs.md",
    "404.html": "404.md"
}

# Auto-discover service pages
service_files = sorted(glob.glob(os.path.join(REPO_DIR, "service-*.html")))
for s in service_files:
    base = os.path.basename(s)
    PAGE_MAPPINGS[base] = base.replace(".html", ".md")

def get_file_hash(path: str) -> str:
    if not os.path.exists(path):
        return ""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def get_file_mtime(path: str) -> float:
    if not os.path.exists(path):
        return 0.0
    return os.path.getmtime(path)

def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_state(state: dict):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"Error saving state: {e}", file=sys.stderr)

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

# ==============================================================================
# HTML -> MARKDOWN GENERATION
# ==============================================================================

import build_md_files

def sync_html_to_md(html_name: str, md_name: str):
    html_path = os.path.join(REPO_DIR, html_name)
    md_path = os.path.join(SYNC_DIR, md_name)
    
    if not os.path.exists(html_path):
        return False
        
    print(f"  [HTML -> MD] Syncing {html_name} -> md-html-sync/{md_name}...")
    
    if html_name == "index.html":
        # Keep home.md updated with index.html
        with open(html_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        
        name = clean_text(soup.find("h1", class_="hero-name").get_text()) if soup.find("h1", class_="hero-name") else "Thomas Cherickal"
        subtitle = clean_text(soup.find("h2", class_="hero-subtitle").get_text()) if soup.find("h2", class_="hero-subtitle") else "Generative AI Consultant"
        eyebrow = clean_text(soup.find("p", class_="hero-eyebrow").get_text()) if soup.find("p", class_="hero-eyebrow") else "// The Digital Futurist · Remote Worldwide"
        tagline = clean_text(soup.find("p", class_="hero-tagline").get_text()) if soup.find("p", class_="hero-tagline") else ""
        bio = clean_text(soup.find("p", class_="hero-bio").get_text()) if soup.find("p", class_="hero-bio") else ""
        
        roles = [clean_text(r.get_text()) for r in soup.find_all("span", class_="role-chip")]
        # Keep unique in order
        uniq_roles = []
        for r in roles:
            if r not in uniq_roles and len(r) > 2:
                uniq_roles.append(r)
                
        # Stats
        stats = []
        stat_items = soup.find_all("div", class_="stat-item")
        for s in stat_items:
            num = clean_text(s.find("div", class_="stat-num").get_text()) if s.find("div", class_="stat-num") else ""
            lbl = clean_text(s.find("div", class_="stat-label").get_text()) if s.find("div", class_="stat-label") else ""
            if num and lbl:
                stats.append((lbl, num))
                
        md_lines = [
            f"# {name}",
            f"## {subtitle}",
            "",
            f"> **{eyebrow}**  ",
            f"> **{tagline}**",
            "",
            "**Location**: Chennai, India (Remote — Worldwide)  ",
            "**Brand**: The Digital Futurist  ",
            "**Email**: [thomascherickal@gmail.com](mailto:thomascherickal@gmail.com)  ",
            "",
            "**Specialized Roles**:"
        ]
        for r in uniq_roles[:8]:
            md_lines.append(f"- `{r}`")
            
        md_lines.extend([
            "",
            "---",
            "",
            "## Bio",
            bio,
            "",
            "### Quick Actions",
            "- [📚 Portfolio](portfolio.html)",
            "- [📝 Contact via Email](contact.html)",
            "- [🐙 GitHub Profile](https://github.com/thomascherickal)",
            "- [📅 Book 1:1 Consult](https://topmate.io/thomascherickal)",
            "",
            "---",
            "",
            "## Key Metrics",
            "| Metric | Value |",
            "| :--- | :--- |"
        ])
        for lbl, num in stats:
            md_lines.append(f"| **{lbl}** | {num} |")
            
        md_lines.extend([
            "",
            "---",
            "",
            "## Published Across 10+ Platforms Since 2020",
            "HackerNoon · Medium · Hashnode · Substack · LinkedIn · Reddit · Quora · WordPress · Topmate · Gumroad · Patreon · Linktree (+ More)",
            "",
            "---",
            "",
            "## Explore My Work (Curated Destinations)",
            "",
            "1. [📚 Portfolio & Case Studies](portfolio.html) — Detailed case studies across Generative AI Systems and Quantum Systems — the brief, the approach, and what shipped, verified in Python and Rust.",
            "2. [✍️ Publications](writing.html) — Selected work from 500+ publications across 10+ platforms since 2020, plus the book RECRUITED.",
            "3. [⚡ Capabilities & Tech Stack](expertise.html) — 10 specialized capability areas, 10 curated tech-stack chip groups, and the Python & Rust execution tooling behind every piece.",
            "4. [🎯 Services & Commissions](services.html) — Service offerings — AI agent orchestration, deep dives, courses, training, mentoring, enterprise LLM deployment, and Generative AI developer content.",
            "5. [🔬 Workflow & Verification](expertise.html#how-this-gets-made) — Research → Build → Run → Verify → Explain. AI accelerates the workflow, while human verification owns the result.",
            "6. [💳 Pricing & Parity Index](pricing.html) — Transparent milestone rates and an interactive 198-country Purchasing Power Parity (PPP) calculator for global equity.",
            "7. [❓ Frequently Asked Questions](faqs.html) — Turnaround timelines, code verification standards, AI workflows, daily service fee calculator, and collaboration details.",
            "8. [📬 Contact & Training](contact.html) — Commission custom emerging technologies training, AI agent orchestration, developer courses, or retainer sprints directly.",
            "9. [📰 HackerNoon Profile](https://hackernoon.com/u/thomascherickal) — Explore featured deep dives, AI exploration, and published articles on HackerNoon covering AI models and devtools.",
            "10. [🐙 GitHub Profile](https://github.com/thomascherickal) — Inspect open-source code bases, Python and Rust scripts, quantum circuit benchmarks, and custom repositories for content.",
            "",
            "---",
            "",
            "## Ready to Begin Your Enterprise AI Transformation?",
            "",
            "Enterprise AI transformation, AI agent orchestration, technical deep dives, generative AI agent consulting, generative AI agent training.",
            "- [Start a Conversation via Email →](contact.html)",
            "- [Free LinkedIn Consultation ↗](https://linkedin.com/in/thomascherickal)",
            "- [Commission an Enterprise AI Transformation →](contact.html)",
            "- [Book a 1:1 Consultation →](https://topmate.io/thomascherickal)",
            "- [Explore My Portfolio →](portfolio.html)",
            "",
            "---",
            "",
            "## Enterprise AI Transformation Services",
            "",
            "1. [🧠 Generative AI Transformation](service-generative-ai-transformation.html) — Convert an entire business that uses AI as a chatbot to one that works with agents, with metrics and safeguards.",
            "2. [📦 Local LLMs & Cost Slashing](service-local-llms-cost-slashing.html) — Slashing cloud API bills by 60–90% via Ollama, vLLM, llama.cpp, and private on-premise model serving.",
            "3. [🤖 AI Agent Orchestration Fundamentals](service-ai-agents-orchestration.html) — Principles of AI agent orchestration, releasing agent swarms at scale, using AI agents for everything with low budgets as well.",
            "4. [🏎️ Agentic AI Assistants (Hermes Agent)](service-agentic-ai-systems.html) — Enterprise deployment of autonomously improving agentic assistants with persistent vector memory.",
            "5. [🤗 Enterprise Coding Model Optimization](service-enterprise-coding-model-optimization.html) — Training to optimize AI tools like Claude Code, OpenAI Codex, Google Antigravity, and OpenCode.",
            "6. [🌐 Training for WorkFlows Automations](service-training-for-workflows-automations.html) — Training in setting up n8n automations, self-hosted, to automate 50% of daily work.",
            "7. [⚛️ Quantum Applications (Experimental)](service-quantum-applications.html) — If required, exploring quantum systems for business case uses and applications.",
            "8. [⚙️ Low Code and No Code Enterprise Automation Training](service-low-code-no-code-automation-training.html) — Setting up no code automations for enterprises to automate 50% of daily work.",
            "9. [💲 AI Budget and Cost Limits Training](service-ai-budget-cost-limits-training.html) — Training employees at all levels on spending limits, budgeting, and how to use free models for minor tasks.",
            "10. [🛡️ Enterprise Workforce AI Training](service-enterprise-workforce-ai-training.html) — Live and remote corporate training for freshers, employees, developers (AI TDD), and CXOs, alongside executive briefings and AI roadmaps.",
            "",
            "---",
            "",
            "## Books & Long-Form",
            "",
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
            "## The Digital Futurist Newsletter",
            "",
            "**The Digital Futurist Newsletter**: How to understand and build emerging technologies.  ",
            "- [✉️ Subscribe Free on Kit](https://thomascherickal.kit.com)",
            "",
            "---",
            "",
            "## Links",
            "",
            "### Find Me Online",
            "- [🌐 Profile (thomascherickal.com)](https://thomascherickal.com)",
            "- [🐙 GitHub (github.com/thomascherickal)](https://github.com/thomascherickal)",
            "- [💼 LinkedIn (in/thomascherickal)](https://linkedin.com/in/thomascherickal)",
            "- [🗞 HackerNoon (u/thomascherickal)](https://hackernoon.com/u/thomascherickal)",
            "- [✍️ Medium (@thomascherickal)](https://thomascherickal.medium.com)",
            "- [🔷 Hashnode (thomascherickal.hashnode.dev)](https://thomascherickal.hashnode.dev)",
            "- [📬 Substack (thesingularitypoint.substack.com)](https://thesingularitypoint.substack.com)",
            "- [❓ Quora (thomascherickal.quora.com)](https://thomascherickal.quora.com)",
            "- [🤖 Reddit (reddit.com/user/thomascherickal1)](https://reddit.com/user/thomascherickal1)",
            "- [🧪 Exercism (exercism.org/profiles/thomascherickal)](https://exercism.org/profiles/thomascherickal)",
            "- [🏅 CodersRank (profile.codersrank.io/user/thomascherickal)](https://profile.codersrank.io/user/thomascherickal/)",
            "- [🧠 Deep-ML (deep-ml.com/profile/thomascherickal)](https://www.deep-ml.com/profile/thomascherickal)",
            "- [✖️ X (@thomazcherickal)](https://x.com/thomazcherickal)",
            "- [👩‍💻 DEV (dev.to/thomascherickal)](https://dev.to/thomascherickal)",
            "- [🦊 GitLab (gitlab.com/thomascherickal)](https://gitlab.com/thomascherickal)",
            "- [✉️ Kit (thomascherickal.kit.com)](https://thomascherickal.kit.com)",
            "- [🔗 Linktree (linktr.ee/thomascherickal)](https://linktr.ee/thomascherickal)",
            "- [🎨 Patreon (patreon.com/thomascherickal)](https://patreon.com/thomascherickal)",
            "- [🛍️ Gumroad (thomascherickal.gumroad.com)](https://thomascherickal.gumroad.com)",
            "- [🤝 Topmate (topmate.io/thomascherickal)](https://topmate.io/thomascherickal)",
            "",
            "---",
            "*© 2026 Thomas Cherickal · The Digital Futurist · Generative AI Consultant · Agentic AI Architect · AI Automation Expert · AI Trainer Live & Remote · Enterprise AI Integration · Local LLM Deployment · AI Agent Orchestration · Remote Worldwide*",
            ""
        ])
        content = "\n".join(md_lines)
    elif html_name == "portfolio.html":
        content = build_md_files.convert_portfolio()
    elif html_name == "writing.html":
        content = build_md_files.convert_writing()
    elif html_name == "services.html":
        content = build_md_files.convert_services()
    elif html_name == "collaboration.html":
        content = build_md_files.convert_collaboration()
    elif html_name == "pricing.html":
        content = build_md_files.convert_pricing()
    elif html_name == "expertise.html":
        content = build_md_files.convert_expertise()
    elif html_name == "contact.html":
        content = build_md_files.convert_contact()
    elif html_name == "faqs.html":
        content = build_md_files.convert_faqs()
    elif html_name == "404.html":
        content = build_md_files.convert_404()
    elif html_name.startswith("service-"):
        content = build_md_files.convert_service_page(html_path)
    else:
        return False
        
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(content)
        
    return True

# ==============================================================================
# MARKDOWN -> HTML PROPAGATION
# ==============================================================================

def sync_md_to_html(md_name: str, html_name: str):
    md_path = os.path.join(SYNC_DIR, md_name)
    html_path = os.path.join(REPO_DIR, html_name)
    
    if not os.path.exists(md_path) or not os.path.exists(html_path):
        return False
        
    print(f"  [MD -> HTML] Syncing md-html-sync/{md_name} -> {html_name}...")
    
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()
        
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
        
    soup = BeautifulSoup(html_content, "html.parser")
    main = soup.find("main")
    if not main:
        return False
        
    # Extract MD headers and key elements
    md_h1_match = re.search(r'^#\s+(.+)$', md_text, re.MULTILINE)
    md_h2_match = re.search(r'^##\s+(.+)$', md_text, re.MULTILINE)
    
    if md_name == "home.md":
        # Synchronize hero elements
        if md_h1_match:
            hero_name = soup.find("h1", class_="hero-name")
            if hero_name:
                hero_name.string = clean_text(md_h1_match.group(1))
        if md_h2_match:
            hero_sub = soup.find("h2", class_="hero-subtitle")
            if hero_sub:
                hero_sub.string = clean_text(md_h2_match.group(1))
                
        # Tagline in blockquote
        tagline_match = re.search(r'>\s+\*\*([^\n]+)\*\*\s*\n>\s+\*\*([^\n]+)\*\*', md_text)
        if tagline_match:
            eyebrow = soup.find("p", class_="hero-eyebrow")
            if eyebrow:
                eyebrow.string = tagline_match.group(1)
            tagline = soup.find("p", class_="hero-tagline")
            if tagline:
                tagline.string = tagline_match.group(2)
                
        # Bio
        bio_match = re.search(r'## Bio\s*\n+([^#\n][^\n]+(?:\n[^#\n][^\n]+)*)', md_text)
        if bio_match:
            bio_p = soup.find("p", class_="hero-bio")
            if bio_p:
                bio_p.string = clean_text(bio_match.group(1))
                
    elif html_name == "faqs.html":
        # Sync FAQ answers if edited in markdown
        faq_sections = re.findall(r'###\s+\d+\.\s+([^\n]+)\n+(?:\*[^\n]+\*\s*\n+)?([^#\n]+(?:\n[^#\n]+)*)', md_text)
        faq_cards = soup.find_all("div", class_="faq-card")
        for (q, ans), card in zip(faq_sections, faq_cards):
            q_el = card.find(["h2", "h3"])
            if q_el:
                q_el.string = clean_text(q)
            ans_el = card.find("p", class_="faq-answer")
            if ans_el:
                # Keep strong start if present
                clean_ans = clean_text(ans)
                ans_el.string = clean_ans
                
    elif html_name == "pricing.html":
        # Sync package prices
        prices = re.findall(r'\*\*Investment\*\*:\s*`([^`]+)`', md_text)
        pkg_cards = soup.find_all("div", class_="package-card")
        for price_val, card in zip(prices, pkg_cards):
            price_div = None
            for div in card.find_all("div"):
                txt = div.get_text().strip()
                if ("$" in txt or "%" in txt) and not div.find("div") and ("–" in txt or "-" in txt or "/mo" in txt or "OFF" in txt):
                    price_div = div
                    break
            if price_div:
                price_div.string = price_val.replace(" USD", "")
                
    elif html_name.startswith("service-"):
        # Sync H1
        if md_h1_match:
            h1_el = soup.find("h1")
            if h1_el:
                h1_el.string = clean_text(md_h1_match.group(1))
                
    # Save modified HTML
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(str(soup))
        
    return True

# ==============================================================================
# BIDIRECTIONAL SYNC ORCHESTRATOR
# ==============================================================================

def run_sync(force_direction: str = None) -> int:
    state = load_state()
    changed_count = 0
    
    print("\n--- Running MD-HTML Sync Pass ---")
    
    for html_name, md_name in PAGE_MAPPINGS.items():
        html_path = os.path.join(REPO_DIR, html_name)
        md_path = os.path.join(SYNC_DIR, md_name)
        
        if not os.path.exists(html_path):
            continue
            
        cur_html_hash = get_file_hash(html_path)
        cur_html_mtime = get_file_mtime(html_path)
        
        cur_md_hash = get_file_hash(md_path)
        cur_md_mtime = get_file_mtime(md_path)
        
        saved = state.get(html_name, {})
        saved_html_hash = saved.get("html_hash", "")
        saved_md_hash = saved.get("md_hash", "")
        
        # Determine direction
        if force_direction == "html_to_md":
            sync_html_to_md(html_name, md_name)
            changed_count += 1
        elif force_direction == "md_to_html":
            sync_md_to_html(md_name, html_name)
            changed_count += 1
        else:
            # Auto bidirectional detection
            if not os.path.exists(md_path):
                sync_html_to_md(html_name, md_name)
                changed_count += 1
            else:
                html_changed = (cur_html_hash != saved_html_hash)
                md_changed = (cur_md_hash != saved_md_hash)
                
                if html_changed and md_changed:
                    # Both changed, resolve by newer mtime
                    if cur_html_mtime >= cur_md_mtime:
                        sync_html_to_md(html_name, md_name)
                    else:
                        sync_md_to_html(md_name, html_name)
                    changed_count += 1
                elif html_changed:
                    sync_html_to_md(html_name, md_name)
                    changed_count += 1
                elif md_changed:
                    sync_md_to_html(md_name, html_name)
                    changed_count += 1
                    
        # Update saved state
        state[html_name] = {
            "html_hash": get_file_hash(html_path),
            "html_mtime": get_file_mtime(html_path),
            "md_hash": get_file_hash(md_path),
            "md_mtime": get_file_mtime(md_path),
            "md_name": md_name
        }
        
    save_state(state)
    print(f"Sync complete. Synchronized {changed_count} file pair(s).\n")
    return changed_count

# ==============================================================================
# WATCH MODE
# ==============================================================================

def watch(interval: float = 1.0):
    print(f"Starting live MD-HTML file watcher (polling every {interval}s)...")
    print("Press Ctrl+C to stop.\n")
    # Initial state record
    run_sync()
    
    try:
        while True:
            time.sleep(interval)
            state = load_state()
            any_change = False
            
            for html_name, md_name in PAGE_MAPPINGS.items():
                html_path = os.path.join(REPO_DIR, html_name)
                md_path = os.path.join(SYNC_DIR, md_name)
                
                if not os.path.exists(html_path) or not os.path.exists(md_path):
                    continue
                    
                cur_html_hash = get_file_hash(html_path)
                cur_md_hash = get_file_hash(md_path)
                
                saved = state.get(html_name, {})
                if cur_html_hash != saved.get("html_hash", "") or cur_md_hash != saved.get("md_hash", ""):
                    any_change = True
                    break
                    
            if any_change:
                run_sync()
                
    except KeyboardInterrupt:
        print("\nWatcher stopped.")

# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================

if __name__ == "__main__":
    if "--watch" in sys.argv:
        watch()
    elif "--html-to-md" in sys.argv:
        run_sync(force_direction="html_to_md")
    elif "--md-to-html" in sys.argv:
        run_sync(force_direction="md_to_html")
    else:
        run_sync()
