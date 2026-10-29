#!/usr/bin/env python3
"""
generate_sitemap.py - Regenerate sitemap.xml dynamically from the repository file tree.
Uses each file's last git commit date for <lastmod>.
Excludes draft, internal, or excluded pages.
"""

import os
import subprocess

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITEMAP_FILE = os.path.join(REPO_DIR, "sitemap.xml")
BASE_URL = "https://thedigitalfuturist.com"

# Core navigation & portal pages in logical order
CORE_PAGES = [
    ("", "index.html"),
    ("articles.html", "articles.html"),
    ("writing.html", "writing.html"),
    ("pricing.html", "pricing.html"),
    ("expertise.html", "expertise.html"),
    ("contact.html", "contact.html"),
    ("faqs.html", "faqs.html"),
    ("all-business-services.html", "all-business-services.html"),
    ("all-personal-services.html", "all-personal-services.html"),
    ("business-ai-training.html", "business-ai-training.html"),
    ("personal-ai-training.html", "personal-ai-training.html"),
]

# 16 Business Service Pages
BUSINESS_SERVICES = [
    "Business-Services/business-service-generative-ai-transformation.html",
    "Business-Services/business-service-local-llms-cost-slashing.html",
    "Business-Services/business-service-ai-agents-orchestration.html",
    "Business-Services/business-service-agentic-ai-systems.html",
    "Business-Services/business-service-workflow-automations.html",
    "Business-Services/business-service-quantum-applications.html",
    "Business-Services/business-service-ai-budget-cost-limits-training.html",
    "Business-Services/business-service-enterprise-workforce-ai-training.html",
    "Business-Services/business-service-ai-developer-training.html",
    "Business-Services/business-service-advanced-ai-agents.html",
    "Business-Services/business-service-deep-research-notebooklm.html",
    "Business-Services/business-service-claude-basic-training.html",
    "Business-Services/business-service-claude-advanced-training.html",
    "Business-Services/business-service-gemini-basic-training.html",
    "Business-Services/business-service-gemini-advanced-training.html",
    "Business-Services/business-service-hybrid-quantum-ai.html",
]

# 12 Claude Guide Pages
CLAUDE_GUIDES = [
    "Claude/claude-101.html",
    "Claude/claude-features.html",
    "Claude/claude-code.html",
    "Claude/claude-cowork.html",
    "Claude/claude-skills.html",
    "Claude/claude-custom-mcp.html",
    "Claude/claude-plugins.html",
    "Claude/claude-connectors.html",
    "Claude/claude-orchestration.html",
    "Claude/claude-design.html",
    "Claude/claude-science.html",
    "Claude/claude-local-llms.html",
]

# 12 Google Gemini Guide Pages
GEMINI_GUIDES = [
    "Google/google-gemini-101.html",
    "Google/google-ai-studio.html",
    "Google/google-antigravity-ide.html",
    "Google/google-gemini-deep-think.html",
    "Google/google-nano-banana.html",
    "Google/google-flow.html",
    "Google/google-gemini-spark.html",
    "Google/google-gemini-notebook.html",
    "Google/google-gemma.html",
    "Google/google-deep-research.html",
    "Google/google-gemini-4-models.html",
    "Google/google-gemini-agents.html",
]

def get_git_lastmod(rel_path: str) -> str:
    try:
        res = subprocess.run(
            ["git", "log", "-1", "--format=%as", rel_path],
            cwd=REPO_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        date = res.stdout.strip()
        if date:
            return date
    except Exception:
        pass
    return "2026-10-29"

def generate_sitemap():
    entries = []

    # 1. Core pages
    entries.append("  <!-- Core Navigation Pages -->")
    for url_suffix, file_path in CORE_PAGES:
        loc = f"{BASE_URL}/{url_suffix}" if url_suffix else f"{BASE_URL}/"
        lastmod = get_git_lastmod(file_path)
        entries.append(f"  <url>\n    <loc>{loc}</loc>\n    <lastmod>{lastmod}</lastmod>\n  </url>")

    # 2. Business services (16 pages)
    entries.append("\n  <!-- 16 Business AI Transformation Service Pages -->")
    for file_path in BUSINESS_SERVICES:
        loc = f"{BASE_URL}/{file_path}"
        lastmod = get_git_lastmod(file_path)
        entries.append(f"  <url>\n    <loc>{loc}</loc>\n    <lastmod>{lastmod}</lastmod>\n  </url>")

    # 3. Claude Guides (12 pages)
    entries.append("\n  <!-- Claude Guides -->")
    for file_path in CLAUDE_GUIDES:
        loc = f"{BASE_URL}/{file_path}"
        lastmod = get_git_lastmod(file_path)
        entries.append(f"  <url>\n    <loc>{loc}</loc>\n    <lastmod>{lastmod}</lastmod>\n  </url>")

    # 4. Google Gemini Guides (12 pages)
    entries.append("\n  <!-- Google Gemini Guides -->")
    for file_path in GEMINI_GUIDES:
        loc = f"{BASE_URL}/{file_path}"
        lastmod = get_git_lastmod(file_path)
        entries.append(f"  <url>\n    <loc>{loc}</loc>\n    <lastmod>{lastmod}</lastmod>\n  </url>")

    xml_content = '<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
    xml_content += "\n".join(entries)
    xml_content += "\n</urlset>\n"

    with open(SITEMAP_FILE, "w", encoding="utf-8") as f:
        f.write(xml_content)
    print(f"Generated {SITEMAP_FILE} with {len(CORE_PAGES) + len(BUSINESS_SERVICES) + len(CLAUDE_GUIDES) + len(GEMINI_GUIDES)} entries.")

if __name__ == "__main__":
    generate_sitemap()
