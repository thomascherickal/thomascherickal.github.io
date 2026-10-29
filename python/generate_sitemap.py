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

# 12 Business Service Pages
BUSINESS_SERVICES = [
    "business-services/business-service-generative-ai-transformation.html",
    "business-services/business-service-local-llms-cost-slashing.html",
    "business-services/business-service-ai-agents-orchestration.html",
    "business-services/business-service-agentic-ai-systems.html",
    "business-services/business-service-workflow-automations.html",
    "business-services/business-service-ai-budget-cost-limits-training.html",
    "business-services/business-service-ai-developer-training.html",
    "business-services/business-service-advanced-ai-agents.html",
    "business-services/business-service-claude-advanced-training.html",
    "business-services/business-service-gemini-advanced-training.html",
    "business-services/business-service-competitor-research.html",
    "business-services/business-service-leads-magnet.html",
]

# 12 claude Guide Pages
CLAUDE_GUIDES = [
    "claude/claude-101.html",
    "claude/claude-features.html",
    "claude/claude-code.html",
    "claude/claude-cowork.html",
    "claude/claude-skills.html",
    "claude/claude-custom-mcp.html",
    "claude/claude-plugins.html",
    "claude/claude-connectors.html",
    "claude/claude-orchestration.html",
    "claude/claude-design.html",
    "claude/claude-science.html",
    "claude/claude-local-llms.html",
]

# 12 google Gemini Guide Pages
GEMINI_GUIDES = [
    "google/google-gemini-101.html",
    "google/google-ai-studio.html",
    "google/google-antigravity-ide.html",
    "google/google-gemini-deep-think.html",
    "google/google-nano-banana.html",
    "google/google-flow.html",
    "google/google-gemini-spark.html",
    "google/google-gemini-notebook.html",
    "google/google-gemma.html",
    "google/google-deep-research.html",
    "google/google-gemini-4-models.html",
    "google/google-gemini-agents.html",
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

    # 3. claude Guides (12 pages)
    entries.append("\n  <!-- claude Guides -->")
    for file_path in CLAUDE_GUIDES:
        loc = f"{BASE_URL}/{file_path}"
        lastmod = get_git_lastmod(file_path)
        entries.append(f"  <url>\n    <loc>{loc}</loc>\n    <lastmod>{lastmod}</lastmod>\n  </url>")

    # 4. google Gemini Guides (12 pages)
    entries.append("\n  <!-- google Gemini Guides -->")
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
