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
    ("guarantees.html", "guarantees.html"),
    ("all-business-services.html", "all-business-services.html"),
    ("all-personal-services.html", "all-personal-services.html"),
]

# 12 Business Service Pages
BUSINESS_SERVICES = [
    "business-services/business-service-generative-ai-transformation.html",
    "business-services/business-service-local-llms-cost-slashing.html",
    "business-services/business-service-claude-agents-orchestration.html",
    "business-services/business-service-agentic-assistants.html",
    "business-services/business-service-workflow-automations.html",
    "business-services/business-service-ai-budget-cost-control.html",
    "business-services/business-service-ai-developer-training.html",
    "business-services/business-service-advanced-ai-agents.html",
    "business-services/business-service-competitor-research.html",
    "business-services/business-service-ai-assisted-leads-magnet.html",
    "business-services/business-service-ai-second-brain.html",
    "business-services/business-service-continuous-learning.html",
    "business-services/business-service-claude-research-training.html",
    "business-services/business-service-gemini-research-training.html",
    "business-services/business-service-knowledge-graphs.html",
    "business-services/business-service-local-ai-for-sensitive-data.html",
]

# 12 claude Guide Pages
CLAUDE_GUIDES = [
    "claude-training/claude-101.html",
    "claude-training/claude-projects.html",
    "claude-training/claude-code.html",
    "claude-training/claude-desktop.html",
    "claude-training/claude-office.html",
    "claude-training/claude-skills.html",
    "claude-training/claude-mcp.html",
    "claude-training/claude-plugins.html",
    "claude-training/claude-connectors.html",
    "claude-training/claude-orchestration.html",
    "claude-training/claude-design.html",
    "claude-training/claude-science.html",
    "claude-training/claude-local-llms.html",
    "claude-training/claude-artifacts.html",
    "claude-training/claude-dashboards.html",
    "claude-training/claude-mods.html",
    "claude-training/claude-whiteboards.html",
    "claude-training/claude-fable.html",
]

# 12 google Gemini Guide Pages
GEMINI_GUIDES = [
    "gemini-training/google-gemini-101.html",
    "gemini-training/google-ai-studio.html",
    "gemini-training/google-antigravity-ide.html",
    "gemini-training/google-gemini-4-argon.html",
    "gemini-training/google-gemini-desktop.html",
    "gemini-training/google-deepmind.html",
    "gemini-training/google-gemini-spark.html",
    "gemini-training/google-gemini-notebook.html",
    "gemini-training/google-gemma-local-models.html",
    "gemini-training/google-multimodality.html",
    "gemini-training/google-gemini-4-carbon.html",
    "gemini-training/google-gemini-4-hydrogen.html",
    "gemini-training/google-code-wiki.html",
    "gemini-training/google-deep-research.html",
    "gemini-training/google-gemini-4-krypton.html",
    "gemini-training/google-gemini-skills.html",
    "gemini-training/google-antigravity-chat.html",
    "gemini-training/google-gemini-4-helix.html",
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
    except Exception:  # pragma: no cover
        pass
    return "2026-10-29"  # pragma: no cover

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

if __name__ == "__main__":  # pragma: no cover
    generate_sitemap()
