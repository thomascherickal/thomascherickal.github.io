#!/usr/bin/env python3
"""
generate_sitemap.py - Dynamically regenerates sitemap.xml from the repository file tree.

Auto-discovers all indexable HTML pages across root, pricing/, business-services/,
personal-services/, claude-training/, and gemini-training/, skips excluded or noindex pages,
and uses each file's real last git commit date for <lastmod>.
"""

import glob
import os
import re
import subprocess

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITEMAP_FILE = os.path.join(REPO_DIR, "sitemap.xml")
BASE_URL = "https://thedigitalfuturist.com"

EXCLUDED_BASENAMES = {"header.html", "footer.html", "404.html", "mermaid.html"}

CORE_ORDER = [
    "index.html",
    "articles.html",
    "writing.html",
    "pricing.html",
    "parity-pricing.html",
    "expertise.html",
    "contact.html",
    "faqs.html",
    "guarantees.html",
    "all-business-services.html",
    "all-personal-services.html",
]


def is_noindex(abs_path: str) -> bool:
    try:
        with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
            head = f.read(4096)
        return bool(re.search(r'<meta[^>]+name=["\']robots["\'][^>]+noindex', head, re.I))
    except OSError:
        return False


def get_git_lastmod(rel_path: str) -> str:
    try:
        res = subprocess.run(
            ["git", "log", "-1", "--format=%as", "--", rel_path],
            cwd=REPO_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        date = res.stdout.strip()
        if date:
            return date
    except Exception:  # pragma: no cover
        pass
    return "2026-11-27"  # pragma: no cover


def discover_dir_html(subdir: str) -> list:
    full_dir = os.path.join(REPO_DIR, subdir)
    if not os.path.isdir(full_dir):
        return []
    found = []
    for abs_p in sorted(glob.glob(os.path.join(full_dir, "*.html"))):
        base = os.path.basename(abs_p)
        if base in EXCLUDED_BASENAMES or base.startswith("google43d") or is_noindex(abs_p):
            continue
        found.append(f"{subdir}/{base}")
    return found


def discover_core_html() -> list:
    ordered = []
    seen = set()
    for fn in CORE_ORDER:
        abs_p = os.path.join(REPO_DIR, fn)
        if os.path.isfile(abs_p) and not is_noindex(abs_p):
            ordered.append(fn)
            seen.add(fn)
    for abs_p in sorted(glob.glob(os.path.join(REPO_DIR, "*.html"))):
        base = os.path.basename(abs_p)
        if base in seen or base in EXCLUDED_BASENAMES or base.startswith("google43d") or is_noindex(abs_p):
            continue
        ordered.append(base)
    return ordered


def _fmt_entry(rel_path: str) -> str:
    loc = f"{BASE_URL}/" if rel_path == "index.html" else f"{BASE_URL}/{rel_path}"
    lastmod = get_git_lastmod(rel_path)
    return f"  <url>\n    <loc>{loc}</loc>\n    <lastmod>{lastmod}</lastmod>\n  </url>"


def generate_sitemap():
    sections = [
        ("Core Navigation Pages", discover_core_html()),
        ("Purchasing Power Parity Pricing Tier Pages", discover_dir_html("pricing")),
        ("Business AI Transformation Service Pages", discover_dir_html("business-services")),
        ("Personal AI Training Service Pages", discover_dir_html("personal-services")),
        ("Claude Engineering Guide Pages", discover_dir_html("claude-training")),
        ("Google Gemini Engineering Guide Pages", discover_dir_html("gemini-training")),
    ]

    entries = []
    total = 0
    for label, files in sections:
        if not files:
            continue
        prefix = "\n  " if entries else "  "
        entries.append(f"{prefix}<!-- {label} ({len(files)}) -->")
        for rel_p in files:
            entries.append(_fmt_entry(rel_p))
            total += 1

    xml_content = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        + "\n".join(entries)
        + "\n</urlset>\n"
    )

    with open(SITEMAP_FILE, "w", encoding="utf-8") as f:
        f.write(xml_content)

    print(f"Generated {SITEMAP_FILE} with {total} indexable URLs.")
    return total


if __name__ == "__main__":  # pragma: no cover
    generate_sitemap()
