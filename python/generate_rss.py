#!/usr/bin/env python3
"""
generate_rss.py - Generates rss.xml (RSS 2.0) for thedigitalfuturist.com.

Publishes all Claude and Gemini engineering guides plus core portal updates
so RSS readers, aggregators, and AI discovery crawlers detect new content.
Strict Repo Compliance: Zero emojis, zero em-dashes.
"""

import glob
import html
import os
import re
import subprocess
from datetime import datetime, timezone
from email.utils import format_datetime

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RSS_FILE = os.path.join(REPO_DIR, "rss.xml")
BASE_URL = "https://thedigitalfuturist.com"


def get_git_rfc2822(rel_path: str) -> str:
    try:
        res = subprocess.run(
            ["git", "log", "-1", "--format=%cI", "--", rel_path],
            cwd=REPO_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        iso = res.stdout.strip()
        if iso:
            dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
            return format_datetime(dt)
    except Exception:  # pragma: no cover
        pass
    return format_datetime(datetime.now(timezone.utc))


def extract_meta(rel_path: str) -> tuple:
    with open(os.path.join(REPO_DIR, rel_path), "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read(16000)
    tm = re.search(r"<title[^>]*>(.*?)</title>", raw, re.I | re.S)
    title = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", tm.group(1))).strip() if tm else rel_path
    dm = re.search(r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']*)["\']', raw, re.I)
    if not dm:
        dm = re.search(r'<meta[^>]+content=["\']([^"\']*)["\'][^>]+name=["\']description["\']', raw, re.I)
    desc = re.sub(r"\s+", " ", dm.group(1)).strip() if dm else ""
    title = title.replace("\u2014", " - ").replace("\u2013", "-")
    desc = desc.replace("\u2014", " - ").replace("\u2013", "-")
    return title, desc


def generate_rss():
    targets = (
        ["index.html", "articles.html", "writing.html", "pricing.html", "guarantees.html", "faqs.html"]
        + sorted(glob.glob("claude-training/*.html", root_dir=REPO_DIR))
        + sorted(glob.glob("gemini-training/*.html", root_dir=REPO_DIR))
        + sorted(glob.glob("business-services/*.html", root_dir=REPO_DIR))
        + sorted(glob.glob("personal-services/*.html", root_dir=REPO_DIR))
    )

    items = []
    for rel in targets:
        if not os.path.isfile(os.path.join(REPO_DIR, rel)):
            continue
        title, desc = extract_meta(rel)
        url = f"{BASE_URL}/" if rel == "index.html" else f"{BASE_URL}/{rel}"
        pub = get_git_rfc2822(rel)
        items.append(
            "    <item>\n"
            f"      <title>{html.escape(title)}</title>\n"
            f"      <link>{url}</link>\n"
            f"      <guid isPermaLink=\"true\">{url}</guid>\n"
            f"      <description>{html.escape(desc)}</description>\n"
            f"      <pubDate>{pub}</pubDate>\n"
            "    </item>"
        )

    now_rfc = format_datetime(datetime.now(timezone.utc))
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">\n'
        "  <channel>\n"
        "    <title>The Digital Futurist - Thomas Cherickal</title>\n"
        f"    <link>{BASE_URL}/</link>\n"
        "    <description>Thomas Cherickal - Generative AI Consultant and The Digital Futurist. Engineering guides, local LLM architectures, Claude and Gemini agent playbooks, and training services.</description>\n"
        "    <language>en-us</language>\n"
        f"    <lastBuildDate>{now_rfc}</lastBuildDate>\n"
        f'    <atom:link href="{BASE_URL}/rss.xml" rel="self" type="application/rss+xml" />\n'
        + "\n".join(items)
        + "\n  </channel>\n"
        "</rss>\n"
    )

    with open(RSS_FILE, "w", encoding="utf-8") as f:
        f.write(xml)
    print(f"Generated {RSS_FILE} with {len(items)} items.")
    return len(items)


if __name__ == "__main__":  # pragma: no cover
    generate_rss()
