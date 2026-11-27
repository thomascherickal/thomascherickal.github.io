#!/usr/bin/env python3
"""
generate_llms_full.py - Builds llms-full.txt deterministically from data/*.json and live pages.

Provides a complete, zero-boilerplate Markdown compendium for AI answer engines
and coding agents (ChatGPT Search, Perplexity, Claude, Gemini) with verified URLs,
SSOT-sourced pricing/terms, and direct md-html-sync/ mirror pointers.
Strict Repo Compliance: Zero emojis, zero em-dashes, no LLM version numbers in prose.
"""

import glob
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from python.ssot.load import SSOTData
from python.ssot.format import (
    tier_price_list,
    seats_phrase,
    duration_phrase,
    installment_phrase,
    reply_sla_phrase,
    quarterly_cap_phrase,
)

BASE_URL = "https://thedigitalfuturist.com"


def clean_prose(text: str) -> str:
    s = str(text or "").replace("\u2014", " - ").replace("\u2013", "-")
    s = re.sub(r"\b(Gemini|Claude|Llama|Qwen)\s+\d+(?:\.\d+)?\b", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_title_and_desc(rel_path: str) -> tuple:
    abs_p = os.path.join(REPO_ROOT, rel_path)
    with open(abs_p, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read(16000)
    tm = re.search(r"<title[^>]*>(.*?)</title>", raw, re.I | re.S)
    title = clean_prose(re.sub(r"<[^>]+>", "", tm.group(1))) if tm else os.path.basename(rel_path)
    title = re.sub(r"\s*\|\s*Thomas Cherickal.*$", "", title)
    dm = re.search(r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']*)["\']', raw, re.I)
    if not dm:
        dm = re.search(r'<meta[^>]+content=["\']([^"\']*)["\'][^>]+name=["\']description["\']', raw, re.I)
    desc = clean_prose(dm.group(1)) if dm else ""
    return title, desc


def generate_llms_full(out_path=None) -> str:
    if out_path is None:
        out_path = os.path.join(REPO_ROOT, "llms-full.txt")
    ssot = SSOTData()

    flag_prices = tier_price_list(ssot, "flagship-business-training")
    pick_prices = tier_price_list(ssot, "individual-service-business")
    flag_seats = seats_phrase(ssot, "flagship-business-training")
    pick_seats = seats_phrase(ssot, "individual-service-business")
    flag_dur = duration_phrase(ssot, "flagship-business-training")
    pick_dur = duration_phrase(ssot, "individual-service-business")
    flag_inst = installment_phrase(ssot, "flagship-business-training")
    sla = reply_sla_phrase(ssot)
    q_cap = quarterly_cap_phrase(ssot)

    lines = [
        "# The Digital Futurist - Full LLM Knowledge Compendium",
        "",
        "> Thomas Cherickal - Generative AI Consultant and The Digital Futurist.",
        "> Canonical Tagline: The Future Is Now",
        f"> Business Website: {BASE_URL}",
        "> Direct Email: thomascherickal@gmail.com",
        f"> Contact Reply SLA: {sla}",
        f"> Concise Index: {BASE_URL}/llms.txt",
        f"> Sitemap: {BASE_URL}/sitemap.xml",
        f"> RSS Feed: {BASE_URL}/rss.xml",
        "",
        "## Authoritative Commercial Terms & Purchasing Power Parity (PPP)",
        "",
        f"- **Business AI Training Flagship**: {flag_prices} USD (Tiers 1-6) | {flag_dur} (30 days live engineering training + 30 calendar days dedicated support; 4 hours/day Mon-Fri; weekends are holidays; 100% remote worldwide) | {flag_seats} | $0 upfront | {flag_inst} (Days 5-50) | Walkaway guarantee | Quarterly cap: {q_cap}.",
        f"- **Personal AI Training Flagship**: {flag_prices} USD (Tiers 1-6) | {flag_dur} (30 days live personal training + 30 calendar days dedicated support; 4 hours/day Mon-Fri) | {flag_seats} | $0 upfront | {flag_inst} | Walkaway guarantee.",
        f"- **Individual Service Picks (Business & Personal)**: {pick_prices} USD (Tiers 1-6) | {pick_dur} (5 days live training + 5 days dedicated support) | {pick_seats} | $0 upfront | {flag_inst} (Days 1-10).",
        "- **Extra Seats**: Billed at the identical calibrated PPP tier rate (+$100 / +$75 / +$50 / +$50 / +$25 / Custom USD per extra seat up to 20 seats max).",
        "- **Licensing & IP Ownership**: All training materials, architectural guides, and custom code are licensed under the Apache 2.0 license with 100% client ownership; Zero Charge Training Materials policy applies upon milestone completion.",
        "- **Tier Precedence Rule**: When a country appears in both Tier 3 and Tier 6 (UN LDCs), Tier 6 (Relief / UN LDCs) takes precedence. Tier 4 (Startups) follows the standard 2-month flagship program timeline.",
        "",
    ]

    sections = [
        ("Core Portal Pages", [
            "index.html", "pricing.html", "parity-pricing.html", "guarantees.html",
            "faqs.html", "contact.html", "all-business-services.html", "all-personal-services.html",
            "expertise.html", "articles.html", "writing.html",
        ]),
        ("6 Purchasing Power Parity Regional Tier Pages", sorted(glob.glob("pricing/*.html", root_dir=REPO_ROOT))),
        ("16 Enterprise Business AI Services", sorted(glob.glob("business-services/*.html", root_dir=REPO_ROOT))),
        ("16 Personal AI Training Services", sorted(glob.glob("personal-services/*.html", root_dir=REPO_ROOT))),
        ("18 Claude Engineering Guides", sorted(glob.glob("claude-training/*.html", root_dir=REPO_ROOT))),
        ("18 Google Gemini & Gemma Engineering Guides", sorted(glob.glob("gemini-training/*.html", root_dir=REPO_ROOT))),
    ]

    for heading, rel_paths in sections:
        lines.append(f"## {heading}")
        lines.append("")
        for rel in rel_paths:
            if not os.path.isfile(os.path.join(REPO_ROOT, rel)):
                continue
            title, desc = extract_title_and_desc(rel)
            url = f"{BASE_URL}/" if rel == "index.html" else f"{BASE_URL}/{rel}"
            md_rel = "md-html-sync/home.md" if rel == "index.html" else f"md-html-sync/{rel[:-5]}.md"
            md_note = f" | [Markdown Mirror]({BASE_URL}/{md_rel})" if os.path.isfile(os.path.join(REPO_ROOT, md_rel)) else ""
            lines.append(f"- **{title}**: {desc} - {url}{md_note}")
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")
    print(f"Generated {os.path.relpath(out_path, REPO_ROOT)} ({len(lines)} lines).")
    return out_path


if __name__ == "__main__":  # pragma: no cover
    generate_llms_full()
