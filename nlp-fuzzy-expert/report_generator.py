#!/usr/bin/env python3
"""
report_generator.py - Markdown and Reference Report Generation for NLP-Fuzzy-Expert.

Strict Repo Compliance:
- Zero emojis.
- Zero em-dashes (spaced hyphens ' - ' or plain hyphens '-' used).
- Strict data/ directory immutability.
"""

import os
from datetime import datetime
from expert_system import save_json, load_json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)

DOMAIN_FILES = {
    "pricing": "pricing_references.json",
    "business_services": "business_services_references.json",
    "personal_services": "personal_services_references.json",
    "tiers": "tiers_references.json",
    "claude": "claude_references.json",
    "gemini": "gemini_references.json",
    "publications": "publications_references.json",
    "config": "config_references.json",
    "faqs": "faqs_references.json",
}

DOMAIN_DATA_SOURCE = {
    "pricing": "data/pricing.json",
    "business_services": "data/business-services.json",
    "personal_services": "data/personal-services.json",
    "tiers": "data/tiers.json",
    "claude": "data/claude.json",
    "gemini": "data/gemini.json",
    "publications": "data/publications.json",
    "config": "data/config.json",
    "faqs": "data/faqs.json",
}


def classify_rule_field(field):
    """Maps a rule-tracked field key to its data domain."""
    if "pricing" in field or "flagship" in field or "seats" in field or "calculator" in field:
        return "pricing"
    if "personal_service" in field:
        return "personal_services"
    if "business_service" in field:
        return "business_services"
    if "tier" in field:
        return "tiers"
    if "claude" in field:
        return "claude"
    if "gemini" in field:
        return "gemini"
    if "publication" in field:
        return "publications"
    if "email" in field or "config" in field:
        return "config"
    if "faq" in field:
        return "faqs"
    return "pricing"


_DOMAIN_REFS_CACHE = {}
_DOMAIN_MD_CACHE = {}


def write_reference_files(domain_refs, force=False):
    """Writes per-domain and master reference JSON files (never touches data/)."""
    global _DOMAIN_REFS_CACHE
    timestamp = datetime.now().isoformat()
    all_refs = []
    any_domain_written = False

    for domain, fname in DOMAIN_FILES.items():
        refs = domain_refs.get(domain, [])
        all_refs.extend(refs)
        fpath = os.path.join(SCRIPT_DIR, fname)

        if not force and os.path.exists(fpath):
            cached_refs = _DOMAIN_REFS_CACHE.get(domain)
            if cached_refs is not None and cached_refs == refs:
                continue
            existing = load_json(fpath)
            if existing.get("all_references") == refs:
                _DOMAIN_REFS_CACHE[domain] = refs
                continue

        out_dict = {
            "metadata": {
                "domain": domain,
                "engine": "nlp-fuzzy-expert",
                "total_references_tracked": len(refs),
                "last_synchronized": timestamp,
            },
            "all_references": refs,
        }
        save_json(fpath, out_dict)
        _DOMAIN_REFS_CACHE[domain] = refs
        any_domain_written = True
        print(f"  -> Wrote {fname} ({len(refs)} references)")

    master_path = os.path.join(SCRIPT_DIR, "references_dictionary.json")
    if force or not os.path.exists(master_path):
        should_write_master = True
    elif not any_domain_written:
        should_write_master = False
    else:
        existing_master = load_json(master_path)
        should_write_master = (existing_master.get("all_references") != all_refs)

    if should_write_master:
        master_dict = {
            "metadata": {
                "engine": "nlp-fuzzy-expert",
                "version": "1.0.0",
                "architecture": "Expert System (Forward Chaining) + Fuzzy Logic + NLP",
                "total_references_tracked": len(all_refs),
                "last_synchronized": timestamp,
            },
            "by_domain": {k: len(domain_refs.get(k, [])) for k in DOMAIN_FILES},
            "all_references": all_refs,
        }
        save_json(master_path, master_dict)
        print(f"  -> Master Dictionary generated: references_dictionary.json ({len(all_refs)} total references)")


def generate_markdown_reports(domain_refs, force=False):
    """Generates domain-specific markdown reports under nlp-fuzzy-expert/."""
    global _DOMAIN_MD_CACHE
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    reports = [
        ("pricing.md", "pricing", "Pricing & Engagement Models"),
        ("business-services.md", "business_services", "Enterprise & Business AI Services"),
        ("personal-services.md", "personal_services", "Personal & Professional AI Services"),
        ("tiers.md", "tiers", "Country Parity Pricing Tiers"),
        ("claude.md", "claude", "Anthropic Claude Engineering Guides"),
        ("gemini.md", "gemini", "Google Gemini Architecture Guides"),
        ("publications.md", "publications", "Technical Publications"),
        ("config.md", "config", "Site Configuration & Brand Identity"),
        ("faqs.md", "faqs", "Frequently Asked Questions & Policies"),
    ]

    for md_name, domain, title in reports:
        md_path = os.path.join(SCRIPT_DIR, md_name)
        filtered = domain_refs.get(domain, [])

        if not force and os.path.exists(md_path):
            cached_filtered = _DOMAIN_MD_CACHE.get(domain)
            if cached_filtered is not None and cached_filtered == filtered:
                continue

        lines = [
            f"# NLP-Fuzzy-Expert Synchronization Report - {title}",
            "",
            f"Generated by `nlp-fuzzy-expert/synchronizer.py` via Expert System forward chaining.",
            "",
            "## Summary Metadata",
            "",
            f"- **Domain**: `{domain}`",
            f"- **Tracked Records / Fields**: {len(filtered)}",
            "- **Inference Status**: Rule Fired & Synchronized",
            "- **Highlight Specification**: Visible in IDE (`rgba(251, 191, 36, 0.25)`), invisible on live production sites",
            f"- **Last Synchronized**: {timestamp}",
            "",
            "## Tracked Occurrences Table",
            "",
            "| Field Key | Tracked Value | File Location | Section | Line |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]

        if not filtered:
            lines.append(f"| {domain}_rules | Authoritative | Target documents | Global Inference | Centralized |")
        else:
            for ref in filtered:
                fk = ref.get("field_key", "")
                val = ref.get("new_value", "")
                fl = ref.get("file", "")
                sec = ref.get("section", "")
                ln = str(ref.get("line_number", 1))
                lines.append(f"| {fk} | {val} | `{fl}` | {sec} | Line {ln} |")

        lines.append("")
        new_content = "\n".join(lines)

        if not force and os.path.exists(md_path):
            with open(md_path, "r", encoding="utf-8") as f:
                old_content = f.read()
            old_body = old_content.split("## Tracked Occurrences Table", 1)[-1] if "## Tracked Occurrences Table" in old_content else ""
            new_body = new_content.split("## Tracked Occurrences Table", 1)[-1]
            if old_body and old_body == new_body:
                _DOMAIN_MD_CACHE[domain] = filtered
                continue

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        _DOMAIN_MD_CACHE[domain] = filtered
        print(f"  -> Wrote {md_name}")

