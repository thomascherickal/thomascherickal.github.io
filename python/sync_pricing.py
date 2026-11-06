#!/usr/bin/env python3
"""
sync_pricing.py - Centralized Pricing & Parity Schema Synchronization Utility.

Defines authoritative pricing structures aligned with GEMINI.md and data/pricing.json:
1. Business AI Training Flagship: $3000 USD for 60 days total (30 days live engineering
   training + 30 days dedicated support), up to 10 participants (+$300 USD per extra seat
   after 10 participants, up to 20 seats max).
2. Personal AI Training Flagship: $3000 USD for 60 days total (30 days live 1-on-1
   training + 30 days dedicated support), strictly 1 person only.
3. Individual Service Picks: $300 USD base for 1 seat (5 days live training + 5 days
   dedicated support; for business cohorts, +$300 USD per extra seat up to 10 seats max).

Strict Repo Compliance:
- Zero emojis.
- Zero em-dashes (spaced hyphens ' - ' or plain hyphens '-' used).
"""

import json
import os
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
SCHEMA_DIR = os.path.join(REPO_DIR, "schema")

OFFERS = [
    {
        "id": "individual-service-personal",
        "name": "Individual Service Pick (Personal / 1-on-1)",
        "headline": "$300 USD",
        "sub": "USD - PER SERVICE - 5 DAYS TRAINING + 5 DAYS SUPPORT",
        "minPrice": 300,
        "maxPrice": 300,
        "description": "Pick individual services at $300 USD per service for individuals, covering 5 days of live training plus 5 days of dedicated support (10 days total coverage per service). Strictly 1 person only. Full 195-country Purchasing Power Parity supported across 6 tiers.",
        "duration": "5 days live training + 5 days dedicated support (10 days total per service)",
        "cohort": "Strictly 1 person only (private 1-on-1 acceleration)",
        "parity": {
            "tier1": 300,
            "tier2": 225,
            "tier3": 150,
            "tier4": 150,
            "tier5": 75,
            "tier6": 0
        }
    },
    {
        "id": "individual-service-business",
        "name": "Individual Service Pick (Business / Teams)",
        "headline": "$300 USD",
        "sub": "USD - PER SERVICE - 5 DAYS TRAINING + 5 DAYS SUPPORT",
        "minPrice": 300,
        "maxPrice": 300,
        "description": "Pick individual services at $300 USD base for 1 seat, with additional seats at $300 USD per seat up to a maximum of 10 seats, covering 5 days of live training plus 5 days of dedicated support (10 days total coverage per service). Full 195-country Purchasing Power Parity supported across 6 tiers.",
        "duration": "5 days live training + 5 days dedicated support (10 days total per service)",
        "cohort": "1 seat base (+$300 USD per extra seat up to 10 seats max)",
        "parity": {
            "tier1": 300,
            "tier2": 225,
            "tier3": 150,
            "tier4": 150,
            "tier5": 75,
            "tier6": 0
        }
    },
    {
        "id": "business-training",
        "name": "Business AI Training Flagship",
        "headline": "$3000 USD",
        "sub": "USD - 60 DAYS (30D LIVE TRAINING + 30D DEDICATED SUPPORT)",
        "minPrice": 3000,
        "maxPrice": 3000,
        "description": "Business AI Training Flagship at $3000 USD for 60 days total (30 days live engineering training + 30 days dedicated support, 4 hours per day, Monday through Friday). Includes up to 10 participants. Extra seats at $300 USD per seat after 10 participants, up to 20 seats max. Full 195-country Purchasing Power Parity supported: Tier 1 ($3000), Tier 2 ($2250), Tier 3 ($1500), Tier 4 ($1500), Tier 5 ($750), Tier 6 (Custom).",
        "duration": "60 days total (30 days live training + 30 days dedicated support, 4 hours per day)",
        "cohort": "Up to 10 participants included (+$300 USD per extra seat up to 20 max)",
        "parity": {
            "tier1": 3000,
            "tier2": 2250,
            "tier3": 1500,
            "tier4": 1500,
            "tier5": 750,
            "tier6": 0
        }
    },
    {
        "id": "personal-training",
        "name": "Personal AI Training Flagship",
        "headline": "$3000 USD",
        "sub": "USD - 60 DAYS (30D LIVE TRAINING + 30D DEDICATED SUPPORT)",
        "minPrice": 3000,
        "maxPrice": 3000,
        "description": "Personal AI Training Flagship at $3000 USD for 60 days total (30 days live 1-on-1 training + 30 days dedicated support, 4 hours per day, Monday through Friday). Strictly one person only. Full 195-country Purchasing Power Parity supported: Tier 1 ($3000), Tier 2 ($2250), Tier 3 ($1500), Tier 4 ($1500), Tier 5 ($750), Tier 6 (Custom).",
        "duration": "60 days total (30 days live training + 30 days dedicated support, 4 hours per day)",
        "cohort": "Strictly one person only (dedicated private acceleration)",
        "parity": {
            "tier1": 3000,
            "tier2": 2250,
            "tier3": 1500,
            "tier4": 1500,
            "tier5": 750,
            "tier6": 0
        }
    }
]

ALL_SERVICES = [
    {"slug": "service-generative-ai-transformation", "title": "Generative AI Transformation"},
    {"slug": "service-local-llms-cost-slashing", "title": "Local LLMs & Cost Slashing"},
    {"slug": "service-ai-agents-orchestration", "title": "Claude Agent Orchestration"},
    {"slug": "service-agentic-ai-systems", "title": "Agentic Assistants Training"},
    {"slug": "service-workflow-automations", "title": "Workflow Automations"},
    {"slug": "service-ai-budget-cost-limits-training", "title": "AI Budget & Cost Control Training"},
    {"slug": "service-ai-developer-training", "title": "AI Developer Training"},
    {"slug": "service-advanced-ai-agents", "title": "Advanced AI Agents"},
    {"slug": "service-claude-advanced-training", "title": "Claude Advanced Training"},
    {"slug": "service-gemini-advanced-training", "title": "Google Gemini Advanced Training"},
    {"slug": "service-ai-based-competitor-research", "title": "Competitor Research & Intelligence"},
    {"slug": "service-ai-assisted-leads-magnet", "title": "AI Leads Magnet & Conversion Systems"},
    {"slug": "service-goals-self-authoring", "title": "Personal Goals Self-Authoring"}
]

clean_desc = (
    "Transparent AI engagement model: Individual Service Picks at $300 USD base rate "
    "(covering 5 days live training + 5 days dedicated support). Flagship programs: "
    "Business AI Training Flagship at $3000 USD for 60 days total (30 days live + 30 days "
    "dedicated support, up to 10 participants, +$300 USD per extra seat up to 20 max) and "
    "Personal AI Training Flagship at $3000 USD for 60 days total (30 days live + 30 days "
    "dedicated support, strictly 1 person only). $0 upfront deposit, invoiced in 5 equal "
    "20% milestone installments. 6 Purchasing Power Parity (PPP) tiers across 195 countries: "
    "Tier 1 ($3000), Tier 2 ($2250), Tier 3 ($1500), Tier 4 ($1500), Tier 5 ($750), Tier 6 (Custom)."
)


def format_price_spec(name, price_val, desc=None):
    """Formats a JSON-LD PriceSpecification object."""
    if price_val == 0:
        return {
            "@type": "PriceSpecification",
            "name": name,
            "price": "0.00",
            "priceCurrency": "USD",
            "description": desc or "Custom subsidized quotes and micro-rates"
        }
    spec = {
        "@type": "PriceSpecification",
        "name": name,
        "price": f"{price_val:.2f}",
        "priceCurrency": "USD"
    }
    if desc:
        spec["description"] = desc
    return spec


def build_offer_catalog_elements():
    """Builds JSON-LD OfferCatalog itemListElement array from OFFERS."""
    elements = []
    for o in OFFERS:
        par = o["parity"]
        price_specs = [
            format_price_spec("Tier 1 (Standard / 100%)", par["tier1"]),
            format_price_spec("Tier 2 (Developing / 25% Off)", par["tier2"]),
            format_price_spec("Tier 3 & 4 (Lower-Middle/Low-Income & Startups / 50% Off)", par["tier3"]),
            format_price_spec("Tier 5 (Academia & Non-Profits / 75% Off)", par["tier5"]),
            format_price_spec("Tier 6 (Custom Subsidized Relief)", 0, "Micro-rates and customized quotes for under-resourced teams")
        ]
        offer = {
            "@type": "Offer",
            "name": o["name"],
            "description": o["description"],
            "price": f"{o['minPrice']:.2f}",
            "priceCurrency": "USD",
            "priceSpecification": price_specs
        }
        elements.append(offer)
    return elements


def get_service_price_specs():
    """Returns full list of PriceSpecification entries for service schemas."""
    return [
        format_price_spec("Individual Service Pick (Tier 1 / 100%)", 300, "5 days live training + 5 days dedicated support (10 days total)"),
        format_price_spec("Individual Service Pick (Tier 2 / 25% Off)", 225),
        format_price_spec("Individual Service Pick (Tier 3 & 4 / 50% Off)", 150),
        format_price_spec("Individual Service Pick (Tier 5 / 75% Off)", 75),
        format_price_spec("Flagship AI Training (Tier 1 / 100%)", 3000, "60 days total: 30 days live training + 30 days dedicated support"),
        format_price_spec("Flagship AI Training (Tier 2 / 25% Off)", 2250),
        format_price_spec("Flagship AI Training (Tier 3 & 4 / 50% Off)", 1500),
        format_price_spec("Flagship AI Training (Tier 5 / 75% Off)", 750),
        format_price_spec("Tier 6 (Custom Subsidized Relief)", 0, "Custom subsidized quotes and relief for under-resourced teams")
    ]


def sync_pricing_schema_json(schema_dir=None):
    """Updates schema/pricing-schema.json with authoritative OfferCatalog."""
    target_dir = schema_dir or SCHEMA_DIR
    pricing_schema_path = os.path.join(target_dir, "pricing-schema.json")
    if not os.path.exists(pricing_schema_path):
        return False
    with open(pricing_schema_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data.get("@graph", []):
        if item.get("@type") == "WebPage":
            item["description"] = clean_desc
        elif item.get("@type") == "OfferCatalog":
            item["description"] = clean_desc
            item["itemListElement"] = build_offer_catalog_elements()
    with open(pricing_schema_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return True


def sync_home_schema_json(schema_dir=None):
    """Updates schema/home-schema.json OfferCatalog."""
    target_dir = schema_dir or SCHEMA_DIR
    home_schema_path = os.path.join(target_dir, "home-schema.json")
    if not os.path.exists(home_schema_path):
        return False
    with open(home_schema_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    updated = False
    for item in data.get("@graph", []):
        if item.get("@type") == "OfferCatalog" and "pricing" in item.get("@id", "").lower():
            item["description"] = clean_desc
            item["itemListElement"] = build_offer_catalog_elements()
            updated = True
            break
    if updated:
        with open(home_schema_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
    return updated


def sync_pricing_html_embedded_schema(repo_dir=None):
    """Updates embedded JSON-LD inside pricing.html."""
    target_repo = repo_dir or REPO_DIR
    pricing_html_path = os.path.join(target_repo, "pricing.html")
    if not os.path.exists(pricing_html_path):
        return False
    with open(pricing_html_path, "r", encoding="utf-8") as f:
        content = f.read()

    schema_pattern = re.compile(r'(<script type="application/ld\+json">)([\s\S]*?)(</script>)')
    match = schema_pattern.search(content)
    if not match:
        return False

    try:
        data = json.loads(match.group(2).strip())
        for item in data.get("@graph", []):
            if item.get("@type") == "WebPage":
                item["description"] = clean_desc
            elif item.get("@type") == "OfferCatalog":
                item["description"] = clean_desc
                item["itemListElement"] = build_offer_catalog_elements()
        new_json_str = json.dumps(data, indent=2, ensure_ascii=False)
        new_content = content[:match.start(2)] + "\n  " + new_json_str.replace("\n", "\n  ") + "\n  " + content[match.end(2):]
        with open(pricing_html_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        return True
    except Exception:
        return False


def sync_service_schema_files(schema_dir=None):
    """Updates service-*.json schema files in schema_dir."""
    target_dir = schema_dir or SCHEMA_DIR
    offer_specs = get_service_price_specs()
    updated_count = 0
    for s in ALL_SERVICES:
        slug = s["slug"]
        schema_path = os.path.join(target_dir, f"{slug}.json")
        if os.path.exists(schema_path):
            with open(schema_path, "r", encoding="utf-8") as f:
                s_data = json.load(f)
            for item in s_data.get("@graph", []):
                if item.get("@type") == "Service" and "offers" in item:
                    item["offers"]["price"] = "300.00"
                    item["offers"]["description"] = clean_desc
                    item["offers"]["priceSpecification"] = offer_specs
            with open(schema_path, "w", encoding="utf-8") as f:
                json.dump(s_data, f, indent=2, ensure_ascii=False)
                f.write("\n")
            updated_count += 1
    return updated_count


def sync_all_pricing(repo_dir=None, schema_dir=None):
    """Runs all schema pricing synchronization passes."""
    s_dir = schema_dir or (os.path.join(repo_dir, "schema") if repo_dir else SCHEMA_DIR)
    r_dir = repo_dir or REPO_DIR
    sync_pricing_schema_json(schema_dir=s_dir)
    sync_home_schema_json(schema_dir=s_dir)
    sync_pricing_html_embedded_schema(repo_dir=r_dir)
    sync_service_schema_files(schema_dir=s_dir)
    return True


if __name__ == "__main__":
    sync_all_pricing()
