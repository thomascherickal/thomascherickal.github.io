#!/usr/bin/env python3
"""
sync_pricing.py - Centralized Pricing & Parity Synchronization Engine.
Defines authoritative pricing data for the two core offerings:
1. Business Training: $5,000 USD for 60 days (30 days training + 30 days guidance and detailed guides and materials),
   up to 20 participants, +$250 USD per extra seat.
2. Personal Training: $2,500 USD for 30 days (15 days training + 15 days guidance and detailed guides and materials),
   strictly 1 person only.

Clients are encouraged to choose services relevant to their work or interests.
All individual service fees have been removed.

Synchronizes across:
- schema/pricing-schema.json
- pricing.html (embedded JSON-LD, calculator, and guarantee sections)
- schema.json
- 16 service-*.html pages (Section 5/6 & embedded JSON-LD)
- 16 schema/service-*.json files
- personal-ai-training.html
- faqs.html
- index.html
- services-summary.md
- llms.txt
"""

import os
import sys
import json
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
SCHEMA_DIR = os.path.join(REPO_DIR, "schema")

# Authoritative Offers
OFFERS = [
    {
        "id": "individual-service-personal",
        "name": "Individual Service Pick (Personal / 1-on-1)",
        "headline": "$1,000 USD",
        "sub": "USD · PER SERVICE · 5 DAYS TRAINING + 5 DAYS SUPPORT",
        "minPrice": 1000,
        "maxPrice": 1000,
        "description": "Pick individual services at $1,000 USD per service for individuals, covering 5 days of live training plus 5 days of dedicated support (10 days total coverage per service). Strictly 1 person only. Clients can pick and choose their services. Full 195-country Purchasing Power Parity supported: Tier 1 $1,000; Tier 2 (25% off) $750; Tier 3 & 4 (50% off) $500; Tier 5 (75% off) $250; Tier 6 custom subsidized relief ($100).",
        "duration": "5 days live training + 5 days dedicated support (10 days total per service)",
        "cohort": "Strictly 1 person only (private 1-on-1 acceleration)",
        "parity": {
            "tier1": 1000,
            "tier2": 750,
            "tier3": 500,
            "tier4": 500,
            "tier5": 250,
            "tier6": 100
        }
    },
    {
        "id": "individual-service-business",
        "name": "Individual Service Pick (Business / Teams)",
        "headline": "$2,000 USD",
        "sub": "USD · PER SERVICE · 5 DAYS TRAINING + 5 DAYS SUPPORT",
        "minPrice": 2000,
        "maxPrice": 2000,
        "description": "Pick individual services at $2,000 USD per service for businesses, covering 5 days of live training plus 5 days of dedicated support (10 days total coverage per service). Covers cohorts up to 20 participants (additional seats at $250 USD per seat). Clients can pick and choose their services. Full 195-country Purchasing Power Parity supported: Tier 1 $2,000; Tier 2 (25% off) $1,500; Tier 3 & 4 (50% off) $1,000; Tier 5 (75% off) $500; Tier 6 custom subsidized relief ($200).",
        "duration": "5 days live training + 5 days dedicated support (10 days total per service)",
        "cohort": "Up to 20 participants included (+$250 USD per extra seat)",
        "parity": {
            "tier1": 2000,
            "tier2": 1500,
            "tier3": 1000,
            "tier4": 1000,
            "tier5": 500,
            "tier6": 200
        }
    },
    {
        "id": "business-training",
        "name": "Business Training (Comprehensive Program)",
        "headline": "$5,000 USD",
        "sub": "USD · 60 DAYS (30D TRAINING + 30D GUIDANCE & MATERIALS)",
        "minPrice": 5000,
        "maxPrice": 5000,
        "description": "Comprehensive Business Training at $5,000 USD for 60 days (30 days training + 30 days guidance and detailed guides and materials). Includes up to 20 participants. Additional seats at $250 USD per seat. Clients can pick and choose their services. Full 195-country Purchasing Power Parity supported: Tier 2 (25% off) $3,750; Tier 3 & 4 (50% off) $2,500; Tier 5 (75% off) $1,250; Tier 6 custom subsidized relief.",
        "duration": "60 days (30 days live training + 30 days guidance and detailed guides and materials)",
        "cohort": "Up to 20 participants included (+$250 USD per extra seat)",
        "parity": {
            "tier1": 5000,
            "tier2": 3750,
            "tier3": 2500,
            "tier4": 2500,
            "tier5": 1250,
            "tier6": 0
        }
    },
    {
        "id": "personal-training",
        "name": "Personal Training (Comprehensive Program)",
        "headline": "$2,500 USD",
        "sub": "USD · 30 DAYS (15D TRAINING + 15D GUIDANCE & MATERIALS)",
        "minPrice": 2500,
        "maxPrice": 2500,
        "description": "Comprehensive Personal Training at $2,500 USD for 30 days (15 days training + 15 days guidance and detailed guides and materials). Strictly for 1 person only. Clients can pick and choose their services. Full 195-country Purchasing Power Parity supported: Tier 2 (25% off) $1,875; Tier 3 & 4 (50% off) $1,250; Tier 5 (75% off) $625; Tier 6 custom subsidized relief ($100-$250).",
        "duration": "30 days (15 days live training + 15 days guidance and detailed guides and materials)",
        "cohort": "Strictly 1 person only (private 1-on-1 acceleration)",
        "parity": {
            "tier1": 2500,
            "tier2": 1875,
            "tier3": 1250,
            "tier4": 1250,
            "tier5": 625,
            "tier6": 0
        }
    }
]

ALL_SERVICES = [
    {"slug": "service-generative-ai-transformation", "title": "Generative AI Transformation"},
    {"slug": "service-local-llms-cost-slashing", "title": "Local LLMs & Cost Slashing"},
    {"slug": "service-ai-agents-orchestration", "title": "Claude Agent Orchestration"},
    {"slug": "service-agentic-ai-systems", "title": "Hermes Agent Training"},
    {"slug": "service-workflow-automations", "title": "Workflow Automations"},
    {"slug": "service-quantum-applications", "title": "Quantum Applications (Experimental)"},
    {"slug": "service-ai-budget-cost-limits-training", "title": "AI Budget & Cost Control Training"},
    {"slug": "service-enterprise-workforce-ai-training", "title": "Enterprise Workforce AI Training"},
    {"slug": "service-ai-developer-training", "title": "AI Developer Training"},
    {"slug": "service-advanced-ai-agents", "title": "Advanced AI Agents"},
    {"slug": "service-claude-basic-training", "title": "Claude Basic Training"},
    {"slug": "service-claude-advanced-training", "title": "Claude Advanced Training"},
    {"slug": "service-gemini-basic-training", "title": "Google Gemini Basic Training"},
    {"slug": "service-gemini-advanced-training", "title": "Google Gemini Advanced Training"},
    {"slug": "service-deep-research-notebooklm", "title": "Deep Research & NotebookLM Intelligence"},
    {"slug": "service-hybrid-quantum-ai", "title": "Hybrid Quantum-Classical AI Architecture"}
]

def format_price_spec(name, price_val, desc=None):
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

clean_desc = "Transparent AI engagement model: Pick individual services at $1,000 USD per service for individuals and $2,000 USD per service for businesses (each covering 5 days of training + 5 days of dedicated support, 10 days total coverage). Clients can also choose our comprehensive programs: Business Training at $5,000 USD for 60 days (30 days training + 30 days guidance/materials, up to 20 participants, +$250 per extra seat) and Personal Training at $2,500 USD for 30 days (15 days training + 15 days guidance/materials, 1 person only). Full 195-country Purchasing Power Parity (PPP) supported across all service pricing."

def sync_pricing_schema_json():
    pricing_schema_path = os.path.join(SCHEMA_DIR, "pricing-schema.json")
    if not os.path.exists(pricing_schema_path):
        return
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
    print(f"Updated {pricing_schema_path}")

def sync_root_schema_json():
    root_schema_path = os.path.join(REPO_DIR, "schema.json")
    if not os.path.exists(root_schema_path):
        return
    with open(root_schema_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data.get("@graph", []):
        if item.get("@type") == "OfferCatalog" and "pricing" in item.get("@id", "").lower():
            item["description"] = clean_desc
            item["itemListElement"] = build_offer_catalog_elements()
            break
    with open(root_schema_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Updated {root_schema_path}")

def sync_pricing_html_embedded_schema():
    pricing_html_path = os.path.join(REPO_DIR, "pricing.html")
    if not os.path.exists(pricing_html_path):
        return
    with open(pricing_html_path, "r", encoding="utf-8") as f:
        content = f.read()

    schema_pattern = re.compile(r'(<script type="application/ld\+json">)([\s\S]*?)(</script>)')
    match = schema_pattern.search(content)
    if not match:
        print("Could not find JSON-LD in pricing.html")
        return

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
        print("Updated embedded JSON-LD in pricing.html")
    except Exception as e:
        print(f"Error updating pricing.html JSON-LD: {e}")

def get_service_price_specs():
    return [
        format_price_spec("Individual Service Pick (Individual Tier 1 / 100%)", 1000, "5 days live training + 5 days dedicated support (10 days total coverage), 1 person only"),
        format_price_spec("Individual Service Pick (Individual Tier 2 / 25% Off)", 750),
        format_price_spec("Individual Service Pick (Individual Tier 3 & 4 / 50% Off)", 500),
        format_price_spec("Individual Service Pick (Individual Tier 5 / 75% Off)", 250),
        format_price_spec("Individual Service Pick (Individual Tier 6 / Subsidized)", 100),
        format_price_spec("Business Service Pick (Business Tier 1 / 100%)", 2000, "5 days live training + 5 days dedicated support (10 days total coverage), up to 20 participants, +$250/extra seat"),
        format_price_spec("Business Service Pick (Business Tier 2 / 25% Off)", 1500),
        format_price_spec("Business Service Pick (Business Tier 3 & 4 / 50% Off)", 1000),
        format_price_spec("Business Service Pick (Business Tier 5 / 75% Off)", 500),
        format_price_spec("Business Service Pick (Business Tier 6 / Subsidized)", 200),
        format_price_spec("Business Training (Tier 1 / 100%)", 5000, "60 days: 30d training + 30d guidance & materials, up to 20 participants, +$250/extra seat"),
        format_price_spec("Business Training (Tier 2 / 25% Off)", 3750),
        format_price_spec("Business Training (Tier 3 & 4 / 50% Off)", 2500),
        format_price_spec("Business Training (Tier 5 / 75% Off)", 1250),
        format_price_spec("Personal Training (Tier 1 / 100%)", 2500, "30 days: 15d training + 15d guidance & materials, 1 person only"),
        format_price_spec("Personal Training (Tier 2 / 25% Off)", 1875),
        format_price_spec("Personal Training (Tier 3 & 4 / 50% Off)", 1250),
        format_price_spec("Personal Training (Tier 5 / 75% Off)", 625),
        format_price_spec("Tier 6 (Custom Subsidized Relief)", 0, "Custom subsidized quotes and relief for under-resourced teams")
    ]

service_offer_desc = "Available as an individual service pick ($1,000 USD for individuals / $2,000 USD for businesses, each for 5 days of live training + 5 days of dedicated support, 10 days total coverage), or as part of our comprehensive programs: Business Training ($5,000 USD for 60 days: 30d training + 30d guidance/materials, up to 20 seats, +$250/extra seat) or Personal Training ($2,500 USD for 30 days: 15d training + 15d guidance/materials, 1 person only). Clients are free to pick and choose their services. Full 195-country Purchasing Power Parity (PPP) supported across all service pricing."

def sync_service_schema_files():
    offer_specs = get_service_price_specs()
    for s in ALL_SERVICES:
        slug = s["slug"]
        schema_path = os.path.join(SCHEMA_DIR, f"{slug}.json")
        if os.path.exists(schema_path):
            with open(schema_path, "r", encoding="utf-8") as f:
                s_data = json.load(f)
            for item in s_data.get("@graph", []):
                if item.get("@type") == "Service" and "offers" in item:
                    item["offers"]["price"] = "1000.00"
                    item["offers"]["description"] = service_offer_desc
                    item["offers"]["priceSpecification"] = offer_specs
            with open(schema_path, "w", encoding="utf-8") as f:
                json.dump(s_data, f, indent=2, ensure_ascii=False)
                f.write("\n")
            print(f"Updated {schema_path}")

def sync_service_html_pages():
    offer_specs = get_service_price_specs()

    pricing_section_html = """
        <ul style="color:var(--text-secondary);font-size:0.95rem;line-height:1.8;margin-left:1.5rem;list-style:disc;">
          <li><strong>Modular Service Selection &amp; Flexible Engagement:</strong> Clients can freely pick and choose their services, either individually or bundled under our comprehensive transformation programs:
            <ul style="margin-left:1.5rem;margin-top:0.5rem;list-style:circle;">
              <li><strong>Individual Service Pick (Personal / 1-on-1):</strong> <strong>$1,000 USD</strong> per service for <strong>5 days of live training + 5 days of dedicated support</strong> (10 days total coverage per service). Strictly 1 person only.</li>
              <li><strong>Individual Service Pick (Business / Teams):</strong> <strong>$2,000 USD</strong> per service for <strong>5 days of live training + 5 days of dedicated support</strong> (10 days total coverage per service). Covers up to 20 participants (additional seats at <strong>$250 USD per seat</strong>).</li>
              <li><strong>Comprehensive Business Training:</strong> <strong>$5,000 USD</strong> for <strong>60 days</strong> (30 days live training + 30 days guidance and detailed guides and materials). Includes cohorts up to 20 participants. Extra seats at <strong>$250 USD per seat</strong>).</li>
              <li><strong>Comprehensive Personal Training:</strong> <strong>$2,500 USD</strong> for <strong>30 days</strong> (15 days live training + 15 days guidance and detailed guides and materials). Strictly for 1 person only (private acceleration).</li>
            </ul>
          </li>
          <li><strong>Global Purchasing Power Parity (PPP) Available for All Service Pricing:</strong>
            <ul style="margin-left:1.5rem;margin-top:0.5rem;list-style:circle;">
              <li><strong>Tier 1 (Standard / 100%):</strong> Individual $1,000 USD | Business $2,000 USD | Business 60d $5,000 USD | Personal 30d $2,500 USD (USA, UK, EU, Canada, Australia)</li>
              <li><strong>Tier 2 (Developing Nations / 25% Off):</strong> Individual $750 USD | Business $1,500 USD | Business 60d $3,750 USD | Personal 30d $1,875 USD (India, Brazil, Mexico, UAE, Poland)</li>
              <li><strong>Tier 3 &amp; 4 (Lower-Middle/Low-Income &amp; Startups / 50% Off):</strong> Individual $500 USD | Business $1,000 USD | Business 60d $2,500 USD | Personal 30d $1,250 USD</li>
              <li><strong>Tier 5 (Academia &amp; Non-Profits / 75% Off):</strong> Individual $250 USD | Business $500 USD | Business 60d $1,250 USD | Personal 30d $625 USD</li>
              <li><strong>Tier 6 (Crisis-Affected &amp; Under-Resourced Regions):</strong> Individual $100 USD | Business $200 USD | Custom Subsidized Relief / Micro-Rates</li>
              <li>Calculate and lock in your exact tier rate live on the <a href="pricing.html" style="color:var(--cyan);text-decoration:underline;">Pricing page</a>.</li>
            </ul>
          </li>
          <li><strong>Payment Schedule:</strong>
            <ul style="margin-left:1.5rem;margin-top:0.5rem;list-style:circle;">
              <li>Zero upfront advance; billed in 5 equal milestone installments due only after verified instruction and runtime delivery.</li>
            </ul>
          </li>
        </ul>"""

    for s in ALL_SERVICES:
        slug = s["slug"]
        html_path = os.path.join(REPO_DIR, f"{slug}.html")
        if not os.path.exists(html_path):
            continue
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        # 1. Update embedded JSON-LD
        match = re.search(r'(<script type="application/ld\+json">)([\s\S]*?)(</script>)', html_content)
        if match:
            try:
                s_data = json.loads(match.group(2).strip())
                for item in s_data.get("@graph", []):
                    if item.get("@type") == "Service" and "offers" in item:
                        item["offers"]["price"] = "1000.00"
                        item["offers"]["description"] = service_offer_desc
                        item["offers"]["priceSpecification"] = offer_specs
                new_json_str = json.dumps(s_data, indent=2, ensure_ascii=False)
                html_content = html_content[:match.start(2)] + "\n  " + new_json_str.replace("\n", "\n  ") + "\n  " + html_content[match.end(2):]
            except Exception as e:
                print(f"Error parsing JSON-LD in {html_path}: {e}")

        # 2. Update Investment & Pricing Quote Section
        section_pattern = re.compile(
            r'(<h2[^>]*>(?:5|6)\.\s*Investment[\s\S]*?</h2>)\s*<ul[\s\S]*?</ul>\s*(?=<h2|</section|</main|<div\s+class="cta|\Z)',
            re.IGNORECASE
        )
        if section_pattern.search(html_content):
            html_content = section_pattern.sub(r'\1\n' + pricing_section_html + '\n\n        ', html_content)

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"Updated service page: {slug}.html")

def sync_llms_txt():
    llms_path = os.path.join(REPO_DIR, "llms.txt")
    if not os.path.exists(llms_path):
        return
    with open(llms_path, "r", encoding="utf-8") as f:
        content = f.read()

    new_block = """- **Individual Service Picks**: Clients can pick and choose their services:
  * **Individual (1-on-1)**: **$1,000 USD** per service for **5 days live training + 5 days dedicated support** (10 days total coverage per service). Strictly 1 person only.
 - Parity: Tier 1: $1,000 | Tier 2 (25% off): $750 | Tier 3 & 4 (50% off): $500 | Tier 5 (75% off): $250 | Tier 6 (Subsidized): $100.
  * **Business (Teams)**: **$2,000 USD** per service for **5 days live training + 5 days dedicated support** (10 days total coverage per service). Covers up to 20 participants (additional seats at **$250 USD per seat**).
 - Parity: Tier 1: $2,000 | Tier 2 (25% off): $1,500 | Tier 3 & 4 (50% off): $1,000 | Tier 5 (75% off): $500 | Tier 6 (Subsidized): $200.
- **Comprehensive Business Training**: **$5,000 USD** for **60 days** (30 days live hands-on engineering training + 30 days guidance and detailed guides and materials). Covers cohorts up to 20 participants. Additional seats at **$250 USD per seat**. Clients are encouraged to choose services relevant to their work or interests.
  * Parity: Tier 1: $5,000 | Tier 2 (25% off): $3,750 | Tier 3 & 4 (50% off): $2,500 | Tier 5 (75% off): $1,250 | Tier 6: Custom Subsidized Relief.
- **Comprehensive Personal Training**: **$2,500 USD** for **30 days** (15 days intensive 1-on-1 sprint + 15 days guidance and detailed guides and materials). Strictly for **one person only**. Clients are encouraged to choose services relevant to their work or interests.
  * Parity: Tier 1: $2,500 | Tier 2 (25% off): $1,875 | Tier 3 & 4 (50% off): $1,250 | Tier 5 (75% off): $625 | Tier 6: Custom Subsidized Relief ($100-$250).
- **Purchasing Power Parity (PPP)**: Available across all service pricing, covering 195 countries and dedicated relief tiers."""

    pattern = re.compile(
        r'-\s*\*\*(?:Individual Service Picks|Core Strategic Packages|Enterprise Services|Business Training)[\s\S]*?(?=\n\n|\n###|\Z)',
        re.MULTILINE
    )
    if pattern.search(content):
        content = pattern.sub(new_block, content)
        with open(llms_path, "w", encoding="utf-8") as f:
            f.write(content)
        print("Updated llms.txt")

def sync_all_pricing():
    print("--- Starting Central Pricing & Parity Sync Pass ---")
    sync_pricing_schema_json()
    sync_root_schema_json()
    sync_pricing_html_embedded_schema()
    sync_service_schema_files()
    sync_service_html_pages()
    sync_llms_txt()
    print("--- Central Pricing & Parity Sync Pass Complete ---")

if __name__ == "__main__":
    sync_all_pricing()
