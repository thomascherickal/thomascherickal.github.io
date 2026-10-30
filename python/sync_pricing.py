#!/usr/bin/env python3
"""
sync_pricing.py - Centralized Pricing & Parity Synchronization Engine.
Defines authoritative pricing data for the offerings:
1. Business Training: $2,000 USD for 40 days (20 days training + 20 days guidance, Q&A and written guides, 20 working days over 4 weeks with 8 - 10 recovery days, 4 hours per day),
   up to 20 participants, +$100 USD per extra seat.
2. Personal Training: $1,000 USD for 40 days (20 days training + 20 days guidance, Q&A and written guides, 20 working days over 4 weeks with 8 - 10 recovery days, 4 hours per day),
   strictly 1 person only.
3. Individual Service Picks: $200 USD per service for businesses (up to 20 participants, +$100/extra seat);
   $100 USD per service for individuals (strictly 1 person).

Clients are encouraged to choose services relevant to their work or interests.

Synchronizes across:
- schema/pricing-schema.json
- pricing.html (embedded JSON-LD, calculator, and guarantee sections)
- schema/home-schema.json
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
        "headline": "$100 USD",
        "sub": "USD · PER SERVICE · 5 DAYS TRAINING + 5 DAYS SUPPORT",
        "minPrice": 100,
        "maxPrice": 100,
        "description": "Pick individual services at $100 USD per service for individuals, covering 5 days of live training plus 5 days of dedicated support (10 days total coverage per service). Strictly 1 person only. Clients can pick and choose their services. Full 195-country Purchasing Power Parity supported: Tier 1 $100; Tier 2 (25% off) $75; Tier 3 & 4 (50% off) $50; Tier 5 (75% off) $25; Tier 6 custom subsidized relief ($10).",
        "duration": "5 days live training + 5 days dedicated support (10 days total per service)",
        "cohort": "Strictly 1 person only (private 1-on-1 acceleration)",
        "parity": {
            "tier1": 100,
            "tier2": 75,
            "tier3": 50,
            "tier4": 50,
            "tier5": 25,
            "tier6": 10
        }
    },
    {
        "id": "individual-service-business",
        "name": "Individual Service Pick (Business / Teams)",
        "headline": "$200 USD",
        "sub": "USD · PER SERVICE · 5 DAYS TRAINING + 5 DAYS SUPPORT",
        "minPrice": 200,
        "maxPrice": 200,
        "description": "Pick individual services at $200 USD per service for businesses, covering 5 days of live training plus 5 days of dedicated support (10 days total coverage per service). Covers cohorts up to 20 participants (additional seats at $100 USD per seat). Clients can pick and choose their services. Full 195-country Purchasing Power Parity supported: Tier 1 $200; Tier 2 (25% off) $150; Tier 3 & 4 (50% off) $100; Tier 5 (75% off) $50; Tier 6 custom subsidized relief ($20).",
        "duration": "5 days live training + 5 days dedicated support (10 days total per service)",
        "cohort": "Up to 20 participants included (+$100 USD per extra seat)",
        "parity": {
            "tier1": 200,
            "tier2": 150,
            "tier3": 100,
            "tier4": 100,
            "tier5": 50,
            "tier6": 20
        }
    },
    {
        "id": "business-training",
        "name": "Business Training (Comprehensive Program)",
        "headline": "$2,000 USD",
        "sub": "USD · 40 DAYS (20D TRAINING + 20D GUIDANCE & MATERIALS)",
        "minPrice": 2000,
        "maxPrice": 2000,
        "description": "Comprehensive Business Training at $2,000 USD for 40 days (20 days training + 20 days guidance, Q&A and written guides, 20 working days over 4 weeks with 8 - 10 recovery days, 4 hours per day). Includes up to 20 participants. Additional seats at $100 USD per seat. Clients can pick and choose their services. Full 195-country Purchasing Power Parity supported: Tier 1 $2,000; Tier 2 (25% off) $1,500; Tier 3 & 4 (50% off) $1,000; Tier 5 (75% off) $500; Tier 6 custom subsidized relief.",
        "duration": "40 days (20 days live training + 20 days guidance, Q&A and written guides, 20 working days over 4 weeks with 8 - 10 recovery days, 4 hours per day)",
        "cohort": "Up to 20 participants included (+$100 USD per extra seat)",
        "parity": {
            "tier1": 2000,
            "tier2": 1500,
            "tier3": 1000,
            "tier4": 1000,
            "tier5": 500,
            "tier6": 0
        }
    },
    {
        "id": "personal-training",
        "name": "Personal Training (Comprehensive Program)",
        "headline": "$1,000 USD",
        "sub": "USD · 40 DAYS (20D TRAINING + 20D GUIDANCE & MATERIALS)",
        "minPrice": 1000,
        "maxPrice": 1000,
        "description": "Comprehensive Personal Training at $1,000 USD for 40 days (20 days training + 20 days guidance, Q&A and written guides, 20 working days over 4 weeks with 8 - 10 recovery days, 4 hours per day). Strictly for 1 person only. Clients can pick and choose their services. Full 195-country Purchasing Power Parity supported: Tier 1 $1,000; Tier 2 (25% off) $750; Tier 3 & 4 (50% off) $500; Tier 5 (75% off) $250; Tier 6 custom subsidized relief ($100).",
        "duration": "40 days (20 days live training + 20 days guidance, Q&A and written guides, 20 working days over 4 weeks with 8 - 10 recovery days, 4 hours per day)",
        "cohort": "Strictly 1 person only (private 1-on-1 acceleration)",
        "parity": {
            "tier1": 1000,
            "tier2": 750,
            "tier3": 500,
            "tier4": 500,
            "tier5": 250,
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
    {"slug": "service-ai-budget-cost-limits-training", "title": "AI Budget & Cost Control Training"},
    {"slug": "service-ai-developer-training", "title": "AI Developer Training"},
    {"slug": "service-advanced-ai-agents", "title": "Advanced AI Agents"},
    {"slug": "service-claude-advanced-training", "title": "Claude Advanced Training"},
    {"slug": "service-gemini-advanced-training", "title": "Google Gemini Advanced Training"},
    {"slug": "service-competitor-research", "title": "Competitor Research & Intelligence"},
    {"slug": "service-leads-magnet", "title": "AI Leads Magnet & Conversion Systems"}
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

clean_desc = "Transparent AI engagement model: Pick individual services at $100 USD per service for individuals and $200 USD per service for businesses (each covering 5 days of training + 5 days of dedicated support, 10 days total coverage). Clients can also choose our comprehensive programs: Business Training at $2,000 USD for 40 days (20 days training + 20 days guidance/materials, 20 working days over 4 weeks with 8 - 10 recovery days, 4 hours per day, up to 20 participants, +$100 per extra seat) and Personal Training at $1,000 USD for 40 days (20 days training + 20 days guidance/materials, 20 working days over 4 weeks with 8 - 10 recovery days, 4 hours per day, 1 person only). Full 195-country Purchasing Power Parity (PPP) supported across all service pricing."

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

def sync_home_schema_json():
    home_schema_path = os.path.join(SCHEMA_DIR, "home-schema.json")
    if not os.path.exists(home_schema_path):
        return
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
        print(f"Updated {home_schema_path}")

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
        format_price_spec("Individual Service Pick (Individual Tier 1 / 100%)", 100, "5 days live training + 5 days dedicated support (10 days total coverage), 1 person only"),
        format_price_spec("Individual Service Pick (Individual Tier 2 / 25% Off)", 75),
        format_price_spec("Individual Service Pick (Individual Tier 3 & 4 / 50% Off)", 50),
        format_price_spec("Individual Service Pick (Individual Tier 5 / 75% Off)", 25),
        format_price_spec("Individual Service Pick (Individual Tier 6 / Subsidized)", 10),
        format_price_spec("Business Service Pick (Business Tier 1 / 100%)", 200, "5 days live training + 5 days dedicated support (10 days total coverage), up to 20 participants, +$100/extra seat"),
        format_price_spec("Business Service Pick (Business Tier 2 / 25% Off)", 150),
        format_price_spec("Business Service Pick (Business Tier 3 & 4 / 50% Off)", 100),
        format_price_spec("Business Service Pick (Business Tier 5 / 75% Off)", 50),
        format_price_spec("Business Service Pick (Business Tier 6 / Subsidized)", 20),
        format_price_spec("Business Training (Tier 1 / 100%)", 2000, "40 days: 20d training + 20d guidance & materials (20 working days over 4 weeks, 4 hours/day), up to 20 participants, +$100/extra seat"),
        format_price_spec("Business Training (Tier 2 / 25% Off)", 1500),
        format_price_spec("Business Training (Tier 3 & 4 / 50% Off)", 1000),
        format_price_spec("Business Training (Tier 5 / 75% Off)", 500),
        format_price_spec("Personal Training (Tier 1 / 100%)", 1000, "40 days: 20d training + 20d guidance & materials (20 working days over 4 weeks, 4 hours/day), 1 person only"),
        format_price_spec("Personal Training (Tier 2 / 25% Off)", 750),
        format_price_spec("Personal Training (Tier 3 & 4 / 50% Off)", 500),
        format_price_spec("Personal Training (Tier 5 / 75% Off)", 250),
        format_price_spec("Tier 6 (Custom Subsidized Relief)", 0, "Custom subsidized quotes and relief for under-resourced teams")
    ]

service_offer_desc = "Available as an individual service pick ($100 USD for individuals / $200 USD for businesses, each for 5 days of live training + 5 days of dedicated support, 10 days total coverage), or as part of our comprehensive programs: Business Training ($2,000 USD for 40 days: 20d training + 20d guidance/materials, 20 working days over 4 weeks with 8 - 10 recovery days, 4 hours/day, up to 20 seats, +$100/extra seat) or Personal Training ($1,000 USD for 40 days: 20d training + 20d guidance/materials, 20 working days over 4 weeks with 8 - 10 recovery days, 4 hours/day, 1 person only). Clients are free to pick and choose their services. Full 195-country Purchasing Power Parity (PPP) supported across all service pricing."

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
                    item["offers"]["price"] = "100.00"
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
              <li><strong>Individual Service Pick (Personal / 1-on-1):</strong> <strong>$100 USD</strong> per service for <strong>5 days of live training + 5 days of dedicated support</strong> (10 days total coverage per service). Strictly 1 person only.</li>
              <li><strong>Individual Service Pick (Business / Teams):</strong> <strong>$200 USD</strong> per service for <strong>5 days of live training + 5 days of dedicated support</strong> (10 days total coverage per service). Covers up to 20 participants (additional seats at <strong>$100 USD per seat</strong>).</li>
              <li><strong>Comprehensive Business Training:</strong> <strong>$2,000 USD</strong> for <strong>40 days</strong> (20 days live training + 20 days guidance, Q&A and written guides, 20 working days over 4 weeks with 8 - 10 recovery days, 4 hours per day). Includes cohorts up to 20 participants. Extra seats at <strong>$100 USD per seat</strong>).</li>
              <li><strong>Comprehensive Personal Training:</strong> <strong>$1,000 USD</strong> for <strong>40 days</strong> (20 days live training + 20 days guidance, Q&A and written guides, 20 working days over 4 weeks with 8 - 10 recovery days, 4 hours per day). Strictly for 1 person only (private acceleration).</li>
            </ul>
          </li>
          <li><strong>Global Purchasing Power Parity (PPP) Available for All Service Pricing:</strong>
            <ul style="margin-left:1.5rem;margin-top:0.5rem;list-style:circle;">
              <li><strong>Tier 1 (Standard / 100%):</strong> Individual $100 USD | Business $200 USD | Business 40d $2,000 USD | Personal 40d $1,000 USD (USA, UK, EU, Canada, Australia)</li>
              <li><strong>Tier 2 (Upper-Middle-Income / 25% Off):</strong> Individual $75 USD | Business $150 USD | Business 40d $1,500 USD | Personal 40d $750 USD: e.g. Brazil, Mexico, Malaysia, Thailand, Türkiye</li>
              <li><strong>Tier 3 &amp; 4 (Lower-Middle/Low-Income &amp; Startups / 50% Off):</strong> Individual $50 USD | Business $100 USD | Business 40d $1,000 USD | Personal 40d $500 USD</li>
              <li><strong>Tier 5 (Academia &amp; Non-Profits / 75% Off):</strong> Individual $25 USD | Business $50 USD | Business 40d $500 USD | Personal 40d $250 USD</li>
              <li><strong>Tier 6 (Crisis-Affected &amp; Under-Resourced Regions):</strong> Individual $10 USD | Business $20 USD | Custom Subsidized Relief / Micro-Rates</li>
              <li>Calculate and lock in your exact tier rate live on the <a href="pricing.html" style="color:var(--cyan);text-decoration:underline;">Pricing page</a>.</li>
            </ul>
          </li>
          <li><strong>Payment Schedule:</strong>
            <ul style="margin-left:1.5rem;margin-top:0.5rem;list-style:circle;">
              <li>Zero upfront advance; billed in 5 equal milestone installments due only after verified instruction and runtime delivery.</li>
            </ul>
          </li>
        </ul>"""

    target_paths = []
    for s in ALL_SERVICES:
        slug = s["slug"]
        for candidate in [
            os.path.join(REPO_DIR, f"{slug}.html"),
            os.path.join(REPO_DIR, "business-services", f"business-{slug}.html"),
            os.path.join(REPO_DIR, "personal-services", f"personal-{slug}.html")
        ]:
            if os.path.exists(candidate):
                target_paths.append(candidate)

    for html_path in target_paths:
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        is_subdir = os.path.dirname(os.path.abspath(html_path)) != os.path.abspath(REPO_DIR)
        pricing_href = "../pricing.html" if is_subdir else "pricing.html"
        current_pricing_section = pricing_section_html.replace('href="pricing.html"', f'href="{pricing_href}"')

        # 1. Update embedded JSON-LD
        match = re.search(r'(<script type="application/ld\+json">)([\s\S]*?)(</script>)', html_content)
        if match:
            try:
                s_data = json.loads(match.group(2).strip())
                for item in s_data.get("@graph", []):
                    if item.get("@type") == "Service" and "offers" in item:
                        item["offers"]["price"] = "100.00"
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
            html_content = section_pattern.sub(r'\1\n' + current_pricing_section + '\n\n        ', html_content)

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"Updated service page: {os.path.basename(html_path)}")

def sync_llms_txt():
    llms_path = os.path.join(REPO_DIR, "llms.txt")
    if not os.path.exists(llms_path):
        return
    with open(llms_path, "r", encoding="utf-8") as f:
        content = f.read()

    new_block = """- **Individual Service Picks**: Clients can pick and choose their services:
  * **Individual (1-on-1)**: **$100 USD** per service for **5 days live training + 5 days dedicated support** (10 days total coverage per service). Strictly 1 person only.
 - Parity: Tier 1: $100 | Tier 2 (25% off): $75 | Tier 3 & 4 (50% off): $50 | Tier 5 (75% off): $25 | Tier 6 (Subsidized): $10.
  * **Business (Teams)**: **$200 USD** per service for **5 days live training + 5 days dedicated support** (10 days total coverage per service). Covers up to 20 participants (additional seats at **$100 USD per seat**).
 - Parity: Tier 1: $200 | Tier 2 (25% off): $150 | Tier 3 & 4 (50% off): $100 | Tier 5 (75% off): $50 | Tier 6 (Subsidized): $20.
- **Comprehensive Business Training**: **$2,000 USD** for **40 days** (20 days live hands-on engineering training + 20 days guidance, Q&A and written guides, 20 working days over 4 weeks with 8 - 10 recovery days, 4 hours per day). Covers cohorts up to 20 participants. Additional seats at **$100 USD per seat**. Clients are encouraged to choose services relevant to their work or interests.
  * Parity: Tier 1: $2,000 | Tier 2 (25% off): $1,500 | Tier 3 & 4 (50% off): $1,000 | Tier 5 (75% off): $500 | Tier 6: Custom Subsidized Relief.
- **Comprehensive Personal Training**: **$1,000 USD** for **40 days** (20 days intensive 1-on-1 sprint + 20 days guidance, Q&A and written guides, 20 working days over 4 weeks with 8 - 10 recovery days, 4 hours per day). Strictly for **one person only**. Clients are encouraged to choose services relevant to their work or interests.
  * Parity: Tier 1: $1,000 | Tier 2 (25% off): $750 | Tier 3 & 4 (50% off): $500 | Tier 5 (75% off): $250 | Tier 6: Custom Subsidized Relief ($100).
- **Purchasing Power Parity (PPP)**: Available across all service pricing, covering 195 countries and dedicated relief tiers."""

    pattern = re.compile(
        r'-\s*\*\*(?:Individual Service Picks|Core Strategic Packages|Business Services|Business Training)[\s\S]*?(?=\n\n|\n###|\Z)',
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
    sync_home_schema_json()
    sync_pricing_html_embedded_schema()
    sync_service_schema_files()
    sync_service_html_pages()
    sync_llms_txt()
    print("--- Central Pricing & Parity Sync Pass Complete ---")

if __name__ == "__main__":
    sync_all_pricing()
