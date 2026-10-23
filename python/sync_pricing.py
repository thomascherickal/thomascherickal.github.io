#!/usr/bin/env python3
"""
sync_pricing.py — Centralized Pricing & Parity Synchronization Engine.
Defines authoritative pricing data, generates parity pricing across 6 tiers
(Standard, 25% off, 50% off, 75% off, and Tier 6 custom subsidized relief),
and synchronizes prices across:
- schema/pricing-schema.json
- pricing.html (embedded JSON-LD and pricing tables)
- schema.json
- 8 service-*.html pages (Section 5 & embedded JSON-LD)
- 8 schema/service-*.json files
- llms.txt
- md-html-sync/pricing.md (via md_html_sync)
"""

import os
import sys
import json
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
SCHEMA_DIR = os.path.join(REPO_DIR, "schema")

# Authoritative Core Packages (Enterprise Services @ $2,000 USD)
CORE_PILLARS = [
    {
        "id": "01",
        "name": "01 — AI Agent Mastery",
        "shortName": "AI Agent Mastery",
        "headline": "$2,000 USD",
        "sub": "USD · FLAT BASE RATE",
        "minPrice": 2000,
        "maxPrice": 2000,
        "description": "Create teams of Claude Agents. Flat base rate: $2,000 USD. Parity rates: Tier 2 (25% off) $1,500; Tier 3 & 4 (50% off) $1,000; Tier 5 (75% off) $500; Tier 6 custom subsidized relief across 195 countries.",
        "bullets": [
            ("Multi-Agent Architecture", "Coordinated teams of Claude Agents."),
            ("Constraint Navigation", "Overcoming context degradation, tool loops & state drift."),
            ("Sample Code & IP Transfer", "Runnable Python/Rust repos, evals & 100% IP transfer.")
        ]
    },
    {
        "id": "02",
        "name": "02 — Workforce AI Training (Live & Remote)",
        "shortName": "Workforce AI Training",
        "headline": "$2,000 USD",
        "sub": "USD · FLAT BASE RATE",
        "minPrice": 2000,
        "maxPrice": 2000,
        "description": "Live and remote training tailored for Employees, Developers, Corporate Staff, and CXOs. Flat base rate: $2,000 USD. Parity rates: Tier 2 (25% off) $1,500; Tier 3 & 4 (50% off) $1,000; Tier 5 (75% off) $500; Tier 6 custom subsidized relief across 195 countries.",
        "bullets": [
            ("Employees & Staff", "AI adoption, prompt craft, and daily productivity tool mastery."),
            ("Engineering & TDD", "AI-assisted dev, test-driven development & agentic loops."),
            ("CXOs & Leaders", "Strategic AI transformation, tech radars & ROI metrics frameworks.")
        ]
    },
    {
        "id": "03",
        "name": "03 — Local LLM Deployment & Inference Cost Slashing",
        "shortName": "Local LLM Deployment",
        "headline": "$2,000 USD",
        "sub": "USD · FLAT BASE RATE",
        "minPrice": 2000,
        "maxPrice": 2000,
        "description": "Slashing recurring AI inference costs by deploying private open-weight models on local/on-prem hardware. Flat base rate: $2,000 USD. Parity rates: Tier 2 (25% off) $1,500; Tier 3 & 4 (50% off) $1,000; Tier 5 (75% off) $500; Tier 6 custom subsidized relief across 195 countries.",
        "bullets": [
            ("API Cost Audit", "Quantitative assessment of token spend and 60–90% savings plan."),
            ("Production Model Serving", "vLLM, Ollama, AWQ/GGUF on consumer GPUs or servers."),
            ("Zero Data Egress", "Fully air-gapped sovereign AI with zero external compliance risk.")
        ]
    }
]

# Authoritative A La Carte Modules (Individual Services - Exactly matching the 12 standalone service offerings)
A_LA_CARTE = [
    {"num": 1, "name": "Generative AI Transformation", "price": 2000, "per": "", "color": "gold"},
    {"num": 2, "name": "Local LLMs & Cost Slashing", "price": 2000, "per": "", "color": "cyan"},
    {"num": 3, "name": "Claude AI Agent Orchestration", "price": 2000, "per": "", "color": "gold"},
    {"num": 4, "name": "Hermes Agent", "price": 2000, "per": "", "color": "cyan"},
    {"num": 5, "name": "WorkFlow Automations", "price": 2000, "per": "", "color": "gold"},
    {"num": 6, "name": "Applied Quantum (Experimental)", "price": 2000, "per": "", "color": "cyan"},
    {"num": 7, "name": "AI Budget Training", "price": 2000, "per": "", "color": "gold"},
    {"num": 8, "name": "Enterprise AI Training", "price": 2000, "per": "", "color": "cyan"},
    {"num": 9, "name": "AI Developer Training", "price": 2000, "per": "", "color": "gold"},
    {"num": 10, "name": "Advanced AI Agents", "price": 2000, "per": "", "color": "cyan"},
    {"num": 11, "name": "Claude Basic Training", "price": 2000, "per": "", "color": "gold"},
    {"num": 12, "name": "Claude Advanced Training", "price": 2000, "per": "", "color": "cyan"}
]

# The 8 Standalone Services
STANDALONE_SERVICES = {
    "service-agentic-ai-systems": {
        "title": "Agentic AI Assistants (Hermes Agent)",
        "min": 3000,
        "max": 6000,
        "unit": "per system rollout"
    },
    "service-ai-agents-orchestration": {
        "title": "Claude AI Agent Orchestration & Tool Loops",
        "min": 3000,
        "max": 6000,
        "unit": "per agent swarm milestone"
    },
    "service-ai-budget-cost-limits-training": {
        "title": "AI Budget Limits, Cost Quotas & Workforce Training",
        "min": 4000,
        "max": 8000,
        "unit": "per corporate workshop program or gateway setup"
    },
    "service-enterprise-workforce-ai-training": {
        "title": "Corporate & Developer AI Training (Live & Remote)",
        "min": 4000,
        "max": 8000,
        "unit": "per workshop series, bootcamp cohort, or executive briefing"
    },
    "service-generative-ai-transformation": {
        "title": "Enterprise Generative AI Transformation & Architecture",
        "min": 3000,
        "max": 6000,
        "unit": "per transformation module or implementation milestone"
    },
    "service-local-llms-cost-slashing": {
        "title": "Local LLM Deployment & Inference Cost Slashing",
        "min": 3000,
        "max": 5000,
        "unit": "per deployment sprint"
    },
    "service-quantum-applications": {
        "title": "Early Quantum Computing Applications & Qiskit Labs",
        "min": 1000,
        "max": 3000,
        "unit": "per advisory assessment or circuit proof-of-concept"
    },
    "service-training-for-workflow-automations": {
        "title": "Training for Workflows, Agent Swarms & Automations",
        "min": 4000,
        "max": 8000,
        "unit": "per enterprise training sprint or workflow rollout"
    },
    "service-ai-developer-training": {
        "title": "AI Developer Training (Coding Agents, MCP & TDD)",
        "min": 2000,
        "max": 4000,
        "unit": "per developer workshop series or bootcamp cohort"
    },
    "service-advanced-ai-agents": {
        "title": "Advanced AI Agents (Swarms, Self-Correction & Autonomy)",
        "min": 2000,
        "max": 5000,
        "unit": "per autonomous agent swarm milestone"
    },
    "service-claude-basic-training": {
        "title": "Claude Basic Training (Foundations, Prompt Craft & Office)",
        "min": 2000,
        "max": 4000,
        "unit": "per corporate workshop series or cohort"
    },
    "service-claude-advanced-training": {
        "title": "Claude Advanced Training (Claude Code, Cowork, MCP & Swarms)",
        "min": 2000,
        "max": 5000,
        "unit": "per advanced developer masterclass"
    }
}

def calc_parity(min_val, max_val):
    return {
        "tier1": {"label": "Tier 1 (Standard / 100%)", "min": min_val, "max": max_val},
        "tier2": {"label": "Tier 2 (Developing / 25% Off)", "min": int(min_val * 0.75), "max": int(max_val * 0.75)},
        "tier3": {"label": "Tier 3 & 4 (Least Developed & Startups / 50% Off)", "min": int(min_val * 0.50), "max": int(max_val * 0.50)},
        "tier5": {"label": "Tier 5 (Academia & Non-Profits / 75% Off)", "min": int(min_val * 0.25), "max": int(max_val * 0.25)},
        "tier6": {"label": "Tier 6 (Underprivileged & Backward Nations)", "min": 0, "max": 0, "custom": True}
    }

def format_price_spec(name, min_val, max_val, desc=None):
    if min_val == 0 and max_val == 0:
        return {
            "@type": "PriceSpecification",
            "name": name,
            "price": "0.00",
            "priceCurrency": "USD",
            "description": desc or "Custom subsidized quotes and micro-rates"
        }
    if min_val == max_val and min_val > 0:
        spec = {
            "@type": "PriceSpecification",
            "name": name,
            "price": f"{min_val:.2f}",
            "priceCurrency": "USD"
        }
    else:
        spec = {
            "@type": "PriceSpecification",
            "name": name,
            "minPrice": f"{min_val:.2f}",
            "maxPrice": f"{max_val:.2f}",
            "priceCurrency": "USD"
        }
    if desc:
        spec["description"] = desc
    return spec

def build_offer_catalog_elements():
    elements = []
    for p in CORE_PILLARS:
        if p["minPrice"] == 0 and p["maxPrice"] == 0:
            price_specs = [
                format_price_spec("Tier 6 Underprivileged & Backward Nations", 0, 0, "Custom subsidized quotes and micro-rates across 195 countries")
            ]
        else:
            par = calc_parity(p["minPrice"], p["maxPrice"])
            price_specs = [
                format_price_spec(par["tier1"]["label"], par["tier1"]["min"], par["tier1"]["max"]),
                format_price_spec(par["tier2"]["label"], par["tier2"]["min"], par["tier2"]["max"]),
                format_price_spec(par["tier3"]["label"], par["tier3"]["min"], par["tier3"]["max"]),
                format_price_spec(par["tier5"]["label"], par["tier5"]["min"], par["tier5"]["max"]),
                format_price_spec("Tier 6 (Custom Subsidized Relief)", 0, 0, "Micro-rates and customized quotes for underprivileged teams")
            ]
        
        offer = {
            "@type": "Offer",
            "name": p["name"],
            "description": p["description"],
            "price": f"{p['minPrice']:.2f}",
            "priceCurrency": "USD",
            "priceSpecification": price_specs
        }
        elements.append(offer)
    return elements

clean_desc = "Transparent flat-rate pricing across Enterprise Services: 01 AI Agent Mastery, 02 Workforce AI Training, 03 Local LLMs & Inference Cost Slashing with Global Parity across 195 countries. Each service lasts for one month and can freely overlap for different customers according to their needs."

def sync_pricing_schema_json():
    pricing_schema_path = os.path.join(SCHEMA_DIR, "pricing-schema.json")
    with open(pricing_schema_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Locate WebPage and OfferCatalog in @graph
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

def sync_pricing_html_cards():
    pricing_html_path = os.path.join(REPO_DIR, "pricing.html")
    with open(pricing_html_path, "r", encoding="utf-8") as f:
        content = f.read()

    pillars_html = """    <!-- ENTERPRISE SERVICES SECTION -->
    <section class="reveal" id="pricing-pillars" style="margin-bottom:3.5rem;">
      <div style="max-width:1380px;margin:0 auto 2.5rem auto;padding:0 1.5rem;text-align:center;">
        <p class="section-label">// STRATEGIC PACKAGES</p>
        <h2 class="section-title">Enterprise <span class="accent-gold">Services</span></h2>
        <p class="section-sub" style="max-width:820px;margin:0.5rem auto 0 auto;line-height:1.7;">
          Turnkey enterprise deployments and high-impact AI capabilities at a flat, transparent base rate of $2,000 USD with global parity pricing across 195 countries.
        </p>
      </div>
      <div class="packages-grid"
        style="display:grid;grid-template-columns:repeat(auto-fit, minmax(min(100%, 340px), 1fr));gap:2.5rem 2rem;max-width:1380px;margin:0 auto;padding:1rem 1.5rem 0 1.5rem;align-items:stretch;">
        <!-- Pillar 01: AI Agent Mastery -->
        <div class="package-card popular-card border-cyan"
          style="background:linear-gradient(155deg, rgba(8, 25, 45, 0.95) 0%, rgba(3, 7, 18, 0.98) 100%);border:2px solid #00d2ff;border-radius:var(--radius-lg);padding:2.2rem 1.85rem;display:flex;flex-direction:column;justify-content:space-between;position:relative;transition:var(--transition);box-shadow:0 0 28px rgba(0, 210, 255, 0.25), 0 10px 30px rgba(0,0,0,0.7);">
          <div
            style="position:absolute;top:-13px;left:50%;transform:translateX(-50%);background:linear-gradient(135deg, #00d2ff 0%, #0284c7 100%);color:#000000;font-family:var(--font-mono);font-size:calc(0.75rem + 3px);font-weight:900;letter-spacing:0.08em;text-transform:uppercase;padding:0.35rem 1.1rem;border-radius:9999px;box-shadow:0 0 18px rgba(0, 210, 255, 0.65);white-space:nowrap;">
            AGENT ORCHESTRATION MASTERY
          </div>
          <div>
            <span
              style="font-size:calc(0.75rem + 3px);font-weight:800;letter-spacing:0.07em;margin-bottom:0.75rem;display:inline-block;background:rgba(0, 210, 255, 0.18);border:1px solid #00d2ff;color:#00f0ff;padding:0.35rem 0.85rem;border-radius:6px;">01
              · AGENT ORCHESTRATION</span>
            <h3 style="font-size:1.55rem;font-weight:800;color:var(--text-primary);margin-bottom:0.35rem;">AI Agent
              Mastery</h3>
            <p style="color:var(--text-muted);font-size:calc(0.88rem + 3px);margin-bottom:1.1rem;line-height:1.55;">Create teams of Claude Agents. Master multi-agent orchestration principles &amp; constraints.</p>
            <div style="margin-bottom:1.1rem;padding-bottom:1rem;border-bottom:1px solid rgba(0, 210, 255, 0.3);">
              <div
                style="font-family:var(--font-display);font-size:2.2rem;font-weight:900;color:#00f0ff;line-height:1.1;text-shadow:0 0 16px rgba(0, 240, 255, 0.4);">
                $2,000 USD</div>
              <span
                style="font-family:var(--font-mono);font-size:calc(0.75rem + 3px);color:#7dd3fc;letter-spacing:0.06em;font-weight:700;">USD
                · FLAT BASE RATE</span>
            </div>
            <!-- Parity Box -->
            <div class="pillar-parity-box">
              <div class="pillar-parity-header">
                <span>PARITY PRICING TIERS</span>
                <span style="color:#00f0ff;font-size:calc(0.65rem + 3px);">195 NATIONS</span>
              </div>
              <div class="pillar-parity-grid">
                <div class="pillar-parity-item">
                  <span class="ppp-tag ppp-t2">25% OFF (TIER 2)</span>
                  <span class="ppp-val">$1,500</span>
                </div>
                <div class="pillar-parity-item">
                  <span class="ppp-tag ppp-t3">50% OFF (T3 &amp; STARTUPS)</span>
                  <span class="ppp-val">$1,000</span>
                </div>
                <div class="pillar-parity-item">
                  <span class="ppp-tag ppp-t5">75% OFF (ACADEMIA)</span>
                  <span class="ppp-val">$500</span>
                </div>
                <div class="pillar-parity-item">
                  <span class="ppp-tag ppp-t6">TIER 6 RELIEF</span>
                  <span class="ppp-val">Custom Quote</span>
                </div>
              </div>
            </div>
            <ul
              style="list-style:none;display:flex;flex-direction:column;gap:0.75rem;margin-bottom:1.5rem;font-size:calc(0.87rem + 3px);color:var(--text-secondary);line-height:1.5;">
              <li style="display:flex;align-items:flex-start;gap:0.6rem;">
                <span style="color:#00f0ff;font-weight:bold;">✓</span>
                <span><strong>Multi-Agent Architecture:</strong> Coordinated teams of Claude Agents.</span>
              </li>
              <li style="display:flex;align-items:flex-start;gap:0.6rem;">
                <span style="color:#00f0ff;font-weight:bold;">✓</span>
                <span><strong>Constraint Navigation:</strong> Overcoming context degradation, tool loops &amp; state drift.</span>
              </li>
              <li style="display:flex;align-items:flex-start;gap:0.6rem;">
                <span style="color:#00f0ff;font-weight:bold;">✓</span>
                <span><strong>Sample Code &amp; IP Transfer:</strong> Runnable Python/Rust repos, evals &amp; 100% IP transfer.</span>
              </li>
            </ul>
          </div>
          <a class="btn btn-glow-cyan" href="contact.html?package=AI%20Agent%20Mastery"
            style="width:100%;justify-content:center;padding:0.85rem 1rem;font-size:calc(0.88rem + 3px);font-weight:800;letter-spacing:0.02em;border:3px solid #00d2ff;">Master
            AI Agents →</a>
        </div>
        <!-- Pillar 02: Workforce AI Training (Live & Remote) -->
        <div class="package-card border-gold"
          style="background:linear-gradient(155deg, rgba(35, 26, 6, 0.95) 0%, rgba(12, 9, 2, 0.98) 100%);border:2px solid #fbbf24;border-radius:var(--radius-lg);padding:2.2rem 1.85rem;display:flex;flex-direction:column;justify-content:space-between;position:relative;transition:var(--transition);box-shadow:0 0 28px rgba(251, 191, 36, 0.25), 0 10px 30px rgba(0,0,0,0.7);">
          <div
            style="position:absolute;top:-13px;left:50%;transform:translateX(-50%);background:linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);color:#000000;font-family:var(--font-mono);font-size:calc(0.75rem + 3px);font-weight:900;letter-spacing:0.08em;text-transform:uppercase;padding:0.35rem 1.1rem;border-radius:9999px;box-shadow:0 0 18px rgba(251, 191, 36, 0.65);white-space:nowrap;">
            LIVE &amp; REMOTE TRAINING
          </div>
          <div>
            <span
              style="font-size:calc(0.75rem + 3px);font-weight:800;letter-spacing:0.07em;margin-bottom:0.75rem;display:inline-block;background:rgba(251, 191, 36, 0.18);border:1px solid #fbbf24;color:#fde047;padding:0.35rem 0.85rem;border-radius:6px;">02
              · CORPORATE &amp; DEV TRAINING</span>
            <h3 style="font-size:1.55rem;font-weight:800;color:var(--text-primary);margin-bottom:0.35rem;">Workforce AI
              Training</h3>
            <p style="color:var(--text-muted);font-size:calc(0.88rem + 3px);margin-bottom:1.1rem;line-height:1.55;">Live and remote
              training tailored for Employees, Developers, Corporate Staff, and CXOs.</p>
            <div style="margin-bottom:1.1rem;padding-bottom:1rem;border-bottom:1px solid rgba(251, 191, 36, 0.3);">
              <div
                style="font-family:var(--font-display);font-size:2.2rem;font-weight:900;color:#fbbf24;line-height:1.1;text-shadow:0 0 16px rgba(251, 191, 36, 0.4);">
                $2,000 USD</div>
              <span
                style="font-family:var(--font-mono);font-size:calc(0.75rem + 3px);color:#fde68a;letter-spacing:0.06em;font-weight:700;">USD
                · FLAT BASE RATE</span>
            </div>
            <!-- Parity Box -->
            <div class="pillar-parity-box">
              <div class="pillar-parity-header">
                <span>PARITY PRICING TIERS</span>
                <span style="color:#fbbf24;font-size:calc(0.65rem + 3px);">195 NATIONS</span>
              </div>
              <div class="pillar-parity-grid">
                <div class="pillar-parity-item">
                  <span class="ppp-tag ppp-t2">25% OFF (TIER 2)</span>
                  <span class="ppp-val">$1,500</span>
                </div>
                <div class="pillar-parity-item">
                  <span class="ppp-tag ppp-t3">50% OFF (T3 &amp; STARTUPS)</span>
                  <span class="ppp-val">$1,000</span>
                </div>
                <div class="pillar-parity-item">
                  <span class="ppp-tag ppp-t5">75% OFF (ACADEMIA)</span>
                  <span class="ppp-val">$500</span>
                </div>
                <div class="pillar-parity-item">
                  <span class="ppp-tag ppp-t6">TIER 6 RELIEF</span>
                  <span class="ppp-val">Custom Quote</span>
                </div>
              </div>
            </div>
            <ul
              style="list-style:none;display:flex;flex-direction:column;gap:0.75rem;margin-bottom:1.5rem;font-size:calc(0.87rem + 3px);color:var(--text-secondary);line-height:1.5;">
              <li style="display:flex;align-items:flex-start;gap:0.6rem;">
                <span style="color:#fbbf24;font-weight:bold;">✓</span>
                <span><strong>Employees &amp; Staff:</strong> AI adoption, prompt craft, and daily productivity tool mastery.</span>
              </li>
              <li style="display:flex;align-items:flex-start;gap:0.6rem;">
                <span style="color:#fbbf24;font-weight:bold;">✓</span>
                <span><strong>Engineering &amp; TDD:</strong> AI-assisted dev, test-driven development &amp; agentic loops.</span>
              </li>
              <li style="display:flex;align-items:flex-start;gap:0.6rem;">
                <span style="color:#fbbf24;font-weight:bold;">✓</span>
                <span><strong>CXOs &amp; Leaders:</strong> Strategic AI transformation, tech radars &amp; ROI metrics frameworks.</span>
              </li>
            </ul>
          </div>
          <a class="btn btn-glow-gold" href="contact.html?package=Workforce%20AI%20Training"
            style="width:100%;justify-content:center;padding:0.85rem 1rem;font-size:calc(0.88rem + 3px);font-weight:800;letter-spacing:0.02em;border:3px solid #fbbf24;">Book
            Training Program →</a>
        </div>
        <!-- Pillar 03: Local LLM Deployment & Inference Cost Slashing -->
        <div class="package-card border-violet"
          style="background:linear-gradient(155deg, rgba(30, 12, 48, 0.95) 0%, rgba(10, 4, 18, 0.98) 100%);border:2px solid #a855f7;border-radius:var(--radius-lg);padding:2.2rem 1.85rem;display:flex;flex-direction:column;justify-content:space-between;position:relative;transition:var(--transition);box-shadow:0 0 28px rgba(168, 85, 247, 0.25), 0 10px 30px rgba(0,0,0,0.7);">
          <div
            style="position:absolute;top:-13px;left:50%;transform:translateX(-50%);background:linear-gradient(135deg, #c084fc 0%, #9333ea 100%);color:#000000;font-family:var(--font-mono);font-size:calc(0.75rem + 3px);font-weight:900;letter-spacing:0.08em;text-transform:uppercase;padding:0.35rem 1.1rem;border-radius:9999px;box-shadow:0 0 18px rgba(168, 85, 247, 0.65);white-space:nowrap;">
            COST SLASHING &amp; PRIVACY
          </div>
          <div>
            <span
              style="font-size:calc(0.75rem + 3px);font-weight:800;letter-spacing:0.07em;margin-bottom:0.75rem;display:inline-block;background:rgba(168, 85, 247, 0.18);border:1px solid #a855f7;color:#d8b4fe;padding:0.35rem 0.85rem;border-radius:6px;">03
              · LOCAL LLMS &amp; INFERENCE</span>
            <h3 style="font-size:1.55rem;font-weight:800;color:var(--text-primary);margin-bottom:0.35rem;">Local LLM
              Deployment</h3>
            <p style="color:var(--text-muted);font-size:calc(0.88rem + 3px);margin-bottom:1.1rem;line-height:1.55;">Slashing recurring
              AI inference costs by deploying private open-weight models on local/on-prem hardware.</p>
            <div style="margin-bottom:1.1rem;padding-bottom:1rem;border-bottom:1px solid rgba(168, 85, 247, 0.3);">
              <div
                style="font-family:var(--font-display);font-size:2.2rem;font-weight:900;color:#c084fc;line-height:1.1;text-shadow:0 0 16px rgba(192, 132, 252, 0.4);">
                $2,000 USD</div>
              <span
                style="font-family:var(--font-mono);font-size:calc(0.75rem + 3px);color:#e9d5ff;letter-spacing:0.06em;font-weight:700;">USD
                · FLAT BASE RATE</span>
            </div>
            <!-- Parity Box -->
            <div class="pillar-parity-box">
              <div class="pillar-parity-header">
                <span>PARITY PRICING TIERS</span>
                <span style="color:#c084fc;font-size:calc(0.65rem + 3px);">195 NATIONS</span>
              </div>
              <div class="pillar-parity-grid">
                <div class="pillar-parity-item">
                  <span class="ppp-tag ppp-t2">25% OFF (TIER 2)</span>
                  <span class="ppp-val">$1,500</span>
                </div>
                <div class="pillar-parity-item">
                  <span class="ppp-tag ppp-t3">50% OFF (T3 &amp; STARTUPS)</span>
                  <span class="ppp-val">$1,000</span>
                </div>
                <div class="pillar-parity-item">
                  <span class="ppp-tag ppp-t5">75% OFF (ACADEMIA)</span>
                  <span class="ppp-val">$500</span>
                </div>
                <div class="pillar-parity-item">
                  <span class="ppp-tag ppp-t6">TIER 6 RELIEF</span>
                  <span class="ppp-val">Custom Quote</span>
                </div>
              </div>
            </div>
            <ul
              style="list-style:none;display:flex;flex-direction:column;gap:0.75rem;margin-bottom:1.5rem;font-size:calc(0.87rem + 3px);color:var(--text-secondary);line-height:1.5;">
              <li style="display:flex;align-items:flex-start;gap:0.6rem;">
                <span style="color:#c084fc;font-weight:bold;">✓</span>
                <span><strong>API Cost Audit:</strong> Quantitative assessment of token spend and 60–90% savings plan.</span>
              </li>
              <li style="display:flex;align-items:flex-start;gap:0.6rem;">
                <span style="color:#c084fc;font-weight:bold;">✓</span>
                <span><strong>Production Model Serving:</strong> vLLM, Ollama, AWQ/GGUF on consumer GPUs or servers.</span>
              </li>
              <li style="display:flex;align-items:flex-start;gap:0.6rem;">
                <span style="color:#c084fc;font-weight:bold;">✓</span>
                <span><strong>Zero Data Egress:</strong> Fully air-gapped sovereign AI with zero external compliance risk.</span>
              </li>
            </ul>
          </div>
          <a class="btn btn-glow-violet" href="contact.html?package=Local%20LLM%20Deployment"
            style="width:100%;justify-content:center;padding:0.85rem 1rem;font-size:calc(0.88rem + 3px);font-weight:800;letter-spacing:0.02em;border:3px solid #c084fc;">Slash
            Inference Costs →</a>
        </div>
      </div>
    </section>"""

    services_html = """    <!-- INDIVIDUAL SERVICES & REFERENCE RATES -->
    <section class="reveal" id="service-rates" style="margin-bottom:4.5rem;">
      <div style="max-width:1380px;margin:0 auto;padding:0 1.5rem;">
        <div style="text-align:center;margin-bottom:2.5rem;">
          <p class="section-label">// REFERENCE SPECIFICATIONS</p>
          <h2 class="section-title">Individual Services &amp; <span class="accent-cyan">Base Rates</span></h2>
          <p class="section-sub" style="max-width:820px;margin:0.5rem auto 0 auto;line-height:1.7;">
            A la carte specifications across our twelve core transformation services. <strong style="color:var(--gold, #fbbf24);font-weight:700;">Every enterprise booking covers strictly 30 days of training and 30 days of included support (total 60 days of enterprise protection) with global parity pricing across 195 countries.</strong>
          </p>
        </div>
        <div class="pricing-grid-faq">
          <!-- Item 1 (Row 1, Card 1) -->
          <div class="price-card-faq border-gold">
            <div class="price-card-title"
              style="color:var(--gold);font-weight:700;font-size:0.98rem;min-height:2.6rem;display:flex;align-items:center;justify-content:center;">
              Generative AI Transformation</div>
            <div class="price-amount"
              style="font-size:1.3rem;font-weight:800;margin:0.4rem 0 0.2rem 0;color:#fbbf24;text-shadow:0 0 12px rgba(251,191,36,0.35);">
              $2,000 USD
            </div>
            <div style="font-size:0.75rem;color:var(--cyan);font-weight:700;margin:0.2rem 0 0.4rem 0;text-transform:uppercase;letter-spacing:0.06em;">Coverage: 30d Training + 30d Support</div>
            <div class="service-parity-strip">
              <span>25% Off: <strong>$1,500</strong></span>
              <span>50% Off: <strong>$1,000</strong></span>
              <span>75% Off: <strong>$500</strong></span>
              <span>T6: <strong>Relief</strong></span>
            </div>
            <div
              style="font-size:calc(0.84rem + 3px);color:var(--text-secondary);line-height:1.5;flex-grow:1;display:flex;align-items:center;justify-content:center;">
              Move from one-off chatbot prompts to coordinated teams of Claude agents with custom MCP servers and bounded execution loops.
            </div>
          </div>
          <!-- Item 2 (Row 1, Card 2) -->
          <div class="price-card-faq border-cyan">
            <div class="price-card-title"
              style="color:var(--cyan);font-weight:700;font-size:0.98rem;min-height:2.6rem;display:flex;align-items:center;justify-content:center;">
              Local LLMs &amp; Cost Slashing</div>
            <div class="price-amount"
              style="font-size:1.3rem;font-weight:800;margin:0.4rem 0 0.2rem 0;color:#00f0ff;text-shadow:0 0 12px rgba(0,240,255,0.35);">
              $2,000 USD
            </div>
            <div style="font-size:0.75rem;color:var(--cyan);font-weight:700;margin:0.2rem 0 0.4rem 0;text-transform:uppercase;letter-spacing:0.06em;">Coverage: 30d Training + 30d Support</div>
            <div class="service-parity-strip">
              <span>25% Off: <strong>$1,500</strong></span>
              <span>50% Off: <strong>$1,000</strong></span>
              <span>75% Off: <strong>$500</strong></span>
              <span>T6: <strong>Relief</strong></span>
            </div>
            <div
              style="font-size:calc(0.84rem + 3px);color:var(--text-secondary);line-height:1.5;flex-grow:1;display:flex;align-items:center;justify-content:center;">
              Cut recurring cloud API bills by 60% to 90% by running open-weight models privately with Ollama, vLLM, and llama.cpp on your hardware.
            </div>
          </div>
          <!-- Item 3 (Row 1, Card 3) -->
          <div class="price-card-faq border-gold">
            <div class="price-card-title"
              style="color:var(--gold);font-weight:700;font-size:0.98rem;min-height:2.6rem;display:flex;align-items:center;justify-content:center;">
              Claude AI Agent Orchestration</div>
            <div class="price-amount"
              style="font-size:1.3rem;font-weight:800;margin:0.4rem 0 0.2rem 0;color:#fbbf24;text-shadow:0 0 12px rgba(251,191,36,0.35);">
              $2,000 USD
            </div>
            <div style="font-size:0.75rem;color:var(--cyan);font-weight:700;margin:0.2rem 0 0.4rem 0;text-transform:uppercase;letter-spacing:0.06em;">Coverage: 30d Training + 30d Support</div>
            <div class="service-parity-strip">
              <span>25% Off: <strong>$1,500</strong></span>
              <span>50% Off: <strong>$1,000</strong></span>
              <span>75% Off: <strong>$500</strong></span>
              <span>T6: <strong>Relief</strong></span>
            </div>
            <div
              style="font-size:calc(0.84rem + 3px);color:var(--text-secondary);line-height:1.5;flex-grow:1;display:flex;align-items:center;justify-content:center;">
              Master Claude agent orchestration to build supervisor-worker swarms, sequential pipelines, and parallel teams that control quality and cost.
            </div>
          </div>
          <!-- Item 4 (Row 1, Card 4) -->
          <div class="price-card-faq border-cyan">
            <div class="price-card-title"
              style="color:var(--cyan);font-weight:700;font-size:0.98rem;min-height:2.6rem;display:flex;align-items:center;justify-content:center;">
              Hermes Agent</div>
            <div class="price-amount"
              style="font-size:1.3rem;font-weight:800;margin:0.4rem 0 0.2rem 0;color:#00f0ff;text-shadow:0 0 12px rgba(0,240,255,0.35);">
              $2,000 USD
            </div>
            <div style="font-size:0.75rem;color:var(--cyan);font-weight:700;margin:0.2rem 0 0.4rem 0;text-transform:uppercase;letter-spacing:0.06em;">Coverage: 30d Training + 30d Support</div>
            <div class="service-parity-strip">
              <span>25% Off: <strong>$1,500</strong></span>
              <span>50% Off: <strong>$1,000</strong></span>
              <span>75% Off: <strong>$500</strong></span>
              <span>T6: <strong>Relief</strong></span>
            </div>
            <div
              style="font-size:calc(0.84rem + 3px);color:var(--text-secondary);line-height:1.5;flex-grow:1;display:flex;align-items:center;justify-content:center;">
              Deploy self-hosted Hermes Agent on Linux with persistent session memory, autonomous skill accumulation, and scheduled background tasks.
            </div>
          </div>
          <!-- Item 5 (Row 2, Card 1) -->
          <div class="price-card-faq border-gold">
            <div class="price-card-title"
              style="color:var(--gold);font-weight:700;font-size:0.98rem;min-height:2.6rem;display:flex;align-items:center;justify-content:center;">
              WorkFlow Automations</div>
            <div class="price-amount"
              style="font-size:1.3rem;font-weight:800;margin:0.4rem 0 0.2rem 0;color:#fbbf24;text-shadow:0 0 12px rgba(251,191,36,0.35);">
              $2,000 USD
            </div>
            <div style="font-size:0.75rem;color:var(--cyan);font-weight:700;margin:0.2rem 0 0.4rem 0;text-transform:uppercase;letter-spacing:0.06em;">Coverage: 30d Training + 30d Support</div>
            <div class="service-parity-strip">
              <span>25% Off: <strong>$1,500</strong></span>
              <span>50% Off: <strong>$1,000</strong></span>
              <span>75% Off: <strong>$500</strong></span>
              <span>T6: <strong>Relief</strong></span>
            </div>
            <div
              style="font-size:calc(0.84rem + 3px);color:var(--text-secondary);line-height:1.5;flex-grow:1;display:flex;align-items:center;justify-content:center;">
              Hands-on training to automate up to 50% of routine office work using Claude skills, plugins, connectors, and Microsoft 365 add-ins.
            </div>
          </div>
          <!-- Item 6 (Row 2, Card 2) -->
          <div class="price-card-faq border-cyan">
            <div class="price-card-title"
              style="color:var(--cyan);font-weight:700;font-size:0.98rem;min-height:2.6rem;display:flex;align-items:center;justify-content:center;">
              Applied Quantum (Experimental)</div>
            <div class="price-amount"
              style="font-size:1.3rem;font-weight:800;margin:0.4rem 0 0.2rem 0;color:#00f0ff;text-shadow:0 0 12px rgba(0,240,255,0.35);">
              $2,000 USD
            </div>
            <div style="font-size:0.75rem;color:var(--cyan);font-weight:700;margin:0.2rem 0 0.4rem 0;text-transform:uppercase;letter-spacing:0.06em;">Coverage: 30d Training + 30d Support</div>
            <div class="service-parity-strip">
              <span>25% Off: <strong>$1,500</strong></span>
              <span>50% Off: <strong>$1,000</strong></span>
              <span>75% Off: <strong>$500</strong></span>
              <span>T6: <strong>Relief</strong></span>
            </div>
            <div
              style="font-size:calc(0.84rem + 3px);color:var(--text-secondary);line-height:1.5;flex-grow:1;display:flex;align-items:center;justify-content:center;">
              Exploratory quantum computing advisory: algorithm simulation with Qiskit and PennyLane, variational circuits, and NIST PQC migration.
            </div>
          </div>
          <!-- Item 7 (Row 2, Card 3) -->
          <div class="price-card-faq border-gold">
            <div class="price-card-title"
              style="color:var(--gold);font-weight:700;font-size:0.98rem;min-height:2.6rem;display:flex;align-items:center;justify-content:center;">
              AI Budget Training</div>
            <div class="price-amount"
              style="font-size:1.3rem;font-weight:800;margin:0.4rem 0 0.2rem 0;color:#fbbf24;text-shadow:0 0 12px rgba(251,191,36,0.35);">
              $2,000 USD
            </div>
            <div style="font-size:0.75rem;color:var(--cyan);font-weight:700;margin:0.2rem 0 0.4rem 0;text-transform:uppercase;letter-spacing:0.06em;">Coverage: 30d Training + 30d Support</div>
            <div class="service-parity-strip">
              <span>25% Off: <strong>$1,500</strong></span>
              <span>50% Off: <strong>$1,000</strong></span>
              <span>75% Off: <strong>$500</strong></span>
              <span>T6: <strong>Relief</strong></span>
            </div>
            <div
              style="font-size:calc(0.84rem + 3px);color:var(--text-secondary);line-height:1.5;flex-grow:1;display:flex;align-items:center;justify-content:center;">
              Train staff across all levels to enforce hard spending caps, manage token quotas, leverage prompt caching, and route to local models.
            </div>
          </div>
          <!-- Item 8 (Row 2, Card 4) -->
          <div class="price-card-faq border-cyan">
            <div class="price-card-title"
              style="color:var(--cyan);font-weight:700;font-size:0.98rem;min-height:2.6rem;display:flex;align-items:center;justify-content:center;">
              Enterprise AI Training</div>
            <div class="price-amount"
              style="font-size:1.3rem;font-weight:800;margin:0.4rem 0 0.2rem 0;color:#00f0ff;text-shadow:0 0 12px rgba(0,240,255,0.35);">
              $2,000 USD
            </div>
            <div style="font-size:0.75rem;color:var(--cyan);font-weight:700;margin:0.2rem 0 0.4rem 0;text-transform:uppercase;letter-spacing:0.06em;">Coverage: 30d Training + 30d Support</div>
            <div class="service-parity-strip">
              <span>25% Off: <strong>$1,500</strong></span>
              <span>50% Off: <strong>$1,000</strong></span>
              <span>75% Off: <strong>$500</strong></span>
              <span>T6: <strong>Relief</strong></span>
            </div>
            <div
              style="font-size:calc(0.84rem + 3px);color:var(--text-secondary);line-height:1.5;flex-grow:1;display:flex;align-items:center;justify-content:center;">
              Role-tailored corporate AI training across distinct tracks: staff workflows, developer AI-assisted TDD, and executive AI roadmaps.
            </div>
          </div>
          <!-- Item 9 (Row 3, Card 1) -->
          <div class="price-card-faq border-gold">
            <div class="price-card-title"
              style="color:var(--gold);font-weight:700;font-size:0.98rem;min-height:2.6rem;display:flex;align-items:center;justify-content:center;">
              AI Developer Training</div>
            <div class="price-amount"
              style="font-size:1.3rem;font-weight:800;margin:0.4rem 0 0.2rem 0;color:#fbbf24;text-shadow:0 0 12px rgba(251,191,36,0.35);">
              $2,000 USD
            </div>
            <div style="font-size:0.75rem;color:var(--cyan);font-weight:700;margin:0.2rem 0 0.4rem 0;text-transform:uppercase;letter-spacing:0.06em;">Coverage: 30d Training + 30d Support</div>
            <div class="service-parity-strip">
              <span>25% Off: <strong>$1,500</strong></span>
              <span>50% Off: <strong>$1,000</strong></span>
              <span>75% Off: <strong>$500</strong></span>
              <span>T6: <strong>Relief</strong></span>
            </div>
            <div
              style="font-size:calc(0.84rem + 3px);color:var(--text-secondary);line-height:1.5;flex-grow:1;display:flex;align-items:center;justify-content:center;">
              Hands-on bootcamps training developers on Claude Code CLI, Google Antigravity, custom Python/Rust MCP servers, and AI-assisted TDD.
            </div>
          </div>
          <!-- Item 10 (Row 3, Card 2) -->
          <div class="price-card-faq border-cyan">
            <div class="price-card-title"
              style="color:var(--cyan);font-weight:700;font-size:0.98rem;min-height:2.6rem;display:flex;align-items:center;justify-content:center;">
              Advanced AI Agents</div>
            <div class="price-amount"
              style="font-size:1.3rem;font-weight:800;margin:0.4rem 0 0.2rem 0;color:#00f0ff;text-shadow:0 0 12px rgba(0,240,255,0.35);">
              $2,000 USD
            </div>
            <div style="font-size:0.75rem;color:var(--cyan);font-weight:700;margin:0.2rem 0 0.4rem 0;text-transform:uppercase;letter-spacing:0.06em;">Coverage: 30d Training + 30d Support</div>
            <div class="service-parity-strip">
              <span>25% Off: <strong>$1,500</strong></span>
              <span>50% Off: <strong>$1,000</strong></span>
              <span>75% Off: <strong>$500</strong></span>
              <span>T6: <strong>Relief</strong></span>
            </div>
            <div
              style="font-size:calc(0.84rem + 3px);color:var(--text-secondary);line-height:1.5;flex-grow:1;display:flex;align-items:center;justify-content:center;">
              Build production autonomous agent systems: hierarchical Claude teams, test-driven self-correction loops, and Docker sandboxed tools.
            </div>
          </div>
          <!-- Item 11 (Row 3, Card 3) -->
          <div class="price-card-faq border-gold">
            <div class="price-card-title"
              style="color:var(--gold);font-weight:700;font-size:0.98rem;min-height:2.6rem;display:flex;align-items:center;justify-content:center;">
              Claude Basic Training</div>
            <div class="price-amount"
              style="font-size:1.3rem;font-weight:800;margin:0.4rem 0 0.2rem 0;color:#fbbf24;text-shadow:0 0 12px rgba(251,191,36,0.35);">
              $2,000 USD
            </div>
            <div style="font-size:0.75rem;color:var(--cyan);font-weight:700;margin:0.2rem 0 0.4rem 0;text-transform:uppercase;letter-spacing:0.06em;">Coverage: 30d Training + 30d Support</div>
            <div class="service-parity-strip">
              <span>25% Off: <strong>$1,500</strong></span>
              <span>50% Off: <strong>$1,000</strong></span>
              <span>75% Off: <strong>$500</strong></span>
              <span>T6: <strong>Relief</strong></span>
            </div>
            <div
              style="font-size:calc(0.84rem + 3px);color:var(--text-secondary);line-height:1.5;flex-grow:1;display:flex;align-items:center;justify-content:center;">
              Foundational enablement: model tier selection (Haiku, Sonnet, Opus), structured XML prompting, Claude Projects, and Artifacts.
            </div>
          </div>
          <!-- Item 12 (Row 3, Card 4) -->
          <div class="price-card-faq border-cyan">
            <div class="price-card-title"
              style="color:var(--cyan);font-weight:700;font-size:0.98rem;min-height:2.6rem;display:flex;align-items:center;justify-content:center;">
              Claude Advanced Training</div>
            <div class="price-amount"
              style="font-size:1.3rem;font-weight:800;margin:0.4rem 0 0.2rem 0;color:#00f0ff;text-shadow:0 0 12px rgba(0,240,255,0.35);">
              $2,000 USD
            </div>
            <div style="font-size:0.75rem;color:var(--cyan);font-weight:700;margin:0.2rem 0 0.4rem 0;text-transform:uppercase;letter-spacing:0.06em;">Coverage: 30d Training + 30d Support</div>
            <div class="service-parity-strip">
              <span>25% Off: <strong>$1,500</strong></span>
              <span>50% Off: <strong>$1,000</strong></span>
              <span>75% Off: <strong>$500</strong></span>
              <span>T6: <strong>Relief</strong></span>
            </div>
            <div
              style="font-size:calc(0.84rem + 3px);color:var(--text-secondary);line-height:1.5;flex-grow:1;display:flex;align-items:center;justify-content:center;">
              Senior engineer masterclass: Claude Code CLI in depth, Claude Cowork in Microsoft 365, custom MCP engineering, and agent swarms.
            </div>
          </div>
        </div>
      </div>
    </section>"""

    # Regex replacement for #pricing-pillars
    pillar_pattern = re.compile(
        r'(?:<!--[^\n]*-->\s*)?<section\b[^>]*id="pricing-pillars"[\s\S]*?</section>',
        re.IGNORECASE
    )
    if pillar_pattern.search(content):
        content = pillar_pattern.sub(pillars_html.strip(), content)
        print("Updated #pricing-pillars in pricing.html")
    else:
        print("Warning: Could not match #pricing-pillars in pricing.html")

    # Regex replacement for #service-rates
    service_pattern = re.compile(
        r'<!-- INDIVIDUAL SERVICES &amp; REFERENCE RATES -->\s*<section\b[^>]*id="service-rates"[\s\S]*?</section>|<!-- INDIVIDUAL SERVICES & REFERENCE RATES -->\s*<section\b[^>]*id="service-rates"[\s\S]*?</section>',
        re.IGNORECASE
    )
    if service_pattern.search(content):
        content = service_pattern.sub(services_html.strip(), content)
        print("Updated #service-rates in pricing.html")
    else:
        print("Warning: Could not match #service-rates in pricing.html")

    with open(pricing_html_path, "w", encoding="utf-8") as f:
        f.write(content)

def sync_standalone_service_files():
    for slug, info in STANDALONE_SERVICES.items():
        min_p = info["min"]
        max_p = info["max"]
        unit = info["unit"]
        par = calc_parity(min_p, max_p)
        
        # 1. Update schema JSON file
        schema_path = os.path.join(SCHEMA_DIR, f"{slug}.json")
        if os.path.exists(schema_path):
            with open(schema_path, "r", encoding="utf-8") as f:
                s_data = json.load(f)
            for item in s_data.get("@graph", []):
                if item.get("@type") == "Service" and "offers" in item:
                    item["offers"]["price"] = f"{min_p:.2f}"
                    item["offers"]["priceSpecification"] = [
                        format_price_spec(par["tier1"]["label"], par["tier1"]["min"], par["tier1"]["max"]),
                        format_price_spec(par["tier2"]["label"], par["tier2"]["min"], par["tier2"]["max"]),
                        format_price_spec(par["tier3"]["label"], par["tier3"]["min"], par["tier3"]["max"]),
                        format_price_spec(par["tier5"]["label"], par["tier5"]["min"], par["tier5"]["max"]),
                        format_price_spec(par["tier6"]["label"], 0, 0, "Custom subsidized quotes for underprivileged teams")
                    ]
            with open(schema_path, "w", encoding="utf-8") as f:
                json.dump(s_data, f, indent=2, ensure_ascii=False)
                f.write("\n")
            print(f"Updated {schema_path}")

        # 2. Update HTML page
        html_path = os.path.join(REPO_DIR, f"{slug}.html")
        if os.path.exists(html_path):
            with open(html_path, "r", encoding="utf-8") as f:
                html_content = f.read()
                
            # Update embedded JSON-LD
            match = re.search(r'(<script type="application/ld\+json">)([\s\S]*?)(</script>)', html_content)
            if match:
                try:
                    s_data = json.loads(match.group(2).strip())
                    for item in s_data.get("@graph", []):
                        if item.get("@type") == "Service" and "offers" in item:
                            item["offers"]["price"] = f"{min_p:.2f}"
                            item["offers"]["priceSpecification"] = [
                                format_price_spec(par["tier1"]["label"], par["tier1"]["min"], par["tier1"]["max"]),
                                format_price_spec(par["tier2"]["label"], par["tier2"]["min"], par["tier2"]["max"]),
                                format_price_spec(par["tier3"]["label"], par["tier3"]["min"], par["tier3"]["max"]),
                                format_price_spec(par["tier5"]["label"], par["tier5"]["min"], par["tier5"]["max"]),
                                format_price_spec(par["tier6"]["label"], 0, 0, "Custom subsidized quotes for underprivileged teams")
                            ]
                    new_json_str = json.dumps(s_data, indent=2, ensure_ascii=False)
                    html_content = html_content[:match.start(2)] + "\n  " + new_json_str.replace("\n", "\n  ") + "\n  " + html_content[match.end(2):]
                except Exception as e:
                    print(f"Error parsing JSON-LD in {html_path}: {e}")

            # Update Section 5 HTML block
            section5_pattern = re.compile(
                r'(<h2[^>]*>5\.\s*Investment\s*&amp;\s*Pricing\s*Quote</h2>\s*<ol[^>]*>)([\s\S]*?)(</ol>)',
                re.IGNORECASE
            )
            s5_match = section5_pattern.search(html_content)
            if s5_match:
                new_s5_body = f"""
<li>Base Investment Range (Tier 1 Standard): <strong>${min_p:,} – ${max_p:,} USD</strong> ({unit}).</li>
<li>Global Purchasing Power Parity (PPP) &amp; Subsidized Rates:
  <ul style="margin-left:1.5rem;margin-top:0.5rem;list-style:disc;">
    <li><strong>Tier 1 (Standard / 100%):</strong> ${par['tier1']['min']:,} – ${par['tier1']['max']:,} USD</li>
    <li><strong>Tier 2 (Developing Nations / 25% Off):</strong> ${par['tier2']['min']:,} – ${par['tier2']['max']:,} USD</li>
    <li><strong>Tier 3 &amp; 4 (Least Developed &amp; Startups / 50% Off):</strong> ${par['tier3']['min']:,} – ${par['tier3']['max']:,} USD</li>
    <li><strong>Tier 5 (Academia &amp; Non-Profits / 75% Off):</strong> ${par['tier5']['min']:,} – ${par['tier5']['max']:,} USD</li>
    <li><strong>Tier 6 (Backward &amp; Underprivileged Nations):</strong> Custom Subsidized Relief / Micro-Rates</li>
    <li>Calculate and lock in your exact tier rate live on the <a href="pricing.html" style="color:var(--cyan);text-decoration:underline;">Pricing page</a>.</li>
  </ul>
</li>
<li>Payment Schedule:
  <ul style="margin-left:1.5rem;margin-top:0.5rem;list-style:disc;">
    <li>50% upfront milestone initialization / 50% upon completed delivery, runtime verification, and client sign-off.</li>
  </ul>
</li>
"""
                html_content = html_content[:s5_match.start(2)] + new_s5_body + html_content[s5_match.end(2):]

            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            print(f"Updated {html_path}")

def sync_llms_txt():
    llms_path = os.path.join(REPO_DIR, "llms.txt")
    if not os.path.exists(llms_path):
        return
    with open(llms_path, "r", encoding="utf-8") as f:
        content = f.read()

    new_packages_block = """- **Enterprise Services ($2,000 USD Flat Base Rate, 195 Countries Supported)**:
  * **01 · AI Agent Mastery**: **$2,000 USD** (Flat Base Rate)
    - Parity: 25% Off (Tier 2): $1,500 | 50% Off (Tier 3/4 Startups): $1,000 | 75% Off (Tier 5): $500 | Tier 6: Subsidized Relief
  * **02 · Workforce AI Training (Live & Remote)**: **$2,000 USD** (Flat Base Rate)
    - Parity: 25% Off (Tier 2): $1,500 | 50% Off (Tier 3/4 Startups): $1,000 | 75% Off (Tier 5): $500 | Tier 6: Subsidized Relief
  * **03 · Local LLM Deployment & Inference Cost Slashing**: **$2,000 USD** (Flat Base Rate)
    - Parity: 25% Off (Tier 2): $1,500 | 50% Off (Tier 3/4 Startups): $1,000 | 75% Off (Tier 5): $500 | Tier 6: Subsidized Relief
- **A La Carte Advisory & Modules**: **$2,000 USD** flat rate per module (AI Agents, Local LLM Setup, AI-Assisted Dev & TDD, Early Quantum Systems, Frontier AI Tools, Technical Deep Dives).
  - Parity: 25% Off: $1,500 USD | 50% Off: $1,000 USD | 75% Off: $500 USD | Tier 6: Subsidized Relief"""

    pattern = re.compile(
        r'-\s*\*\*(?:Core Strategic Packages|Enterprise Services)[\s\S]*?-\s*\*\*A La Carte Advisory & Modules\*\*:[\s\S]*?(?=\n\n|\n-|\n###)',
        re.MULTILINE
    )
    if pattern.search(content):
        new_content = pattern.sub(new_packages_block, content)
        with open(llms_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print("Updated llms.txt")
    else:
        print("Could not match packages block in llms.txt")

def sync_all_pricing():
    print("--- Starting Central Pricing & Parity Sync Pass ---")
    sync_pricing_schema_json()
    sync_pricing_html_embedded_schema()
    sync_pricing_html_cards()
    sync_root_schema_json()
    # Note: STANDALONE_SERVICES are skipped to strictly modify pricing.html and linked files only
    sync_llms_txt()
    print("--- Central Pricing & Parity Sync Pass Complete ---")

if __name__ == "__main__":
    sync_all_pricing()
