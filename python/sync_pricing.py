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

# Authoritative Core Packages
CORE_PILLARS = [
    {
        "id": "01",
        "name": "01 — AI Agent Mastery",
        "shortName": "AI Agent Mastery",
        "headline": "$3,000 – $6,000 USD",
        "sub": "USD · SPRINT ($3,000) OR FULL DEPLOYMENT ($6,000)",
        "minPrice": 3000,
        "maxPrice": 6000,
        "description": "Create teams of AI agents with Claude, OpenAI, Gemini, or Grok. Standard Sprint: $3,000 USD | Full Deployment: $6,000 USD. Parity rates: Tier 2 (25% off) $2,250 / $4,500; Tier 3 & 4 (50% off) $1,500 / $3,000; Tier 5 (75% off) $750 / $1,500; Tier 6 custom subsidized relief.",
        "bullets": [
            ("Multi-Agent Team Architecture", "Architecting coordinated teams using Claude, OpenAI, Gemini, or Grok."),
            ("Constraint Navigation", "Overcoming context degradation, tool loops, and state drift."),
            ("Domain Task Grounding", "Grounding workflows in evaluation schemas and validation gates."),
            ("Runnable Repositories", "Python/Rust sample code, orchestration guides, and tests.")
        ]
    },
    {
        "id": "02",
        "name": "02 — Workforce AI Training (Live & Remote)",
        "shortName": "Workforce AI Training",
        "headline": "$4,000 – $8,000 USD",
        "sub": "USD · STANDARD ($4,000) OR ENTERPRISE ($8,000)",
        "minPrice": 4000,
        "maxPrice": 8000,
        "description": "Live and remote training for Employees, Developers, Staff, and CXOs. Standard Cohort: $4,000 USD | Enterprise Cohorts: $8,000 USD. Parity rates: Tier 2 (25% off) $3,000 / $6,000; Tier 3 & 4 (50% off) $2,000 / $4,000; Tier 5 (75% off) $1,000 / $2,000; Tier 6 custom subsidized relief.",
        "bullets": [
            ("Employees & Staff", "AI adoption, prompt craft, and daily productivity tool mastery."),
            ("Engineering Teams", "AI-assisted development, TDD pairing, and agentic workflows."),
            ("CXOs & Leadership", "Strategic AI transformation, technology radars, and ROI frameworks."),
            ("Hands-On Sandboxes", "Interactive Jupyter notebooks and reproducible curriculum materials.")
        ]
    },
    {
        "id": "03",
        "name": "03 — Local LLM Deployment & Inference Cost Slashing",
        "shortName": "Local LLM Deployment",
        "headline": "$3,000 – $5,000 USD",
        "sub": "USD · PRIVATE CLUSTER ($3,000) OR FULL DEPLOYMENT ($5,000)",
        "minPrice": 3000,
        "maxPrice": 5000,
        "description": "Slashing recurring AI API costs 60–90% with open-weight model serving (vLLM, Ollama, llama.cpp), quantization, and zero-egress pipelines. Private Cluster Setup: $3,000 USD | Full Production Pipeline: $5,000 USD. Parity rates: Tier 2 (25% off) $2,250 / $3,750; Tier 3 & 4 (50% off) $1,500 / $2,500; Tier 5 (75% off) $750 / $1,250; Tier 6 custom subsidized relief.",
        "bullets": [
            ("API Cost Audit", "Quantitative assessment of current token spend and 60–90% savings plan."),
            ("Open-Weight Model Serving", "Production vLLM, Ollama, and llama.cpp on consumer GPUs or servers."),
            ("Quantization & Tuning", "GGUF, AWQ, and EXL2 compression for high token throughput."),
            ("Air-Gapped Sovereign AI", "Complete data privacy with zero external API calls or compliance risk.")
        ]
    },
    {
        "id": "04",
        "name": "04 — Dedicated AI Sprints",
        "shortName": "Dedicated AI Sprints",
        "headline": "$3,000 – $10,000 USD",
        "sub": "USD · SPRINT ($3,000) · MONTHLY ($5,000) · QUARTERLY ($10,000)",
        "minPrice": 3000,
        "maxPrice": 10000,
        "description": "Continuous engineering capacity for agent swarms, AI workflows, and TDD transformation: Fortnightly Sprints ($3,000/2-wks), Monthly Retainers ($5,000/mo), and Enterprise Quarterly Retainers ($10,000/qtr). Parity rates: Tier 2 (25% off) $2,250 / $3,750 / $7,500; Tier 3 & 4 (50% off) $1,500 / $2,500 / $5,000; Tier 5 (75% off) $750 / $1,250 / $2,500; Tier 6 custom subsidized relief.",
        "bullets": [
            ("Reserved Sprint Capacity", "Guaranteed bandwidth for agent swarms, architecture, and evaluations."),
            ("Direct Async Communication", "Dedicated private Slack/Discord channel with <4-hour response SLA."),
            ("Weekly Architecture Sync", "Alignment on milestones, reviews, and rapid deployment cycles."),
            ("AI-Assisted TDD Enablement", "Embedding test-driven development practices with Claude Code & Antigravity.")
        ]
    },
    {
        "id": "05",
        "name": "05 — Startup Enablement (50% Off)",
        "shortName": "Startup Enablement",
        "headline": "50% FLAT DISCOUNT",
        "sub": "FLAT DISCOUNT ACROSS PILLARS 01, 02, 03 & 04 (<$1M RAISED)",
        "minPrice": 1500,
        "maxPrice": 5000,
        "description": "Flat 50% discount on any consulting, training, or development sprint package for early-stage startups with <$1M funding raised. Rates from $1,500 USD (Sprints) to $5,000 USD (Quarterly). Stacks with regional Purchasing Power Parity.",
        "bullets": [
            ("Universal Discount", "Flat 50% reduction on any consulting, training, or development sprint."),
            ("Accessible Frontier Rates", "Complete agent deployments and workshops tailored to lean budgets."),
            ("Investor & Accelerator Ready", "Built for Pre-Seed, Seed, and bootstrapped teams (<$1M funding)."),
            ("Zero Equity Required", "100% cash fee milestone model with complete IP transfer upon handover.")
        ]
    },
    {
        "id": "06",
        "name": "06 — Global Equity & Parity Tiers",
        "shortName": "Parity & Custom Tiers",
        "headline": "25% – 75% OFF + SUBSIDIZED",
        "sub": "WORLD BANK PPP INDEX ACROSS 198 COUNTRIES",
        "minPrice": 0,
        "maxPrice": 0,
        "description": "6 Tiers of Payment & Parity: Tier 1 Developed (100%), Tier 2 Developing (75% / 25% off), Tier 3 Least Developed (50% / 50% off), Tier 4 Startups (50% off), Tier 5 Non-Profits & Academia (25% / 75% off), Tier 6 Underprivileged & Backward Nations (Custom subsidized quote).",
        "bullets": [
            ("198 Countries Supported", "Rates automatically calibrated using World Bank PPP indices."),
            ("Frictionless Verification", "Simple intake declaration with zero bureaucratic overhead."),
            ("Radical Tech Equity", "Ensuring emerging builders access frontier Generative AI agents."),
            ("Equal Scope Guarantee", "Exact same deliverables, sandboxes, and IP rights across all tiers.")
        ]
    }
]

# Authoritative A La Carte Modules (Individual Services)
A_LA_CARTE = [
    {"num": 1, "name": "🤖 AI Agent Mastery & Team Orchestration", "price": 1000, "per": "", "color": "gold"},
    {"num": 2, "name": "🎓 Corporate & Dev Training (Live & Remote)", "price": 1000, "per": "", "color": "cyan"},
    {"num": 3, "name": "📦 Local LLM Deployment & Cost Slashing", "price": 1000, "per": "", "color": "gold"},
    {"num": 4, "name": "🤖 AI Agent Orchestration & Tool Loops", "price": 1000, "per": "", "color": "violet"},
    {"num": 5, "name": "🧪 AI-Assisted Dev & Test-Driven Dev (TDD)", "price": 1000, "per": "", "color": "cyan"},
    {"num": 6, "name": "📊 AI Benchmarks & Evaluation Metrics (Evals)", "price": 1000, "per": "", "color": "gold"},
    {"num": 7, "name": "⚛️ Early Quantum Systems & Qiskit Labs", "price": 1000, "per": "", "color": "violet"},
    {"num": 8, "name": "🛠️ Frontier AI Tools Advisory", "price": 1000, "per": "", "color": "cyan"},
    {"num": 9, "name": "💼 Monthly AI Consulting Retainer", "price": 5000, "per": " / mo", "color": "gold"},
    {"num": 10, "name": "🔬 Technical Deep Dives & Architecture Docs", "price": 1000, "per": "", "color": "violet"}
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
        "title": "AI Agent Orchestration & Tool Loops",
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
    "service-training-for-workflows-automations": {
        "title": "Training for Workflows, Agent Swarms & Automations",
        "min": 4000,
        "max": 8000,
        "unit": "per enterprise training sprint or workflow rollout"
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
                format_price_spec("Tier 6 Underprivileged & Backward Nations", 0, 0, "Custom subsidized quotes and micro-rates across 198 countries")
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
            "priceSpecification": price_specs
        }
        elements.append(offer)
    return elements

def sync_pricing_schema_json():
    pricing_schema_path = os.path.join(SCHEMA_DIR, "pricing-schema.json")
    with open(pricing_schema_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Locate OfferCatalog in @graph
    for item in data.get("@graph", []):
        if item.get("@type") == "OfferCatalog":
            item["itemListElement"] = build_offer_catalog_elements()
            break
            
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
            if item.get("@type") == "OfferCatalog":
                item["itemListElement"] = build_offer_catalog_elements()
                break
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

    pillars_html = """    <!-- 6-PILLAR PRICING SECTION -->
    <section class="reveal" id="pricing-pillars" style="margin-bottom:3.5rem;">
      <div class="packages-grid"
        style="display:grid;grid-template-columns:repeat(auto-fit, minmax(min(100%, 340px), 1fr));gap:2.5rem 2rem;max-width:1380px;margin:0 auto;padding:1rem 1.5rem 0 1.5rem;align-items:stretch;">
        <!-- Pillar 01: AI Agent Mastery & Team Orchestration -->
        <div class="package-card popular-card border-cyan"
          style="background:linear-gradient(155deg, rgba(8, 25, 45, 0.95) 0%, rgba(3, 7, 18, 0.98) 100%);border:2px solid #00d2ff;border-radius:var(--radius-lg);padding:2.2rem 1.85rem;display:flex;flex-direction:column;justify-content:space-between;position:relative;transition:var(--transition);box-shadow:0 0 28px rgba(0, 210, 255, 0.25), 0 10px 30px rgba(0,0,0,0.7);">
          <div
            style="position:absolute;top:-13px;left:50%;transform:translateX(-50%);background:linear-gradient(135deg, #00d2ff 0%, #0284c7 100%);color:#000000;font-family:var(--font-mono);font-size:calc(0.75rem + 3px);font-weight:900;letter-spacing:0.08em;text-transform:uppercase;padding:0.35rem 1.1rem;border-radius:9999px;box-shadow:0 0 18px rgba(0, 210, 255, 0.65);white-space:nowrap;">
            ★ AGENT ORCHESTRATION MASTERY
          </div>
          <div>
            <span
              style="font-size:calc(0.75rem + 3px);font-weight:800;letter-spacing:0.07em;margin-bottom:0.75rem;display:inline-block;background:rgba(0, 210, 255, 0.18);border:1px solid #00d2ff;color:#00f0ff;padding:0.35rem 0.85rem;border-radius:6px;">01
              · AGENT ORCHESTRATION</span>
            <h3 style="font-size:1.55rem;font-weight:800;color:var(--text-primary);margin-bottom:0.35rem;">AI Agent
              Mastery</h3>
            <p style="color:var(--text-muted);font-size:calc(0.88rem + 3px);margin-bottom:1.1rem;line-height:1.55;">Create teams of AI
              agents with Claude, OpenAI, Gemini, or Grok. Master multi-agent orchestration principles &amp; constraints.</p>
            <div style="margin-bottom:1.1rem;padding-bottom:1rem;border-bottom:1px solid rgba(0, 210, 255, 0.3);">
              <div
                style="font-family:var(--font-display);font-size:2rem;font-weight:900;color:#00f0ff;line-height:1.1;text-shadow:0 0 16px rgba(0, 240, 255, 0.4);">
                $3,000 – $6,000</div>
              <span
                style="font-family:var(--font-mono);font-size:calc(0.75rem + 3px);color:#7dd3fc;letter-spacing:0.06em;font-weight:700;">USD
                · SPRINT ($3,000) OR DEPLOYMENT ($6,000)</span>
            </div>
            <!-- Parity Box -->
            <div class="pillar-parity-box">
              <div class="pillar-parity-header">
                <span>PARITY PRICING TIERS</span>
                <span style="color:#00f0ff;font-size:calc(0.65rem + 3px);">198 NATIONS</span>
              </div>
              <div class="pillar-parity-grid">
                <div class="pillar-parity-item">
                  <span class="ppp-tag ppp-t2">25% OFF (TIER 2)</span>
                  <span class="ppp-val">$2,250 / $4,500</span>
                </div>
                <div class="pillar-parity-item">
                  <span class="ppp-tag ppp-t3">50% OFF (T3 &amp; STARTUPS)</span>
                  <span class="ppp-val">$1,500 / $3,000</span>
                </div>
                <div class="pillar-parity-item">
                  <span class="ppp-tag ppp-t5">75% OFF (ACADEMIA)</span>
                  <span class="ppp-val">$750 / $1,500</span>
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
                <span><strong>Multi-Agent Architecture:</strong> Coordinated teams using Claude, OpenAI, Gemini, or Grok.</span>
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
        <!-- Pillar 02: Corporate & Developer AI Training (Live & Remote) -->
        <div class="package-card border-gold"
          style="background:linear-gradient(155deg, rgba(35, 26, 6, 0.95) 0%, rgba(12, 9, 2, 0.98) 100%);border:2px solid #fbbf24;border-radius:var(--radius-lg);padding:2.2rem 1.85rem;display:flex;flex-direction:column;justify-content:space-between;position:relative;transition:var(--transition);box-shadow:0 0 28px rgba(251, 191, 36, 0.25), 0 10px 30px rgba(0,0,0,0.7);">
          <div
            style="position:absolute;top:-13px;left:50%;transform:translateX(-50%);background:linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);color:#000000;font-family:var(--font-mono);font-size:calc(0.75rem + 3px);font-weight:900;letter-spacing:0.08em;text-transform:uppercase;padding:0.35rem 1.1rem;border-radius:9999px;box-shadow:0 0 18px rgba(251, 191, 36, 0.65);white-space:nowrap;">
            ★ LIVE &amp; REMOTE TRAINING
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
                style="font-family:var(--font-display);font-size:2rem;font-weight:900;color:#fbbf24;line-height:1.1;text-shadow:0 0 16px rgba(251, 191, 36, 0.4);">
                $4,000 – $8,000</div>
              <span
                style="font-family:var(--font-mono);font-size:calc(0.75rem + 3px);color:#fde68a;letter-spacing:0.06em;font-weight:700;">USD
                · STANDARD ($4,000) OR ENTERPRISE ($8,000)</span>
            </div>
            <!-- Parity Box -->
            <div class="pillar-parity-box">
              <div class="pillar-parity-header">
                <span>PARITY PRICING TIERS</span>
                <span style="color:#fbbf24;font-size:calc(0.65rem + 3px);">198 NATIONS</span>
              </div>
              <div class="pillar-parity-grid">
                <div class="pillar-parity-item">
                  <span class="ppp-tag ppp-t2">25% OFF (TIER 2)</span>
                  <span class="ppp-val">$3,000 / $6,000</span>
                </div>
                <div class="pillar-parity-item">
                  <span class="ppp-tag ppp-t3">50% OFF (T3 &amp; STARTUPS)</span>
                  <span class="ppp-val">$2,000 / $4,000</span>
                </div>
                <div class="pillar-parity-item">
                  <span class="ppp-tag ppp-t5">75% OFF (ACADEMIA)</span>
                  <span class="ppp-val">$1,000 / $2,000</span>
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
            ★ COST SLASHING &amp; PRIVACY
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
                style="font-family:var(--font-display);font-size:2rem;font-weight:900;color:#c084fc;line-height:1.1;text-shadow:0 0 16px rgba(192, 132, 252, 0.4);">
                $3,000 – $5,000</div>
              <span
                style="font-family:var(--font-mono);font-size:calc(0.75rem + 3px);color:#e9d5ff;letter-spacing:0.06em;font-weight:700;">USD
                · CLUSTER ($3,000) OR PIPELINE ($5,000)</span>
            </div>
            <!-- Parity Box -->
            <div class="pillar-parity-box">
              <div class="pillar-parity-header">
                <span>PARITY PRICING TIERS</span>
                <span style="color:#c084fc;font-size:calc(0.65rem + 3px);">198 NATIONS</span>
              </div>
              <div class="pillar-parity-grid">
                <div class="pillar-parity-item">
                  <span class="ppp-tag ppp-t2">25% OFF (TIER 2)</span>
                  <span class="ppp-val">$2,250 / $3,750</span>
                </div>
                <div class="pillar-parity-item">
                  <span class="ppp-tag ppp-t3">50% OFF (T3 &amp; STARTUPS)</span>
                  <span class="ppp-val">$1,500 / $2,500</span>
                </div>
                <div class="pillar-parity-item">
                  <span class="ppp-tag ppp-t5">75% OFF (ACADEMIA)</span>
                  <span class="ppp-val">$750 / $1,250</span>
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
        <!-- Pillar 04: Dedicated Sprints, Agent Swarms & Retainers -->
        <div class="package-card border-emerald"
          style="background:linear-gradient(155deg, rgba(4, 38, 26, 0.95) 0%, rgba(2, 18, 12, 0.98) 100%);border:2px solid #10b981;border-radius:var(--radius-lg);padding:2.2rem 1.85rem;display:flex;flex-direction:column;justify-content:space-between;position:relative;transition:var(--transition);box-shadow:0 0 28px rgba(16, 185, 129, 0.25), 0 10px 30px rgba(0,0,0,0.7);">
          <div
            style="position:absolute;top:-13px;left:50%;transform:translateX(-50%);background:linear-gradient(135deg, #34d399 0%, #059669 100%);color:#000000;font-family:var(--font-mono);font-size:calc(0.75rem + 3px);font-weight:900;letter-spacing:0.08em;text-transform:uppercase;padding:0.35rem 1.1rem;border-radius:9999px;box-shadow:0 0 18px rgba(16, 185, 129, 0.65);white-space:nowrap;">
            ★ DEDICATED SPRINT CAPACITY
          </div>
          <div>
            <span
              style="font-size:calc(0.75rem + 3px);font-weight:800;letter-spacing:0.07em;margin-bottom:0.75rem;display:inline-block;background:rgba(16, 185, 129, 0.2);border:1px solid #10b981;color:#34d399;padding:0.35rem 0.85rem;border-radius:6px;">04
              · CONTINUOUS EXECUTION</span>
            <h3 style="font-size:1.55rem;font-weight:800;color:var(--text-primary);margin-bottom:0.35rem;">Dedicated AI
              Sprints</h3>
            <p style="color:var(--text-muted);font-size:calc(0.88rem + 3px);margin-bottom:1.1rem;line-height:1.55;">Continuous
              engineering capacity for agent swarms, AI workflows, and TDD transformation.</p>
            <div style="margin-bottom:1.1rem;padding-bottom:1rem;border-bottom:1px solid rgba(16, 185, 129, 0.3);">
              <div
                style="font-family:var(--font-display);font-size:2rem;font-weight:900;color:#34d399;line-height:1.1;text-shadow:0 0 16px rgba(52, 211, 153, 0.45);">
                $3,000 – $10,000</div>
              <span
                style="font-family:var(--font-mono);font-size:calc(0.75rem + 3px);color:#6ee7b7;letter-spacing:0.06em;font-weight:700;">USD
                · 2-WK ($3K) · MONTHLY ($5K) · QTR ($10K)</span>
            </div>
            <!-- Parity Box -->
            <div class="pillar-parity-box">
              <div class="pillar-parity-header">
                <span>PARITY PRICING TIERS</span>
                <span style="color:#34d399;font-size:calc(0.65rem + 3px);">198 NATIONS</span>
              </div>
              <div class="pillar-parity-grid">
                <div class="pillar-parity-item">
                  <span class="ppp-tag ppp-t2">25% OFF (TIER 2)</span>
                  <span class="ppp-val">$2,250 / $3,750 / $7,500</span>
                </div>
                <div class="pillar-parity-item">
                  <span class="ppp-tag ppp-t3">50% OFF (T3 &amp; STARTUPS)</span>
                  <span class="ppp-val">$1,500 / $2,500 / $5,000</span>
                </div>
                <div class="pillar-parity-item">
                  <span class="ppp-tag ppp-t5">75% OFF (ACADEMIA)</span>
                  <span class="ppp-val">$750 / $1,250 / $2,500</span>
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
                <span style="color:#10b981;font-weight:bold;">✓</span>
                <span><strong>Reserved Sprint Capacity:</strong> Guaranteed bandwidth for agent swarms and evals.</span>
              </li>
              <li style="display:flex;align-items:flex-start;gap:0.6rem;">
                <span style="color:#10b981;font-weight:bold;">✓</span>
                <span><strong>Direct Async Channel:</strong> Dedicated private Slack with &lt;4-hr response SLA.</span>
              </li>
              <li style="display:flex;align-items:flex-start;gap:0.6rem;">
                <span style="color:#10b981;font-weight:bold;">✓</span>
                <span><strong>Weekly Architecture Sync:</strong> Technical alignment, backlog prioritization &amp; reviews.</span>
              </li>
            </ul>
          </div>
          <a class="btn btn-glow-emerald" href="contact.html?package=Dedicated%20AI%20Sprints"
            style="width:100%;justify-content:center;padding:0.85rem 1rem;font-size:calc(0.88rem + 3px);font-weight:800;letter-spacing:0.02em;border:3px solid #34d399;">Reserve
            Sprint Capacity →</a>
        </div>
        <!-- Pillar 05: Startup Enablement (50% Off) -->
        <div class="package-card popular-card border-gold"
          style="background:linear-gradient(155deg, rgba(40, 30, 8, 0.96) 0%, rgba(15, 11, 2, 0.98) 100%);border:2px solid #fbbf24;border-radius:var(--radius-lg);padding:2.2rem 1.85rem;display:flex;flex-direction:column;justify-content:space-between;position:relative;transition:var(--transition);box-shadow:0 0 35px rgba(251, 191, 36, 0.4), 0 12px 35px rgba(0,0,0,0.8);transform:scale(1.02);">
          <div
            style="position:absolute;top:-13px;left:50%;transform:translateX(-50%);background:linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);color:#000000;font-family:var(--font-mono);font-size:calc(0.75rem + 3px);font-weight:900;letter-spacing:0.08em;text-transform:uppercase;padding:0.35rem 1.1rem;border-radius:9999px;box-shadow:0 0 18px rgba(251, 191, 36, 0.65);white-space:nowrap;">
            ★ 50% STARTUP DISCOUNT
          </div>
          <div>
            <span
              style="font-size:calc(0.75rem + 3px);font-weight:800;letter-spacing:0.07em;margin-bottom:0.75rem;display:inline-block;background:rgba(251, 191, 36, 0.25);border:1px solid #fbbf24;color:#fde047;padding:0.35rem 0.85rem;border-radius:6px;">05
              · EARLY-STAGE ENABLEMENT</span>
            <h3 style="font-size:1.55rem;font-weight:800;color:var(--text-primary);margin-bottom:0.35rem;">Startup
              Enablement</h3>
            <p style="color:var(--text-muted);font-size:calc(0.88rem + 3px);margin-bottom:1.1rem;line-height:1.55;">50% off any of the
              above consulting and training pillars for early-stage startups.</p>
            <div style="margin-bottom:1.1rem;padding-bottom:1rem;border-bottom:1px solid rgba(251, 191, 36, 0.3);">
              <div
                style="font-family:var(--font-display);font-size:2rem;font-weight:900;color:#fbbf24;line-height:1.1;text-shadow:0 0 16px rgba(251, 191, 36, 0.45);">
                50% OFF</div>
              <span
                style="font-family:var(--font-mono);font-size:calc(0.75rem + 3px);color:#fde68a;letter-spacing:0.06em;font-weight:700;">FLAT
                DISCOUNT ON PILLARS 01, 02, 03 &amp; 04</span>
            </div>
            <!-- Parity Box -->
            <div class="pillar-parity-box">
              <div class="pillar-parity-header">
                <span>STARTUP PARITY RATES</span>
                <span style="color:#fbbf24;font-size:calc(0.65rem + 3px);">&lt;$1M RAISED</span>
              </div>
              <div class="pillar-parity-grid">
                <div class="pillar-parity-item">
                  <span class="ppp-tag ppp-t2">DEVELOPED STARTUPS</span>
                  <span class="ppp-val">From $1,500 USD</span>
                </div>
                <div class="pillar-parity-item">
                  <span class="ppp-tag ppp-t3">DEVELOPING (T2)</span>
                  <span class="ppp-val">From $1,125 USD</span>
                </div>
                <div class="pillar-parity-item">
                  <span class="ppp-tag ppp-t5">LDC FOUNDERS (T3)</span>
                  <span class="ppp-val">From $750 USD</span>
                </div>
                <div class="pillar-parity-item">
                  <span class="ppp-tag ppp-t6">TIER 6 RELIEF</span>
                  <span class="ppp-val">Micro-Grants</span>
                </div>
              </div>
            </div>
            <ul
              style="list-style:none;display:flex;flex-direction:column;gap:0.75rem;margin-bottom:1.5rem;font-size:calc(0.87rem + 3px);color:var(--text-secondary);line-height:1.5;">
              <li style="display:flex;align-items:flex-start;gap:0.6rem;">
                <span style="color:#fbbf24;font-weight:bold;">✓</span>
                <span><strong>Applies to Any Offering:</strong> Half-price on Agent Mastery, Training, Local LLMs, or Retainers.</span>
              </li>
              <li style="display:flex;align-items:flex-start;gap:0.6rem;">
                <span style="color:#fbbf24;font-weight:bold;">✓</span>
                <span><strong>Stacks with Parity:</strong> Seamlessly stacks with Country Parity Pricing for global founders.</span>
              </li>
              <li style="display:flex;align-items:flex-start;gap:0.6rem;">
                <span style="color:#fbbf24;font-weight:bold;">🚀</span>
                <span><strong>Eligibility:</strong> Pre-Seed, Seed-stage, or bootstrapped teams under $1M total funding raised.</span>
              </li>
            </ul>
          </div>
          <a class="btn btn-glow-gold" href="contact.html?package=Startup%20Enablement"
            style="width:100%;justify-content:center;padding:0.85rem 1rem;font-size:calc(0.88rem + 3px);font-weight:800;letter-spacing:0.02em;border:3px solid #fbbf24;">Apply
            for Startup Pricing →</a>
        </div>
        <!-- Pillar 06: Global Equity & Custom Tier -->
        <div class="package-card popular-card border-cyan"
          style="background:linear-gradient(155deg, rgba(8, 25, 45, 0.96) 0%, rgba(3, 7, 18, 0.98) 100%);border:2px solid #00d2ff;border-radius:var(--radius-lg);padding:2.2rem 1.85rem;display:flex;flex-direction:column;justify-content:space-between;position:relative;transition:var(--transition);box-shadow:0 0 35px rgba(0, 210, 255, 0.4), 0 12px 35px rgba(0,0,0,0.8);transform:scale(1.02);">
          <div
            style="position:absolute;top:-13px;left:50%;transform:translateX(-50%);background:linear-gradient(135deg, #00d2ff 0%, #3b82f6 100%);color:#000000;font-family:var(--font-mono);font-size:calc(0.75rem + 3px);font-weight:900;letter-spacing:0.08em;text-transform:uppercase;padding:0.35rem 1.1rem;border-radius:9999px;box-shadow:0 0 18px rgba(0, 210, 255, 0.65);white-space:nowrap;">
            🌐 6 TIERS OF PAYMENT &amp; PARITY
          </div>
          <div>
            <span
              style="font-size:calc(0.75rem + 3px);font-weight:800;letter-spacing:0.07em;margin-bottom:0.75rem;display:inline-block;background:rgba(0, 210, 255, 0.25);border:1px solid #00d2ff;color:#00f0ff;padding:0.35rem 0.85rem;border-radius:6px;">06
              · GLOBAL EQUITY</span>
            <h3 style="font-size:1.55rem;font-weight:800;color:var(--text-primary);margin-bottom:0.35rem;">Parity &amp;
              Custom Tiers</h3>
            <p style="color:var(--text-muted);font-size:calc(0.88rem + 3px);margin-bottom:1.1rem;line-height:1.55;">Purchasing Power
              Parity across 198 countries and subsidized rates for academia, non-profits, and underprivileged teams.</p>
            <div style="margin-bottom:1.1rem;padding-bottom:1rem;border-bottom:1px solid rgba(0, 210, 255, 0.3);">
              <div
                style="font-family:var(--font-display);font-size:2rem;font-weight:900;color:#00f0ff;line-height:1.1;text-shadow:0 0 16px rgba(0, 240, 255, 0.45);">
                25% – 75% OFF</div>
              <span
                style="font-family:var(--font-mono);font-size:calc(0.75rem + 3px);color:#7dd3fc;letter-spacing:0.06em;font-weight:700;">+
                SUBSIDIZED RATES FOR ACADEMIA &amp; TIER 6</span>
            </div>
            <!-- Parity Box -->
            <div class="pillar-parity-box">
              <div class="pillar-parity-header">
                <span>6 PAYMENT TIERS</span>
                <span style="color:#00f0ff;font-size:calc(0.65rem + 3px);">CALIBRATED PPP</span>
              </div>
              <div class="pillar-parity-grid">
                <div class="pillar-parity-item">
                  <span class="ppp-tag ppp-t2">TIER 2 (DEVELOPING)</span>
                  <span class="ppp-val">25% Discount</span>
                </div>
                <div class="pillar-parity-item">
                  <span class="ppp-tag ppp-t3">TIER 3 &amp; 4 (LDC / STARTUP)</span>
                  <span class="ppp-val">50% Discount</span>
                </div>
                <div class="pillar-parity-item">
                  <span class="ppp-tag ppp-t5">TIER 5 (ACADEMIA / NGO)</span>
                  <span class="ppp-val">75% Subsidy</span>
                </div>
                <div class="pillar-parity-item">
                  <span class="ppp-tag ppp-t6">TIER 6 (UNDERPRIVILEGED)</span>
                  <span class="ppp-val">Custom Subsidized</span>
                </div>
              </div>
            </div>
            <ul
              style="list-style:none;display:flex;flex-direction:column;gap:0.75rem;margin-bottom:1.5rem;font-size:calc(0.87rem + 3px);color:var(--text-secondary);line-height:1.5;">
              <li style="display:flex;align-items:flex-start;gap:0.6rem;">
                <span style="color:#00f0ff;font-weight:bold;">✓</span>
                <span><strong>198 Countries Supported:</strong> Rates automatically calibrated using World Bank PPP indices.</span>
              </li>
              <li style="display:flex;align-items:flex-start;gap:0.6rem;">
                <span style="color:#00f0ff;font-weight:bold;">✓</span>
                <span><strong>75% Academia Subsidy:</strong> Accessible for universities, clinical labs, and registered 501(c)(3)s.</span>
              </li>
              <li style="display:flex;align-items:flex-start;gap:0.6rem;">
                <span style="color:#00f0ff;font-weight:bold;">🤝</span>
                <span><strong>Radical Equity Commitment:</strong> Bespoke micro-rates and relief for distress &amp; conflict zones.</span>
              </li>
            </ul>
          </div>
          <a class="btn btn-glow-cyan" href="contact.html?package=Parity%20Quote"
            style="width:100%;justify-content:center;padding:0.85rem 1rem;font-size:calc(0.88rem + 3px);font-weight:800;letter-spacing:0.02em;border:3px solid #00d2ff;">Request
            Parity Quote →</a>
        </div>
      </div>
    </section>"""

    services_html = """    <!-- INDIVIDUAL SERVICES & REFERENCE RATES -->
    <section class="reveal" id="service-rates" style="margin-bottom:4.5rem;">
      <div style="max-width:1380px;margin:0 auto;padding:0 1.5rem;">
        <div style="text-align:center;margin-bottom:2.5rem;">
          <p class="section-label">// REFERENCE SPECIFICATIONS</p>
          <h2 class="section-title">Individual Services &amp; <span class="accent-cyan">Base Rates</span></h2>
          <p class="section-sub" style="max-width:760px;margin:0.5rem auto 0 auto;line-height:1.7;">
            A la carte specifications across AI Agent Mastery, live/remote training, local LLMs, AI agent swarms, and
            quantum systems.
          </p>
        </div>
        <div class="pricing-grid-faq">
          <!-- Item 1 -->
          <div class="price-card-faq border-gold">
            <div class="price-card-title"
              style="color:var(--gold);font-weight:700;font-size:0.98rem;min-height:2.6rem;display:flex;align-items:center;justify-content:center;">
              🤖 AI Agent Mastery &amp; Team Orchestration</div>
            <div class="price-amount"
              style="font-size:1.3rem;font-weight:800;margin:0.4rem 0 0.2rem 0;color:#fbbf24;text-shadow:0 0 12px rgba(251,191,36,0.35);">
              $1,000 USD
            </div>
            <div class="service-parity-strip">
              <span>25% Off: <strong>$750</strong></span>
              <span>50% Off: <strong>$500</strong></span>
              <span>75% Off: <strong>$250</strong></span>
              <span>T6: <strong>Relief</strong></span>
            </div>
            <div
              style="font-size:calc(0.84rem + 3px);color:var(--text-secondary);line-height:1.5;flex-grow:1;display:flex;align-items:center;justify-content:center;">
              Build multi-agent teams with Claude, OpenAI, Gemini, Grok; master orchestration rules &amp; domain tasks.
            </div>
          </div>
          <!-- Item 2 -->
          <div class="price-card-faq border-cyan">
            <div class="price-card-title"
              style="color:var(--cyan);font-weight:700;font-size:0.98rem;min-height:2.6rem;display:flex;align-items:center;justify-content:center;">
              🎓 Corporate &amp; Dev Training (Live &amp; Remote)</div>
            <div class="price-amount"
              style="font-size:1.3rem;font-weight:800;margin:0.4rem 0 0.2rem 0;color:#00f0ff;text-shadow:0 0 12px rgba(0,240,255,0.35);">
              $1,000 USD
            </div>
            <div class="service-parity-strip">
              <span>25% Off: <strong>$750</strong></span>
              <span>50% Off: <strong>$500</strong></span>
              <span>75% Off: <strong>$250</strong></span>
              <span>T6: <strong>Relief</strong></span>
            </div>
            <div
              style="font-size:calc(0.84rem + 3px);color:var(--text-secondary);line-height:1.5;flex-grow:1;display:flex;align-items:center;justify-content:center;">
              Live &amp; remote workshops for Employees (adoption), Developers (TDD pairing), Staff, and Executive Leaders.
            </div>
          </div>
          <!-- Item 3 -->
          <div class="price-card-faq border-gold">
            <div class="price-card-title"
              style="color:var(--gold);font-weight:700;font-size:0.98rem;min-height:2.6rem;display:flex;align-items:center;justify-content:center;">
              📦 Local LLM Deployment &amp; Cost Slashing</div>
            <div class="price-amount"
              style="font-size:1.3rem;font-weight:800;margin:0.4rem 0 0.2rem 0;color:#fbbf24;text-shadow:0 0 12px rgba(251,191,36,0.35);">
              $1,000 USD
            </div>
            <div class="service-parity-strip">
              <span>25% Off: <strong>$750</strong></span>
              <span>50% Off: <strong>$500</strong></span>
              <span>75% Off: <strong>$250</strong></span>
              <span>T6: <strong>Relief</strong></span>
            </div>
            <div
              style="font-size:calc(0.84rem + 3px);color:var(--text-secondary);line-height:1.5;flex-grow:1;display:flex;align-items:center;justify-content:center;">
              Lowering AI inference costs 60–90% with Ollama, vLLM, llama.cpp, GGUF quantization &amp; on-prem private clusters.
            </div>
          </div>
          <!-- Item 4 -->
          <div class="price-card-faq border-violet">
            <div class="price-card-title"
              style="color:var(--violet);font-weight:700;font-size:0.98rem;min-height:2.6rem;display:flex;align-items:center;justify-content:center;">
              🤖 AI Agent Orchestration &amp; Tool Loops</div>
            <div class="price-amount"
              style="font-size:1.3rem;font-weight:800;margin:0.4rem 0 0.2rem 0;color:#c084fc;text-shadow:0 0 12px rgba(192,132,252,0.35);">
              $1,000 USD
            </div>
            <div class="service-parity-strip">
              <span>25% Off: <strong>$750</strong></span>
              <span>50% Off: <strong>$500</strong></span>
              <span>75% Off: <strong>$250</strong></span>
              <span>T6: <strong>Relief</strong></span>
            </div>
            <div
              style="font-size:calc(0.84rem + 3px);color:var(--text-secondary);line-height:1.5;flex-grow:1;display:flex;align-items:center;justify-content:center;">
              Multi-agent architecture design, LangGraph, CrewAI, AutoGen, OpenClaw, tool calling &amp; stateful workflows.
            </div>
          </div>
          <!-- Item 5 -->
          <div class="price-card-faq border-cyan">
            <div class="price-card-title"
              style="color:var(--cyan);font-weight:700;font-size:0.98rem;min-height:2.6rem;display:flex;align-items:center;justify-content:center;">
              🧪 AI-Assisted Dev &amp; Test-Driven Dev (TDD)</div>
            <div class="price-amount"
              style="font-size:1.3rem;font-weight:800;margin:0.4rem 0 0.2rem 0;color:#00f0ff;text-shadow:0 0 12px rgba(0,240,255,0.35);">
              $1,000 USD
            </div>
            <div class="service-parity-strip">
              <span>25% Off: <strong>$750</strong></span>
              <span>50% Off: <strong>$500</strong></span>
              <span>75% Off: <strong>$250</strong></span>
              <span>T6: <strong>Relief</strong></span>
            </div>
            <div
              style="font-size:calc(0.84rem + 3px);color:var(--text-secondary);line-height:1.5;flex-grow:1;display:flex;align-items:center;justify-content:center;">
              Claude Code, Google Antigravity &amp; OpenCode pairing, automated pytest/cargo test suites &amp; TDD workflows.
            </div>
          </div>
          <!-- Item 6 -->
          <div class="price-card-faq border-gold">
            <div class="price-card-title"
              style="color:var(--gold);font-weight:700;font-size:0.98rem;min-height:2.6rem;display:flex;align-items:center;justify-content:center;">
              📊 AI Benchmarks &amp; Evaluation Metrics (Evals)</div>
            <div class="price-amount"
              style="font-size:1.3rem;font-weight:800;margin:0.4rem 0 0.2rem 0;color:#fbbf24;text-shadow:0 0 12px rgba(251,191,36,0.35);">
              $1,000 USD
            </div>
            <div class="service-parity-strip">
              <span>25% Off: <strong>$750</strong></span>
              <span>50% Off: <strong>$500</strong></span>
              <span>75% Off: <strong>$250</strong></span>
              <span>T6: <strong>Relief</strong></span>
            </div>
            <div
              style="font-size:calc(0.84rem + 3px);color:var(--text-secondary);line-height:1.5;flex-grow:1;display:flex;align-items:center;justify-content:center;">
              Evaluation harnesses using Ragas, DeepEval, Promptfoo, Langfuse; latency, token cost &amp; hallucination benchmarks.
            </div>
          </div>
          <!-- Item 7 -->
          <div class="price-card-faq border-violet">
            <div class="price-card-title"
              style="color:var(--violet);font-weight:700;font-size:0.98rem;min-height:2.6rem;display:flex;align-items:center;justify-content:center;">
              ⚛️ Early Quantum Systems &amp; Qiskit Labs</div>
            <div class="price-amount"
              style="font-size:1.3rem;font-weight:800;margin:0.4rem 0 0.2rem 0;color:#c084fc;text-shadow:0 0 12px rgba(192,132,252,0.35);">
              $1,000 USD
            </div>
            <div class="service-parity-strip">
              <span>25% Off: <strong>$750</strong></span>
              <span>50% Off: <strong>$500</strong></span>
              <span>75% Off: <strong>$250</strong></span>
              <span>T6: <strong>Relief</strong></span>
            </div>
            <div
              style="font-size:calc(0.84rem + 3px);color:var(--text-secondary);line-height:1.5;flex-grow:1;display:flex;align-items:center;justify-content:center;">
              IBM Qiskit &amp; PennyLane quantum circuits, variational algorithms (VQE, QAOA) &amp; quantum readiness roadmaps.
            </div>
          </div>
          <!-- Item 8 -->
          <div class="price-card-faq border-cyan">
            <div class="price-card-title"
              style="color:var(--cyan);font-weight:700;font-size:0.98rem;min-height:2.6rem;display:flex;align-items:center;justify-content:center;">
              🛠️ Frontier AI Tools Advisory</div>
            <div class="price-amount"
              style="font-size:1.3rem;font-weight:800;margin:0.4rem 0 0.2rem 0;color:#00f0ff;text-shadow:0 0 12px rgba(0,240,255,0.35);">
              $1,000 USD
            </div>
            <div class="service-parity-strip">
              <span>25% Off: <strong>$750</strong></span>
              <span>50% Off: <strong>$500</strong></span>
              <span>75% Off: <strong>$250</strong></span>
              <span>T6: <strong>Relief</strong></span>
            </div>
            <div
              style="font-size:calc(0.84rem + 3px);color:var(--text-secondary);line-height:1.5;flex-grow:1;display:flex;align-items:center;justify-content:center;">
              Setup &amp; advisory for Claude Code, Google Antigravity, ChatGPT, HuggingFace Spaces, OpenAI Codex &amp; OpenCode.
            </div>
          </div>
          <!-- Item 9 -->
          <div class="price-card-faq border-gold">
            <div class="price-card-title"
              style="color:var(--gold);font-weight:700;font-size:0.98rem;min-height:2.6rem;display:flex;align-items:center;justify-content:center;">
              💼 Monthly AI Consulting Retainer</div>
            <div class="price-amount"
              style="font-size:1.3rem;font-weight:800;margin:0.4rem 0 0.2rem 0;color:#fbbf24;text-shadow:0 0 12px rgba(251,191,36,0.35);">
              $5,000 USD
              <span style="font-size:calc(0.75rem + 3px);font-weight:400;color:var(--text-muted);">/ mo</span>
            </div>
            <div class="service-parity-strip">
              <span>25% Off: <strong>$3,750</strong></span>
              <span>50% Off: <strong>$2,500</strong></span>
              <span>75% Off: <strong>$1,250</strong></span>
              <span>T6: <strong>Relief</strong></span>
            </div>
            <div
              style="font-size:calc(0.84rem + 3px);color:var(--text-secondary);line-height:1.5;flex-grow:1;display:flex;align-items:center;justify-content:center;">
              Dedicated monthly engineering capacity for AI Agent Mastery, team upskilling, cost audits &amp; architecture leadership.
            </div>
          </div>
          <!-- Item 10 -->
          <div class="price-card-faq border-violet">
            <div class="price-card-title"
              style="color:var(--violet);font-weight:700;font-size:0.98rem;min-height:2.6rem;display:flex;align-items:center;justify-content:center;">
              🔬 Technical Deep Dives &amp; Architecture Docs</div>
            <div class="price-amount"
              style="font-size:1.3rem;font-weight:800;margin:0.4rem 0 0.2rem 0;color:#c084fc;text-shadow:0 0 12px rgba(192,132,252,0.35);">
              $1,000 USD
            </div>
            <div class="service-parity-strip">
              <span>25% Off: <strong>$750</strong></span>
              <span>50% Off: <strong>$500</strong></span>
              <span>75% Off: <strong>$250</strong></span>
              <span>T6: <strong>Relief</strong></span>
            </div>
            <div
              style="font-size:calc(0.84rem + 3px);color:var(--text-secondary);line-height:1.5;flex-grow:1;display:flex;align-items:center;justify-content:center;">
              Long-form architectural teardowns, technical whitepapers, Diátaxis documentation, training modules &amp; code guides.
            </div>
          </div>
        </div>
      </div>
    </section>"""

    # Regex replacement for #pricing-pillars
    pillar_pattern = re.compile(
        r'<!-- 6-PILLAR PRICING SECTION -->\s*<section\b[^>]*id="pricing-pillars"[\s\S]*?</section>',
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

    new_packages_block = """- **Core Strategic Packages & Parity Rates (198 Countries Supported)**:
  * **01 · AI Agent Mastery Sprint**: **$3,000 USD** | Full Deployment: **$6,000 USD**
    - Parity: 25% Off (Tier 2): $2,250 / $4,500 | 50% Off (Tier 3/4 Startups): $1,500 / $3,000 | 75% Off (Tier 5): $750 / $1,500 | Tier 6: Subsidized Relief
  * **02 · Workforce AI Training**: **$4,000 USD** (Standard Cohort) | **$8,000 USD** (Enterprise Cohorts)
    - Parity: 25% Off (Tier 2): $3,000 / $6,000 | 50% Off (Tier 3/4 Startups): $2,000 / $4,000 | 75% Off (Tier 5): $1,000 / $2,000 | Tier 6: Subsidized Relief
  * **03 · Local LLM Deployment & Cost Slashing**: **$5,000 USD** | Private On-Prem Cluster Deployment: **$3,000 USD**
    - Parity: 25% Off (Tier 2): $2,250 / $3,750 | 50% Off (Tier 3/4 Startups): $1,500 / $2,500 | 75% Off (Tier 5): $750 / $1,250 | Tier 6: Subsidized Relief
  * **04 · Dedicated Fortnightly AI Sprint**: **$3,000 USD** / 2-wks
    - Parity: 25% Off: $2,250 | 50% Off: $1,500 | 75% Off: $750 | Tier 6: Subsidized Relief
  * **05 · Monthly Engineering Consulting Retainer**: **$5,000 USD** / mo
    - Parity: 25% Off: $3,750 | 50% Off: $2,500 | 75% Off: $1,250 | Tier 6: Subsidized Relief
  * **06 · Enterprise Quarterly Retainer**: **$10,000 USD** / qtr
    - Parity: 25% Off: $7,500 | 50% Off: $5,000 | 75% Off: $2,500 | Tier 6: Subsidized Relief
- **A La Carte Advisory & Modules**: **$1,000 USD** flat rate per module (AI Agents, Local LLM Setup, AI-Assisted Dev & TDD, Early Quantum Systems, Frontier AI Tools, Technical Deep Dives).
  - Parity: 25% Off: $750 USD | 50% Off: $500 USD | 75% Off: $250 USD | Tier 6: Subsidized Relief"""

    pattern = re.compile(
        r'-\s*\*\*Core Strategic Packages[\s\S]*?-\s*\*\*A La Carte Advisory & Modules\*\*:[\s\S]*?(?=\n\n|\n-|\n###)',
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
    sync_standalone_service_files()
    sync_llms_txt()
    print("--- Central Pricing & Parity Sync Pass Complete ---")

if __name__ == "__main__":
    sync_all_pricing()
