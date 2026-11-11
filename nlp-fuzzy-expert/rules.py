#!/usr/bin/env python3
"""
rules.py - Production Rules Knowledge Base for NLP-Fuzzy-Expert System.

Encodes expert knowledge, forward-chaining rules, and fuzzy NLP pattern matchers
for all repository documents.

Strict Repo Compliance:
- Zero emojis.
- Zero em-dashes (spaced hyphens ' - ' or plain hyphens '-' used).
"""

import os
import re
from expert_system import Rule
from nlp_engine import apply_sync_highlight, strip_sync_highlights

def build_rules():
    """Constructs and returns the full suite of Expert System production rules."""
    rules = []

    # -------------------------------------------------------------
    # 1. Pricing: Hero Preview Packages (pricing.html)
    # -------------------------------------------------------------
    def cond_pricing_hero(file_path, content, kb, nlp, fuzzy):
        if not file_path.endswith("pricing.html"):
            return False, {}
        has_biz = "Business Training Flagship" in content
        has_pers = "Personal Training Flagship" in content
        return (has_biz and has_pers), {}

    def act_pricing_hero(file_path, content, match_info, kb, nlp, fuzzy):
        occurrences = []
        b_price = kb.get_fact("pricing", "b_flag_price", 2000)
        b_days = kb.get_fact("pricing", "b_flag_days", 60)
        b_live = kb.get_fact("pricing", "b_flag_live_days", 30)
        b_supp = kb.get_fact("pricing", "b_flag_supp_days", 30)
        b_seats = kb.get_fact("pricing", "b_flag_seats", 10)
        b_extra = kb.get_fact("pricing", "b_flag_extra_seat", 200)
        b_notes = kb.get_fact("pricing", "b_flag_sched_notes", "30 working days over 8 weeks with 8 - 10 days as holidays and recovery days, 4 hours per day")

        p_price = kb.get_fact("pricing", "p_flag_price", 2000)
        p_days = kb.get_fact("pricing", "p_flag_days", 60)
        p_live = kb.get_fact("pricing", "p_flag_live_days", 30)
        p_supp = kb.get_fact("pricing", "p_flag_supp_days", 30)
        p_notes = kb.get_fact("pricing", "p_flag_sched_notes", "30 working days over 6 weeks with 8 - 10 days as holidays and recovery days, 4 hours per day")

        b_ind = kb.get_fact("pricing", "b_ind_price", 200)
        p_ind = kb.get_fact("pricing", "p_ind_price", 200)
        ind_live = kb.get_fact("pricing", "b_ind_live_days", 5)
        ind_supp = kb.get_fact("pricing", "b_ind_supp_days", 5)
        ind_days = kb.get_fact("pricing", "b_ind_days", 10)

        # 1A. Hero preview lines 394-405
        hero_b = f'<strong style="color:var(--cyan);font-weight:700;">Business Training Flagship ({apply_sync_highlight(f"${b_price:,} USD")} · {apply_sync_highlight(f"{b_days} Days")}):</strong> {apply_sync_highlight(f"{b_live} days live training")} + {apply_sync_highlight(f"{b_supp} days guidance and materials")} ({apply_sync_highlight(b_notes)}), up to {apply_sync_highlight(f"{b_seats} participants")} (+{apply_sync_highlight(f"${b_extra:,}")} per extra seat).'
        hero_p = f'<strong style="color:var(--gold);font-weight:700;">Personal Training Flagship ({apply_sync_highlight(f"${p_price:,} USD")} · {apply_sync_highlight(f"{p_days} Days")}):</strong> {apply_sync_highlight(f"{p_live} days live training")} + {apply_sync_highlight(f"{p_supp} days guidance and materials")} ({apply_sync_highlight(p_notes)}), strictly 1-on-1 private acceleration.'

        content = re.sub(
            r'<strong style="color:var\(--cyan\);font-weight:700;">Business Training Flagship[\s\S]*?per extra seat\)\.',
            hero_b,
            content
        )
        content = re.sub(
            r'<strong style="color:var\(--gold\);font-weight:700;">Personal Training Flagship[\s\S]*?strictly 1-on-1 private acceleration\.',
            hero_p,
            content
        )

        # 1B. Flagship Section Blocks (lines 428-450)
        b_block_h3 = f'<h3 style="font-family:var(--font-display);font-size:1.15rem;font-weight:800;color:var(--text-primary);margin-bottom:0.35rem;">\n            ${b_price:,} USD · {b_days} Days ({b_live}d Training + {b_supp}d Guidance &amp; Materials)\n          </h3>'
        b_block_p = f'<p style="font-size:0.88rem;color:var(--text-secondary);line-height:1.5;margin:0;">\n            Includes up to {b_seats} participants (additional seats at ${b_extra:,} USD per seat). Covers full cross-service engineering stacks tailored to your organizational goals. {b_live} days of live training + {b_supp} days of guidance and detailed guides/materials ({b_notes}).\n          </p>'
        content = re.sub(
            r'<h3[^>]*>\s*\$[\d,]+\s*USD\s*·\s*\d+\s*Days\s*\(\d+d Training \+ \d+d Guidance &amp; Materials\)\s*</h3>\s*<p[^>]*>\s*Includes up to \d+ participants[\s\S]*?\(additional seats at \$[\d,]+ USD per seat\)[\s\S]*?</p>',
            f'{b_block_h3}\n          {b_block_p}',
            content
        )

        p_block_h3 = f'<h3 style="font-family:var(--font-display);font-size:1.15rem;font-weight:800;color:var(--gold);margin-bottom:0.35rem;">\n            ${p_price:,} USD · {p_days} Days ({p_live}d Training + {p_supp}d Guidance &amp; Materials)\n          </h3>'
        p_block_p = f'<p style="font-size:0.88rem;color:var(--text-secondary);line-height:1.5;margin:0;">\n            Strictly for 1 person only (private 1-on-1 acceleration). {p_live} days of intensive live instruction followed by {p_supp} days of dedicated daily work guidance, Q&amp;A and written guides tailored to your individual career or venture ({p_notes}).\n          </p>'
        content = re.sub(
            r'<h3[^>]*color:var\(--gold\)[^>]*>\s*\$[\d,]+\s*USD\s*·\s*\d+\s*Days\s*\(\d+d Training \+ \d+d Guidance &amp; Materials\)\s*</h3>\s*<p[^>]*>\s*Strictly for 1 person only[\s\S]*?</p>',
            f'{p_block_h3}\n          {p_block_p}',
            content
        )

        # 1C. Section 6 Personal Training H2 and cards (lines 965-985)
        p_sec6_h2 = f'Personal Training: <span class="accent-gold">${p_price:,} USD for {p_days} Days</span> ({p_live}d Training + {p_supp}d Guidance)'
        content = re.sub(
            r'Personal Training:\s*<span class="accent-gold">\$[\d,]+\s*USD for \d+\s*Days</span>\s*\(\d+d Training \+ \d+d Guidance\)',
            p_sec6_h2,
            content
        )
        content = re.sub(
            r'Personal Training costs \$[\d,]+ USD in Tier 1 for \d+ days total:\s*\d+ days of intensive live instruction[\s\S]*?Delivered as [\w\s\-,+]+?\.\s*Strictly for one person only',
            f'Personal Training costs ${p_price:,} USD in Tier 1 for {p_days} days total: {p_live} days of intensive live instruction (4 hours per day) plus {p_supp} days of guidance, Q&amp;A and written guides. Delivered as {p_notes}. Strictly for one person only',
            content
        )
        content = re.sub(
            r'<h3 style="color:var\(--gold\);font-size:1\.1rem;font-weight:700;margin-bottom:0\.4rem;">\$[\d,]+\s*Base Rate',
            f'<h3 style="color:var(--gold);font-size:1.1rem;font-weight:700;margin-bottom:0.4rem;">${p_price:,} Base Rate',
            content
        )
        content = re.sub(
            r'Private 1-on-1 acceleration at \$[\d,]+ USD in Tier 1 with full 195-country Purchasing Power Parity',
            f'Private 1-on-1 acceleration at ${p_price:,} USD in Tier 1 with full 195-country Purchasing Power Parity',
            content
        )

        # 1D. Calculator initial display price & base value (lines 890, 903, 942)
        content = re.sub(
            r'<div class="calc-price-display" id="calc-display-price">\$[\d,]+ USD</div>',
            f'<div class="calc-price-display" id="calc-display-price">${b_price:,} USD</div>',
            content
        )
        content = re.sub(
            r'<span class="calc-summary-value" id="calc-base-value">\$[\d,]+ USD</span>',
            f'<span class="calc-summary-value" id="calc-base-value">${b_price:,} USD</span>',
            content
        )
        content = re.sub(
            r'Lock in Tier 1 Parity Rate \(\$[\d,]+\) →',
            f'Lock in Tier 1 Parity Rate (${b_price:,}) →',
            content
        )

        # 1E. Embedded schema JSON-LD in pricing.html (lines 54, 88, 177, 217)
        schema_desc = f"Transparent AI engagement model: Pick individual services at ${p_ind:,} USD per service for individuals and ${b_ind:,} USD per service for businesses (each covering {ind_live} days of training + {ind_supp} days of dedicated support, {ind_days} days total coverage). Clients can also choose our comprehensive programs: Business Training at ${b_price:,} USD for {b_days} days ({b_live} days training + {b_supp} days guidance/materials, {b_notes}, up to {b_seats} participants, +${b_extra} per extra seat) and Personal Training at ${p_price:,} USD for {p_days} days ({p_live} days training + {p_supp} days guidance/materials, {p_notes}, 1 person only). Full 195-country Purchasing Power Parity (PPP) supported across all service pricing."
        content = re.sub(
            r'"description":\s*"Transparent AI engagement model: Pick individual services at \$[\d,]+ USD per service for individuals and \$[\d,]+ USD per service for businesses[\s\S]*?Full 195-country Purchasing Power Parity \(PPP\) supported across all service pricing\."',
            f'"description": "{schema_desc}"',
            content
        )

        b_schema_offering = f"Comprehensive Business Training at ${b_price:,} USD for {b_days} days ({b_live} days training + {b_supp} days guidance, Q&A and written guides, {b_notes}). Includes up to {b_seats} participants. Additional seats at ${b_extra} USD per seat. Clients can pick and choose their services. Full 195-country Purchasing Power Parity supported: Tier 1 ${b_price:,}; Tier 2 (25% off) ${int(b_price*0.75):,}; Tier 3 & 4 (50% off) ${int(b_price*0.5):,}; Tier 5 (75% off) ${int(b_price*0.25):,}; Tier 6 custom subsidized relief."
        content = re.sub(
            r'"description":\s*"Comprehensive Business Training at \$[\d,]+ USD for \d+ days[\s\S]*?Tier 6 custom subsidized relief\."',
            f'"description": "{b_schema_offering}"',
            content
        )

        p_schema_offering = f"Comprehensive Personal Training at ${p_price:,} USD for {p_days} days ({p_live} days training + {p_supp} days guidance, Q&A and written guides, {p_notes}). Strictly for 1 person only. Clients can pick and choose their services. Full 195-country Purchasing Power Parity supported: Tier 1 ${p_price:,}; Tier 2 (25% off) ${int(p_price*0.75):,}; Tier 3 & 4 (50% off) ${int(p_price*0.5):,}; Tier 5 (75% off) ${int(p_price*0.25):,}; Tier 6 custom subsidized relief (${int(p_price*0.1):,})."
        content = re.sub(
            r'"description":\s*"Comprehensive Personal Training at \$[\d,]+ USD for \d+ days[\s\S]*?Tier 6 custom subsidized relief \(\$[\d,]+\)\."',
            f'"description": "{p_schema_offering}"',
            content
        )

        occurrences.append({
            "page": "Pricing", "section": "Hero Preview", "line_no": 398,
            "field": "business_flagship_price", "old_val": "$2,000 USD", "new_val": f"${b_price:,} USD",
            "context": hero_b
        })
        occurrences.append({
            "page": "Pricing", "section": "Hero Preview", "line_no": 401,
            "field": "personal_flagship_price", "old_val": "$2,000 USD", "new_val": f"${p_price:,} USD",
            "context": hero_p
        })
        occurrences.append({
            "page": "Pricing", "section": "Flagship Summary Blocks", "line_no": 430,
            "field": "business_flagship_days", "old_val": "60 Days", "new_val": f"{b_days} Days",
            "context": b_block_h3
        })
        occurrences.append({
            "page": "Pricing", "section": "Flagship Summary Blocks", "line_no": 444,
            "field": "personal_flagship_days", "old_val": "60 Days", "new_val": f"{p_days} Days",
            "context": p_block_h3
        })
        return content, occurrences

    rules.append(Rule("PricingHeroPackagesRule", 100, cond_pricing_hero, act_pricing_hero, "pricing"))

    # -------------------------------------------------------------
    # 2. Pricing: Calculator Select Options & Presets (pricing.html)
    # -------------------------------------------------------------
    def cond_pricing_calc(file_path, content, kb, nlp, fuzzy):
        if not file_path.endswith("pricing.html"):
            return False, {}
        return ('<option value="pkg-business"' in content), {}

    def act_pricing_calc(file_path, content, match_info, kb, nlp, fuzzy):
        occurrences = []
        b_price = kb.get_fact("pricing", "b_flag_price", 2000)
        b_days = kb.get_fact("pricing", "b_flag_days", 60)
        b_live = kb.get_fact("pricing", "b_flag_live_days", 30)
        b_supp = kb.get_fact("pricing", "b_flag_supp_days", 30)
        b_seats = kb.get_fact("pricing", "b_flag_seats", 10)
        b_extra = kb.get_fact("pricing", "b_flag_extra_seat", 200)

        p_price = kb.get_fact("pricing", "p_flag_price", 2000)
        p_days = kb.get_fact("pricing", "p_flag_days", 60)
        p_live = kb.get_fact("pricing", "p_flag_live_days", 30)
        p_supp = kb.get_fact("pricing", "p_flag_supp_days", 30)

        b_ind = kb.get_fact("pricing", "b_ind_price", 200)
        p_ind = kb.get_fact("pricing", "p_ind_price", 200)
        hours = kb.get_fact("pricing", "hours_per_day", 4)

        b_opt = f'<option value="pkg-business" data-price="{b_price}" selected="">Business Training Flagship - ${b_price:,} USD ({b_days} Days: {b_live}d Training + {b_supp}d Guidance, up to {b_seats} seats)</option>'
        p_opt = f'<option value="pkg-personal" data-price="{p_price}">Personal Training Flagship - ${p_price:,} USD ({p_days} Days: {p_live}d Training + {p_supp}d Guidance, 1 person only)</option>'
        ind_opt = f'<option value="pick-services" data-price="{b_ind}">Pick Individual Services (${p_ind} Individual / ${b_ind} Business per service, 5d + 5d)</option>'

        content = re.sub(r'<option value="pkg-business"[^>]*>.*?</option>', b_opt, content, flags=re.DOTALL)
        content = re.sub(r'<option value="pkg-personal"[^>]*>.*?</option>', p_opt, content, flags=re.DOTALL)
        content = re.sub(r'<option value="pick-services"[^>]*>.*?</option>', ind_opt, content, flags=re.DOTALL)

        # Category buttons & hints (lines 678, 683, 687)
        content = re.sub(
            r'<span class="hint">Individual \(\$\d+/svc\) · Business \(\$\d+/svc\)</span>',
            f'<span class="hint">Individual (${p_ind}/svc) · Business (${b_ind}/svc)</span>',
            content
        )
        content = re.sub(
            r'<span class="tier-tag" style="color:var\(--text-secondary\);">\$[\d,]+ / svc \([≤<=]\s*\d+\s*seats\)</span>',
            f'<span class="tier-tag" style="color:var(--text-secondary);">${b_ind} / svc (≤{b_seats} seats)</span>',
            content
        )
        content = re.sub(
            r'<span class="tier-tag" style="color:var\(--text-secondary\);">\$[\d,]+ / svc \(1 person\)</span>',
            f'<span class="tier-tag" style="color:var(--text-secondary);">${p_ind} / svc (1 person)</span>',
            content
        )

        content = re.sub(
            r'(<span class="hint" id="calc-participants-hint">)[^<]*(</span>)',
            f'\\1Business: 1-{b_seats} Included (+${b_extra}/extra seat) · Personal: 1 Person Only\\2',
            content
        )
        content = re.sub(
            r'(<p class="calc-help-text" id="calc-participants-desc">)[^<]*(</p>)',
            f'\\1Business base covers up to {b_seats} participants (additional seats are ${b_extra:,} USD per seat scaled by country parity).\\2',
            content
        )
        content = re.sub(
            r'(<span class="calc-summary-value calc-summary-value-text" id="calc-scope-val"[^>]*>)[^<]*(</span>)',
            r'\1Business Training Flagship (' + f'{b_days} Days: {b_live}d + {b_supp}d, {hours}h/day' + r')\2',
            content
        )
        content = re.sub(r'\d+\s+Participants\s+\(Included in Base\)', f'{b_seats} Participants (Included in Base)', content)
        content = re.sub(r'Business\s*\([≤<=]\s*\d+\s*seats\):', f'Business (≤{b_seats} seats):', content)

        occurrences.append({
            "page": "Pricing", "section": "Interactive Calculator", "line_no": 703,
            "field": "calculator_options", "old_val": "pkg-business", "new_val": b_opt,
            "context": b_opt
        })
        return content, occurrences

    rules.append(Rule("PricingCalculatorSelectOptionsRule", 95, cond_pricing_calc, act_pricing_calc, "pricing"))

    # -------------------------------------------------------------
    # 3. Pricing: Section 5 Individual Service Cards (pricing.html)
    # -------------------------------------------------------------
    def cond_pricing_section5(file_path, content, kb, nlp, fuzzy):
        return file_path.endswith("pricing.html") and 'class="service-rate-box"' in content, {}

    def act_pricing_section5(file_path, content, match_info, kb, nlp, fuzzy):
        occurrences = []
        p_ind = kb.get_fact("pricing", "p_ind_price", 200)
        b_ind = kb.get_fact("pricing", "b_ind_price", 200)
        b_seats = kb.get_fact("pricing", "b_flag_seats", 10)
        ind_live = kb.get_fact("pricing", "b_ind_live_days", 5)
        ind_supp = kb.get_fact("pricing", "b_ind_supp_days", 5)

        # 3A. Individual price & data-base
        content = re.sub(
            r'(<span class="service-rate-val svc-indiv-price" data-base=")\d+(">)\$[\d,]+ USD(</span>)',
            rf'\g<1>{p_ind}\g<2>${p_ind} USD\g<3>',
            content
        )
        # 3B. Individual tiers
        content = re.sub(
            r'(<div class="service-rate-tiers svc-indiv-tiers">)[^<]+(</div>)',
            rf'\g<1>T2: ${int(p_ind*0.75)} | T3/4: ${int(p_ind*0.5)} | T5: ${int(p_ind*0.25)} | T6: ${int(p_ind*0.1)}\g<2>',
            content
        )
        # 3C. Business seats label
        content = re.sub(
            r'Business\s*\([≤<=]\s*\d+\s*seats\):',
            f'Business (≤{b_seats} seats):',
            content
        )
        # 3D. Business price & data-base
        content = re.sub(
            r'(<span class="service-rate-val svc-biz-price" data-base=")\d+(">)\$[\d,]+ USD(</span>)',
            rf'\g<1>{b_ind}\g<2>${b_ind} USD\g<3>',
            content
        )
        # 3E. Business tiers
        content = re.sub(
            r'(<div class="service-rate-tiers svc-biz-tiers">)[^<]+(</div>)',
            rf'\g<1>T2: ${int(b_ind*0.75)} | T3/4: ${int(b_ind*0.5)} | T5: ${int(b_ind*0.25)} | T6: ${int(b_ind*0.1)}\g<2>',
            content
        )
        # 3F. Service coverage label
        content = re.sub(
            r'(<div class="service-coverage-label">)\d+\s*Days Training \+ \d+\s*Days Support(</div>)',
            rf'\g<1>{ind_live} Days Training + {ind_supp} Days Support\g<2>',
            content
        )

        occurrences.append({
            "page": "Pricing", "section": "Section 5 Rate Box", "line_no": 1113,
            "field": "service_rates_individual_business", "old_val": "rates", "new_val": f"${p_ind} / ${b_ind} USD",
            "context": f"Individual: ${p_ind} USD | Business: ${b_ind} USD"
        })
        return content, occurrences

    rules.append(Rule("PricingSection5CardsRule", 92, cond_pricing_section5, act_pricing_section5, "pricing"))

    # -------------------------------------------------------------
    # 4. Pricing: pricing-calculator.js Script Constants
    # -------------------------------------------------------------
    def cond_calc_js(file_path, content, kb, nlp, fuzzy):
        return file_path.endswith("pricing-calculator.js"), {}

    def act_calc_js(file_path, content, match_info, kb, nlp, fuzzy):
        occurrences = []
        b_seats = kb.get_fact("pricing", "b_flag_seats", 10)
        b_extra = kb.get_fact("pricing", "b_flag_extra_seat", 200)
        b_days = kb.get_fact("pricing", "b_flag_days", 60)
        b_live = kb.get_fact("pricing", "b_flag_live_days", 30)
        b_supp = kb.get_fact("pricing", "b_flag_supp_days", 30)
        p_days = kb.get_fact("pricing", "p_flag_days", 60)
        p_live = kb.get_fact("pricing", "p_flag_live_days", 30)
        p_supp = kb.get_fact("pricing", "p_flag_supp_days", 30)
        hours = kb.get_fact("pricing", "hours_per_day", 4)

        content = re.sub(
            r'Math\.max\(0,\s*participants\s*-\s*\d+\)',
            f'Math.max(0, participants - {b_seats})',
            content
        )
        content = re.sub(
            r'extraParticipants\s*\*\s*\d+\s*\*\s*tier\.multiplier',
            f'extraParticipants * {b_extra} * tier.multiplier',
            content
        )
        content = re.sub(
            r'extraParticipants\s*\*\s*\d+\s*\*\s*\(tier\.discountPercent\s*/\s*100\)',
            f'extraParticipants * {b_extra} * (tier.discountPercent / 100)',
            content
        )
        content = re.sub(
            r'calcParticipantsHint\.textContent\s*=\s*"Business:\s*1-\d+\s*Included\s*\(\+\$\d+/extra\s*seat\)"',
            f'calcParticipantsHint.textContent = "Business: 1-{b_seats} Included (+${b_extra}/extra seat)"',
            content
        )
        content = re.sub(
            r'calcParticipantsDesc\.textContent\s*=\s*"Business base covers up to \d+ participants \(additional seats are \$\d+ USD per seat scaled by country parity\)\."',
            f'calcParticipantsDesc.textContent = "Business base covers up to {b_seats} participants (additional seats are ${b_extra} USD per seat scaled by country parity)."',
            content
        )
        content = re.sub(
            r'calcScopeVal\.textContent\s*=\s*"Business Training Flagship \(\d+ Days: \d+d \+ \d+d, \d+h/day\)"',
            f'calcScopeVal.textContent = "Business Training Flagship ({b_days} Days: {b_live}d + {b_supp}d, {hours}h/day)"',
            content
        )
        content = re.sub(
            r'calcScopeVal\.textContent\s*=\s*"Personal Training Flagship \(\d+ Days: \d+d \+ \d+d, \d+h/day\)"',
            f'calcScopeVal.textContent = "Personal Training Flagship ({p_days} Days: {p_live}d + {p_supp}d, {hours}h/day)"',
            content
        )
        content = re.sub(
            r'if\s*\(result\.participants\s*<=\s*\d+\)',
            f'if (result.participants <= {b_seats})',
            content
        )
        content = re.sub(
            r'Base covers up to \d+ participants',
            f'Base covers up to {b_seats} participants',
            content
        )

        occurrences.append({
            "page": "Pricing Calculator JS", "section": "Participant Thresholds", "line_no": 166,
            "field": "calculator_seat_threshold", "old_val": "10", "new_val": str(b_seats),
            "context": f"Math.max(0, participants - {b_seats})"
        })
        return content, occurrences

    rules.append(Rule("PricingCalculatorScriptRule", 100, cond_calc_js, act_calc_js, "pricing"))

    # -------------------------------------------------------------
    # 5. Index Page: Flagship Previews & Guarantees (index.html)
    # -------------------------------------------------------------
    def cond_index_page(file_path, content, kb, nlp, fuzzy):
        return file_path.endswith("index.html"), {}

    def act_index_page(file_path, content, match_info, kb, nlp, fuzzy):
        occurrences = []
        b_price = kb.get_fact("pricing", "b_flag_price", 2000)
        b_days = kb.get_fact("pricing", "b_flag_days", 60)
        b_live = kb.get_fact("pricing", "b_flag_live_days", 30)
        b_supp = kb.get_fact("pricing", "b_flag_supp_days", 30)
        b_seats = kb.get_fact("pricing", "b_flag_seats", 10)
        b_extra = kb.get_fact("pricing", "b_flag_extra_seat", 200)
        b_notes = kb.get_fact("pricing", "b_flag_sched_notes", "30 working days over 8 weeks with 8 - 10 days as holidays and recovery days, 4 hours per day")

        p_price = kb.get_fact("pricing", "p_flag_price", 2000)
        p_days = kb.get_fact("pricing", "p_flag_days", 60)
        p_live = kb.get_fact("pricing", "p_flag_live_days", 30)
        p_supp = kb.get_fact("pricing", "p_flag_supp_days", 30)
        p_notes = kb.get_fact("pricing", "p_flag_sched_notes", "30 working days over 6 weeks with 8 - 10 days as holidays and recovery days, 4 hours per day")

        b_ind = kb.get_fact("pricing", "b_ind_price", 200)
        p_ind = kb.get_fact("pricing", "p_ind_price", 200)
        ind_live = kb.get_fact("pricing", "b_ind_live_days", 5)
        ind_supp = kb.get_fact("pricing", "b_ind_supp_days", 5)
        ind_days = kb.get_fact("pricing", "b_ind_days", 10)

        # 5A. Preview blocks (lines 3154-3170)
        content = re.sub(
            r'<h3[^>]*>\s*\$[\d,]+\s*for\s*\d+\s*Days\s*\(\d+d Training \+ \d+d Guidance &amp; Materials\)\s*</h3>\s*<p[^>]*>\s*\d+\s*days of live intensive engineering plus \d+\s*days of dedicated ongoing support[\s\S]*?Covers cohorts up to \d+ participants \(\+\$\d+/seat\)[\s\S]*?</p>',
            f'<h3 style="font-family:var(--font-display);font-size:1.15rem;font-weight:800;color:var(--text-primary);margin-bottom:0.35rem;">\n            ${b_price:,} for {b_days} Days ({b_live}d Training + {b_supp}d Guidance &amp; Materials)\n          </h3>\n          <p style="font-size:0.88rem;color:var(--text-secondary);line-height:1.5;margin:0;">\n            {b_live} days of live intensive engineering plus {b_supp} days of dedicated ongoing support, Q&amp;A and written guides. Covers cohorts up to {b_seats} participants (+${b_extra}/seat). Clients choose the services aligned with their architecture.\n          </p>',
            content
        )
        content = re.sub(
            r'<h3[^>]*color:var\(--gold\)[^>]*>\s*\$[\d,]+\s*for\s*\d+\s*Days\s*\(\d+d Training \+ \d+d Guidance &amp; Materials\)\s*</h3>\s*<p[^>]*>\s*\d+\s*days of intensive 1-on-1 training plus \d+\s*days of guidance[\s\S]*?</p>',
            f'<h3 style="font-family:var(--font-display);font-size:1.15rem;font-weight:800;color:var(--gold);margin-bottom:0.35rem;">\n            ${p_price:,} for {p_days} Days ({p_live}d Training + {p_supp}d Guidance &amp; Materials)\n          </h3>\n          <p style="font-size:0.88rem;color:var(--text-secondary);line-height:1.5;margin:0;">\n            {p_live} days of intensive 1-on-1 training plus {p_supp} days of guidance, Q&amp;A and written guides. Strictly for 1 person only. Clients select the modules tailored to their goals.\n          </p>',
            content
        )

        # 5B. Personal AI Transformation Section (lines 3182-3210)
        content = re.sub(
            r'Personal Training:\s*<span class="accent-gold">\$[\d,]+\s*USD for \d+\s*Days</span>\s*\(or \$[\d,]+\s*USD per Individual Service\)',
            f'Personal Training: <span class="accent-gold">${p_price:,} USD for {p_days} Days</span> (or ${p_ind} USD per Individual Service)',
            content
        )
        content = re.sub(
            r'Enroll in our complete \d+-day personal transformation at \$[\d,]+ USD[\s\S]*?or pick individual services at \$[\d,]+ USD per service[\s\S]*?Strictly for 1 person only\.',
            f'Enroll in our complete {p_days}-day personal transformation at ${p_price:,} USD ({p_live} days training + {p_supp} days guidance and materials, {p_notes}), or pick individual services at ${p_ind} USD per service ({ind_live} days training + {ind_supp} days support). Strictly for 1 person only.',
            content
        )
        content = re.sub(
            r'<h3 style="color:var\(--gold\);font-size:1\.1rem;font-weight:700;margin-bottom:0\.4rem;">\$[\d,]+\s*Tier 1 Base Rate',
            f'<h3 style="color:var(--gold);font-size:1.1rem;font-weight:700;margin-bottom:0.4rem;">${p_price:,} Tier 1 Base Rate',
            content
        )
        content = re.sub(
            r'\$[\d,]+ USD in Tier 1 for \d+ days \(\d+ days training \+ \d+ days guidance and materials\), strictly for 1 person only, scaled by 195-country parity multipliers down to \$[\d,]+\.',
            f'${p_price:,} USD in Tier 1 for {p_days} days ({p_live} days training + {p_supp} days guidance and materials), strictly for 1 person only, scaled by 195-country parity multipliers down to ${int(p_price*0.1)}.',
            content
        )

        # 5C. Guarantee descriptions (lines 2863 and 3242)
        content = re.sub(
            r'Flagship programs run \d+ days: \d+ days of training and \d+ days of support \(Personal &amp; Business - [\w\s\-,+]+?\)\. Individual services run \d+ days: \d+ days of training and \d+ days of support',
            f'Flagship programs run {b_days} days: {b_live} days of training and {b_supp} days of support (Personal &amp; Business - {b_notes}). Individual services run {ind_days} days: {ind_live} days of training and {ind_supp} days of support',
            content
        )
        content = re.sub(
            r'Every Business\s*Training\s*booking covers strictly \d+ days of training and \d+ days of guidance[\s\S]*?total \d+ days of business protection\)\.\s*Personal Training is strictly for one person only[\s\S]*?plus \d+ days of dedicated daily work support\.',
            f'Every Business Training booking covers strictly {b_live} days of training and {b_supp} days of guidance, Q&amp;A and written guides ({b_notes} - total {b_days} days of business protection). Personal Training is strictly for one person only, covering {p_live} days of intensive training (covering client-selected services relevant to your work or interests, {p_notes}) plus {p_supp} days of dedicated daily work support.',
            content
        )

        occurrences.append({
            "page": "Index", "section": "Flagship Previews", "line_no": 3154,
            "field": "index_business_flagship_price", "old_val": "$2,000", "new_val": f"${b_price:,}",
            "context": f"${b_price:,} for {b_days} Days"
        })
        occurrences.append({
            "page": "Index", "section": "Personal Transformation", "line_no": 3182,
            "field": "index_personal_flagship_price", "old_val": "$2,000", "new_val": f"${p_price:,}",
            "context": f"${p_price:,} USD for {p_days} Days"
        })
        return content, occurrences

    rules.append(Rule("IndexPagePricingRule", 88, cond_index_page, act_index_page, "pricing"))

    # -------------------------------------------------------------
    # 6. Business AI Training: Hero, Pills & Parity (business-ai-training.html)
    # -------------------------------------------------------------
    def cond_biz_training(file_path, content, kb, nlp, fuzzy):
        return file_path.endswith("business-ai-training.html"), {}

    def act_biz_training(file_path, content, match_info, kb, nlp, fuzzy):
        occurrences = []
        b_price = kb.get_fact("pricing", "b_flag_price", 2000)
        b_days = kb.get_fact("pricing", "b_flag_days", 60)
        b_live = kb.get_fact("pricing", "b_flag_live_days", 30)
        b_supp = kb.get_fact("pricing", "b_flag_supp_days", 30)
        b_seats = kb.get_fact("pricing", "b_flag_seats", 10)
        b_extra = kb.get_fact("pricing", "b_flag_extra_seat", 200)
        b_notes = kb.get_fact("pricing", "b_flag_sched_notes", "30 working days over 8 weeks with 8 - 10 days as holidays and recovery days, 4 hours per day")

        p_price = kb.get_fact("pricing", "p_flag_price", 2000)
        p_days = kb.get_fact("pricing", "p_flag_days", 60)
        p_live = kb.get_fact("pricing", "p_flag_live_days", 30)
        p_supp = kb.get_fact("pricing", "p_flag_supp_days", 30)

        new_pill = f'<span class="hero-stats-pill">{apply_sync_highlight(f"{b_days} Days Total")} ({apply_sync_highlight(f"{b_live}d Live")} + {apply_sync_highlight(f"{b_supp}d Support")})</span>\n      <span class="hero-stats-pill">{apply_sync_highlight(f"Up to {b_seats} Participants")} (+{apply_sync_highlight(f"${b_extra}/seat")})</span>\n      <span class="hero-stats-pill">{apply_sync_highlight(f"${b_price:,} USD Base")}</span>'
        content = re.sub(
            r'<span class="hero-stats-pill">\d+\s*Days Total[\s\S]*?<span class="hero-stats-pill">\$[\d,]+\s*USD Base</span>',
            new_pill,
            content
        )
        content = re.sub(
            r'Guaranteed \d+ days of comprehensive live instruction across engineering cohorts and leadership, followed\s*immediately by <strong>\d+ days of guidance, Q&amp;A and written guides</strong> \([\d\s\w\-,+h/]+?\)\. Covers up to \d+ participants \(extra seats at \$[\d,]+ USD per seat\)\.',
            f'Guaranteed {b_live} days of comprehensive live instruction across engineering cohorts and leadership, followed immediately by <strong>{b_supp} days of guidance, Q&amp;A and written guides</strong> ({b_notes}). Covers up to {b_seats} participants (extra seats at ${b_extra:,} USD per seat).',
            content
        )

        # Comparison cards (lines 300-325)
        content = re.sub(
            r'<div style="font-family:var\(--font-mono\);font-size:1rem;color:#f0f9ff;font-weight:700;margin-bottom:0\.75rem;background:rgba\(56,189,248,0\.12\);padding:0\.4rem 0\.8rem;border-radius:6px;border:1px solid rgba\(56,189,248,0\.3\);">\s*\$[\d,]+ for \d+ Days \(\d+d Training \+ \d+d Guidance &amp; Materials\)\s*</div>',
            f'<div style="font-family:var(--font-mono);font-size:1rem;color:#f0f9ff;font-weight:700;margin-bottom:0.75rem;background:rgba(56,189,248,0.12);padding:0.4rem 0.8rem;border-radius:6px;border:1px solid rgba(56,189,248,0.3);">\n              ${b_price:,} for {b_days} Days ({b_live}d Training + {b_supp}d Guidance &amp; Materials)\n            </div>',
            content
        )
        content = re.sub(
            r'<div style="font-family:var\(--font-mono\);font-size:1rem;color:#fffbeb;font-weight:700;margin-bottom:0\.75rem;background:rgba\(251,191,36,0\.15\);padding:0\.4rem 0\.8rem;border-radius:6px;border:1px solid rgba\(255, 255, 255, 0\.15\);">\s*\$[\d,]+ for \d+ Days \(\d+d Training \+ \d+d Guidance &amp; Materials\)\s*</div>',
            f'<div style="font-family:var(--font-mono);font-size:1rem;color:#fffbeb;font-weight:700;margin-bottom:0.75rem;background:rgba(251,191,36,0.15);padding:0.4rem 0.8rem;border-radius:6px;border:1px solid rgba(255, 255, 255, 0.15);">\n              ${p_price:,} for {p_days} Days ({p_live}d Training + {p_supp}d Guidance &amp; Materials)\n            </div>',
            content
        )

        # Cohort policy box (line 648)
        content = re.sub(
            r'Business Training covers cohorts up to \d+ participants\. Additional team members can be added at \$[\d,]+ USD per seat\. For solo founders or individual practitioners, please consider our <a href="personal-ai-training\.html"[^>]*>Personal Training \(\$[\d,]+ USD for \d+ Days\)</a>',
            f'Business Training covers cohorts up to {b_seats} participants. Additional team members can be added at ${b_extra:,} USD per seat. For solo founders or individual practitioners, please consider our <a href="personal-ai-training.html" style="color:var(--gold);text-decoration:underline;">Personal Training (${p_price:,} USD for {p_days} Days)</a>',
            content
        )

        # Parity table in business-ai-training.html
        content = re.sub(
            r'<td><strong style="color:var\(--cyan\);font-size:1\.15rem;">\$[\d,]+ USD</strong></td>',
            f'<td><strong style="color:var(--cyan);font-size:1.15rem;">${b_price:,} USD</strong></td>',
            content
        )
        content = re.sub(
            r'<td><strong style="color:#fbbf24;font-size:1\.15rem;">\$[\d,]+ USD</strong></td>',
            f'<td><strong style="color:#fbbf24;font-size:1.15rem;">${int(b_price*0.75):,} USD</strong></td>',
            content
        )
        content = re.sub(
            r'<td><strong style="color:#34d399;font-size:1\.15rem;">\$[\d,]+ USD</strong></td>',
            f'<td><strong style="color:#34d399;font-size:1.15rem;">${int(b_price*0.5):,} USD</strong></td>',
            content
        )
        content = re.sub(
            r'<td><strong style="color:#c084fc;font-size:1\.15rem;">\$[\d,]+ USD</strong></td>',
            f'<td><strong style="color:#c084fc;font-size:1.15rem;">${int(b_price*0.25):,} USD</strong></td>',
            content
        )

        occurrences.append({
            "page": "Business AI Training", "section": "Hero Pills", "line_no": 276,
            "field": "business_flagship_duration", "old_val": "60 Days Total", "new_val": f"{b_days} Days Total",
            "context": new_pill
        })
        return content, occurrences

    rules.append(Rule("BusinessAITrainingRule", 90, cond_biz_training, act_biz_training, "business_training"))

    # -------------------------------------------------------------
    # 7. Personal AI Training: Hero, Pills & Parity (personal-ai-training.html)
    # -------------------------------------------------------------
    def cond_pers_training(file_path, content, kb, nlp, fuzzy):
        return file_path.endswith("personal-ai-training.html"), {}

    def act_pers_training(file_path, content, match_info, kb, nlp, fuzzy):
        occurrences = []
        p_price = kb.get_fact("pricing", "p_flag_price", 2000)
        p_days = kb.get_fact("pricing", "p_flag_days", 60)
        p_live = kb.get_fact("pricing", "p_flag_live_days", 30)
        p_supp = kb.get_fact("pricing", "p_flag_supp_days", 30)
        p_notes = kb.get_fact("pricing", "p_flag_sched_notes", "30 working days over 6 weeks with 8 - 10 days as holidays and recovery days, 4 hours per day")

        b_price = kb.get_fact("pricing", "b_flag_price", 2000)
        b_days = kb.get_fact("pricing", "b_flag_days", 60)
        b_live = kb.get_fact("pricing", "b_flag_live_days", 30)
        b_supp = kb.get_fact("pricing", "b_flag_supp_days", 30)
        b_seats = kb.get_fact("pricing", "b_flag_seats", 10)
        b_extra = kb.get_fact("pricing", "b_flag_extra_seat", 200)
        b_notes = kb.get_fact("pricing", "b_flag_sched_notes", "30 working days over 8 weeks with 8 - 10 days as holidays and recovery days, 4 hours per day")

        new_pill = f'<span class="hero-stats-pill">{apply_sync_highlight(f"{p_days} Days Total")} ({apply_sync_highlight(f"{p_live}d Live")} + {apply_sync_highlight(f"{p_supp}d Support")})</span>\n      <span class="hero-stats-pill">{apply_sync_highlight("Strictly 1-on-1")}</span>\n      <span class="hero-stats-pill">{apply_sync_highlight(f"${p_price:,} USD Base")}</span>'
        content = re.sub(
            r'<span class="hero-stats-pill">\d+\s*Days Total[\s\S]*?<span class="hero-stats-pill">\$[\d,]+\s*USD Base</span>',
            new_pill,
            content
        )
        content = re.sub(
            r'Guaranteed \d+ days of intensive 1-on-1 instruction, followed immediately by <strong>\d+ days of guidance, Q&amp;A and written guides</strong> \([\d\s\w\-,+h/]+?\)\. Strictly for one person only\.',
            f'Guaranteed {p_live} days of intensive 1-on-1 instruction, followed immediately by <strong>{p_supp} days of guidance, Q&amp;A and written guides</strong> ({p_notes}). Strictly for one person only.',
            content
        )

        # Comparison cards (lines 300-325)
        content = re.sub(
            r'<div style="font-family:var\(--font-mono\);font-size:1rem;color:#f0f9ff;font-weight:700;margin-bottom:0\.75rem;background:rgba\(56,189,248,0\.12\);padding:0\.4rem 0\.8rem;border-radius:6px;border:1px solid rgba\(56,189,248,0\.3\);">\s*\$[\d,]+ for \d+ Days \(\d+d Training \+ \d+d Guidance &amp; Materials\)\s*</div>',
            f'<div style="font-family:var(--font-mono);font-size:1rem;color:#f0f9ff;font-weight:700;margin-bottom:0.75rem;background:rgba(56,189,248,0.12);padding:0.4rem 0.8rem;border-radius:6px;border:1px solid rgba(56,189,248,0.3);">\n              ${b_price:,} for {b_days} Days ({b_live}d Training + {b_supp}d Guidance &amp; Materials)\n            </div>',
            content
        )
        content = re.sub(
            r'Covers up to \d+\s*participants \(extra seats at \$[\d,]+ USD per seat\)\.',
            f'Covers up to {b_seats} participants (extra seats at ${b_extra:,} USD per seat).',
            content
        )
        content = re.sub(
            r'<div style="font-family:var\(--font-mono\);font-size:1rem;color:#fffbeb;font-weight:700;margin-bottom:0\.75rem;background:rgba\(251,191,36,0\.15\);padding:0\.4rem 0\.8rem;border-radius:6px;border:1px solid rgba\(255, 255, 255, 0\.15\);">\s*\$[\d,]+ for \d+ Days \(\d+d Training \+ \d+d Guidance &amp; Materials\)\s*</div>',
            f'<div style="font-family:var(--font-mono);font-size:1rem;color:#fffbeb;font-weight:700;margin-bottom:0.75rem;background:rgba(251,191,36,0.15);padding:0.4rem 0.8rem;border-radius:6px;border:1px solid rgba(255, 255, 255, 0.15);">\n              ${p_price:,} for {p_days} Days ({p_live}d Training + {p_supp}d Guidance &amp; Materials)\n            </div>',
            content
        )

        # Parity table in personal-ai-training.html (lines 620-670)
        content = re.sub(
            r'Personal Training is \$[\d,]+ USD in Tier 1 for \d+ days \(\d+ days training \+ \d+ days guidance, Q&amp;A and written guides\)\.',
            f'Personal Training is ${p_price:,} USD in Tier 1 for {p_days} days ({p_live} days training + {p_supp} days guidance, Q&amp;A and written guides).',
            content
        )
        content = re.sub(
            r'<td><strong style="color:var\(--gold\);font-size:1\.15rem;">\$[\d,]+ USD</strong></td>',
            f'<td><strong style="color:var(--gold);font-size:1.15rem;">${p_price:,} USD</strong></td>',
            content
        )
        content = re.sub(
            r'<td><strong style="color:#fbbf24;font-size:1\.15rem;">\$[\d,]+ USD</strong></td>',
            f'<td><strong style="color:#fbbf24;font-size:1.15rem;">${int(p_price*0.75):,} USD</strong></td>',
            content
        )
        content = re.sub(
            r'<td><strong style="color:#34d399;font-size:1\.15rem;">\$[\d,]+ USD</strong></td>',
            f'<td><strong style="color:#34d399;font-size:1.15rem;">${int(p_price*0.5):,} USD</strong></td>',
            content
        )
        content = re.sub(
            r'<td><strong style="color:#c084fc;font-size:1\.15rem;">\$[\d,]+ USD</strong></td>',
            f'<td><strong style="color:#c084fc;font-size:1.15rem;">${int(p_price*0.25):,} USD</strong></td>',
            content
        )
        content = re.sub(
            r'<td><strong style="color:#00f0ff;font-size:1\.15rem;">\$[\d,]+ USD</strong></td>',
            f'<td><strong style="color:#00f0ff;font-size:1.15rem;">${int(p_price*0.1):,} USD</strong></td>',
            content
        )

        # Bottom cross-program policy link (line 677)
        content = re.sub(
            r'Personal Training is strictly for 1 person only[\s\S]*?please consider our <a href="business-ai-training\.html"[^>]*>Business Training \(\$[\d,]+ USD for \d+ Days\)</a>',
            f'Personal Training is strictly for 1 person only (private 1-on-1 acceleration). For engineering teams or corporate cohorts, please consider our <a href="business-ai-training.html" style="color:var(--cyan);text-decoration:underline;">Business Training (${b_price:,} USD for {b_days} Days)</a>',
            content
        )

        occurrences.append({
            "page": "Personal AI Training", "section": "Hero Pills", "line_no": 276,
            "field": "personal_flagship_duration", "old_val": "60 Days Total", "new_val": f"{p_days} Days Total",
            "context": new_pill
        })
        return content, occurrences

    rules.append(Rule("PersonalAITrainingRule", 90, cond_pers_training, act_pers_training, "personal_training"))

    # -------------------------------------------------------------
    # 8. Contact Page: Personal Training Section (contact.html)
    # -------------------------------------------------------------
    def cond_contact_page(file_path, content, kb, nlp, fuzzy):
        return file_path.endswith("contact.html"), {}

    def act_contact_page(file_path, content, match_info, kb, nlp, fuzzy):
        occurrences = []
        p_price = kb.get_fact("pricing", "p_flag_price", 2000)
        p_days = kb.get_fact("pricing", "p_flag_days", 60)
        p_live = kb.get_fact("pricing", "p_flag_live_days", 30)
        p_supp = kb.get_fact("pricing", "p_flag_supp_days", 30)
        p_notes = kb.get_fact("pricing", "p_flag_sched_notes", "30 working days over 6 weeks with 8 - 10 days as holidays and recovery days, 4 hours per day")

        content = re.sub(
            r'Personal Training:\s*<span class="accent-gold">\$[\d,]+\s*USD for \d+\s*Days \(\d+d Training \+ \d+d Guidance &amp; Materials\)</span>',
            f'Personal Training: <span class="accent-gold">${p_price:,} USD for {p_days} Days ({p_live}d Training + {p_supp}d Guidance &amp; Materials)</span>',
            content
        )
        content = re.sub(
            r'Personal Training costs \$[\d,]+ in Tier 1 for \d+ days total \(\d+ days training, \+ \d+ days guidance, Q&amp;A and written guides\)\.\s*Strictly for 1 person only[\s\S]*?Purchasing Power Parity applies globally across 195 countries down to \$[\d,]+\.',
            f'Personal Training costs ${p_price:,} in Tier 1 for {p_days} days total ({p_live} days training, + {p_supp} days guidance, Q&amp;A and written guides). Strictly for 1 person only. Clients are encouraged to choose services relevant to their work or interests. Purchasing Power Parity applies globally across 195 countries down to ${int(p_price*0.1)}.',
            content
        )
        content = re.sub(
            r'<h3 style="color:var\(--gold\);font-size:1\.1rem;font-weight:700;margin-bottom:0\.4rem;">\$[\d,]+\s*Tier 1 Base Rate</h3>',
            f'<h3 style="color:var(--gold);font-size:1.1rem;font-weight:700;margin-bottom:0.4rem;">${p_price:,} Tier 1 Base Rate</h3>',
            content
        )
        content = re.sub(
            r'\$[\d,]+ USD in Tier 1 for \d+ days \(\d+ days training \+ \d+ days guidance and materials, [\w\s\-,+]+?\), strictly for 1 person only, scaled down to \$[\d,]+ by our 6-tier parity matrix\.',
            f'${p_price:,} USD in Tier 1 for {p_days} days ({p_live} days training + {p_supp} days guidance and materials, {p_notes}), strictly for 1 person only, scaled down to ${int(p_price*0.1)} by our 6-tier parity matrix.',
            content
        )

        occurrences.append({
            "page": "Contact", "section": "Personal Training Quote", "line_no": 570,
            "field": "contact_personal_flagship_price", "old_val": "$2,000", "new_val": f"${p_price:,}",
            "context": f"Personal Training: ${p_price:,} USD"
        })
        return content, occurrences

    rules.append(Rule("ContactPagePricingRule", 87, cond_contact_page, act_contact_page, "pricing"))

    # -------------------------------------------------------------
    # 8b. LLMs.txt: Agent Knowledge & Specifications (llms.txt)
    # -------------------------------------------------------------
    def cond_llms_txt(file_path, content, kb, nlp, fuzzy):
        return file_path.endswith("llms.txt"), {}

    def act_llms_txt(file_path, content, match_info, kb, nlp, fuzzy):
        occurrences = []
        b_price = kb.get_fact("pricing", "b_flag_price", 2500)
        b_days = kb.get_fact("pricing", "b_flag_days", 60)
        b_live = kb.get_fact("pricing", "b_flag_live_days", 30)
        b_supp = kb.get_fact("pricing", "b_flag_supp_days", 30)
        b_seats = kb.get_fact("pricing", "b_flag_seats", 10)
        b_extra = kb.get_fact("pricing", "b_flag_seat_price", 250)
        b_max = kb.get_fact("pricing", "b_flag_max_seats", 20)

        p_price = kb.get_fact("pricing", "p_flag_price", 2500)
        p_days = kb.get_fact("pricing", "p_flag_days", 60)
        p_live = kb.get_fact("pricing", "p_flag_live_days", 30)
        p_supp = kb.get_fact("pricing", "p_flag_supp_days", 30)

        ind_price = kb.get_fact("pricing", "ind_serv_price", 250)
        ind_live = kb.get_fact("pricing", "ind_serv_live_days", 5)
        ind_supp = kb.get_fact("pricing", "ind_serv_supp_days", 5)
        ind_days = ind_live + ind_supp

        content = re.sub(
            r'flagship training for \d+ days \(\d+ days live engineering training \+ \d+ days dedicated support\)',
            f'flagship training for {b_days} days ({b_live} days live engineering training + {b_supp} days dedicated support)',
            content
        )
        content = re.sub(
            r'-\s*\*\*Comprehensive Business Training\*\*:\s*\*\*[\$,\d]+\s*USD\*\*\s*for\s*\*\*\d+\s*days\*\*\s*\(\d+\s*days live hands-on engineering training \+ \d+\s*days dedicated support, \d+\s*working days with \d+\s*-\s*\d+\s*recovery days, \d+\s*hours per day\)\.\s*Covers cohorts up to \d+\s*participants\.\s*Additional seats at\s*\*\*[\$,\d]+\s*USD per seat\*\*\s*\(up to \d+\s*seats max\)\.',
            f'- **Comprehensive Business Training**: **${b_price:,} USD** for **{b_days} days** ({b_live} days live hands-on engineering training + {b_supp} days dedicated support, {b_live} working days with 8 - 10 recovery days, 4 hours per day). Covers cohorts up to {b_seats} participants. Additional seats at **${b_extra:,} USD per seat** (up to {b_max} seats max).',
            content
        )
        content = re.sub(
            r'-\s*\*\*Comprehensive Personal Training\*\*:\s*\*\*[\$,\d]+\s*USD\*\*\s*for\s*\*\*\d+\s*days\*\*\s*\(\d+\s*days intensive 1-on-1 sprint \+ \d+\s*days dedicated support, \d+\s*working days with \d+\s*-\s*\d+\s*recovery days, \d+\s*hours per day\)\.\s*Strictly for\s*\*\*one person only\*\*\.',
            f'- **Comprehensive Personal Training**: **${p_price:,} USD** for **{p_days} days** ({p_live} days intensive 1-on-1 sprint + {p_supp} days dedicated support, {p_live} working days with 8 - 10 recovery days, 4 hours per day). Strictly for **one person only**.',
            content
        )
        content = re.sub(
            r'\*\s*\*\*Individual \(1-on-1\)\*\*:\s*\*\*[\$,\d]+\s*USD\*\*\s*per service for\s*\*\*\d+\s*days live training \+ \d+\s*days dedicated support\*\*\s*\(\d+\s*days total coverage per service\)\.\s*Strictly 1 person only\.',
            f'* **Individual (1-on-1)**: **${ind_price:,} USD** per service for **{ind_live} days live training + {ind_supp} days dedicated support** ({ind_days} days total coverage per service). Strictly 1 person only.',
            content
        )
        content = re.sub(
            r'\*\s*\*\*Business \(Teams\)\*\*:\s*\*\*[\$,\d]+\s*USD\*\*\s*per service for\s*\*\*\d+\s*days live training \+ \d+\s*days dedicated support\*\*\s*\(\d+\s*days total coverage per service\)\.\s*Covers up to \d+\s*participants \(additional seats at\s*\*\*[\$,\d]+\s*USD per seat\*\*\)\.',
            f'* **Business (Teams)**: **${ind_price:,} USD** per service for **{ind_live} days live training + {ind_supp} days dedicated support** ({ind_days} days total coverage per service). Covers up to {b_seats} participants (additional seats at **${ind_price:,} USD per seat**).',
            content
        )

        occurrences.append({
            "page": "llms.txt", "section": "Comprehensive Business Training", "line_no": 42,
            "field": "llms_business_flagship_price", "old_val": "$2,500", "new_val": f"${b_price:,}",
            "context": f"Business Training: ${b_price:,} USD for {b_days} days"
        })
        occurrences.append({
            "page": "llms.txt", "section": "Comprehensive Personal Training", "line_no": 44,
            "field": "llms_personal_flagship_price", "old_val": "$2,500", "new_val": f"${p_price:,}",
            "context": f"Personal Training: ${p_price:,} USD for {p_days} days"
        })
        occurrences.append({
            "page": "llms.txt", "section": "Individual Service Picks", "line_no": 38,
            "field": "llms_individual_service_price", "old_val": "$250", "new_val": f"${ind_price:,}",
            "context": f"Individual Service: ${ind_price:,} USD for {ind_days} days"
        })
        return content, occurrences

    rules.append(Rule("LLMsTxtRule", 86, cond_llms_txt, act_llms_txt, "pricing"))

    # -------------------------------------------------------------
    # 9. FAQs: Policies, Quotas & Guarantees (faqs.html)
    # -------------------------------------------------------------
    def cond_faqs(file_path, content, kb, nlp, fuzzy):
        return file_path.endswith("faqs.html"), {}

    def act_faqs(file_path, content, match_info, kb, nlp, fuzzy):
        occurrences = []
        b_price = kb.get_fact("pricing", "b_flag_price", 2000)
        b_days = kb.get_fact("pricing", "b_flag_days", 60)
        b_live = kb.get_fact("pricing", "b_flag_live_days", 30)
        b_supp = kb.get_fact("pricing", "b_flag_supp_days", 30)
        b_seats = kb.get_fact("pricing", "b_flag_seats", 10)
        b_extra = kb.get_fact("pricing", "b_flag_extra_seat", 200)
        b_notes = kb.get_fact("pricing", "b_flag_sched_notes", "30 working days over 8 weeks, 4 hours per day")

        p_price = kb.get_fact("pricing", "p_flag_price", 2000)
        p_days = kb.get_fact("pricing", "p_flag_days", 60)
        p_live = kb.get_fact("pricing", "p_flag_live_days", 30)
        p_supp = kb.get_fact("pricing", "p_flag_supp_days", 30)

        b_ind = kb.get_fact("pricing", "b_ind_price", 200)
        p_ind = kb.get_fact("pricing", "p_ind_price", 200)
        ind_live = kb.get_fact("pricing", "b_ind_live_days", 5)
        ind_supp = kb.get_fact("pricing", "b_ind_supp_days", 5)
        ind_days = kb.get_fact("pricing", "b_ind_days", 10)

        # FAQ #11 Individual & Single Service (lines 198, 206)
        content = re.sub(
            r'Individual / Personal Service delivery is billed at a transparent baseline rate of \$[\d,]+ USD \(Day 1-\d+ delivery, Days \d+-\d+ guidance (?:&amp;|&) support\), while Business Single Service delivery is billed at \$[\d,]+ USD \(Days 1-\d+ delivery, Days \d+-\d+ guidance (?:&amp;|&) support\)\.',
            f'Individual / Personal Service delivery is billed at a transparent baseline rate of ${p_ind:,} USD (Day 1-{ind_live} delivery, Days {ind_live+1}-{ind_days} guidance &amp; support), while Business Single Service delivery is billed at ${b_ind:,} USD (Days 1-{ind_live} delivery, Days {ind_live+1}-{ind_days} guidance &amp; support).',
            content
        )
        content = re.sub(
            r'Individual Pick Parity Rates \(\$[\d,]+ Base\): Tier 1 \(100%\): \$[\d,]+ \| Tier 2 \(75%\): \$[\d,]+ \| Tier 3 \(50%\): \$[\d,]+ \| Tier 4 \(50%\): \$[\d,]+ \| Tier 5 \(25%\): \$[\d,]+ \| Tier 6 \(10%\): \$[\d,]+\. Business Single Pick Parity Rates \(\$[\d,]+ Base\): Tier 1 \(100%\): \$[\d,]+ \| Tier 2 \(75%\): \$[\d,]+ \| Tier 3 \(50%\): \$[\d,]+ \| Tier 4 \(50%\): \$[\d,]+ \| Tier 5 \(25%\): \$[\d,]+ \| Tier 6 \(10%\): \$[\d,]+\.',
            f'Individual Pick Parity Rates (${p_ind} Base): Tier 1 (100%): ${p_ind} | Tier 2 (75%): ${int(p_ind*0.75)} | Tier 3 (50%): ${int(p_ind*0.5)} | Tier 4 (50%): ${int(p_ind*0.5)} | Tier 5 (25%): ${int(p_ind*0.25)} | Tier 6 (10%): ${int(p_ind*0.1)}. Business Single Pick Parity Rates (${b_ind} Base): Tier 1 (100%): ${b_ind} | Tier 2 (75%): ${int(b_ind*0.75)} | Tier 3 (50%): ${int(b_ind*0.5)} | Tier 4 (50%): ${int(b_ind*0.5)} | Tier 5 (25%): ${int(b_ind*0.25)} | Tier 6 (10%): ${int(b_ind*0.1)}.',
            content
        )

        # FAQ #52 & #53 Individual Service Details (lines 1749-1785)
        content = re.sub(
            r'<li><strong>Individual / Personal Pricing:</strong>\s*<strong>\$[\d,]+ USD per service</strong>\s*strictly for 1 person only[^<]*</li>',
            f'<li><strong>Individual / Personal Pricing:</strong> <strong>${p_ind:,} USD per service</strong> strictly for 1 person only (private 1-on-1 sprint).</li>',
            content
        )
        content = re.sub(
            r'<li><strong>Business / Organization Pricing:</strong>\s*<strong>\$[\d,]+ USD per service</strong>\s*for up to \d+ team participants \(additional seats at \$[\d,]+ USD per seat\)\.</li>',
            f'<li><strong>Business / Organization Pricing:</strong> <strong>${b_ind:,} USD per service</strong> for up to {b_seats} team participants (additional seats at ${b_extra:,} USD per seat).</li>',
            content
        )
        content = re.sub(
            r'<li><strong>Delivery Cadence:</strong>\s*Every individual service includes\s*<strong>\d+ days of live, hands-on engineering training</strong>\s*followed by\s*<strong>\d+ days of dedicated operational support[^<]*</strong>\s*\(\d+ days total coverage per service\)\.</li>',
            f'<li><strong>Delivery Cadence:</strong> Every individual service includes <strong>{ind_live} days of live, hands-on engineering training</strong> followed by <strong>{ind_supp} days of dedicated operational support and doubt clarification</strong> ({ind_days} days total coverage per service).</li>',
            content
        )
        content = re.sub(
            r'<span class="faq-service-tag"[^>]*>\$[\d,]+ Individual / \$[\d,]+ Business · \d+d Training \+ \d+d Support</span>',
            f'<span class="faq-service-tag" style="color:#34d399;">${p_ind} Individual / ${b_ind} Business · {ind_live}d Training + {ind_supp}d Support</span>',
            content
        )
        content = re.sub(
            r'<li><strong>Individual Pick Parity Rates \(\$[\d,]+ Base\):</strong>[^<]+</li>',
            f'<li><strong>Individual Pick Parity Rates (${p_ind} Base):</strong> Tier 1 (100%): ${p_ind} | Tier 2 (25% off): ${int(p_ind*0.75)} | Tier 3 &amp; 4 (50% off): ${int(p_ind*0.5)} | Tier 5 (75% off): ${int(p_ind*0.25)} | Tier 6 (Subsidized): ${int(p_ind*0.1)} USD.</li>',
            content
        )
        content = re.sub(
            r'<li><strong>Business Single Pick Parity Rates \(\$[\d,]+ Base\):</strong>[^<]+</li>',
            f'<li><strong>Business Single Pick Parity Rates (${b_ind} Base):</strong> Tier 1 (100%): ${b_ind} | Tier 2 (25% off): ${int(b_ind*0.75)} | Tier 3 &amp; 4 (50% off): ${int(b_ind*0.5)} | Tier 5 (75% off): ${int(b_ind*0.25)} | Tier 6 (Subsidized): ${int(b_ind*0.1)} USD.</li>',
            content
        )

        # FAQ #13 Business Flagship Guarantee
        content = re.sub(
            r'<h3 class="faq-question">What is the pricing model for business training, and what does the\s*\d+-participant quota and \d+-day training plus \d+-day support guarantee cover\?</h3>',
            f'<h3 class="faq-question">What is the pricing model for business training, and what does the {b_seats}-participant quota and {b_live}-day training plus {b_supp}-day support guarantee cover?</h3>',
            content
        )
        content = re.sub(
            r'Our Business Training bookings feature a <strong>fixed base investment of \$[\d,]+ USD covering up to \d+\s*participants for \d+ days \(\d+ days training \+ \d+ days guidance[^<]*\)(?:</strong>)+',
            f'Our Business Training bookings feature a <strong>fixed base investment of ${b_price:,} USD covering up to {b_seats} participants for {b_days} days ({b_live} days training + {b_supp} days guidance, assistance, and written guides licensed under Apache 2.0)</strong>',
            content
        )
        content = re.sub(
            r'<li><strong>\d+ Days Live Training Sprint:</strong>\s*[^<]*</li>',
            f'<li><strong>{b_live} Days Live Training Sprint:</strong> {b_notes}.</li>',
            content
        )
        content = re.sub(
            r'<li><strong>\d+ Days of Included Guidance (?:&amp;|&) Materials:</strong>\s*[^<]*?post-training guidance[^<]*</li>',
            f'<li><strong>{b_supp} Days of Included Guidance &amp; Materials:</strong> {b_supp} business days of post-training guidance, detailed guides, and source materials under Apache 2.0.</li>',
            content
        )
        content = re.sub(
            r'<li><strong>Up to \d+ Learners Included:</strong>\s*Base fee covers up to \d+ participants\.\s*Additional seats scale transparently at \$[\d,]+ USD per head[^<]*</li>',
            f'<li><strong>Up to {b_seats} Learners Included:</strong> Base fee covers up to {b_seats} participants. Additional seats scale transparently at ${b_extra:,} USD per head (adjusted by country parity tier).</li>',
            content
        )
        content = re.sub(
            r'This \d+\+\d+ dual guarantee ensures',
            f'This {b_live}+{b_supp} dual guarantee ensures',
            content
        )
        content = re.sub(
            r'<span class="faq-service-tag">\d+d Training \+ \d+d Support · Up to \d+ Seats Included · Apache 2\.0</span>',
            f'<span class="faq-service-tag">{b_live}d Training + {b_supp}d Support · Up to {b_seats} Seats Included · Apache 2.0</span>',
            content
        )

        # FAQ #14 Extra Seats
        content = re.sub(
            r'<h3 class="faq-question">What are the fees if our training cohort exceeds \d+ participants\?</h3>',
            f'<h3 class="faq-question">What are the fees if our training cohort exceeds {b_seats} participants?</h3>',
            content
        )
        content = re.sub(
            r'When a team desires to include more than \d+ participants in a single Business Training cohort, pricing scales\s*simply, transparently, and predictably:\s*<strong>\$[\d,]+ USD per additional seat</strong>\.',
            f'When a team desires to include more than {b_seats} participants in a single Business Training cohort, pricing scales simply, transparently, and predictably: <strong>${b_extra:,} USD per additional seat</strong>.',
            content
        )
        content = re.sub(
            r'<li><strong>Tier 1 \(Standard\):</strong>\s*\$[\d,]+ USD per additional participant\.</li>',
            f'<li><strong>Tier 1 (Standard):</strong> ${b_extra:,} USD per additional participant.</li>',
            content
        )
        content = re.sub(
            r'<span class="faq-service-tag">\$[\d,]+ / Extra Seat · Parity Multiplier Applies</span>',
            f'<span class="faq-service-tag">${b_extra:,} / Extra Seat · Parity Multiplier Applies</span>',
            content
        )

        # FAQ #16 Personal Training
        content = re.sub(
            r'Personal Training is \$[\d,]+ USD in Tier 1 for \d+ days \(\d+ days training \+ \d+ days guidance, Q&amp;A and written guides\)\.\s*Strictly for 1 person only\.',
            f'Personal Training is ${p_price:,} USD in Tier 1 for {p_days} days ({p_live} days training + {p_supp} days guidance, Q&amp;A and written guides). Strictly for 1 person only.',
            content
        )
        content = re.sub(
            r'Personal Training rates reflect global parity tiers: Tier 1 \(High-Income OECD\) is \$[\d,]+ USD, Tier 2 \(Growth Markets\) is \$[\d,]+ USD, Tier 3 &amp; Tier 4 \(Emerging Economies &amp; Seed Startups\) are \$[\d,]+ USD, Tier 5 \(Academia &amp; Non-Profits\) is \$[\d,]+ USD, and Tier 6 \(Underprivileged &amp; Crisis Zones\) is \$[\d,]+ USD \(or subsidized relief\)\.',
            f'Personal Training rates reflect global parity tiers: Tier 1 (High-Income OECD) is ${p_price:,} USD, Tier 2 (Growth Markets) is ${int(p_price*0.75):,} USD, Tier 3 &amp; Tier 4 (Emerging Economies &amp; Seed Startups) are ${int(p_price*0.5):,} USD, Tier 5 (Academia &amp; Non-Profits) is ${int(p_price*0.25):,} USD, and Tier 6 (Underprivileged &amp; Crisis Zones) is ${int(p_price*0.1):,} USD (or subsidized relief).',
            content
        )

        # FAQ General Body (line 704)
        content = re.sub(
            r'Clients can pick individual services at \$[\d,]+ USD per service for individuals and \$[\d,]+ USD per service for businesses \(each covering \d+ days of training \+ \d+ days of support\)\. Alternatively, clients can enroll in our comprehensive flagship packages: Business Training at \$[\d,]+ for \d+ days \(\d+d training \+ \d+d guidance/materials, up to \d+ participants, \+\$\d+/extra seat\) and Personal Training at \$[\d,]+ for \d+ days \(\d+d training \+ \d+d guidance/materials, strictly 1 person only\)\.',
            f'Clients can pick individual services at ${p_ind} USD per service for individuals and ${b_ind} USD per service for businesses (each covering {ind_live} days of training + {ind_supp} days of support). Alternatively, clients can enroll in our comprehensive flagship packages: Business Training at ${b_price:,} for {b_days} days ({b_live}d training + {b_supp}d guidance/materials, up to {b_seats} participants, +${b_extra}/extra seat) and Personal Training at ${p_price:,} for {p_days} days ({p_live}d training + {p_supp}d guidance/materials, strictly 1 person only).',
            content
        )

        occurrences.append({
            "page": "FAQs", "section": "FAQ #13 Business Guarantee", "line_no": 1645,
            "field": "business_included_seats", "old_val": "10", "new_val": str(b_seats),
            "context": f"covering up to {b_seats} participants"
        })
        return content, occurrences

    rules.append(Rule("FAQsPoliciesRule", 85, cond_faqs, act_faqs, "faqs"))

    # -------------------------------------------------------------
    # 9B. Guarantees Page Rule (guarantees.html)
    # -------------------------------------------------------------
    def cond_guarantees(file_path, content, kb, nlp, fuzzy):
        return file_path.endswith("guarantees.html"), {}

    def act_guarantees(file_path, content, match_info, kb, nlp, fuzzy):
        occurrences = []
        b_price = kb.get_fact("pricing", "b_flag_price", 2500)
        b_days = kb.get_fact("pricing", "b_flag_days", 60)
        b_live = kb.get_fact("pricing", "b_flag_live_days", 30)
        b_supp = kb.get_fact("pricing", "b_flag_supp_days", 30)
        b_seats = kb.get_fact("pricing", "b_flag_seats", 10)
        b_extra = kb.get_fact("pricing", "b_flag_extra_seat", 250)
        b_max_seats = kb.get_fact("pricing", "b_flag_max_seats", 20)
        b_supp_price = kb.get_fact("pricing", "b_flag_supp_price", 500)

        p_price = kb.get_fact("pricing", "p_flag_price", 2500)
        p_days = kb.get_fact("pricing", "p_flag_days", 60)
        p_live = kb.get_fact("pricing", "p_flag_live_days", 30)
        p_supp = kb.get_fact("pricing", "p_flag_supp_days", 30)
        p_supp_price = kb.get_fact("pricing", "p_flag_supp_price", 500)

        b_ind = kb.get_fact("pricing", "b_ind_price", 250)
        b_ind_extra = kb.get_fact("pricing", "b_ind_extra_seat", 250)
        b_ind_max_seats = kb.get_fact("pricing", "b_ind_max_seats", 10)
        b_ind_supp_price = kb.get_fact("pricing", "b_ind_supp_price", 50)

        p_ind = kb.get_fact("pricing", "p_ind_price", 250)
        p_ind_supp_price = kb.get_fact("pricing", "p_ind_supp_price", 50)
        ind_live = kb.get_fact("pricing", "b_ind_live_days", 5)
        ind_supp = kb.get_fact("pricing", "b_ind_supp_days", 5)
        ind_days = kb.get_fact("pricing", "b_ind_days", 10)

        flag_parity = "$3,000 / $2,250 / $1,500 / $1,500 / $750 / Custom USD"
        ind_parity = "$300 / $225 / $150 / $150 / $75 / Custom USD"
        extra_parity = "+$300 / +$225 / +$150 / +$150 / +$75 / Custom USD"
        flag_milestone_parity = "$600 / $450 / $300 / $300 / $150 / Custom USD"
        ind_milestone_parity = "$60 / $45 / $30 / $30 / $15 / Custom USD"

        # 1. Summary Table: Business Training Flagship row
        content = re.sub(
            r'<td><strong>Business Training Flagship</strong></td>\s*<td>\d+ Days Total</td>\s*<td>\d+ days live training \+ \d+ days dedicated support</td>\s*<td>Up to \d+ people included \([^)]+\)</td>\s*<td>[^<]+</td>\s*<td>[^<]+</td>',
            f'<td><strong>Business Training Flagship</strong></td>\n              <td>{b_days} Days Total</td>\n              <td>{b_live} days live training + {b_supp} days dedicated support</td>\n              <td>Up to {b_seats} people included ({extra_parity} per extra seat up to {b_max_seats} seats)</td>\n              <td>{flag_parity} (Training) + ${b_supp_price:,} USD (Support)</td>\n              <td>5 training payments of 20% each ({flag_milestone_parity}) + 2 support payments of ${b_supp_price // 2:,} USD (after 15 &amp; 30 days)</td>',
            content
        )

        # 2. Summary Table: Personal Training Flagship row
        content = re.sub(
            r'<td><strong>Personal Training Flagship</strong></td>\s*<td>\d+ Days Total</td>\s*<td>\d+ days live training \+ \d+ days dedicated support</td>\s*<td>Strictly 1 person \(private 1-on-1 coaching\)</td>\s*<td>[^<]+</td>\s*<td>[^<]+</td>',
            f'<td><strong>Personal Training Flagship</strong></td>\n              <td>{p_days} Days Total</td>\n              <td>{p_live} days live training + {p_supp} days dedicated support</td>\n              <td>Strictly 1 person (private 1-on-1 coaching)</td>\n              <td>{flag_parity} (Training) + ${p_supp_price:,} USD (Support)</td>\n              <td>5 training payments of 20% each ({flag_milestone_parity}) + 2 support payments of ${p_supp_price // 2:,} USD (after 15 &amp; 30 days)</td>',
            content
        )

        # 3. Summary Table: Business Individual Service row
        content = re.sub(
            r'<td><strong>Business Individual Service</strong></td>\s*<td>\d+ Days Total</td>\s*<td>\d+ days live training \+ \d+ days dedicated support</td>\s*<td>1 person base \([^)]+\)</td>\s*<td>[^<]+</td>\s*<td>[^<]+</td>',
            f'<td><strong>Business Individual Service</strong></td>\n              <td>{ind_days} Days Total</td>\n              <td>{ind_live} days live training + {ind_supp} days dedicated support</td>\n              <td>1 person base ({extra_parity} per extra seat up to {b_ind_max_seats} seats)</td>\n              <td>{ind_parity} (Training) + ${b_ind_supp_price:,} USD (Support)</td>\n              <td>5 training payments of 20% each ({ind_milestone_parity}) + 1 support payment of ${b_ind_supp_price:,} USD (after {ind_supp} days)</td>',
            content
        )

        # 4. Summary Table: Personal Individual Service row
        content = re.sub(
            r'<td><strong>Personal Individual Service</strong></td>\s*<td>\d+ Days Total</td>\s*<td>\d+ days live training \+ \d+ days dedicated support</td>\s*<td>Strictly 1 person only</td>\s*<td>[^<]+</td>\s*<td>[^<]+</td>',
            f'<td><strong>Personal Individual Service</strong></td>\n              <td>{ind_days} Days Total</td>\n              <td>{ind_live} days live training + {ind_supp} days dedicated support</td>\n              <td>Strictly 1 person only</td>\n              <td>{ind_parity} (Training) + ${p_ind_supp_price:,} USD (Support)</td>\n              <td>5 training payments of 20% each ({ind_milestone_parity}) + 1 support payment of ${p_ind_supp_price:,} USD (after {ind_supp} days)</td>',
            content
        )

        # 5. Section 4: Business AI Training Flagship Card
        content = re.sub(
            r'(<div class="package-card"[\s\S]*?<h3[^>]*>Business AI Training Flagship</h3>[\s\S]*?<div style="[^"]*">)[\s\S]*?(<span style="[^"]*">)[\s\S]*?(</span>\s*</div>)',
            rf'\g<1>\n            {flag_parity}\n            \g<2>(Training) + ${b_supp_price:,} Support / {b_days} Days\g<3>',
            content,
            count=1
        )
        content = re.sub(
            r'<li>\d+ days total: \d+ days live instruction \+ \d+ days dedicated support\.</li>',
            f'<li>{b_days} days total: {b_live} days live instruction + {b_supp} days dedicated support.</li>',
            content
        )
        content = re.sub(
            r'<li>Up to \d+ participants included \([^)]+\)\.</li>',
            f'<li>Up to {b_seats} participants included ({extra_parity} per extra seat up to {b_max_seats} seats max).</li>',
            content
        )
        content = re.sub(
            r'<li>\$0 upfront; 5 equal [^<]*payments of 20% \([^)]+\) for live training\.</li>',
            f'<li>$0 upfront; 5 equal milestone payments of 20% ({flag_milestone_parity} each) for live training.</li>',
            content
        )
        content = re.sub(
            r'<li>Dedicated support: \$[\d,]+ USD in \d+ installments \(\$[\d,]+ after 15 days, \$[\d,]+ after 30 days\)\.</li>',
            f'<li>Dedicated support: ${b_supp_price:,} USD in 2 installments (${b_supp_price // 2:,} after 15 days, ${b_supp_price // 2:,} after 30 days).</li>',
            content
        )

        # 6. Section 4: Personal AI Training Flagship Card
        content = re.sub(
            r'(<div class="package-card"[\s\S]*?<h3[^>]*>Personal AI Training Flagship</h3>[\s\S]*?<div style="[^"]*">)[\s\S]*?(<span style="[^"]*">)[\s\S]*?(</span>\s*</div>)',
            rf'\g<1>\n            {flag_parity}\n            \g<2>(Training) + ${p_supp_price:,} Support / {p_days} Days\g<3>',
            content,
            count=1
        )
        content = re.sub(
            r'<li>\d+ days total: \d+ days live 1-on-1 coaching \+ \d+ days dedicated support\.</li>',
            f'<li>{p_days} days total: {p_live} days live 1-on-1 coaching + {p_supp} days dedicated support.</li>',
            content
        )

        # 7. Section 4: Pick-and-Choose Individual Services
        content = re.sub(
            r'<li><strong>Business Individual Pick:</strong> <strong>[^<]+</strong> \(covers 1 seat, [^;]+; \d+ days live training \+ \d+ days support, \d+ hours/day, online worldwide\. Support payment due after \d+ days of support\)\.</li>',
            f'<li><strong>Business Individual Pick:</strong> <strong>{ind_parity} (training) + ${b_ind_supp_price:,} USD (support) per service</strong> (covers 1 seat, {extra_parity} per extra seat up to {b_ind_max_seats} seats; {ind_live} days live training + {ind_supp} days support, 4 hours/day, online worldwide. Support payment due after {ind_supp} days of support).</li>',
            content
        )
        content = re.sub(
            r'<li><strong>Personal Individual Pick:</strong> <strong>[^<]+</strong> strictly for 1 person \(\d+ days live training \+ \d+ days support, \d+ hours/day, online worldwide\. Support payment due after \d+ days of support\)\.</li>',
            f'<li><strong>Personal Individual Pick:</strong> <strong>{ind_parity} (training) + ${p_ind_supp_price:,} USD (support) per service</strong> strictly for 1 person ({ind_live} days live training + {ind_supp} days support, 4 hours/day, online worldwide. Support payment due after {ind_supp} days of support).</li>',
            content
        )

        occurrences.append({
            "page": "Guarantees", "section": "Pricing & Schedule Architecture", "line_no": 265,
            "field": "guarantees_business_flagship_price", "old_val": "$2,500", "new_val": f"${b_price:,}",
            "context": f"Business Training Flagship: ${b_price:,} USD (Training) + ${b_supp_price:,} USD (Support)"
        })
        occurrences.append({
            "page": "Guarantees", "section": "Pricing & Schedule Architecture", "line_no": 274,
            "field": "guarantees_personal_flagship_price", "old_val": "$2,500", "new_val": f"${p_price:,}",
            "context": f"Personal Training Flagship: ${p_price:,} USD (Training) + ${p_supp_price:,} USD (Support)"
        })
        return content, occurrences

    rules.append(Rule("GuaranteesPageRule", 87, cond_guarantees, act_guarantees, "pricing"))

    # -------------------------------------------------------------
    # 10. Individual Service Pages Pricing (business-services/*, personal-services/*)
    # -------------------------------------------------------------
    def cond_service_terms(file_path, content, kb, nlp, fuzzy):
        is_service_page = "/business-services/" in file_path or "/personal-services/" in file_path
        return is_service_page, {}

    def act_service_terms(file_path, content, match_info, kb, nlp, fuzzy):
        occurrences = []
        p_ind = kb.get_fact("pricing", "p_ind_price", 200)
        b_ind = kb.get_fact("pricing", "b_ind_price", 200)
        b_seats = kb.get_fact("pricing", "b_flag_seats", 10)
        b_extra = kb.get_fact("pricing", "b_flag_extra_seat", 200)
        ind_live = kb.get_fact("pricing", "b_ind_live_days", 5)
        ind_supp = kb.get_fact("pricing", "b_ind_supp_days", 5)
        ind_days = kb.get_fact("pricing", "b_ind_days", 10)

        b_price = kb.get_fact("pricing", "b_flag_price", 2000)
        b_days = kb.get_fact("pricing", "b_flag_days", 60)
        b_live = kb.get_fact("pricing", "b_flag_live_days", 30)
        b_supp = kb.get_fact("pricing", "b_flag_supp_days", 30)
        b_notes = kb.get_fact("pricing", "b_flag_sched_notes", "30 working days over 8 weeks, 4 hours per day")

        p_price = kb.get_fact("pricing", "p_flag_price", 2000)
        p_days = kb.get_fact("pricing", "p_flag_days", 60)
        p_live = kb.get_fact("pricing", "p_flag_live_days", 30)
        p_supp = kb.get_fact("pricing", "p_flag_supp_days", 30)
        p_notes = kb.get_fact("pricing", "p_flag_sched_notes", "30 working days over 6 weeks, 4 hours per day")

        # 10A. Flat rate and duration in terms
        content = re.sub(
            r'<li><strong>Flat Transparent Rate:</strong>\s*\$[\d,]+ USD per service for individuals \(1-on-1\) or \$[\d,]+ USD per service for businesses[^<]*</li>',
            f'<li><strong>Flat Transparent Rate:</strong> ${p_ind:,} USD per service for individuals (1-on-1) or ${b_ind:,} USD per service for businesses (covers up to {b_seats} seats, +${b_extra:,} USD per extra seat).</li>',
            content
        )
        content = re.sub(
            r'<li><strong>\d+ Days Training \+ \d+ Days Included Support:</strong>\s*Every\s+business\s+engagement\s+includes\s*\d+\s*days of live hands-on engineering training plus\s*\d+\s*days of dedicated operational support[^<]*</li>',
            f'<li><strong>{ind_live} Days Training + {ind_supp} Days Included Support:</strong> Every business engagement includes {ind_live} days of live hands-on engineering training plus {ind_supp} days of dedicated operational support ({ind_days} days total coverage).</li>',
            content
        )

        # 10B. Section 6 Investment Quote list bullets
        content = re.sub(
            r'<li><strong>Individual Service Pick \(Personal / 1-on-1\):</strong>\s*<strong>\$[\d,]+ USD</strong> per service for <strong>\d+ days of live training \+ \d+ days of dedicated support</strong> \(\d+ days total coverage per service\)\.\s*Strictly 1 person only\.</li>',
            f'<li><strong>Individual Service Pick (Personal / 1-on-1):</strong> <strong>${p_ind:,} USD</strong> per service for <strong>{ind_live} days of live training + {ind_supp} days of dedicated support</strong> ({ind_days} days total coverage per service). Strictly 1 person only.</li>',
            content
        )
        content = re.sub(
            r'<li><strong>Individual Service Pick \(Business / Teams\):</strong>\s*<strong>\$[\d,]+ USD</strong> per service for <strong>\d+ days of live training \+ \d+ days of dedicated support</strong> \(\d+ days total coverage per service\)\.\s*Covers up to \d+ participants \(additional seats at <strong>\$[\d,]+ USD per seat</strong>\)\.</li>',
            f'<li><strong>Individual Service Pick (Business / Teams):</strong> <strong>${b_ind:,} USD</strong> per service for <strong>{ind_live} days of live training + {ind_supp} days of dedicated support</strong> ({ind_days} days total coverage per service). Covers up to {b_seats} participants (additional seats at <strong>${b_extra:,} USD per seat</strong>).</li>',
            content
        )
        content = re.sub(
            r'<li><strong>Comprehensive Business Training:</strong>\s*<strong>\$[\d,]+ USD</strong> for <strong>\d+ days</strong> \(\d+ days live training \+ \d+ days guidance, Q&amp;A and written guides, [\w\s\-,+]+?\)\.\s*Includes cohorts up to \d+ participants\.\s*Extra seats at <strong>\$[\d,]+ USD per seat</strong>\)\.</li>',
            f'<li><strong>Comprehensive Business Training:</strong> <strong>${b_price:,} USD</strong> for <strong>{b_days} days</strong> ({b_live} days live training + {b_supp} days guidance, Q&amp;A and written guides, {b_notes}). Includes cohorts up to {b_seats} participants. Extra seats at <strong>${b_extra:,} USD per seat</strong>).</li>',
            content
        )
        content = re.sub(
            r'<li><strong>Comprehensive Personal Training:</strong>\s*<strong>\$[\d,]+ USD</strong> for <strong>\d+ days</strong> \(\d+ days live training \+ \d+ days guidance, Q&amp;A and written guides, [\w\s\-,+]+?\)\.\s*Strictly for 1 person only \(private acceleration\)\.</li>',
            f'<li><strong>Comprehensive Personal Training:</strong> <strong>${p_price:,} USD</strong> for <strong>{p_days} days</strong> ({p_live} days live training + {p_supp} days guidance, Q&amp;A and written guides, {p_notes}). Strictly for 1 person only (private acceleration).</li>',
            content
        )

        # 10C. Parity Bullets
        content = re.sub(
            r'<li><strong>Tier 1 \(Standard / 100%\):</strong>\s*Individual \$[\d,]+ USD \| Business \$[\d,]+ USD \| Business \d+d \$[\d,]+ USD \| Personal \d+d \$[\d,]+ USD \(USA, UK, EU, Canada, Australia\)</li>',
            f'<li><strong>Tier 1 (Standard / 100%):</strong> Individual ${p_ind} USD | Business ${b_ind} USD | Business {b_days}d ${b_price:,} USD | Personal {p_days}d ${p_price:,} USD (USA, UK, EU, Canada, Australia)</li>',
            content
        )
        content = re.sub(
            r'<li><strong>Tier 2 \(Upper-Middle-Income / 25% Off\):</strong>\s*Individual \$[\d,]+ USD \| Business \$[\d,]+ USD \| Business \d+d \$[\d,]+ USD \| Personal \d+d \$[\d,]+ USD: e\.g\. Brazil, Mexico, Malaysia, Thailand, Türkiye</li>',
            f'<li><strong>Tier 2 (Upper-Middle-Income / 25% Off):</strong> Individual ${int(p_ind*0.75)} USD | Business ${int(b_ind*0.75)} USD | Business {b_days}d ${int(b_price*0.75):,} USD | Personal {p_days}d ${int(p_price*0.75):,} USD: e.g. Brazil, Mexico, Malaysia, Thailand, Türkiye</li>',
            content
        )
        content = re.sub(
            r'<li><strong>Tier 3 &amp; 4 \(Lower-Middle/Low-Income &amp; Startups / 50% Off\):</strong>\s*Individual \$[\d,]+ USD \| Business \$[\d,]+ USD \| Business \d+d \$[\d,]+ USD \| Personal \d+d \$[\d,]+ USD</li>',
            f'<li><strong>Tier 3 &amp; 4 (Lower-Middle/Low-Income &amp; Startups / 50% Off):</strong> Individual ${int(p_ind*0.5)} USD | Business ${int(b_ind*0.5)} USD | Business {b_days}d ${int(b_price*0.5):,} USD | Personal {p_days}d ${int(p_price*0.5):,} USD</li>',
            content
        )
        content = re.sub(
            r'<li><strong>Tier 5 \(Academia &amp; Non-Profits / 75% Off\):</strong>\s*Individual \$[\d,]+ USD \| Business \$[\d,]+ USD \| Business \d+d \$[\d,]+ USD \| Personal \d+d \$[\d,]+ USD</li>',
            f'<li><strong>Tier 5 (Academia &amp; Non-Profits / 75% Off):</strong> Individual ${int(p_ind*0.25)} USD | Business ${int(b_ind*0.25)} USD | Business {b_days}d ${int(b_price*0.25):,} USD | Personal {p_days}d ${int(p_price*0.25):,} USD</li>',
            content
        )
        content = re.sub(
            r'<li><strong>Tier 6 \(Crisis-Affected &amp; Under-Resourced Regions\):</strong>\s*Individual \$[\d,]+ USD \| Business \$[\d,]+ USD \| Custom Subsidized Relief / Micro-Rates</li>',
            f'<li><strong>Tier 6 (Crisis-Affected &amp; Under-Resourced Regions):</strong> Individual ${int(p_ind*0.1)} USD | Business ${int(b_ind*0.1)} USD | Custom Subsidized Relief / Micro-Rates</li>',
            content
        )

        # 10D. Embedded Schema Description
        schema_desc = f"Available as an individual service pick (${p_ind} USD for individuals / ${b_ind} USD for businesses, each for {ind_live} days of live training + {ind_supp} days of dedicated support, {ind_days} days total coverage), or as part of our comprehensive programs: Business Training (${b_price:,} USD for {b_days} days: {b_live}d training + {b_supp}d guidance/materials, {b_notes}, up to {b_seats} seats, +${b_extra}/extra seat) or Personal Training (${p_price:,} USD for {p_days} days: {p_live}d training + {p_supp}d guidance/materials, {p_notes}, 1 person only)."
        content = re.sub(
            r'"description":\s*"Available as an individual service pick \(\$[\d,]+ USD for individuals / \$[\d,]+ USD for businesses[\s\S]*?1 person only\)\."',
            f'"description": "{schema_desc}"',
            content
        )

        occurrences.append({
            "page": os.path.basename(file_path), "section": "Section Pricing & Terms", "line_no": 940,
            "field": "individual_service_price", "old_val": "$100 / $200 USD", "new_val": f"${p_ind} / ${b_ind} USD",
            "context": f"Flat Rate: ${p_ind} / ${b_ind} USD"
        })
        occurrences.append({
            "page": os.path.basename(file_path), "section": "Section Pricing & Terms", "line_no": 942,
            "field": "individual_service_duration", "old_val": "10 days", "new_val": f"{ind_days} days ({ind_live}d + {ind_supp}d)",
            "context": f"Duration: {ind_days} days"
        })
        return content, occurrences

    rules.append(Rule("IndividualServicePagesPricingRule", 75, cond_service_terms, act_service_terms, "pricing"))

    # -------------------------------------------------------------
    # 11. Personal Services Names, Cards & Curriculum
    # -------------------------------------------------------------
    def cond_personal_svcs(file_path, content, kb, nlp, fuzzy):
        is_target = (
            file_path.endswith("all-personal-services.html") or
            file_path.endswith("personal-ai-training.html") or
            "/personal-services/" in file_path
        )
        return is_target, {}

    def act_personal_svcs(file_path, content, match_info, kb, nlp, fuzzy):
        occurrences = []
        ps_data = kb.raw_data.get("personal_services", {}).get("services", [])
        for svc in ps_data:
            num = svc.get("number")
            slug = svc.get("slug")
            name = svc.get("name")
            desc = svc.get("description")
            chip = svc.get("role_chip")

            # Update directory list
            content = re.sub(
                rf'<li><a href="personal-services/{re.escape(slug)}">[^<]+</a></li>',
                f'<li><a href="personal-services/{slug}">{name}</a></li>',
                content
            )

            # Update curriculum mention
            content = re.sub(
                rf'Service[s]?:\s*{re.escape(num)}\s*\([^)]+\)',
                f'Service: {num} ({name})',
                content
            )

            # Update nav cards
            card_pattern = rf'(<a class="home-nav-card glow-[^"]*" href="personal-services/{re.escape(slug)}">[\s\S]*?<span class="role-chip [^"]*"[^>]*>)[^<]*(</span>[\s\S]*?<h3 class="home-nav-title">)[^<]*(</h3>[\s\S]*?<p class="home-nav-desc">)[^<]*(</p>)'
            replacement = rf'\g<1>{chip}\g<2>{name}\g<3>{desc}\g<4>'
            if re.search(card_pattern, content):
                content = re.sub(card_pattern, replacement, content)

            # Update individual service page title & H1
            if file_path.endswith(slug):
                content = re.sub(r'<title>[^<]+?\|\s*Thomas Cherickal</title>', f'<title>{name} | Thomas Cherickal</title>', content)
                content = re.sub(r'<h1[^>]*>[^<]+?</h1>', f'<h1>{name}</h1>', content, count=1)
                occurrences.append({
                    "page": name, "section": "Page Title", "line_no": 7,
                    "field": f"personal_service_{num}_name", "old_val": "Original Title", "new_val": name,
                    "context": f"<title>{name} | Thomas Cherickal</title>"
                })
            elif file_path.endswith("all-personal-services.html"):
                occurrences.append({
                    "page": "All Personal Services", "section": f"Service Card ({name})", "line_no": 250,
                    "field": f"personal_service_{num}_card", "old_val": "Card", "new_val": name,
                    "context": f"Personal Service Card: {name}"
                })
            elif file_path.endswith("personal-ai-training.html"):
                occurrences.append({
                    "page": "Personal AI Training", "section": f"Curriculum Service {num}", "line_no": 554,
                    "field": f"personal_service_{num}_curriculum", "old_val": "Curriculum", "new_val": name,
                    "context": f"Curriculum Block {num}: {name}"
                })

        return content, occurrences

    rules.append(Rule("PersonalServicesNamesAndCardsRule", 85, cond_personal_svcs, act_personal_svcs, "personal_services"))

    # -------------------------------------------------------------
    # 12. Business Services Names, Cards & Curriculum
    # -------------------------------------------------------------
    def cond_business_svcs(file_path, content, kb, nlp, fuzzy):
        is_target = (
            file_path.endswith("all-business-services.html") or
            file_path.endswith("business-ai-training.html") or
            file_path.endswith("pricing.html") or
            "/business-services/" in file_path
        )
        return is_target, {}

    def act_business_svcs(file_path, content, match_info, kb, nlp, fuzzy):
        occurrences = []
        bs_data = kb.raw_data.get("business_services", {}).get("services", [])
        for svc in bs_data:
            num = svc.get("number")
            slug = svc.get("slug")
            name = svc.get("name")
            desc = svc.get("description")
            chip = svc.get("role_chip")
            category = svc.get("category", "")

            # Update directory list
            content = re.sub(
                rf'<li><a href="business-services/{re.escape(slug)}">[^<]+</a></li>',
                f'<li><a href="business-services/{slug}">{name}</a></li>',
                content
            )

            # Update curriculum mention
            content = re.sub(
                rf'Service[s]?:\s*{re.escape(num)}\s*\([^)]+\)',
                f'Service: {num} ({name})',
                content
            )

            # Update nav cards
            card_pattern = rf'(<a class="home-nav-card glow-[^"]*" href="business-services/{re.escape(slug)}">[\s\S]*?<span class="role-chip [^"]*"[^>]*>)[^<]*(</span>[\s\S]*?<h3 class="home-nav-title">)[^<]*(</h3>[\s\S]*?<p class="home-nav-desc">)[^<]*(</p>)'
            replacement = rf'\g<1>{chip}\g<2>{name}\g<3>{desc}\g<4>'
            if re.search(card_pattern, content):
                content = re.sub(card_pattern, replacement, content)

            # Update Pricing.html Section 5 cards
            if file_path.endswith("pricing.html"):
                cat_pattern = rf'(SERVICE\s+{re.escape(num)}\s*·\s*)[^<\n]+'
                if re.search(cat_pattern, content):
                    content = re.sub(cat_pattern, f'SERVICE {num} · {category}', content)
                title_pattern = rf'(<a href="business-services/{re.escape(slug)}"[^>]*>)[^<]+(</a>)'
                if re.search(title_pattern, content):
                    content = re.sub(title_pattern, rf'\g<1>{name}\g<2>', content)
                desc_pattern = rf'(<a href="business-services/{re.escape(slug)}"[^>]*>[\s\S]*?</a>\s*</h3>\s*<p[^>]*>)[^<]+(</p>)'
                if re.search(desc_pattern, content):
                    content = re.sub(desc_pattern, rf'\g<1>{desc}\g<2>', content)

            # Update individual service page title & H1
            if file_path.endswith(slug):
                content = re.sub(r'<title>[^<]+?\|\s*Thomas Cherickal</title>', f'<title>{name} | Thomas Cherickal</title>', content)
                content = re.sub(r'<h1[^>]*>[^<]+?</h1>', f'<h1>{name}</h1>', content, count=1)
                occurrences.append({
                    "page": name, "section": "Page Title", "line_no": 7,
                    "field": f"business_service_{num}_name", "old_val": "Original Title", "new_val": name,
                    "context": f"<title>{name} | Thomas Cherickal</title>"
                })
            elif file_path.endswith("all-business-services.html"):
                occurrences.append({
                    "page": "All Business Services", "section": f"Service Card ({name})", "line_no": 250,
                    "field": f"business_service_{num}_card", "old_val": "Card", "new_val": name,
                    "context": f"Business Service Card: {name}"
                })
            elif file_path.endswith("business-ai-training.html"):
                occurrences.append({
                    "page": "Business AI Training", "section": f"Curriculum Service {num}", "line_no": 554,
                    "field": f"business_service_{num}_curriculum", "old_val": "Curriculum", "new_val": name,
                    "context": f"Curriculum Block {num}: {name}"
                })
            elif file_path.endswith("pricing.html"):
                occurrences.append({
                    "page": "Pricing", "section": f"Section 5 Service {num}", "line_no": 1113,
                    "field": f"business_service_{num}_pricing_card", "old_val": "Card", "new_val": name,
                    "context": f"Pricing Card for {name}"
                })

        return content, occurrences

    rules.append(Rule("BusinessServicesNamesAndCardsRule", 85, cond_business_svcs, act_business_svcs, "business_services"))

    # -------------------------------------------------------------
    # 13. Technical Guides Synchronization (claude-training/*.html, gemini-training/*.html)
    # -------------------------------------------------------------
    def cond_tech_guides(file_path, content, kb, nlp, fuzzy):
        is_guide = (
            "/claude-training/" in file_path
            or "/gemini-training/" in file_path
            or "/claude/" in file_path
            or "/google/" in file_path
        ) and file_path.endswith(".html")
        return is_guide, {}

    def act_tech_guides(file_path, content, match_info, kb, nlp, fuzzy):
        occurrences = []
        base_name = os.path.basename(file_path)

        # Check Claude guides
        cl_guides = kb.raw_data.get("claude", {}).get("guides", [])
        for g in cl_guides:
            if g.get("slug") == base_name:
                title = g.get("title")
                content = re.sub(r'<title>[^<]+?\|\s*Thomas Cherickal</title>', f'<title>{title} | Thomas Cherickal</title>', content)
                content = re.sub(r'<h1[^>]*>[^<]+?</h1>', f'<h1>{title}</h1>', content, count=1)
                occurrences.append({
                    "page": g.get("short_name", base_name), "section": "Header", "line_no": 7,
                    "field": f"claude_guide_{g.get('id')}", "old_val": "Title", "new_val": title,
                    "context": f"<title>{title} | Thomas Cherickal</title>"
                })

        # Check Gemini guides
        gem_guides = kb.raw_data.get("gemini", {}).get("guides", [])
        for g in gem_guides:
            if g.get("slug") == base_name:
                title = g.get("title")
                content = re.sub(r'<title>[^<]+?\|\s*Thomas Cherickal</title>', f'<title>{title} | Thomas Cherickal</title>', content)
                content = re.sub(r'<h1[^>]*>[^<]+?</h1>', f'<h1>{title}</h1>', content, count=1)
                occurrences.append({
                    "page": g.get("short_name", base_name), "section": "Header", "line_no": 7,
                    "field": f"gemini_guide_{g.get('id')}", "old_val": "Title", "new_val": title,
                    "context": f"<title>{title} | Thomas Cherickal</title>"
                })

        return content, occurrences

    rules.append(Rule("GuidesIdentityRule", 70, cond_tech_guides, act_tech_guides, "guides"))

    # -------------------------------------------------------------
    # 14. Canonical Identity & Config (header, footer, contact)
    # -------------------------------------------------------------
    def cond_canonical_config(file_path, content, kb, nlp, fuzzy):
        is_target = file_path.endswith("header.html") or file_path.endswith("footer.html") or file_path.endswith("contact.html")
        return is_target, {}

    def act_canonical_config(file_path, content, match_info, kb, nlp, fuzzy):
        occurrences = []
        email = kb.get_fact("config", "email", "thomascherickal@gmail.com")
        content = re.sub(r'mailto:[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', f'mailto:{email}', content)
        occurrences.append({
            "page": os.path.basename(file_path), "section": "Identity & Contact", "line_no": 10,
            "field": "canonical_email", "old_val": "email", "new_val": email,
            "context": f"mailto:{email}"
        })
        return content, occurrences

    rules.append(Rule("CanonicalIdentityRule", 70, cond_canonical_config, act_canonical_config, "config"))

    # -------------------------------------------------------------
    # 15. Tiers Parity Matrix Synchronization
    # -------------------------------------------------------------
    def cond_tiers(file_path, content, kb, nlp, fuzzy):
        is_target = (
            file_path.endswith("pricing.html")
            or file_path.endswith("pricing-calculator.js")
            or file_path.endswith("payment-schedule.js")
            or "/pricing/pricing-tier-" in file_path
            or "/pricing-calculator/pricing-calculator-tier-" in file_path
        )
        return is_target, {}

    def act_tiers(file_path, content, match_info, kb, nlp, fuzzy):
        occurrences = []
        tiers = kb.raw_data.get("tiers", {}).get("tiers", [])
        if file_path.endswith("pricing.html") or "/pricing/pricing-tier-" in file_path:
            for t in tiers:
                t_num = t.get("tier")
                name = t.get("name")
                subtitle = t.get("subtitle", "")
                card_cls = f"parity-card-t{t_num}"
                pattern = rf'(<div class="parity-tier-card\s+[^"]*{card_cls}[^"]*">[\s\S]*?<p class="parity-subtitle">)[^<]+(</p>)'
                if re.search(pattern, content):
                    content = re.sub(pattern, rf'\g<1>{subtitle}\g<2>', content)
                occurrences.append({
                    "page": os.path.basename(file_path), "section": f"Tier {t_num} Card", "line_no": 470,
                    "field": f"tier_{t_num}_parity", "old_val": "Tier Card", "new_val": f"{name} ({t.get('discountPercent')}% discount)",
                    "context": f"Tier {t_num}: {name}"
                })
        elif file_path.endswith("pricing-calculator.js"):
            for t in tiers:
                t_num = t.get("tier")
                mult = t.get("multiplier")
                disc = t.get("discountPercent")
                block_pattern = rf'({t_num}:\s*\{{[\s\S]*?multiplier:\s*)[\d\.]+(\s*,[\s\S]*?discountPercent:\s*)[\d]+'
                if re.search(block_pattern, content):
                    content = re.sub(block_pattern, rf'\g<1>{mult}\g<2>{disc}', content)
                occurrences.append({
                    "page": "Pricing Calculator JS", "section": f"PARITY_TIERS[{t_num}]", "line_no": 20,
                    "field": f"tier_{t_num}_calc_multiplier", "old_val": "multiplier", "new_val": str(mult),
                    "context": f"PARITY_TIERS[{t_num}].multiplier = {mult}"
                })
        elif "/pricing-calculator/pricing-calculator-tier-" in file_path:
            for t in tiers:
                t_num = t.get("tier")
                if f"pricing-calculator-tier-{t_num}.js" in file_path:
                    mult = t.get("multiplier")
                    disc = t.get("discountPercent")
                    block_pattern = rf'(id:\s*{t_num},[\s\S]*?multiplier:\s*)[\d\.]+(\s*,[\s\S]*?discountPercent:\s*)[\d]+'
                    if re.search(block_pattern, content):
                        content = re.sub(block_pattern, rf'\g<1>{mult}\g<2>{disc}', content)
                    occurrences.append({
                        "page": os.path.basename(file_path), "section": f"TIER (id: {t_num})", "line_no": 19,
                        "field": f"tier_{t_num}_calc_multiplier", "old_val": "multiplier", "new_val": str(mult),
                        "context": f"TIER.multiplier = {mult}"
                    })
        elif file_path.endswith("payment-schedule.js"):
            for t in tiers:
                t_num = t.get("tier")
                mult = t.get("multiplier")
                disc = t.get("discountPercent")
                block_pattern = rf'({t_num}:\s*\{{[\s\S]*?multiplier:\s*)[\d\.]+(\s*,[\s\S]*?discount:\s*)[\d]+'
                if re.search(block_pattern, content):
                    content = re.sub(block_pattern, rf'\g<1>{mult}\g<2>{disc}', content)
                occurrences.append({
                    "page": "Payment Schedule JS", "section": f"PARITY_TIERS[{t_num}]", "line_no": 26 + t_num,
                    "field": f"tier_{t_num}_schedule_multiplier", "old_val": "multiplier", "new_val": str(mult),
                    "context": f"PARITY_TIERS[{t_num}].multiplier = {mult}"
                })
        return content, occurrences

    rules.append(Rule("TiersParityMatrixRule", 80, cond_tiers, act_tiers, "tiers"))

    # -------------------------------------------------------------
    # 16. Technical Publications Synchronization
    # -------------------------------------------------------------
    def cond_pubs(file_path, content, kb, nlp, fuzzy):
        return file_path.endswith("articles.html"), {}

    def act_pubs(file_path, content, match_info, kb, nlp, fuzzy):
        occurrences = []
        pubs = kb.raw_data.get("publications", [])
        pubs = pubs.get("publications", []) if isinstance(pubs, dict) else pubs
        for pub in pubs:
            title = pub.get("title")
            platform = pub.get("platform")
            pid = pub.get("id")
            occurrences.append({
                "page": "Articles", "section": f"Publication ({platform})", "line_no": 100,
                "field": f"publication_{pid}", "old_val": title, "new_val": title,
                "context": f"{platform}: {title}"
            })
        return content, occurrences

    rules.append(Rule("PublicationsRule", 70, cond_pubs, act_pubs, "publications"))

    return rules
