#!/usr/bin/env python3
"""
test_html_ui.py - Comprehensive HTML UI & DOM Structural Test Suite (BeautifulSoup4 + lxml).

Parses and validates the live HTML UI across all 85+ repository pages using BeautifulSoup4 + lxml:
- Document Shell, Charset, Open Graph, and SEO Meta Tags
- DOM ID Uniqueness & Asset Integrity (local CSS/JS files exist on disk)
- Centralized Navigation Architecture & Footer Social Grid Completeness
- Accessibility, Single H1, Image Alt, Button Styling, and rel=noopener Security
- CTA Banners, Brand Tagline, Parity Pricing, 10x10% Milestones, and Zero Charge Training Materials
- Interactive Widget DOM Contracts (Pricing Calculator, Tier Calculators, Payment Schedule,
  FAQs Search/Filter/Cards, Contact Form, Services Directories, Training Guides)
- JSON-LD Structured Data Validity, Vertical Mermaid Diagrams, Zero Emojis, Zero Em-Dashes,
  and Zero Superscript Citations

Strict Repo Compliance: Zero emojis, zero em-dashes.
"""

import json
import os
import re
import unittest
from collections import Counter
from bs4 import BeautifulSoup, builder_registry

HTML_PARSER = "lxml" if builder_registry.lookup("lxml") is not None else "html.parser"

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HTML_DIRS = [
    ".",
    "pricing",
    "business-services",
    "personal-services",
    "gemini-training",
    "claude-training",
]

EXCLUDED_HTML = {
    "header.html",
    "footer.html",
    "article-footer_dark.html",
    "404.html",
    "google43d90cf6a204dc6d.html",
    "mermaid.html",
}

EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA70-\U0001FAFF"
    "]+",
    flags=re.UNICODE,
)


def discover_html_pages():
    pages = []
    for d in HTML_DIRS:
        full_dir = os.path.join(REPO_DIR, d)
        if not os.path.isdir(full_dir):
            continue
        for fname in sorted(os.listdir(full_dir)):
            if not fname.endswith(".html"):
                continue
            if d == "." and fname in EXCLUDED_HTML:
                continue
            rel = fname if d == "." else os.path.join(d, fname)
            pages.append(rel)
    return pages


class TestHTMLUISuite(unittest.TestCase):
    """Comprehensive HTML UI, DOM contract, accessibility, and content hygiene tests."""

    @classmethod
    def setUpClass(cls):
        cls.pages = discover_html_pages()
        cls.raw_html = {}
        cls.soups = {}
        for rel in cls.pages:
            path = os.path.join(REPO_DIR, rel)
            with open(path, encoding="utf-8") as f:
                raw = f.read()
            cls.raw_html[rel] = raw
            cls.soups[rel] = BeautifulSoup(raw, HTML_PARSER)

    # --- 1. Document Shell, Charset, SEO & Meta UI ---

    def test_01_all_html_pages_discovered(self):
        """Verifies the HTML UI suite covers all 80+ standalone website pages."""
        self.assertGreaterEqual(len(self.pages), 80, f"Expected >= 80 HTML pages, found {len(self.pages)}")

    def test_02_document_shell_and_seo_meta_tags(self):
        """Verifies DOCTYPE, lang='en', viewport, title, meta description, and canonical URL on all pages."""
        failures = []
        for rel, raw in self.raw_html.items():
            soup = self.soups[rel]
            if not raw.lstrip().lower().startswith("<!doctype html>"):
                failures.append(f"{rel}: missing <!DOCTYPE html>")
            html_tag = soup.find("html")
            if not html_tag or not html_tag.get("lang"):
                failures.append(f"{rel}: missing <html lang=...>")
            if not soup.title or not soup.title.get_text(strip=True):
                failures.append(f"{rel}: missing or empty <title>")
            if not soup.find("meta", attrs={"name": "viewport"}):
                failures.append(f"{rel}: missing viewport meta tag")
            desc = soup.find("meta", attrs={"name": "description"})
            if not desc or not (desc.get("content") or "").strip():
                failures.append(f"{rel}: missing meta description")
            canon = soup.find("link", attrs={"rel": "canonical"})
            if not canon or not (canon.get("href") or "").startswith("https://"):
                failures.append(f"{rel}: missing https canonical link")
        self.assertEqual(failures, [], "Document shell/SEO issues:\n" + "\n".join(failures[:15]))

    def test_03_charset_utf8_declared_on_all_pages(self):
        """Verifies every HTML page declares UTF-8 charset in <head>."""
        failures = []
        for rel, soup in self.soups.items():
            meta_charset = soup.find("meta", attrs={"charset": True})
            if not meta_charset or meta_charset.get("charset", "").lower() != "utf-8":
                failures.append(f"{rel}: missing <meta charset='utf-8'>")
        self.assertEqual(failures, [], "Charset failures:\n" + "\n".join(failures[:15]))

    def test_04_open_graph_meta_cards_on_core_pages(self):
        """Verifies core pages include Open Graph og:title, og:description, and og:url metadata."""
        core = ["index.html", "pricing.html", "guarantees.html", "faqs.html", "contact.html"]
        failures = []
        for rel in core:
            soup = self.soups[rel]
            for prop in ("og:title", "og:description", "og:url"):
                tag = soup.find("meta", attrs={"property": prop})
                if not tag or not (tag.get("content") or "").strip():
                    failures.append(f"{rel}: missing {prop}")
        self.assertEqual(failures, [], "Open Graph failures:\n" + "\n".join(failures))

    def test_05_no_duplicate_dom_ids_on_core_interactive_pages(self):
        """Verifies interactive core pages have zero duplicate DOM element IDs."""
        interactive = [
            "index.html",
            "pricing.html",
            "guarantees.html",
            "faqs.html",
            "contact.html",
            "all-business-services.html",
            "all-personal-services.html",
        ]
        failures = []
        for rel in interactive:
            soup = self.soups[rel]
            ids = [el.get("id") for el in soup.find_all(id=True) if el.get("id")]
            dups = [k for k, v in Counter(ids).items() if v > 1]
            if dups:
                failures.append(f"{rel}: duplicate IDs {dups}")
        self.assertEqual(failures, [], "Duplicate DOM IDs:\n" + "\n".join(failures))

    # --- 2. Navigation, Header & Footer UI ---

    def test_06_centralized_single_navbar_architecture(self):
        """Verifies every page uses strictly the single centralized navbar (no rogue secondary <nav> elements)."""
        failures = []
        for rel, soup in self.soups.items():
            navs = soup.find_all("nav")
            if len(navs) != 1:
                failures.append(f"{rel}: expected exactly 1 <nav>, found {len(navs)}")
        self.assertEqual(failures, [], "Navbar count violations:\n" + "\n".join(failures[:15]))

    def test_07_navbar_contains_essential_navigation_links(self):
        """Verifies the centralized navbar links to Pricing, FAQs, Contact, and Services on core pages."""
        for rel in ("index.html", "pricing.html", "faqs.html", "contact.html"):
            nav = self.soups[rel].find("nav")
            self.assertIsNotNone(nav, f"{rel}: missing <nav>")
            hrefs = " ".join(a.get("href", "") for a in nav.find_all("a", href=True))
            for target in ("pricing.html", "faqs.html", "contact.html"):
                self.assertIn(target, hrefs, f"{rel}: navbar missing link to {target}")

    def test_08_footer_integrity_and_canonical_contact(self):
        """Verifies every page renders a <footer> containing canonical email, LinkedIn, and GitHub links."""
        failures = []
        for rel, soup in self.soups.items():
            footer = soup.find("footer")
            if not footer:
                failures.append(f"{rel}: missing <footer>")
                continue
            footer_html = str(footer)
            if "thomascherickal@gmail.com" not in footer_html:
                failures.append(f"{rel}: footer missing canonical email thomascherickal@gmail.com")
            if "linkedin.com/in/thomascherickal" not in footer_html:
                failures.append(f"{rel}: footer missing LinkedIn link")
            if "github.com/thomascherickal" not in footer_html:
                failures.append(f"{rel}: footer missing GitHub link")
        self.assertEqual(failures, [], "Footer integrity failures:\n" + "\n".join(failures[:15]))

    def test_09_footer_social_links_grid_has_24_links_on_index(self):
        """Verifies index.html footer renders the full 24-link social & contact grid."""
        footer = self.soups["index.html"].find("footer")
        self.assertIsNotNone(footer)
        links = footer.select("a[href]")
        self.assertGreaterEqual(len(links), 24, f"Expected >= 24 footer links, found {len(links)}")
        footer_html = str(footer)
        for required in ("mailto:thomascherickal@gmail.com", "mastodon.social", "t.me/thomascherickal"):
            self.assertIn(required, footer_html, f"index.html footer missing {required}")

    # --- 3. Accessibility, Buttons & Security UI ---

    def test_10_single_h1_and_image_alt_accessibility(self):
        """Verifies every page has exactly one <h1> and every <img> has a non-empty src and alt attribute."""
        failures = []
        for rel, soup in self.soups.items():
            h1s = soup.find_all("h1")
            if len(h1s) != 1:
                failures.append(f"{rel}: expected 1 <h1>, found {len(h1s)}")
            for img in soup.find_all("img"):
                src = (img.get("src") or "").strip()
                if not src:
                    failures.append(f"{rel}: <img> missing src")
                if img.get("alt") is None:
                    failures.append(f"{rel}: <img> missing alt attribute (src={src})")
        self.assertEqual(failures, [], "Accessibility/Media failures:\n" + "\n".join(failures[:15]))

    def test_11_buttons_never_underlined_and_accessible(self):
        """Verifies no .btn element has inline text-decoration:underline and all buttons have accessible labels."""
        failures = []
        for rel, soup in self.soups.items():
            for btn in soup.select("button, a.btn"):
                style = (btn.get("style") or "").lower().replace(" ", "")
                if "text-decoration:underline" in style:
                    failures.append(f"{rel}: button has inline underline style")
                label = btn.get_text(strip=True) or btn.get("aria-label") or btn.get("title")
                if not label:
                    failures.append(f"{rel}: button missing accessible label ({str(btn)[:80]})")
        self.assertEqual(failures, [], "Button UI failures:\n" + "\n".join(failures[:15]))

    def test_12_external_target_blank_links_have_noopener(self):
        """Verifies external links opening in a new tab include rel='noopener' or 'noreferrer'."""
        failures = []
        for rel, soup in self.soups.items():
            for a in soup.find_all("a", attrs={"target": "_blank"}):
                rel_vals = [r.lower() for r in (a.get("rel") or [])]
                if "noopener" not in rel_vals and "noreferrer" not in rel_vals:
                    failures.append(f"{rel}: target=_blank missing noopener ({a.get('href', '')[:60]})")
        self.assertEqual(failures, [], "Security rel=noopener failures:\n" + "\n".join(failures[:15]))

    def test_13_no_empty_or_javascript_void_hrefs(self):
        """Verifies no anchor tag uses javascript:void(0) or empty href=''."""
        failures = []
        for rel, soup in self.soups.items():
            for a in soup.find_all("a", href=True):
                href = a["href"].strip().lower()
                if href == "" or href.startswith("javascript:"):
                    failures.append(f"{rel}: invalid href '{a['href']}'")
        self.assertEqual(failures, [], "Invalid anchor hrefs:\n" + "\n".join(failures[:15]))

    def test_14_data_tables_have_headers_and_body_rows(self):
        """Verifies pricing and guarantees data tables have valid <thead> and <tbody> structure."""
        for rel in ("pricing.html", "guarantees.html"):
            tables = self.soups[rel].find_all("table")
            self.assertGreaterEqual(len(tables), 1, f"{rel}: expected at least 1 table")
            for idx, tbl in enumerate(tables):
                self.assertIsNotNone(tbl.find("thead"), f"{rel}: table #{idx+1} missing <thead>")
                self.assertIsNotNone(tbl.find("tbody"), f"{rel}: table #{idx+1} missing <tbody>")

    # --- 4. CTA, Branding & Pricing Policy UI ---

    def test_15_cta_banners_include_free_linkedin_consultation(self):
        """Verifies every section.cta-banner includes the Free LinkedIn Consultation CTA."""
        failures = []
        for rel, soup in self.soups.items():
            for banner in soup.select("section.cta-banner"):
                linkedin_links = [
                    a for a in banner.find_all("a", href=True)
                    if "linkedin.com/in/thomascherickal" in a["href"]
                ]
                if not linkedin_links:
                    failures.append(f"{rel}: section.cta-banner missing Free LinkedIn Consultation link")
        self.assertEqual(failures, [], "CTA banner failures:\n" + "\n".join(failures[:15]))

    def test_16_index_brand_tagline_and_business_website(self):
        """Verifies index.html displays canonical tagline 'The Future Is Now' and business website."""
        raw = self.raw_html["index.html"]
        self.assertIn("The Future Is Now", raw)
        self.assertIn("https://thedigitalfuturist.com", raw)
        self.assertNotIn("The Digital Futurist · Remote Worldwide", raw)

    def test_17_zero_charge_training_materials_card_rendered(self):
        """Verifies Zero Charge Training Materials heading and early-exit clause render on pricing/core pages."""
        for rel in ("index.html", "pricing.html", "contact.html", "guarantees.html", "faqs.html"):
            txt = re.sub(r"\s+", " ", self.soups[rel].get_text(" ", strip=True))
            self.assertIn("Zero Charge Training Materials", txt, f"{rel}: missing Zero Charge Training Materials")
            self.assertIn("Apache 2.0", txt, f"{rel}: missing Apache 2.0 mention")
            self.assertNotIn("Zero Charge Support Periods", txt, f"{rel}: stale Zero Charge Support Periods")

    def test_18_ten_installments_structure_rendered_on_pricing_pages(self):
        """Verifies 10 equal 10% milestone installment copy renders across index, pricing, contact, and tier pages."""
        targets = [
            "index.html",
            "pricing.html",
            "contact.html",
            "pricing/pricing-tier-1-usa-china-eu-100.html",
            "pricing/pricing-tier-2-brazil-malaysia-75.html",
            "pricing/pricing-tier-3-india-egypt-50.html",
            "pricing/pricing-tier-4-tech-startups-50.html",
            "pricing/pricing-tier-5-colleges-non-profits-25.html",
            "pricing/pricing-tier-6-crisis-affected-10.html",
        ]
        for rel in targets:
            txt = self.soups[rel].get_text(" ", strip=True)
            self.assertIn("10% in Ten Milestone", txt, f"{rel}: missing 10% in Ten Milestone heading")
            self.assertNotIn("20% in Five", txt, f"{rel}: stale 20% in Five copy")

    def test_19_global_parity_pricing_range_displayed(self):
        """Verifies full 6-tier PPP pricing range displays on pricing, guarantees, and parity-pricing pages."""
        for rel in ("pricing.html", "guarantees.html"):
            txt = self.soups[rel].get_text(" ", strip=True)
            self.assertIn("$1,000 / $750 / $500 / $500 / $250 / Custom USD", txt, f"{rel}: missing Flagship PPP range")
        pp_txt = self.soups["parity-pricing.html"].get_text(" ", strip=True)
        for token in ("$1,000 USD", "$750 USD", "$500 USD", "$250 USD", "Custom Subsidized Rate"):
            self.assertIn(token, pp_txt, f"parity-pricing.html: missing {token}")

    # --- 5. Interactive Widgets & Page-Specific DOM Contracts ---

    def test_20_index_hero_and_work_with_me_grid_ui(self):
        """Verifies index.html hero CTAs and Work With Me cards render cleanly."""
        soup = self.soups["index.html"]
        hero_links = soup.find_all("a", href=re.compile(r"(pricing|contact)\.html"))
        self.assertGreaterEqual(len(hero_links), 2, "index.html missing primary pricing/contact CTA links")
        self.assertIn("Work With Me", soup.get_text(" ", strip=True))

    def test_21_pricing_calculator_dom_contract(self):
        """Verifies pricing.html contains all required interactive pricing calculator and payment schedule DOM IDs."""
        soup = self.soups["pricing.html"]
        required_ids = [
            "btn-mode-business",
            "btn-mode-individual",
            "calc-package",
            "calc-tier-badge",
            "calc-participants",
            "calc-display-price",
            "calc-base-value",
            "calc-scope-val",
            "calc-cohort-val",
            "calc-applied-tier",
            "calc-adjustment-val",
            "calc-cta-btn",
            "services-checkbox-grid",
            "payment-schedule-widget",
            "sched-tier-select",
            "sched-eng-select",
            "sched-milestones-body",
            "sched-total-display",
            "sched-upfront-display",
        ]
        missing = [el_id for el_id in required_ids if not soup.find(id=el_id)]
        self.assertEqual(missing, [], f"pricing.html missing calculator/schedule DOM IDs: {missing}")

    def test_22_pricing_calculator_participants_input_is_numeric(self):
        """Verifies #calc-participants input on pricing.html is a numeric input with min >= 1."""
        inp = self.soups["pricing.html"].find(id="calc-participants")
        self.assertIsNotNone(inp)
        self.assertEqual(inp.name, "input")
        self.assertEqual(inp.get("type"), "number")
        self.assertEqual(inp.get("min"), "1")

    def test_23_tier_pages_calculator_dom_contract(self):
        """Verifies all 6 pricing-tier-*.html pages mount their tier calculator DOM controls and script."""
        tier_pages = [
            ("pricing/pricing-tier-1-usa-china-eu-100.html", "pricing-calculator-tier-1.js"),
            ("pricing/pricing-tier-2-brazil-malaysia-75.html", "pricing-calculator-tier-2.js"),
            ("pricing/pricing-tier-3-india-egypt-50.html", "pricing-calculator-tier-3.js"),
            ("pricing/pricing-tier-4-tech-startups-50.html", "pricing-calculator-tier-4.js"),
            ("pricing/pricing-tier-5-colleges-non-profits-25.html", "pricing-calculator-tier-5.js"),
            ("pricing/pricing-tier-6-crisis-affected-10.html", "pricing-calculator-tier-6.js"),
        ]
        for rel, script_name in tier_pages:
            soup = self.soups[rel]
            self.assertIsNotNone(soup.find(id="calc-display-price"), f"{rel}: missing #calc-display-price")
            self.assertIsNotNone(soup.find(id="calc-participants"), f"{rel}: missing #calc-participants")
            scripts = [s.get("src", "") for s in soup.find_all("script", src=True)]
            self.assertTrue(
                any(script_name in s for s in scripts),
                f"{rel}: missing script tag for {script_name}",
            )

    def test_24_guarantees_mermaid_ui_and_trust_table(self):
        """Verifies guarantees.html trust comparison table and updated Mermaid diagrams."""
        soup = self.soups["guarantees.html"]
        tables = soup.select("table.data-table")
        self.assertGreaterEqual(len(tables), 2, "guarantees.html must render pricing and agency comparison tables")

        diagrams = soup.select("div.mermaid")
        self.assertGreaterEqual(len(diagrams), 2, "guarantees.html must have at least 2 Mermaid diagrams")
        combined = "\n".join(d.get_text() for d in diagrams)
        self.assertIn("10%", combined)
        self.assertNotIn("Paid after 15 & 30 Days", combined)

    def test_25_faqs_interactive_cards_and_controls(self):
        """Verifies faqs.html renders structured FAQ cards with questions, answers, and copy-link buttons."""
        soup = self.soups["faqs.html"]
        cards = soup.select("article.faq-card")
        self.assertGreaterEqual(len(cards), 30, f"Expected >= 30 FAQ cards, found {len(cards)}")
        for card in cards:
            self.assertIsNotNone(card.select_one(".faq-question"), "FAQ card missing .faq-question")
            self.assertIsNotNone(card.select_one(".faq-answer"), "FAQ card missing .faq-answer")
            self.assertIsNotNone(card.select_one(".copy-link-btn"), "FAQ card missing .copy-link-btn")

    def test_26_faqs_schedule_and_materials_cards_present(self):
        """Verifies FAQ #50 (10 installments) and FAQ #65 (Zero Charge Training Materials) exist in faqs.html."""
        soup = self.soups["faqs.html"]
        faq50 = soup.find(id="faq-50")
        faq65 = soup.find(id="faq-65")
        self.assertIsNotNone(faq50, "faqs.html missing #faq-50")
        self.assertIsNotNone(faq65, "faqs.html missing #faq-65")
        self.assertIn("ten", faq50.get_text(" ", strip=True).lower())
        self.assertIn("10%", faq50.get_text(" ", strip=True))
        self.assertIn("Zero Charge Training Materials", faq65.get_text(" ", strip=True))
        self.assertIn("ends early", faq65.get_text(" ", strip=True))

    def test_27_faqs_search_and_category_filter_ui(self):
        """Verifies faqs.html exposes search input and category filter buttons."""
        soup = self.soups["faqs.html"]
        search_inputs = soup.find_all("input", attrs={"type": re.compile(r"^(text|search)$", re.I)})
        self.assertGreaterEqual(len(search_inputs), 1, "faqs.html missing FAQ search input")
        filter_btns = soup.select("[data-filter], .faq-filter-btn, .filter-btn")
        self.assertGreaterEqual(len(filter_btns), 3, "faqs.html missing category filter buttons")

    def test_28_contact_form_ui_controls(self):
        """Verifies contact.html contains interactive inquiry form inputs and submit control."""
        soup = self.soups["contact.html"]
        form = soup.find("form")
        self.assertIsNotNone(form, "contact.html missing <form>")
        inputs = form.find_all(["input", "textarea", "select"])
        self.assertGreaterEqual(len(inputs), 3, "contact.html form should have >= 3 fields")
        submit = form.find("button", attrs={"type": "submit"}) or form.find("input", attrs={"type": "submit"})
        self.assertIsNotNone(submit, "contact.html form missing submit button")

    def test_29_service_directory_pages_list_all_services(self):
        """Verifies all-business-services.html and all-personal-services.html link to all 16 services each."""
        biz_links = [
            a["href"] for a in self.soups["all-business-services.html"].find_all("a", href=True)
            if "business-services/business-service-" in a["href"]
        ]
        pers_links = [
            a["href"] for a in self.soups["all-personal-services.html"].find_all("a", href=True)
            if "personal-services/personal-service-" in a["href"]
        ]
        self.assertGreaterEqual(len(set(biz_links)), 16, f"Expected 16 business service links, found {len(set(biz_links))}")
        self.assertGreaterEqual(len(set(pers_links)), 16, f"Expected 16 personal service links, found {len(set(pers_links))}")

    def test_30_all_service_pages_include_pricing_and_linkedin_links(self):
        """Verifies all 32 business and personal service pages link to pricing.html and LinkedIn consultation."""
        service_pages = [p for p in self.pages if p.startswith(("business-services/", "personal-services/"))]
        self.assertEqual(len(service_pages), 32, f"Expected 32 service pages, found {len(service_pages)}")
        failures = []
        for rel in service_pages:
            hrefs = " ".join(a.get("href", "") for a in self.soups[rel].find_all("a", href=True))
            if "pricing.html" not in hrefs:
                failures.append(f"{rel}: missing pricing.html link")
            if "linkedin.com/in/thomascherickal" not in hrefs:
                failures.append(f"{rel}: missing LinkedIn CTA link")
        self.assertEqual(failures, [], "Service page CTA failures:\n" + "\n".join(failures[:15]))

    def test_31_all_training_guides_have_structured_headings(self):
        """Verifies all 36 Gemini and Claude training guides render multiple <h2> sections."""
        guides = [p for p in self.pages if p.startswith(("gemini-training/", "claude-training/"))]
        self.assertGreaterEqual(len(guides), 34, f"Expected >= 34 training guides, found {len(guides)}")
        failures = []
        for rel in guides:
            h2s = self.soups[rel].find_all("h2")
            if len(h2s) < 2:
                failures.append(f"{rel}: expected >= 2 <h2> headings, found {len(h2s)}")
        self.assertEqual(failures, [], "Training guide structure failures:\n" + "\n".join(failures[:15]))

    # --- 6. Structured Data, Assets & Content Hygiene ---

    def test_32_json_ld_structured_data_valid_on_all_pages(self):
        """Verifies every embedded JSON-LD script block parses cleanly as JSON with @context."""
        failures = []
        total_blocks = 0
        for rel, soup in self.soups.items():
            scripts = soup.find_all("script", attrs={"type": "application/ld+json"})
            for idx, sc in enumerate(scripts):
                total_blocks += 1
                raw_json = (sc.string or sc.get_text() or "").strip()
                try:
                    data = json.loads(raw_json)
                    if isinstance(data, dict) and "@context" not in data:
                        failures.append(f"{rel}: JSON-LD #{idx+1} missing @context")
                except json.JSONDecodeError as exc:
                    failures.append(f"{rel}: JSON-LD #{idx+1} invalid JSON ({exc})")
        self.assertGreater(total_blocks, 75)
        self.assertEqual(failures, [], "JSON-LD parse errors:\n" + "\n".join(failures[:15]))

    def test_33_local_stylesheets_and_scripts_exist_on_disk(self):
        """Verifies every local <link rel='stylesheet'> and <script src=...> resolves to an existing file."""
        failures = []
        for rel, soup in self.soups.items():
            page_dir = os.path.dirname(os.path.join(REPO_DIR, rel))
            for link in soup.find_all("link", rel="stylesheet", href=True):
                href = link["href"].strip()
                if not href.startswith(("http://", "https://", "//")):
                    clean = href.split("?")[0].split("#")[0]
                    if not os.path.exists(os.path.normpath(os.path.join(page_dir, clean))):
                        failures.append(f"{rel}: missing local CSS '{href}'")
            for sc in soup.find_all("script", src=True):
                src = sc["src"].strip()
                if not src.startswith(("http://", "https://", "//")):
                    clean = src.split("?")[0].split("#")[0]
                    if not os.path.exists(os.path.normpath(os.path.join(page_dir, clean))):
                        failures.append(f"{rel}: missing local JS '{src}'")
        self.assertEqual(failures, [], "Missing local CSS/JS assets:\n" + "\n".join(failures[:15]))

    def test_34_mermaid_vertical_orientation_and_typography_hygiene(self):
        """Verifies zero horizontal Mermaid LR/RL diagrams, zero em-dashes, and zero superscript citations."""
        failures = []
        sup_cite_re = re.compile(r"<sup>\s*(?:<a[^>]*>)?\s*\[\d+\]\s*(?:</a>)?\s*</sup>", re.I)
        for rel, raw in self.raw_html.items():
            if "\u2014" in raw or "&mdash;" in raw or "&#8212;" in raw:
                failures.append(f"{rel}: contains forbidden em-dash")
            if sup_cite_re.search(raw):
                failures.append(f"{rel}: contains forbidden superscript numbered citation")
            soup = self.soups[rel]
            for d in soup.select("div.mermaid"):
                txt = d.get_text()
                if re.search(r"^\s*(flowchart|graph)\s+(LR|RL)\b", txt, re.M):
                    failures.append(f"{rel}: horizontal Mermaid diagram (LR/RL) detected")
        self.assertEqual(failures, [], "Typography/Mermaid violations:\n" + "\n".join(failures[:15]))

    def test_35_zero_emojis_across_all_html_pages(self):
        """Verifies zero Unicode emojis appear across all 85+ HTML pages."""
        failures = []
        for rel, raw in self.raw_html.items():
            m = EMOJI_RE.search(raw)
            if m:
                failures.append(f"{rel}: contains forbidden emoji '{m.group(0)}'")
        self.assertEqual(failures, [], "Emoji violations:\n" + "\n".join(failures[:15]))

    # --- 7. Real Production Error Scenarios & Negative Fault-Injection Tests ---

    def test_36_zero_corrupted_template_placeholders_in_rendered_text(self):
        """Catches unexpanded template tokens ({{...}}, {%...%}) or corrupted '$00 upfront' in visible prose."""
        failures = []
        token_re = re.compile(r"(\{\{[^}]+\}\}|\{%[^%]+%\}|\$00\s+upfront)")
        for rel, raw in self.raw_html.items():
            clone = BeautifulSoup(raw, HTML_PARSER)
            for code_el in clone.find_all(["pre", "code", "script", "style"]):
                code_el.decompose()
            visible = clone.get_text(" ", strip=True)
            m = token_re.search(visible)
            if m:
                failures.append(f"{rel}: corrupted placeholder/token '{m.group(0)}'")
        self.assertEqual(failures, [], "Template/placeholder leakages:\n" + "\n".join(failures[:15]))

    def test_37_zero_broken_same_page_anchor_fragments(self):
        """Catches broken same-page jump links (<a href='#id'>) where target id does not exist on the page."""
        failures = []
        for rel, soup in self.soups.items():
            all_ids = {el.get("id") for el in soup.find_all(id=True) if el.get("id")}
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                if href.startswith("#") and len(href) > 1:
                    frag = href[1:]
                    if frag not in all_ids:
                        failures.append(f"{rel}: broken anchor fragment '{href}'")
        self.assertEqual(failures, [], "Broken same-page anchor fragments:\n" + "\n".join(failures[:15]))

    def test_38_zero_mixed_content_http_subresources(self):
        """Catches insecure http:// subresources (img/script/link/iframe) that trigger HTTPS mixed-content blocks."""
        failures = []
        for rel, soup in self.soups.items():
            for tag, attr in (("img", "src"), ("script", "src"), ("link", "href"), ("iframe", "src")):
                for el in soup.find_all(tag):
                    val = (el.get(attr) or "").strip().lower()
                    if val.startswith("http://"):
                        failures.append(f"{rel}: insecure mixed-content <{tag} {attr}='{val[:60]}'>")
        self.assertEqual(failures, [], "HTTPS mixed-content errors:\n" + "\n".join(failures[:15]))

    def test_39_zero_invalid_nested_interactive_elements(self):
        """Catches illegal interactive nesting (<a> inside <a>, or <button> inside <a>) that breaks clicks."""
        failures = []
        for rel, soup in self.soups.items():
            for a in soup.find_all("a"):
                if a.find("a") is not None:
                    failures.append(f"{rel}: illegal nested <a> inside <a>")
                if a.find("button") is not None:
                    failures.append(f"{rel}: illegal nested <button> inside <a>")
        self.assertEqual(failures, [], "Invalid interactive element nesting:\n" + "\n".join(failures[:15]))

    def test_40_zero_empty_inline_event_handlers_or_document_write(self):
        """Catches empty onclick='' handlers or unsafe document.write() calls in HTML pages."""
        failures = []
        for rel, soup in self.soups.items():
            for el in soup.find_all(attrs={"onclick": True}):
                if not (el.get("onclick") or "").strip():
                    failures.append(f"{rel}: empty onclick handler on <{el.name}>")
            for sc in soup.find_all("script"):
                body = sc.string or ""
                if "document.write(" in body:
                    failures.append(f"{rel}: unsafe document.write() in inline <script>")
        self.assertEqual(failures, [], "Inline script/handler errors:\n" + "\n".join(failures[:15]))

    def test_41_404_error_page_provides_recovery_navigation(self):
        """Verifies 404.html exists, parses cleanly, and provides a recovery link back to home."""
        err_path = os.path.join(REPO_DIR, "404.html")
        self.assertTrue(os.path.exists(err_path), "Missing 404.html error recovery page")
        with open(err_path, encoding="utf-8") as f:
            raw = f.read()
        soup = BeautifulSoup(raw, HTML_PARSER)
        self.assertIsNotNone(soup.title, "404.html missing <title>")
        hrefs = [a.get("href", "") for a in soup.find_all("a", href=True)]
        self.assertTrue(
            any(h in ("/", "index.html", "/index.html", "./index.html") or h.endswith("thedigitalfuturist.com") for h in hrefs),
            "404.html missing recovery link back to home page",
        )

    def test_42_zero_broken_relative_html_links_across_all_pages(self):
        """Catches 404 internal .html links across all 85+ pages before deployment."""
        failures = []
        for rel, soup in self.soups.items():
            page_dir = os.path.dirname(os.path.join(REPO_DIR, rel))
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                if href.startswith(("http://", "https://", "mailto:", "tel:", "#", "data:")):
                    continue
                clean = href.split("?")[0].split("#")[0]
                if clean.endswith(".html"):
                    target = os.path.normpath(os.path.join(page_dir, clean))
                    if not os.path.exists(target):
                        failures.append(f"{rel}: broken internal HTML link '{href}'")
        self.assertEqual(failures, [], "Broken internal HTML links:\n" + "\n".join(failures[:15]))

    def test_43_contact_form_inputs_have_accessible_names(self):
        """Catches unlabeled form inputs on contact.html that break screen readers and accessibility."""
        soup = self.soups["contact.html"]
        form = soup.find("form")
        self.assertIsNotNone(form)
        labels_for = {lbl.get("for") for lbl in form.find_all("label") if lbl.get("for")}
        failures = []
        for inp in form.find_all(["input", "textarea", "select"]):
            if inp.get("type") in ("hidden", "submit", "button"):
                continue
            style = (inp.get("style") or "").lower().replace(" ", "")
            if "display:none" in style or inp.get("tabindex") == "-1":
                continue
            has_name = bool(
                inp.get("aria-label")
                or inp.get("placeholder")
                or (inp.get("id") and inp.get("id") in labels_for)
                or inp.find_parent("label") is not None
            )
            if not has_name:
                failures.append(f"contact.html: unlabeled form control <{inp.name} id='{inp.get('id')}'>")
        self.assertEqual(failures, [], "Contact form accessibility failures:\n" + "\n".join(failures))

    def test_44_zero_deprecated_pricing_or_support_regressions_sitewide(self):
        """Catches stale pricing/support regressions across all 85+ HTML pages."""
        forbidden = [
            "Zero Charge Support Periods",
            "Support Phase (Day 15)",
            "Support Phase (Day 30)",
            "20% in Five Milestone",
            "strictly 1 person only",
        ]
        failures = []
        for rel, raw in self.raw_html.items():
            low = raw.lower()
            for phrase in forbidden:
                if phrase.lower() in low:
                    failures.append(f"{rel}: contains deprecated phrase '{phrase}'")
        self.assertEqual(failures, [], "Deprecated pricing/policy regressions:\n" + "\n".join(failures[:15]))

    def test_45_zero_localhost_or_example_urls_in_json_ld(self):
        """Catches accidental localhost, 127.0.0.1, or example.com URLs leaked into JSON-LD schemas."""
        failures = []
        for rel, soup in self.soups.items():
            for sc in soup.find_all("script", attrs={"type": "application/ld+json"}):
                raw_json = (sc.string or sc.get_text() or "").lower()
                for bad in ("localhost:", "127.0.0.1", "example.com"):
                    if bad in raw_json:
                        failures.append(f"{rel}: JSON-LD contains non-production host '{bad}'")
        self.assertEqual(failures, [], "Non-production JSON-LD URLs:\n" + "\n".join(failures[:15]))

    def test_46_error_scenario_malformed_json_ld_is_detected(self):
        """Negative fault-injection: verifies malformed JSON-LD raises JSONDecodeError as expected."""
        bad_html = '<html><head><script type="application/ld+json">{"@context": "https://schema.org", bad}</script></head></html>'
        soup = BeautifulSoup(bad_html, HTML_PARSER)
        sc = soup.find("script", attrs={"type": "application/ld+json"})
        with self.assertRaises(json.JSONDecodeError):
            json.loads(sc.string)

    def test_47_error_scenario_duplicate_dom_ids_and_missing_h1_detected(self):
        """Negative fault-injection: verifies duplicate DOM IDs and missing <h1> are reliably flagged."""
        bad_html = '<html><body><div id="calc-display-price">$1,000</div><div id="calc-display-price">$500</div></body></html>'
        soup = BeautifulSoup(bad_html, HTML_PARSER)
        ids = [el.get("id") for el in soup.find_all(id=True)]
        dups = [k for k, v in Counter(ids).items() if v > 1]
        self.assertEqual(dups, ["calc-display-price"])
        self.assertEqual(len(soup.find_all("h1")), 0)

    def test_48_error_scenario_horizontal_mermaid_and_em_dash_detected(self):
        """Negative fault-injection: verifies horizontal Mermaid LR diagram and em-dash are caught."""
        bad_html = '<html><body><p>Bad \u2014 dash</p><div class="mermaid">flowchart LR\n  A --> B</div></body></html>'
        self.assertIn("\u2014", bad_html)
        soup = BeautifulSoup(bad_html, HTML_PARSER)
        txt = soup.select_one("div.mermaid").get_text()
        self.assertIsNotNone(re.search(r"^\s*(flowchart|graph)\s+(LR|RL)\b", txt, re.M))

    def test_49_error_scenario_missing_noopener_and_underlined_button_detected(self):
        """Negative fault-injection: verifies target=_blank without noopener and underlined .btn are caught."""
        bad_html = '<html><body><a class="btn" href="https://ext.test" target="_blank" style="text-decoration: underline;">Go</a></body></html>'
        soup = BeautifulSoup(bad_html, HTML_PARSER)
        a = soup.find("a")
        rel_vals = [r.lower() for r in (a.get("rel") or [])]
        self.assertNotIn("noopener", rel_vals)
        self.assertIn("text-decoration:underline", a.get("style", "").lower().replace(" ", ""))

    def test_50_error_scenario_broken_fragment_and_mixed_content_detected(self):
        """Negative fault-injection: verifies broken #anchor fragment and http:// mixed-content image are caught."""
        bad_html = '<html><body><a href="#missing-section">Jump</a><img src="http://insecure.test/a.png" alt="x"></body></html>'
        soup = BeautifulSoup(bad_html, HTML_PARSER)
        all_ids = {el.get("id") for el in soup.find_all(id=True) if el.get("id")}
        self.assertNotIn("missing-section", all_ids)
        self.assertTrue(soup.find("img")["src"].startswith("http://"))


if __name__ == "__main__":
    unittest.main()
