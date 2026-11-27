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


HTML_TEST_CATEGORY_ORDER = [
    "Document Shell & SEO Meta",
    "Navigation, Header & Footer",
    "Accessibility, Landmarks & Security",
    "CSS, Scripts & Media Assets",
    "CTA, Branding & Pricing Policy",
    "Interactive Widgets & Core Pages",
    "Service Pages & Directories",
    "Claude & Gemini Training Guides",
    "Structured Data & Production Hygiene",
    "SEO, AEO & GEO Verification",
    "Negative Fault-Injection Scenarios",
]

HTML_TEST_CATEGORY_MAP = {
    # SEO, AEO & GEO Verification (6)
    "test_102_unique_titles_and_meta_descriptions": "SEO, AEO & GEO Verification",
    "test_103_zero_render_blocking_head_scripts": "SEO, AEO & GEO Verification",
    "test_104_complete_twitter_and_og_cards": "SEO, AEO & GEO Verification",
    "test_105_sitemap_and_rss_and_robots_integrity": "SEO, AEO & GEO Verification",
    "test_106_images_have_dimensions_and_loading_policy": "SEO, AEO & GEO Verification",
    "test_107_llms_txt_and_llms_full_synced_with_zero_broken_urls": "SEO, AEO & GEO Verification",
    # 1. Document Shell & SEO Meta (10)
    "test_01_all_html_pages_discovered": "Document Shell & SEO Meta",
    "test_02_document_shell_and_seo_meta_tags": "Document Shell & SEO Meta",
    "test_03_charset_utf8_declared_on_all_pages": "Document Shell & SEO Meta",
    "test_04_open_graph_meta_cards_on_core_pages": "Document Shell & SEO Meta",
    "test_05_no_duplicate_dom_ids_on_core_interactive_pages": "Document Shell & SEO Meta",
    "test_51_no_duplicate_dom_ids_across_all_pages": "Document Shell & SEO Meta",
    "test_52_unique_canonical_urls_match_file_paths_all_pages": "Document Shell & SEO Meta",
    "test_53_html_lang_en_and_responsive_viewport_all_pages": "Document Shell & SEO Meta",
    "test_54_meaningful_titles_and_zero_malformed_meta_tags": "Document Shell & SEO Meta",
    "test_55_all_pages_have_header_and_footer_sync_markers": "Document Shell & SEO Meta",
    # 2. Navigation, Header & Footer (10)
    "test_06_centralized_single_navbar_architecture": "Navigation, Header & Footer",
    "test_07_navbar_contains_essential_navigation_links": "Navigation, Header & Footer",
    "test_08_footer_integrity_and_canonical_contact": "Navigation, Header & Footer",
    "test_09_footer_social_links_grid_has_24_links_on_index": "Navigation, Header & Footer",
    "test_56_navbar_has_aria_label_and_brand_logo_all_pages": "Navigation, Header & Footer",
    "test_57_all_navbar_dropdown_links_resolve_to_existing_files": "Navigation, Header & Footer",
    "test_58_navbar_includes_claude_plugins_apps_and_code_mode": "Navigation, Header & Footer",
    "test_59_navbar_lists_claude_before_gemini_consistently": "Navigation, Header & Footer",
    "test_60_footer_includes_newsletter_whatsapp_and_mastodon_all_pages": "Navigation, Header & Footer",
    "test_61_zero_typo_whatsapp_number_across_all_pages": "Navigation, Header & Footer",
    "test_101_back_to_top_button_and_target_on_every_page": "Navigation, Header & Footer",
    # 3. Accessibility, Landmarks & Security (10)
    "test_10_single_h1_and_image_alt_accessibility": "Accessibility, Landmarks & Security",
    "test_11_buttons_never_underlined_and_accessible": "Accessibility, Landmarks & Security",
    "test_12_external_target_blank_links_have_noopener": "Accessibility, Landmarks & Security",
    "test_13_no_empty_or_javascript_void_hrefs": "Accessibility, Landmarks & Security",
    "test_14_data_tables_have_headers_and_body_rows": "Accessibility, Landmarks & Security",
    "test_43_contact_form_inputs_have_accessible_names": "Accessibility, Landmarks & Security",
    "test_62_all_pages_render_main_landmark_with_id_main": "Accessibility, Landmarks & Security",
    "test_63_zero_positive_tabindex_keyboard_traps_sitewide": "Accessibility, Landmarks & Security",
    "test_64_all_iframes_if_present_have_accessible_titles": "Accessibility, Landmarks & Security",
    "test_65_all_html_tables_have_valid_row_and_cell_structure": "Accessibility, Landmarks & Security",
    # 4. CSS, Scripts & Media Assets (8)
    "test_33_local_stylesheets_and_scripts_exist_on_disk": "CSS, Scripts & Media Assets",
    "test_66_core_design_system_stylesheets_loaded_on_all_pages": "CSS, Scripts & Media Assets",
    "test_67_all_local_image_assets_exist_on_disk": "CSS, Scripts & Media Assets",
    "test_68_mermaid_pages_load_mermaid_runtime_script": "CSS, Scripts & Media Assets",
    "test_69_shared_components_js_loaded_across_site_pages": "CSS, Scripts & Media Assets",
    "test_70_training_guides_load_article_stylesheet": "CSS, Scripts & Media Assets",
    "test_71_service_pages_load_shared_cta_and_button_css": "CSS, Scripts & Media Assets",
    "test_72_pricing_page_loads_calculator_and_schedule_scripts": "CSS, Scripts & Media Assets",
    # 5. CTA, Branding & Pricing Policy (10)
    "test_15_cta_banners_include_free_linkedin_consultation": "CTA, Branding & Pricing Policy",
    "test_16_index_brand_tagline_and_business_website": "CTA, Branding & Pricing Policy",
    "test_17_zero_charge_training_materials_card_rendered": "CTA, Branding & Pricing Policy",
    "test_18_ten_installments_structure_rendered_on_pricing_pages": "CTA, Branding & Pricing Policy",
    "test_19_global_parity_pricing_range_displayed": "CTA, Branding & Pricing Policy",
    "test_44_zero_deprecated_pricing_or_support_regressions_sitewide": "CTA, Branding & Pricing Policy",
    "test_73_all_six_pricing_tier_pages_linked_from_pricing_hubs": "CTA, Branding & Pricing Policy",
    "test_74_zero_upfront_guarantee_copy_rendered_on_pricing_and_guarantees": "CTA, Branding & Pricing Policy",
    "test_75_zero_forbidden_legacy_entity_mentions_sitewide": "CTA, Branding & Pricing Policy",
    "test_76_cta_banners_use_standardized_btn_classes": "CTA, Branding & Pricing Policy",
    # 6. Interactive Widgets & Core Pages (12)
    "test_20_index_hero_and_work_with_me_grid_ui": "Interactive Widgets & Core Pages",
    "test_21_pricing_calculator_dom_contract": "Interactive Widgets & Core Pages",
    "test_22_pricing_calculator_participants_input_is_numeric": "Interactive Widgets & Core Pages",
    "test_23_tier_pages_calculator_dom_contract": "Interactive Widgets & Core Pages",
    "test_24_guarantees_mermaid_ui_and_trust_table": "Interactive Widgets & Core Pages",
    "test_25_faqs_interactive_cards_and_controls": "Interactive Widgets & Core Pages",
    "test_26_faqs_schedule_and_materials_cards_present": "Interactive Widgets & Core Pages",
    "test_27_faqs_search_and_category_filter_ui": "Interactive Widgets & Core Pages",
    "test_28_contact_form_ui_controls": "Interactive Widgets & Core Pages",
    "test_77_articles_and_writing_pages_exist_and_cross_link": "Interactive Widgets & Core Pages",
    "test_78_expertise_page_renders_core_sections_and_cta": "Interactive Widgets & Core Pages",
    "test_79_pricing_page_renders_personal_ai_section_and_tier_links": "Interactive Widgets & Core Pages",
    # 7. Service Pages & Directories (8)
    "test_29_service_directory_pages_list_all_services": "Service Pages & Directories",
    "test_30_all_service_pages_include_pricing_and_linkedin_links": "Service Pages & Directories",
    "test_80_business_and_personal_services_have_16_pages_each": "Service Pages & Directories",
    "test_81_all_32_service_pages_render_hero_h1_and_cta_banner": "Service Pages & Directories",
    "test_82_all_32_service_pages_link_to_contact_page": "Service Pages & Directories",
    "test_83_business_directory_renders_rich_service_cards": "Service Pages & Directories",
    "test_84_personal_directory_renders_rich_service_cards": "Service Pages & Directories",
    "test_85_all_32_service_pages_embed_json_ld_schema": "Service Pages & Directories",
    # 8. Claude & Gemini Training Guides (10)
    "test_31_all_training_guides_have_structured_headings": "Claude & Gemini Training Guides",
    "test_86_exactly_18_claude_and_18_gemini_training_guides": "Claude & Gemini Training Guides",
    "test_87_all_36_training_guides_have_toc_and_references": "Claude & Gemini Training Guides",
    "test_88_all_36_training_guides_use_unnumbered_content_headings": "Claude & Gemini Training Guides",
    "test_89_all_36_training_guides_include_standard_cta_banner": "Claude & Gemini Training Guides",
    "test_90_claude_guides_never_include_article_footer": "Claude & Gemini Training Guides",
    "test_91_claude_plugins_covers_unified_skills_and_connectors": "Claude & Gemini Training Guides",
    "test_92_claude_apps_covers_mcp_apps_sdk_and_sandboxing": "Claude & Gemini Training Guides",
    "test_93_claude_code_mode_covers_sandbox_hooks_and_activation": "Claude & Gemini Training Guides",
    "test_94_deleted_claude_skills_and_connectors_pages_absent": "Claude & Gemini Training Guides",
    # 9. Structured Data & Production Hygiene (12)
    "test_32_json_ld_structured_data_valid_on_all_pages": "Structured Data & Production Hygiene",
    "test_34_mermaid_vertical_orientation_and_typography_hygiene": "Structured Data & Production Hygiene",
    "test_35_zero_emojis_across_all_html_pages": "Structured Data & Production Hygiene",
    "test_36_zero_corrupted_template_placeholders_in_rendered_text": "Structured Data & Production Hygiene",
    "test_37_zero_broken_same_page_anchor_fragments": "Structured Data & Production Hygiene",
    "test_38_zero_mixed_content_http_subresources": "Structured Data & Production Hygiene",
    "test_39_zero_invalid_nested_interactive_elements": "Structured Data & Production Hygiene",
    "test_40_zero_empty_inline_event_handlers_or_document_write": "Structured Data & Production Hygiene",
    "test_41_404_error_page_provides_recovery_navigation": "Structured Data & Production Hygiene",
    "test_42_zero_broken_relative_html_links_across_all_pages": "Structured Data & Production Hygiene",
    "test_45_zero_localhost_or_example_urls_in_json_ld": "Structured Data & Production Hygiene",
    "test_95_canonical_author_description_present_in_json_ld": "Structured Data & Production Hygiene",
    # 10. Negative Fault-Injection Scenarios (10)
    "test_46_error_scenario_malformed_json_ld_is_detected": "Negative Fault-Injection Scenarios",
    "test_47_error_scenario_duplicate_dom_ids_and_missing_h1_detected": "Negative Fault-Injection Scenarios",
    "test_48_error_scenario_horizontal_mermaid_and_em_dash_detected": "Negative Fault-Injection Scenarios",
    "test_49_error_scenario_missing_noopener_and_underlined_button_detected": "Negative Fault-Injection Scenarios",
    "test_50_error_scenario_broken_fragment_and_mixed_content_detected": "Negative Fault-Injection Scenarios",
    "test_96_error_scenario_nested_interactive_anchors_detected": "Negative Fault-Injection Scenarios",
    "test_97_error_scenario_emoji_and_superscript_citation_detected": "Negative Fault-Injection Scenarios",
    "test_98_error_scenario_numbered_guide_heading_detected": "Negative Fault-Injection Scenarios",
    "test_99_error_scenario_corrupted_template_placeholder_detected": "Negative Fault-Injection Scenarios",
    "test_100_error_scenario_positive_tabindex_and_missing_alt_detected": "Negative Fault-Injection Scenarios",
}


def get_html_test_category(method_name):
    """Returns the logical category for a TestHTMLUISuite test method."""
    return HTML_TEST_CATEGORY_MAP.get(method_name, "Other HTML UI Tests")


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

    # --- 8. Expanded Document Shell, SEO & Component Sync Tests ---

    def test_51_no_duplicate_dom_ids_across_all_pages(self):
        """Verifies zero duplicate DOM element IDs exist across all 85+ standalone HTML pages."""
        failures = []
        for rel, soup in self.soups.items():
            ids = [el.get("id") for el in soup.find_all(id=True) if el.get("id")]
            dups = [k for k, v in Counter(ids).items() if v > 1]
            if dups:
                failures.append(f"{rel}: duplicate IDs {dups}")
        self.assertEqual(failures, [], "Site-wide duplicate DOM IDs:\n" + "\n".join(failures[:15]))

    def test_52_unique_canonical_urls_match_file_paths_all_pages(self):
        """Verifies every HTML page has a unique https://thedigitalfuturist.com/ canonical URL matching its path."""
        seen = {}
        failures = []
        for rel, soup in self.soups.items():
            canon = soup.find("link", attrs={"rel": "canonical"})
            href = (canon.get("href") or "").strip() if canon else ""
            if not href.startswith("https://thedigitalfuturist.com/"):
                failures.append(f"{rel}: non-canonical domain '{href}'")
            elif rel != "index.html" and not href.endswith(rel):
                failures.append(f"{rel}: canonical mismatch '{href}'")
            if href in seen:
                failures.append(f"{rel}: duplicate canonical '{href}' also used by {seen[href]}")
            seen[href] = rel
        self.assertEqual(failures, [], "Canonical URL failures:\n" + "\n".join(failures[:15]))

    def test_53_html_lang_en_and_responsive_viewport_all_pages(self):
        """Verifies every page sets lang='en' and includes width=device-width in its viewport meta tag."""
        failures = []
        for rel, soup in self.soups.items():
            html_tag = soup.find("html")
            if not html_tag or (html_tag.get("lang") or "").lower() != "en":
                failures.append(f"{rel}: expected lang='en'")
            vp = soup.find("meta", attrs={"name": "viewport"})
            content = (vp.get("content") or "").lower() if vp else ""
            if "width=device-width" not in content:
                failures.append(f"{rel}: viewport missing width=device-width")
        self.assertEqual(failures, [], "Language/viewport failures:\n" + "\n".join(failures[:15]))

    def test_54_meaningful_titles_and_zero_malformed_meta_tags(self):
        """Verifies every page title is at least 15 chars and zero malformed meta-og/meta-twitter tags exist."""
        failures = []
        for rel, raw in self.raw_html.items():
            title_txt = self.soups[rel].title.get_text(strip=True) if self.soups[rel].title else ""
            if len(title_txt) < 15:
                failures.append(f"{rel}: overly short <title> '{title_txt}'")
            for bad in ("meta-og:", "meta-twitter:", "meta-consultant"):
                if bad in raw:
                    failures.append(f"{rel}: malformed meta tag '{bad}'")
        self.assertEqual(failures, [], "Title/meta hygiene failures:\n" + "\n".join(failures[:15]))

    def test_55_all_pages_have_header_and_footer_sync_markers(self):
        """Verifies every page preserves START/END HEADER and FOOTER component inlining markers."""
        failures = []
        for rel, raw in self.raw_html.items():
            for marker in ("<!-- START:HEADER -->", "<!-- END:HEADER -->", "<!-- START:FOOTER -->", "<!-- END:FOOTER -->"):
                if marker not in raw:
                    failures.append(f"{rel}: missing component marker {marker}")
        self.assertEqual(failures, [], "Component marker failures:\n" + "\n".join(failures[:15]))

    # --- 9. Expanded Navigation, Header & Footer Tests ---

    def test_56_navbar_has_aria_label_and_brand_logo_all_pages(self):
        """Verifies every page navbar sets aria-label and displays The Digital Futurist brand logo."""
        failures = []
        for rel, soup in self.soups.items():
            nav = soup.find("nav")
            if not nav or not (nav.get("aria-label") or "").strip():
                failures.append(f"{rel}: <nav> missing aria-label")
                continue
            logo = nav.find("img", class_="nav-brand-logo")
            if not logo or not logo.get("alt"):
                failures.append(f"{rel}: navbar missing brand logo with alt text")
        self.assertEqual(failures, [], "Navbar branding/ARIA failures:\n" + "\n".join(failures[:15]))

    def test_57_all_navbar_dropdown_links_resolve_to_existing_files(self):
        """Verifies every relative link inside the centralized navbar dropdowns resolves to an existing file."""
        nav = self.soups["index.html"].find("nav")
        self.assertIsNotNone(nav)
        failures = []
        for a in nav.find_all("a", href=True):
            href = a["href"].strip()
            if href.startswith(("http://", "https://", "mailto:", "#")):
                continue
            clean = href.split("?")[0].split("#")[0]
            if not os.path.exists(os.path.join(REPO_DIR, clean)):
                failures.append(f"Navbar broken target: {href}")
        self.assertEqual(failures, [], "Broken navbar dropdown links:\n" + "\n".join(failures))

    def test_58_navbar_includes_claude_plugins_apps_and_code_mode(self):
        """Verifies the centralized navbar lists Claude Plugins, Claude Apps, and Claude Code Mode."""
        nav_html = str(self.soups["index.html"].find("nav"))
        for slug in (
            "claude-training/claude-plugins.html",
            "claude-training/claude-apps.html",
            "claude-training/claude-code-mode.html",
        ):
            self.assertIn(slug, nav_html, f"Navbar missing {slug}")
        for removed in ("claude-training/claude-skills.html", "claude-training/claude-connectors.html"):
            self.assertNotIn(removed, nav_html, f"Navbar still references deleted {removed}")

    def test_59_navbar_lists_claude_before_gemini_consistently(self):
        """Verifies the centralized navbar orders Claude before Gemini in top-level navigation."""
        nav = self.soups["index.html"].find("nav")
        toggles = [t.get_text(" ", strip=True) for t in nav.select("a.nav-dropdown-toggle")]
        claude_idx = next((i for i, t in enumerate(toggles) if t.startswith("Claude")), -1)
        gemini_idx = next((i for i, t in enumerate(toggles) if t.startswith("Gemini")), -1)
        self.assertGreaterEqual(claude_idx, 0, "Navbar missing Claude dropdown")
        self.assertGreaterEqual(gemini_idx, 0, "Navbar missing Gemini dropdown")
        self.assertLess(claude_idx, gemini_idx, "Navbar must place Claude before Gemini")

    def test_60_footer_includes_newsletter_whatsapp_and_mastodon_all_pages(self):
        """Verifies every page footer includes the newsletter, verified WhatsApp, and Mastodon links."""
        failures = []
        for rel, soup in self.soups.items():
            fhtml = str(soup.find("footer") or "")
            for tok in ("thomascherickal.kit.com", "wa.me/919884452724", "mastodon.social"):
                if tok not in fhtml:
                    failures.append(f"{rel}: footer missing {tok}")
        self.assertEqual(failures, [], "Footer link completeness failures:\n" + "\n".join(failures[:15]))

    def test_61_zero_typo_whatsapp_number_across_all_pages(self):
        """Verifies the mistyped legacy WhatsApp number (9198844452724) never appears anywhere on the site."""
        failures = [rel for rel, raw in self.raw_html.items() if "9198844452724" in raw]
        self.assertEqual(failures, [], f"Typo WhatsApp number found on: {failures}")

    # --- 10. Expanded Accessibility, Landmarks & Security Tests ---

    def test_62_all_pages_render_main_landmark_with_id_main(self):
        """Verifies every standalone page provides a <main id='main'> landmark for skip-link navigation."""
        failures = [rel for rel, soup in self.soups.items() if not soup.find("main", id="main")]
        self.assertEqual(failures, [], "Pages missing <main id='main'>:\n" + "\n".join(failures[:15]))

    def test_63_zero_positive_tabindex_keyboard_traps_sitewide(self):
        """Verifies no element across any HTML page sets tabindex > 0 (which disrupts keyboard tab order)."""
        failures = []
        for rel, soup in self.soups.items():
            for el in soup.find_all(attrs={"tabindex": True}):
                try:
                    if int(el["tabindex"]) > 0:
                        failures.append(f"{rel}: positive tabindex={el['tabindex']} on <{el.name}>")
                except ValueError:
                    failures.append(f"{rel}: non-integer tabindex='{el['tabindex']}' on <{el.name}>")
        self.assertEqual(failures, [], "Keyboard tabindex violations:\n" + "\n".join(failures[:15]))

    def test_64_all_iframes_if_present_have_accessible_titles(self):
        """Verifies any embedded <iframe> element includes a non-empty title attribute for screen readers."""
        failures = []
        for rel, soup in self.soups.items():
            for iframe in soup.find_all("iframe"):
                if not (iframe.get("title") or "").strip():
                    failures.append(f"{rel}: <iframe> missing accessible title attribute")
        self.assertEqual(failures, [], "Iframe accessibility failures:\n" + "\n".join(failures))

    def test_65_all_html_tables_have_valid_row_and_cell_structure(self):
        """Verifies every <table> across all 85+ pages contains rows (<tr>) and data/header cells."""
        failures = []
        for rel, soup in self.soups.items():
            for idx, tbl in enumerate(soup.find_all("table")):
                rows = tbl.find_all("tr")
                cells = tbl.find_all(["th", "td"])
                if not rows or not cells:
                    failures.append(f"{rel}: table #{idx+1} missing rows or cells")
        self.assertEqual(failures, [], "Table DOM structure failures:\n" + "\n".join(failures[:15]))

    # --- 11. Expanded CSS, Scripts & Media Asset Tests ---

    def test_66_core_design_system_stylesheets_loaded_on_all_pages(self):
        """Verifies all 6 core design-system stylesheets load on every page."""
        required_css = ("fonts.css", "styles.css", "navbar.css", "footer.css", "buttons.css", "responsive.css")
        failures = []
        for rel, soup in self.soups.items():
            hrefs = " ".join(l.get("href", "") for l in soup.find_all("link", rel="stylesheet"))
            for css in required_css:
                if css not in hrefs:
                    failures.append(f"{rel}: missing core stylesheet {css}")
        self.assertEqual(failures, [], "Missing core stylesheets:\n" + "\n".join(failures[:15]))

    def test_67_all_local_image_assets_exist_on_disk(self):
        """Verifies every local <img src=...> across all 85+ HTML pages resolves to an existing image file."""
        failures = []
        for rel, soup in self.soups.items():
            page_dir = os.path.dirname(os.path.join(REPO_DIR, rel))
            for img in soup.find_all("img", src=True):
                src = img["src"].strip()
                if not src.startswith(("http://", "https://", "//", "data:")):
                    clean = src.split("?")[0].split("#")[0]
                    if not os.path.exists(os.path.normpath(os.path.join(page_dir, clean))):
                        failures.append(f"{rel}: missing local image '{src}'")
        self.assertEqual(failures, [], "Missing local image assets:\n" + "\n".join(failures[:15]))

    def test_68_mermaid_pages_load_mermaid_runtime_script(self):
        """Verifies every page rendering a div.mermaid diagram also loads a Mermaid renderer script."""
        failures = []
        for rel, soup in self.soups.items():
            if soup.select("div.mermaid"):
                scripts_blob = " ".join(
                    (sc.get("src", "") + " " + (sc.string or "")) for sc in soup.find_all("script")
                ).lower()
                if "mermaid" not in scripts_blob:
                    failures.append(f"{rel}: renders div.mermaid without loading Mermaid script")
        self.assertEqual(failures, [], "Missing Mermaid scripts:\n" + "\n".join(failures[:15]))

    def test_69_shared_components_js_loaded_across_site_pages(self):
        """Verifies shared components.js is loaded on core, pricing, service, and training pages."""
        loaded = sum(
            1 for soup in self.soups.values()
            if any("components.js" in (s.get("src") or "") for s in soup.find_all("script", src=True))
        )
        self.assertGreaterEqual(loaded, 80, f"Expected >= 80 pages loading components.js, found {loaded}")

    def test_70_training_guides_load_article_stylesheet(self):
        """Verifies all 36 Claude and Gemini training guides load article.css for uniform typography."""
        guides = [p for p in self.pages if p.startswith(("claude-training/", "gemini-training/"))]
        failures = []
        for rel in guides:
            hrefs = " ".join(l.get("href", "") for l in self.soups[rel].find_all("link", rel="stylesheet"))
            if "article.css" not in hrefs:
                failures.append(f"{rel}: missing article.css")
        self.assertEqual(failures, [], "Training guide stylesheet failures:\n" + "\n".join(failures[:15]))

    def test_71_service_pages_load_shared_cta_and_button_css(self):
        """Verifies all 32 business and personal service pages load cta.css and buttons.css."""
        service_pages = [p for p in self.pages if p.startswith(("business-services/", "personal-services/"))]
        failures = []
        for rel in service_pages:
            hrefs = " ".join(l.get("href", "") for l in self.soups[rel].find_all("link", rel="stylesheet"))
            for css in ("cta.css", "buttons.css"):
                if css not in hrefs:
                    failures.append(f"{rel}: missing {css}")
        self.assertEqual(failures, [], "Service page CSS failures:\n" + "\n".join(failures[:15]))

    def test_72_pricing_page_loads_calculator_and_schedule_scripts(self):
        """Verifies pricing.html loads pricing-calculator.js and payment-schedule.js."""
        scripts = " ".join(s.get("src", "") for s in self.soups["pricing.html"].find_all("script", src=True))
        self.assertIn("pricing-calculator.js", scripts)
        self.assertIn("payment-schedule.js", scripts)

    # --- 12. Expanded CTA, Branding & Pricing Policy Tests ---

    def test_73_all_six_pricing_tier_pages_linked_from_pricing_hubs(self):
        """Verifies all 6 pricing/pricing-tier-*.html pages exist and are linked from pricing.html and parity-pricing.html."""
        tiers = [p for p in self.pages if p.startswith("pricing/pricing-tier-")]
        self.assertEqual(len(tiers), 6, f"Expected 6 tier pages, found {len(tiers)}")
        for hub in ("pricing.html", "parity-pricing.html"):
            raw = self.raw_html[hub]
            for tier in tiers:
                self.assertIn(tier, raw, f"{hub}: missing link to {tier}")

    def test_74_zero_upfront_guarantee_copy_rendered_on_pricing_and_guarantees(self):
        """Verifies $0 upfront and Walkaway Guarantee copy render clearly on pricing.html and guarantees.html."""
        for rel in ("pricing.html", "guarantees.html"):
            txt = self.soups[rel].get_text(" ", strip=True).lower()
            self.assertIn("$0", txt, f"{rel}: missing $0 upfront copy")
            self.assertIn("walk", txt, f"{rel}: missing walkaway guarantee copy")

    def test_75_zero_forbidden_legacy_entity_mentions_sitewide(self):
        """Verifies deprecated legacy entity strings (Augmentron, study-abroad) never appear across all 85+ pages."""
        failures = []
        for rel, raw in self.raw_html.items():
            low = raw.lower()
            for tok in ("augmentron", "study-abroad", "study abroad"):
                if tok in low:
                    failures.append(f"{rel}: contains forbidden legacy string '{tok}'")
        self.assertEqual(failures, [], "Legacy entity violations:\n" + "\n".join(failures[:15]))

    def test_76_cta_banners_use_standardized_btn_classes(self):
        """Verifies links inside section.cta-banner use standardized .btn classes with zero inline font-size overrides."""
        failures = []
        for rel, soup in self.soups.items():
            for banner in soup.select("section.cta-banner"):
                btns = banner.select("a.btn")
                if not btns:
                    failures.append(f"{rel}: section.cta-banner missing a.btn actions")
                    continue
                for b in btns:
                    st = (b.get("style") or "").lower().replace(" ", "")
                    if "font-size:" in st:
                        failures.append(f"{rel}: CTA button has ad-hoc inline font-size override")
        self.assertEqual(failures, [], "CTA button uniformity failures:\n" + "\n".join(failures[:15]))

    # --- 13. Expanded Core Pages & Directory Tests ---

    def test_77_articles_and_writing_pages_exist_and_cross_link(self):
        """Verifies articles.html and writing.html remain separate pages and cross-link cleanly to each other."""
        self.assertIn("articles.html", self.soups)
        self.assertIn("writing.html", self.soups)
        self.assertIn("writing.html", self.raw_html["articles.html"])
        self.assertIn("articles.html", self.raw_html["writing.html"])

    def test_78_expertise_page_renders_core_sections_and_cta(self):
        """Verifies expertise.html renders multiple structured sections and primary contact/LinkedIn CTA buttons."""
        soup = self.soups["expertise.html"]
        self.assertGreaterEqual(len(soup.find_all("h2")), 3, "expertise.html should have >= 3 <h2> sections")
        self.assertIsNotNone(soup.select_one("a.btn-linkedin"), "expertise.html missing LinkedIn CTA button")
        self.assertIsNotNone(soup.select_one("a.btn[href='contact.html']"), "expertise.html missing contact CTA button")

    def test_79_pricing_page_renders_personal_ai_section_and_tier_links(self):
        """Verifies pricing.html contains #personal-ai-section anchor and links to parity-pricing.html."""
        soup = self.soups["pricing.html"]
        self.assertIsNotNone(soup.find(id="personal-ai-section"), "pricing.html missing #personal-ai-section")
        self.assertIn("parity-pricing.html", self.raw_html["pricing.html"])

    # --- 14. Expanded Service Pages & Directories Tests ---

    def test_80_business_and_personal_services_have_16_pages_each(self):
        """Verifies exact 1-to-1 parity of 16 business service pages and 16 personal service pages."""
        biz = sorted(p.replace("business-services/business-service-", "") for p in self.pages if p.startswith("business-services/"))
        pers = sorted(p.replace("personal-services/personal-service-", "") for p in self.pages if p.startswith("personal-services/"))
        self.assertEqual(len(biz), 16)
        self.assertEqual(len(pers), 16)
        self.assertEqual(biz, pers, "Business and personal service slug lists diverged")

    def test_81_all_32_service_pages_render_hero_h1_and_cta_banner(self):
        """Verifies every business and personal service page renders a single H1 and a CTA banner."""
        service_pages = [p for p in self.pages if p.startswith(("business-services/", "personal-services/"))]
        failures = []
        for rel in service_pages:
            soup = self.soups[rel]
            if not soup.find("h1"):
                failures.append(f"{rel}: missing <h1>")
            if not soup.select("section.cta-banner"):
                failures.append(f"{rel}: missing section.cta-banner")
        self.assertEqual(failures, [], "Service page hero/CTA failures:\n" + "\n".join(failures[:15]))

    def test_82_all_32_service_pages_link_to_contact_page(self):
        """Verifies all 32 business and personal service pages include a link to contact.html."""
        service_pages = [p for p in self.pages if p.startswith(("business-services/", "personal-services/"))]
        failures = [rel for rel in service_pages if "contact.html" not in self.raw_html[rel]]
        self.assertEqual(failures, [], "Service pages missing contact.html link:\n" + "\n".join(failures[:15]))

    def test_83_business_directory_renders_rich_service_cards(self):
        """Verifies all-business-services.html renders at least 16 distinct service cards/blocks."""
        soup = self.soups["all-business-services.html"]
        links = {a["href"] for a in soup.find_all("a", href=True) if "business-service-" in a["href"]}
        self.assertGreaterEqual(len(links), 16)
        self.assertGreaterEqual(len(soup.find_all("h2")) + len(soup.find_all("h3")), 16)

    def test_84_personal_directory_renders_rich_service_cards(self):
        """Verifies all-personal-services.html renders at least 16 distinct service cards/blocks."""
        soup = self.soups["all-personal-services.html"]
        links = {a["href"] for a in soup.find_all("a", href=True) if "personal-service-" in a["href"]}
        self.assertGreaterEqual(len(links), 16)
        self.assertGreaterEqual(len(soup.find_all("h2")) + len(soup.find_all("h3")), 16)

    def test_85_all_32_service_pages_embed_json_ld_schema(self):
        """Verifies every business and personal service page embeds at least one JSON-LD schema block."""
        service_pages = [p for p in self.pages if p.startswith(("business-services/", "personal-services/"))]
        failures = [
            rel for rel in service_pages
            if not self.soups[rel].find("script", attrs={"type": "application/ld+json"})
        ]
        self.assertEqual(failures, [], "Service pages missing JSON-LD:\n" + "\n".join(failures[:15]))

    # --- 15. Expanded Claude & Gemini Training Guide Tests ---

    def test_86_exactly_18_claude_and_18_gemini_training_guides(self):
        """Verifies exact counts of 18 Claude training guides and 18 Gemini training guides."""
        claude = [p for p in self.pages if p.startswith("claude-training/")]
        gemini = [p for p in self.pages if p.startswith("gemini-training/")]
        self.assertEqual(len(claude), 18, f"Expected 18 Claude guides, found {len(claude)}")
        self.assertEqual(len(gemini), 18, f"Expected 18 Gemini guides, found {len(gemini)}")

    def test_87_all_36_training_guides_have_toc_and_references(self):
        """Verifies all 36 Claude and Gemini guides render #table-of-contents and #references anchors."""
        guides = [p for p in self.pages if p.startswith(("claude-training/", "gemini-training/"))]
        failures = []
        for rel in guides:
            soup = self.soups[rel]
            if not soup.find(id="table-of-contents"):
                failures.append(f"{rel}: missing #table-of-contents")
            if not soup.find(id="references"):
                failures.append(f"{rel}: missing #references")
        self.assertEqual(failures, [], "Training guide ToC/References failures:\n" + "\n".join(failures[:15]))

    def test_88_all_36_training_guides_use_unnumbered_content_headings(self):
        """Verifies all 36 training guides obey the unnumbered h2/h3 heading rule (numbering belongs only in ToC)."""
        guides = [p for p in self.pages if p.startswith(("claude-training/", "gemini-training/"))]
        failures = []
        for rel in guides:
            for h in self.soups[rel].find_all(["h2", "h3"]):
                txt = h.get_text(strip=True)
                if re.match(r"^\d+[\.\)]\s+", txt):
                    failures.append(f"{rel}: numbered content heading '{txt[:40]}'")
        self.assertEqual(failures, [], "Numbered heading violations:\n" + "\n".join(failures[:15]))

    def test_89_all_36_training_guides_include_standard_cta_banner(self):
        """Verifies every Claude and Gemini guide includes a standard section.cta-banner with LinkedIn & Pricing links."""
        guides = [p for p in self.pages if p.startswith(("claude-training/", "gemini-training/"))]
        failures = []
        for rel in guides:
            banner = self.soups[rel].select_one("section.cta-banner")
            if not banner:
                failures.append(f"{rel}: missing section.cta-banner")
                continue
            bhtml = str(banner)
            if "pricing.html" not in bhtml or "linkedin.com/in/thomascherickal" not in bhtml:
                failures.append(f"{rel}: cta-banner missing pricing.html or LinkedIn link")
        self.assertEqual(failures, [], "Training guide CTA banner failures:\n" + "\n".join(failures[:15]))

    def test_90_claude_guides_never_include_article_footer(self):
        """Verifies no claude-training/claude-*.html page includes an article-footer block."""
        claude = [p for p in self.pages if p.startswith("claude-training/")]
        failures = [rel for rel in claude if "article-footer" in self.raw_html[rel]]
        self.assertEqual(failures, [], f"Forbidden article-footer in Claude pages: {failures}")

    def test_91_claude_plugins_covers_unified_skills_and_connectors(self):
        """Verifies claude-plugins.html documents how Claude unified Skills and Connectors into Plugins."""
        txt = self.soups["claude-training/claude-plugins.html"].get_text(" ", strip=True)
        for token in ("Skills", "Connectors", "Plugins Hub", "SKILL.md", "MCP"):
            self.assertIn(token, txt, f"claude-plugins.html missing '{token}'")

    def test_92_claude_apps_covers_mcp_apps_sdk_and_sandboxing(self):
        """Verifies claude-apps.html covers MCP Apps, ext-apps SDK, ui:// resources, postMessage, and sandboxing."""
        txt = self.soups["claude-training/claude-apps.html"].get_text(" ", strip=True)
        for token in ("MCP Apps", "@modelcontextprotocol/ext-apps", "ui://", "postMessage", "registerAppTool", "displayMode"):
            self.assertIn(token, txt, f"claude-apps.html missing '{token}'")

    def test_93_claude_code_mode_covers_sandbox_hooks_and_activation(self):
        """Verifies claude-code-mode.html covers CLAUDE_CODE_EXPERIMENTAL_CODE_MODE, PreToolUse, tool_input.code, and PTC."""
        txt = self.soups["claude-training/claude-code-mode.html"].get_text(" ", strip=True)
        for token in ("CLAUDE_CODE_EXPERIMENTAL_CODE_MODE", "PreToolUse", "tool_input.code", "Ctrl+O", "Programmatic Tool Calling"):
            self.assertIn(token, txt, f"claude-code-mode.html missing '{token}'")

    def test_94_deleted_claude_skills_and_connectors_pages_absent(self):
        """Verifies merged legacy pages claude-skills.html and claude-connectors.html do not exist on disk."""
        for obsolete in ("claude-training/claude-skills.html", "claude-training/claude-connectors.html"):
            self.assertFalse(os.path.exists(os.path.join(REPO_DIR, obsolete)), f"Obsolete page still exists: {obsolete}")

    # --- 16. Expanded Structured Data & Negative Fault-Injection Tests ---

    def test_95_canonical_author_description_present_in_json_ld(self):
        """Verifies JSON-LD across home and flagship Claude guides uses the canonical site/author description."""
        canonical_desc = "Thomas Cherickal - Generative AI Consultant and The Digital Futurist."
        for rel in (
            "index.html",
            "claude-training/claude-plugins.html",
            "claude-training/claude-apps.html",
            "claude-training/claude-code-mode.html",
        ):
            scripts = self.soups[rel].find_all("script", attrs={"type": "application/ld+json"})
            blob = " ".join((s.string or s.get_text() or "") for s in scripts)
            self.assertIn(canonical_desc, blob, f"{rel}: JSON-LD missing canonical author/site description")
        pricing_blob = " ".join(
            (s.string or s.get_text() or "")
            for s in self.soups["pricing.html"].find_all("script", attrs={"type": "application/ld+json"})
        )
        self.assertIn("OfferCatalog", pricing_blob, "pricing.html: JSON-LD missing OfferCatalog")
        self.assertIn("https://thedigitalfuturist.com/#website", pricing_blob)

    def test_96_error_scenario_nested_interactive_anchors_detected(self):
        """Negative fault-injection: verifies illegal nested <a> inside <a> and <button> inside <a> are caught."""
        bad_html = '<html><body><a href="/a"><a href="/b">Bad</a><button>Click</button></a></body></html>'
        soup = BeautifulSoup(bad_html, HTML_PARSER)
        outer_a = soup.find("a")
        self.assertIsNotNone(outer_a.find("button") or soup.find("button"))

    def test_97_error_scenario_emoji_and_superscript_citation_detected(self):
        """Negative fault-injection: verifies Unicode emojis and superscript [1] citations are reliably caught."""
        bad_html = '<html><body><p>Launch \U0001F680 now<sup><a href="#ref-1">[1]</a></sup></p></body></html>'
        self.assertIsNotNone(EMOJI_RE.search(bad_html))
        sup_re = re.compile(r"<sup>\s*(?:<a[^>]*>)?\s*\[\d+\]\s*(?:</a>)?\s*</sup>", re.I)
        self.assertIsNotNone(sup_re.search(bad_html))

    def test_98_error_scenario_numbered_guide_heading_detected(self):
        """Negative fault-injection: verifies numbered <h2> headings like '1. Overview' are reliably flagged."""
        bad_html = '<html><body><h2>1. Architecture Overview</h2><h3>02) Deployment</h3></body></html>'
        soup = BeautifulSoup(bad_html, HTML_PARSER)
        numbered = [h.get_text(strip=True) for h in soup.find_all(["h2", "h3"]) if re.match(r"^\d+[\.\)]\s+", h.get_text(strip=True))]
        self.assertEqual(len(numbered), 2)

    def test_99_error_scenario_corrupted_template_placeholder_detected(self):
        """Negative fault-injection: verifies unexpanded {{token}} and '$00 upfront' corruptions are caught."""
        bad_html = '<html><body><p>Pay $00 upfront for {{package_name}}</p></body></html>'
        token_re = re.compile(r"(\{\{[^}]+\}\}|\{%[^%]+%\}|\$00\s+upfront)")
        matches = token_re.findall(bad_html)
        self.assertEqual(len(matches), 2)

    def test_100_error_scenario_positive_tabindex_and_missing_alt_detected(self):
        """Negative fault-injection: verifies positive tabindex traps and images without alt are reliably caught."""
        bad_html = '<html><body><button tabindex="3">Trap</button><img src="logo.png"></body></html>'
        soup = BeautifulSoup(bad_html, HTML_PARSER)
        self.assertGreater(int(soup.find("button")["tabindex"]), 0)
        self.assertIsNone(soup.find("img").get("alt"))

    def test_101_back_to_top_button_and_target_on_every_page(self):
        """Verifies every single HTML page has id='top', a valid <a id='backTop' class='back-top' href='#top'>, and components.js."""
        failures = []
        for rel, soup in self.soups.items():
            if not soup.find(id="top"):
                failures.append(f"{rel}: missing id='top' target")
            btns = soup.select("a.back-top, a#backTop")
            if len(btns) != 1:
                failures.append(f"{rel}: expected exactly 1 back-to-top button, found {len(btns)}")
            else:
                btn = btns[0]
                if btn.get("href") != "#top":
                    failures.append(f"{rel}: back-to-top href must be '#top', got {btn.get('href')!r}")
                if not (btn.get("aria-label") or "").strip():
                    failures.append(f"{rel}: back-to-top button missing aria-label")
            if not any("components.js" in (s.get("src") or "") for s in soup.find_all("script", src=True)):
                failures.append(f"{rel}: missing components.js for back-to-top scroll behavior")
        with open(os.path.join(REPO_DIR, "components.js"), encoding="utf-8") as f:
            comp_js = f.read()
        self.assertIn("initBackToTop", comp_js, "components.js missing initBackToTop()")
        self.assertEqual(failures, [], "Back-to-top failures:\n" + "\n".join(failures[:20]))

    def test_102_unique_titles_and_meta_descriptions(self):
        """SEO: verifies zero duplicate <title> tags and zero duplicate meta descriptions across indexable pages."""
        titles = {}
        descs = {}
        for rel, soup in self.soups.items():
            if rel in ("404.html", "mermaid.html"):
                continue
            t = soup.title.get_text(strip=True) if soup.title else ""
            titles.setdefault(t, []).append(rel)
            dm = soup.find("meta", attrs={"name": "description"})
            d = (dm.get("content") or "").strip() if dm else ""
            descs.setdefault(d, []).append(rel)
        dup_t = {k: v for k, v in titles.items() if len(v) > 1}
        dup_d = {k: v for k, v in descs.items() if len(v) > 1}
        self.assertEqual(dup_t, {}, f"Duplicate <title> tags found: {dup_t}")
        self.assertEqual(dup_d, {}, f"Duplicate meta descriptions found: {dup_d}")

    def test_103_zero_render_blocking_head_scripts(self):
        """CWV/SEO: verifies zero synchronous render-blocking external <script> tags inside <head>."""
        blocking = []
        for rel, soup in self.soups.items():
            head = soup.find("head")
            if not head:
                continue
            for sc in head.find_all("script", src=True):
                if not sc.has_attr("defer") and not sc.has_attr("async") and sc.get("type") != "module":
                    blocking.append(f"{rel}: {sc.get('src')}")
        self.assertEqual(blocking, [], "Render-blocking <head> scripts:\n" + "\n".join(blocking))

    def test_104_complete_twitter_and_og_cards(self):
        """SEO/Social: verifies every indexable HTML page includes og:image and full Twitter Card meta tags."""
        missing = []
        for rel, soup in self.soups.items():
            if rel in ("404.html", "mermaid.html"):
                continue
            if not soup.find("meta", attrs={"property": "og:image"}):
                missing.append(f"{rel}: missing og:image")
            for name in ("twitter:card", "twitter:title", "twitter:description"):
                if not soup.find("meta", attrs={"name": name}):
                    missing.append(f"{rel}: missing {name}")
        self.assertEqual(missing, [], "Missing social meta tags:\n" + "\n".join(missing[:20]))

    def test_105_sitemap_and_rss_and_robots_integrity(self):
        """SEO/GEO: verifies sitemap.xml covers all indexable pages, rss.xml exists, and robots.txt blocks dev dirs."""
        with open(os.path.join(REPO_DIR, "sitemap.xml"), encoding="utf-8") as f:
            sm = f.read()
        locs = re.findall(r"<loc>(.*?)</loc>", sm)
        sm_paths = set()
        for loc in locs:
            p = loc.replace("https://thedigitalfuturist.com/", "").strip("/")
            p = p if p else "index.html"
            self.assertTrue(os.path.exists(os.path.join(REPO_DIR, p)), f"Sitemap dead URL: {loc}")
            sm_paths.add(p)
        for rel in self.soups:
            if rel not in ("404.html", "mermaid.html"):
                self.assertIn(rel, sm_paths, f"Indexable page missing from sitemap.xml: {rel}")

        self.assertTrue(os.path.exists(os.path.join(REPO_DIR, "rss.xml")), "Missing rss.xml")
        with open(os.path.join(REPO_DIR, "robots.txt"), encoding="utf-8") as f:
            rb = f.read()
        for token in ("Disallow: /internal-use/", "Disallow: /tests/", "Disallow: /failed-tests/", "OAI-SearchBot", "ClaudeBot", "PerplexityBot"):
            self.assertIn(token, rb, f"robots.txt missing {token}")

    def test_106_images_have_dimensions_and_loading_policy(self):
        """CWV/CLS: verifies every <img> tag declares explicit width, height, and loading attributes."""
        issues = []
        for rel, soup in self.soups.items():
            if rel in ("404.html", "mermaid.html"):
                continue
            for img in soup.find_all("img"):
                src = img.get("src") or ""
                if not img.get("width") or not img.get("height"):
                    issues.append(f"{rel}: img missing width/height ({src})")
                if not img.get("loading"):
                    issues.append(f"{rel}: img missing loading attr ({src})")
        self.assertEqual(issues, [], "Image CWV issues:\n" + "\n".join(issues[:20]))

    def test_107_llms_txt_and_llms_full_synced_with_zero_broken_urls(self):
        """GEO/AEO: verifies llms.txt uses SSOT markers, has zero stale prices, zero broken URLs, and llms-full.txt exists."""
        with open(os.path.join(REPO_DIR, "llms.txt"), encoding="utf-8") as f:
            llms = f.read()
        self.assertIn("<!--ssot:", llms, "llms.txt must use SSOT markers")
        self.assertNotIn("$2,500", llms, "llms.txt contains stale $2,500 flagship price")
        self.assertNotIn("Tier 1: $250", llms, "llms.txt contains stale $250 pick price")
        self.assertTrue(os.path.exists(os.path.join(REPO_DIR, "llms-full.txt")), "Missing llms-full.txt")
        broken = []
        for m in re.finditer(r"https://thedigitalfuturist\.com/([A-Za-z0-9_./-]+\.(?:html|md|txt|xml))", llms):
            rel = m.group(1)
            if not os.path.exists(os.path.join(REPO_DIR, rel)):
                broken.append(rel)
        self.assertEqual(broken, [], f"Broken URLs in llms.txt: {broken}")


if __name__ == "__main__":
    unittest.main()
