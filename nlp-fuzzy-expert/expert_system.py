#!/usr/bin/env python3
"""
expert_system.py - Expert System Knowledge Base & Forward-Chaining Inference Engine.

Implements facts, propositions, rule base with NLP & fuzzy condition antecedents,
conflict resolution, and consequents that synchronize target documents.

Strict Repo Compliance:
- Zero emojis.
- Zero em-dashes (spaced hyphens ' - ' or plain hyphens '-' used).
"""

import os
import sys
import json
import re
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(REPO_DIR, "data")

from fuzzy_engine import fuzzy_matcher
from nlp_engine import nlp_engine, apply_sync_highlight, strip_sync_highlights

import difflib

def compute_changed_lines(old_text, new_text):
    """Computes formatted ranges of 1-based line numbers in new_text that changed."""
    old_lines = old_text.splitlines()
    new_lines = new_text.splitlines()

    matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
    changed_line_numbers = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ('replace', 'insert'):
            for line_idx in range(j1 + 1, j2 + 1):
                changed_line_numbers.append(line_idx)
        elif tag == 'delete':
            changed_line_numbers.append(max(1, j1))

    changed_line_numbers = sorted(set(changed_line_numbers))
    if not changed_line_numbers:
        return "None"

    ranges = []
    start = changed_line_numbers[0]
    prev = start
    for num in changed_line_numbers[1:]:
        if num == prev + 1:
            prev = num
        else:
            if start == prev:
                ranges.append(f"{start}")
            else:
                ranges.append(f"{start}-{prev}")
            start = num
            prev = num
    if start == prev:
        ranges.append(f"{start}")
    else:
        ranges.append(f"{start}-{prev}")

    return ", ".join(ranges)

_JSON_CACHE = {}

def load_json(filepath):
    """Safely loads a JSON file with retry on transient concurrent read and mtime-based caching."""
    if not os.path.exists(filepath):
        return {}
    try:
        st = os.stat(filepath)
        cache_key = (os.path.abspath(filepath), st.st_mtime_ns, st.st_size)
        if cache_key in _JSON_CACHE:
            return _JSON_CACHE[cache_key]
    except OSError:
        pass

    for attempt in range(5):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    time.sleep(0.05)
                    continue
                data = json.loads(content)
                try:
                    st = os.stat(filepath)
                    cache_key = (os.path.abspath(filepath), st.st_mtime_ns, st.st_size)
                    _JSON_CACHE[cache_key] = data
                except OSError:
                    pass
                return data
        except (json.JSONDecodeError, OSError):
            if attempt == 4:
                raise
            time.sleep(0.05)
    return {}

def save_json(filepath, data):
    """Safely saves pretty-printed JSON using atomic file replacement.
    Strictly forbids mutating the authoritative data/ directory.
    Enforces unidirectional synchronization: data/*.json -> website only.
    """
    abs_fp = os.path.abspath(filepath)
    abs_data_dir = os.path.abspath(DATA_DIR)
    try:
        if os.path.commonpath([abs_fp, abs_data_dir]) == abs_data_dir:
            raise PermissionError(
                f"MUTATION BLOCKED: '{filepath}' is inside authoritative data/ directory. "
                f"Sync is strictly unidirectional (data/*.json -> website). "
                f"Website changes must never be reflected back into the data directory."
            )
    except ValueError:
        pass

    tmp_path = f"{filepath}.{os.getpid()}_{time.time_ns()}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, filepath)
        try:
            st = os.stat(filepath)
            _JSON_CACHE[(os.path.abspath(filepath), st.st_mtime_ns, st.st_size)] = data
        except OSError:
            pass
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

def verify_unidirectional_sync(source_data_dir=DATA_DIR, target_files=None):
    """Verifies that synchronization is strictly unidirectional.
    Ensures that data/*.json files are exclusively read-only sources of truth
    and that no target file path points inside the data directory.
    """
    abs_data_dir = os.path.abspath(source_data_dir)
    if target_files:
        for fpath in target_files:
            abs_fp = os.path.abspath(fpath)
            try:
                if os.path.commonpath([abs_fp, abs_data_dir]) == abs_data_dir:
                    raise PermissionError(
                        f"Unidirectional violation: target file '{fpath}' is inside data directory '{source_data_dir}'."
                    )
            except ValueError:
                continue
    return True


class Fact:
    """Represents an atomic proposition in the knowledge base."""

    def __init__(self, domain, predicate, value, metadata=None):
        self.domain = domain
        self.predicate = predicate
        self.value = value
        self.metadata = metadata or {}

    def __repr__(self):
        return f"<Fact [{self.domain}] {self.predicate} = {self.value}>"

class KnowledgeBase:
    """Working memory containing all authoritative facts from data/*.json."""

    def __init__(self, data_dir=None):
        self.data_dir = data_dir or DATA_DIR
        self.facts = {}
        self.raw_data = {}
        self.reload()

    def add_fact(self, domain, predicate, value, metadata=None):
        fact = Fact(domain, predicate, value, metadata)
        key = f"{domain}:{predicate}"
        self.facts[key] = fact
        return fact

    def get_fact(self, domain, predicate, default=None):
        key = f"{domain}:{predicate}"
        fact = self.facts.get(key)
        return fact.value if fact else default

    def reload(self):
        """Loads all facts from the 9 authoritative JSON files in data/."""
        self.facts.clear()
        self.raw_data.clear()

        # 1. Pricing
        pricing_file = os.path.join(self.data_dir, "pricing.json")
        p_data = load_json(pricing_file)
        self.raw_data["pricing"] = p_data

        offerings = {off["id"]: off for off in p_data.get("offerings", [])}
        b_flag = offerings.get("flagship-business-training", {})
        p_flag = offerings.get("flagship-personal-training", {})
        b_ind = offerings.get("individual-service-business", {})
        p_ind = offerings.get("individual-service-personal", {})
        rules = p_data.get("general_payment_rules", {})

        self.add_fact("pricing", "b_flag_price", int(b_flag.get("tier1_base_price_usd", 2000)))
        self.add_fact("pricing", "b_flag_days", int(b_flag.get("schedule", {}).get("total_calendar_duration_days", 60)))
        self.add_fact("pricing", "b_flag_live_days", int(b_flag.get("schedule", {}).get("live_training_days", 30)))
        self.add_fact("pricing", "b_flag_supp_days", int(b_flag.get("schedule", {}).get("support_phase_days", 30)))
        self.add_fact("pricing", "b_flag_hours", int(b_flag.get("schedule", {}).get("live_hours_per_day", 4)))
        self.add_fact("pricing", "b_flag_seats", int(b_flag.get("cohort_capacity", {}).get("included_seats", 10)))
        self.add_fact("pricing", "b_flag_extra_seat", int(b_flag.get("cohort_capacity", {}).get("additional_seat_fee_usd", 200)))
        self.add_fact("pricing", "b_flag_sched_notes", b_flag.get("schedule", {}).get("schedule_notes", "30 working days over 8 weeks with 8 - 10 days as holidays and recovery days, 4 hours per day"))

        self.add_fact("pricing", "p_flag_price", int(p_flag.get("tier1_base_price_usd", 1000)))
        self.add_fact("pricing", "p_flag_days", int(p_flag.get("schedule", {}).get("total_calendar_duration_days", 60)))
        self.add_fact("pricing", "p_flag_live_days", int(p_flag.get("schedule", {}).get("live_training_days", 30)))
        self.add_fact("pricing", "p_flag_supp_days", int(p_flag.get("schedule", {}).get("support_phase_days", 30)))
        self.add_fact("pricing", "p_flag_hours", int(p_flag.get("schedule", {}).get("live_hours_per_day", 4)))
        self.add_fact("pricing", "p_flag_sched_notes", p_flag.get("schedule", {}).get("schedule_notes", "30 working days over 6 weeks with 8 - 10 days as holidays and recovery days, 4 hours per day"))

        self.add_fact("pricing", "b_ind_price", int(b_ind.get("tier1_base_price_usd", 200)))
        self.add_fact("pricing", "b_ind_days", int(b_ind.get("schedule", {}).get("total_calendar_duration_days", 10)))
        self.add_fact("pricing", "b_ind_live_days", int(b_ind.get("schedule", {}).get("live_training_days", 5)))
        self.add_fact("pricing", "b_ind_supp_days", int(b_ind.get("schedule", {}).get("support_phase_days", 5)))
        self.add_fact("pricing", "b_ind_extra_seat", int(b_ind.get("cohort_capacity", {}).get("additional_seat_fee_usd", 200)))

        self.add_fact("pricing", "p_ind_price", int(p_ind.get("tier1_base_price_usd", 100)))
        self.add_fact("pricing", "p_ind_days", int(p_ind.get("schedule", {}).get("total_calendar_duration_days", 10)))
        self.add_fact("pricing", "p_ind_live_days", int(p_ind.get("schedule", {}).get("live_training_days", 5)))
        self.add_fact("pricing", "p_ind_supp_days", int(p_ind.get("schedule", {}).get("support_phase_days", 5)))

        self.add_fact("pricing", "currency_acceptance", rules.get("currency_acceptance", "USD, EUR, GBP, AUD, CAD, INR (via PayPal, international wire, or approved payment rails)"))
        self.add_fact("pricing", "hours_per_day", int(rules.get("hours_per_day", 4)))

        # 2. Personal Services
        ps_file = os.path.join(self.data_dir, "personal-services.json")
        ps_data = load_json(ps_file)
        self.raw_data["personal_services"] = ps_data
        for s in ps_data.get("services", []):
            num = s.get("number")
            self.add_fact("personal_services", f"{num}_name", s.get("name"))
            self.add_fact("personal_services", f"{num}_category", s.get("category"))
            self.add_fact("personal_services", f"{num}_description", s.get("description"))
            self.add_fact("personal_services", f"{num}_slug", s.get("slug"))
            self.add_fact("personal_services", f"{num}_chip", s.get("role_chip"))

        # 3. Business Services
        bs_file = os.path.join(self.data_dir, "business-services.json")
        bs_data = load_json(bs_file)
        self.raw_data["business_services"] = bs_data
        for s in bs_data.get("services", []):
            num = s.get("number")
            self.add_fact("business_services", f"{num}_name", s.get("name"))
            self.add_fact("business_services", f"{num}_category", s.get("category"))
            self.add_fact("business_services", f"{num}_description", s.get("description"))
            self.add_fact("business_services", f"{num}_slug", s.get("slug"))
            self.add_fact("business_services", f"{num}_chip", s.get("role_chip"))

        # 4. Tiers
        tiers_file = os.path.join(self.data_dir, "tiers.json")
        t_data = load_json(tiers_file)
        self.raw_data["tiers"] = t_data
        for t in t_data.get("tiers", []):
            t_num = t.get("tier")
            self.add_fact("tiers", f"tier_{t_num}_multiplier", t.get("multiplier"))
            self.add_fact("tiers", f"tier_{t_num}_discount", t.get("discountPercent"))
            self.add_fact("tiers", f"tier_{t_num}_name", t.get("name"))
            self.add_fact("tiers", f"tier_{t_num}_subtitle", t.get("subtitle"))
            self.add_fact("tiers", f"tier_{t_num}_examples", ", ".join(t.get("examples", [])))

        # 5. Config
        cfg_file = os.path.join(self.data_dir, "config.json")
        c_data = load_json(cfg_file)
        self.raw_data["config"] = c_data
        self.add_fact("config", "author", c_data.get("author", "Thomas Cherickal"))
        self.add_fact("config", "email", c_data.get("email", "thomascherickal@gmail.com"))
        self.add_fact("config", "domain", c_data.get("domain", "https://thedigitalfuturist.com"))
        self.add_fact("config", "brand", c_data.get("brand", "The Digital Futurist"))
        self.add_fact("config", "linkedin", c_data.get("linkedin", "https://linkedin.com/in/thomascherickal"))

        # 6. FAQs
        faq_file = os.path.join(self.data_dir, "faqs.json")
        faq_data = load_json(faq_file)
        self.raw_data["faqs"] = faq_data
        policies = faq_data.get("core_policies", {})
        self.add_fact("faqs", "walkaway_summary", policies.get("walkaway_guarantee", {}).get("summary", "Walk away owing nothing if not satisfied after the first 5 days of completed training."))
        self.add_fact("faqs", "ip_summary", policies.get("ip_licensing", {}).get("summary", "All code, workflows, custom tooling, and multi-agent infrastructure developed belong 100% to the client."))

        # 7. Claude Guides
        claude_file = os.path.join(self.data_dir, "claude.json")
        cl_data = load_json(claude_file)
        self.raw_data["claude"] = cl_data
        for g in cl_data.get("guides", []):
            self.add_fact("claude", f"guide_{g.get('id')}_title", g.get("title"))
            self.add_fact("claude", f"guide_{g.get('id')}_slug", g.get("slug"))

        # 8. Gemini Guides
        gemini_file = os.path.join(self.data_dir, "gemini.json")
        gem_data = load_json(gemini_file)
        self.raw_data["gemini"] = gem_data
        for g in gem_data.get("guides", []):
            self.add_fact("gemini", f"guide_{g.get('id')}_title", g.get("title"))
            self.add_fact("gemini", f"guide_{g.get('id')}_slug", g.get("slug"))

        # 9. Publications
        pub_file = os.path.join(self.data_dir, "publications.json")
        pub_data = load_json(pub_file)
        self.raw_data["publications"] = pub_data

class Rule:
    """An Expert System Production Rule with NLP & Fuzzy antecedents."""

    def __init__(self, name, salience, condition_fn, action_fn, domain="general"):
        self.name = name
        self.salience = salience
        self.condition_fn = condition_fn
        self.action_fn = action_fn
        self.domain = domain

    def evaluate(self, file_path, content, kb, nlp, fuzzy):
        """Evaluates antecedent conditions."""
        return self.condition_fn(file_path, content, kb, nlp, fuzzy)

    def execute(self, file_path, content, match_info, kb, nlp, fuzzy):
        """Fires consequent action."""
        return self.action_fn(file_path, content, match_info, kb, nlp, fuzzy)

class InferenceEngine:
    """Forward-chaining inference engine executing knowledge-based rules."""

    def __init__(self, kb):
        self.kb = kb
        self.rules = []
        self.tracked_occurrences = []

    def register_rule(self, rule):
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.salience, reverse=True)

    def record_occurrence(self, file_path, page, section, line_no, field, old_v, new_v, context):
        rel_path = os.path.relpath(file_path, REPO_DIR) if os.path.isabs(file_path) else file_path
        self.tracked_occurrences.append({
            "file": rel_path,
            "page": page,
            "section": section,
            "line_number": line_no,
            "field_key": field,
            "old_value": str(old_v),
            "new_value": str(new_v),
            "context": context.strip()
        })

    def process_file(self, file_path):
        """Applies all matching rules to file_path in priority order.
        Strictly enforces unidirectional synchronization: website target documents
        are updated based on authoritative data, but data/ files can never be processed
        or mutated by the engine.
        Returns a structured report dictionary containing modification status,
        fired rules, changed line numbers, and synchronized fields.
        """
        # Strict Unidirectional Guard: Never allow processing or writing to files inside data/
        abs_fp = os.path.abspath(file_path)
        abs_data_dir = os.path.abspath(self.kb.data_dir if hasattr(self.kb, "data_dir") else DATA_DIR)
        try:
            if os.path.commonpath([abs_fp, abs_data_dir]) == abs_data_dir:
                raise PermissionError(
                    f"MUTATION BLOCKED: Unidirectional sync violation. '{file_path}' is inside the data directory. "
                    f"Sync is strictly one-way (data/*.json -> website). "
                    f"Website files cannot modify or write back to data/*.json."
                )
        except ValueError:
            pass

        rel_path = os.path.relpath(file_path, REPO_DIR) if os.path.isabs(file_path) else file_path
        if not os.path.exists(file_path):
            return {
                "file": rel_path,
                "modified": False,
                "fired_rules": [],
                "changed_lines": "None",
                "fields_synchronized": []
            }

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        clean_content = strip_sync_highlights(content)
        fired_rules = []
        file_occurrences = []

        for rule in self.rules:
            matched, match_info = rule.evaluate(file_path, clean_content, self.kb, nlp_engine, fuzzy_matcher)
            if matched:
                new_content, occurrences = rule.execute(file_path, clean_content, match_info, self.kb, nlp_engine, fuzzy_matcher)
                if new_content != clean_content:
                    clean_content = new_content
                    if rule.name not in fired_rules:
                        fired_rules.append(rule.name)
                for occ in occurrences:
                    file_occurrences.append(occ)
                    self.record_occurrence(
                        file_path, occ.get("page", os.path.basename(file_path)),
                        occ.get("section", rule.name), occ.get("line_no", 1),
                        occ.get("field", "synced_field"), occ.get("old_val", ""),
                        occ.get("new_val", ""), occ.get("context", "")
                    )

        modified = (clean_content != content)
        changed_lines = "None"
        if modified:
            changed_lines = compute_changed_lines(content, clean_content)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(clean_content)

        fields_synced = sorted(list(set([occ.get("field", "") for occ in file_occurrences if occ.get("field")])))

        return {
            "file": rel_path,
            "modified": modified,
            "fired_rules": fired_rules,
            "changed_lines": changed_lines,
            "fields_synchronized": fields_synced
        }
