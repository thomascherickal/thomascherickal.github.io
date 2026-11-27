#!/usr/bin/env python3
"""
validate.py - JSON Schema & Cross-File Invariant Validator for data/*.json.

Exits 0 when all schemas and cross-file invariants pass.
Exits non-zero when any schema violation or invariant failure is detected.
Strict Repo Compliance: Zero emojis, zero em-dashes.
"""

import json
import os
import re
import sys
from jsonschema import Draft202012Validator

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(REPO_ROOT, "data")
SCHEMA_DIR = os.path.join(DATA_DIR, "schema")

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

MODEL_VERSION_RE = re.compile(
    r"\b(?:Claude\s+\d|Gemini\s+\d+\.\d+|GPT-\d|Llama\s+\d|Qwen\s+\d|DeepSeek\s+V\d)\b",
    re.IGNORECASE,
)


def _walk_strings(obj, path=""):
    if isinstance(obj, str):
        yield path, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk_strings(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_strings(v, f"{path}[{i}]")


def validate_all(data_dir=DATA_DIR, schema_dir=SCHEMA_DIR):
    errors = []
    loaded = {}

    if not os.path.isdir(data_dir):
        return [f"Missing data directory: {data_dir}"]

    json_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".json"))
    for fname in json_files:
        fpath = os.path.join(data_dir, fname)
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
            loaded[fname] = data
        except Exception as exc:
            errors.append(f"{fname}: invalid JSON ({exc})")
            continue

        schema_name = fname.replace(".json", ".schema.json")
        spath = os.path.join(schema_dir, schema_name)
        if not os.path.exists(spath):
            errors.append(f"{fname}: missing schema {schema_name}")
            continue
        try:
            with open(spath, encoding="utf-8") as sf:
                schema = json.load(sf)
            validator = Draft202012Validator(schema)
            for err in sorted(validator.iter_errors(data), key=lambda e: list(e.path)):
                loc = ".".join(str(p) for p in err.path) or "<root>"
                errors.append(f"{fname} ({loc}): {err.message}")
        except Exception as exc:
            errors.append(f"{schema_name}: schema error ({exc})")

    # Repo hygiene invariants across all data/*.json strings
    for fname, data in loaded.items():
        for path, val in _walk_strings(data):
            if "\u2014" in val or "&mdash;" in val or "&#8212;" in val:
                errors.append(f"{fname} ({path}): forbidden em-dash")
            if EMOJI_RE.search(val):
                errors.append(f"{fname} ({path}): forbidden emoji")
            if fname not in ("publications.json",) and not path.endswith(("url", "href", "slug")):
                m = MODEL_VERSION_RE.search(val)
                if m:
                    errors.append(f"{fname} ({path}): forbidden LLM version number '{m.group(0)}'")

    # Cross-file & pricing invariants
    pricing = loaded.get("pricing.json", {})
    for off in pricing.get("offerings", []):
        oid = off.get("id", "<unknown>")
        cap = off.get("cohort_capacity", {})
        inc = cap.get("included_seats", 0)
        mx = cap.get("max_seats", 0)
        if inc > mx:
            errors.append(f"pricing.json ({oid}): included_seats ({inc}) > max_seats ({mx})")

        sched = off.get("schedule", {})
        total_days = sched.get("total_calendar_duration_days", 0)
        ms = off.get("payment_rules", {}).get("milestone_schedule", [])
        if ms:
            pct_sum = sum(m.get("percentage", 0) for m in ms)
            if pct_sum != 100:
                errors.append(f"pricing.json ({oid}): milestone percentages sum to {pct_sum}, expected 100")
            for idx, m in enumerate(ms, 1):
                if m.get("installment") != idx:
                    errors.append(f"pricing.json ({oid}): milestone installment {m.get('installment')} != {idx}")
                day = m.get("due_after_day", 0)
                if day <= 0 or day > total_days:
                    errors.append(f"pricing.json ({oid}): due_after_day ({day}) outside 1..{total_days}")

    # Tier invariants: consistent multipliers & no duplicate countries unless precedence_rule exists
    tiers_data = loaded.get("tiers.json", {})
    has_precedence = bool(tiers_data.get("precedence_rule", "").strip())
    seen_countries = {}
    for t in tiers_data.get("tiers", []):
        tnum = t.get("tier")
        mult = t.get("multiplier")
        disc = t.get("discountPercent")
        if mult is not None and disc is not None:
            expected_mult = round(1.0 - (disc / 100.0), 4)
            if tnum != 6 and abs(mult - expected_mult) > 1e-4:
                errors.append(f"tiers.json (tier {tnum}): multiplier {mult} != 1 - {disc}/100")
        for c in t.get("examples", []):
            key = c.strip().lower()
            if key in seen_countries and not has_precedence:
                errors.append(f"tiers.json: country '{c}' in tier {seen_countries[key]} and tier {tnum} without precedence_rule")
            seen_countries[key] = tnum

    # Publications slug-vs-title consistency (catches renames like D9)
    pubs = loaded.get("publications.json", {})
    pub_list = pubs.get("publications", pubs) if isinstance(pubs, dict) else pubs
    if isinstance(pub_list, list):
        for p in pub_list:
            if not isinstance(p, dict):
                continue
            title = (p.get("title") or "").lower()
            url = (p.get("url") or "").lower()
            if "hermes-agent" in url and "hermes" not in title:
                errors.append(f"publications.json: URL '{url}' has 'hermes-agent' but title '{p.get('title')}' does not")

    return errors


def main():
    errors = validate_all()
    if errors:
        print("SSOT VALIDATION FAILED:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("SSOT VALIDATION PASSED: All data/*.json schemas and cross-file invariants valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
