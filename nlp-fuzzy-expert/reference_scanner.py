#!/usr/bin/env python3
"""
reference_scanner.py - Exhaustive Tracked-Term Location Scanner.

Reads every leaf value in data/*.json (read-only) and records every location
(file, page, section, line) where that term appears in the repository's HTML/JS
pages. Output records follow the existing *_references.json schema and add:
  - match_type:  "exact" (case-sensitive match) or "fuzzy" (case/whitespace/entity
                 normalized match); records produced by rules use "rule".
  - data_source: "data/<file>.json#<path>" of the authoritative value.

This module never writes to data/ and never modifies any page.

Strict Repo Compliance:
- Zero emojis.
- Zero em-dashes (spaced hyphens ' - ' or plain hyphens '-' used).
"""

import bisect
import hashlib
import html
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(REPO_DIR, "data")
CACHE_FILE = os.path.join(SCRIPT_DIR, ".scanner_cache.json")

DOMAIN_BY_FILE = {
    "pricing.json": "pricing",
    "business-services.json": "business_services",
    "personal-services.json": "personal_services",
    "tiers.json": "tiers",
    "claude.json": "claude",
    "gemini.json": "gemini",
    "publications.json": "publications",
    "config.json": "config",
    "faqs.json": "faqs",
    "contact.json": "contact",
    "guarantees.json": "guarantees",
}

SKIP_KEYS = {"directory", "border", "id", "badge_class", "icon"}
STOP_STRINGS = {"custom", "none", "true", "false", "null", "usd", "flagship", "service", "tier"}
MIN_STRING_LEN = 4
CONTEXT_MAX = 200

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")
HEADING_RE = re.compile(r"<h[1-3][^>]*>(.*?)</h[1-3]>", re.IGNORECASE | re.DOTALL)
TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
H1_RE = re.compile(r"<h1[^>]*>(.*?)</h1>", re.IGNORECASE | re.DOTALL)


def visible(text):
    """Strips tags, unescapes entities, collapses whitespace."""
    return WS_RE.sub(" ", html.unescape(TAG_RE.sub(" ", text))).strip()


def _fmt_number(n):
    if float(n).is_integer():
        return str(int(n))
    return ("%f" % n).rstrip("0").rstrip(".")


def numeric_patterns(key, value):
    """Returns regexes for a numeric leaf based on its key name, or []."""
    k = key.lower()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return []
    n = _fmt_number(value)
    if k in ("id", "number", "tier", "installment", "multiplier", "installments_count"):
        return []
    if any(t in k for t in ("usd", "price", "fee", "amount", "cost")):
        if value < 1:
            return []
        grouped = f"{int(value):,}" if float(value).is_integer() else n
        alts = {re.escape(n), re.escape(grouped)}
        return [r"(?<![\d,.])\$\s?(?:%s)(?![\d]|,\d)" % "|".join(sorted(alts))]
    if "percent" in k:
        return [r"(?<![\d.])%s\s?%%" % re.escape(n)]
    if "day" in k:
        return [r"(?<![\d.])%s[\s-]?(?:days?|d)\b" % re.escape(n)]
    if "seat" in k:
        return [r"(?<![\d.])%s\s*(?:seats?|participants?|people|persons?)\b" % re.escape(n)]
    if "hour" in k:
        return [r"(?<![\d.])%s\s?(?:hours?|hrs?|h)\b" % re.escape(n)]
    if "year" in k.lower():
        return [r"(?<![\d.])%s(?![\d])" % re.escape(n)]
    return []


def flatten(node, path=""):
    """Yields (path, key, value) for every scalar leaf in a JSON tree."""
    if isinstance(node, dict):
        label = node.get("id") or node.get("slug")
        for k, v in node.items():
            yield from flatten(v, f"{path}.{k}" if path else k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            tag = f"[{v.get('id')}]" if isinstance(v, dict) and v.get("id") not in (None, "") \
                and not isinstance(v.get("id"), int) else f"[{i}]"
            yield from flatten(v, f"{path}{tag}")
    else:
        key = path.rsplit(".", 1)[-1].split("[")[0]
        yield path, key, node


def compute_data_digest(data_dir):
    """Computes a combined SHA-256 digest of all data/*.json files for cache validation."""
    h = hashlib.sha256()
    if not os.path.exists(data_dir):
        return ""
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith(".json"):
            fpath = os.path.join(data_dir, fname)
            with open(fpath, "rb") as f:
                h.update(f.read())
    return h.hexdigest()


_MEMORY_SCANNER_CACHE = None
_MEMORY_SCANNER_CACHE_MTIME = None
_TERMS_CACHE = {}


def prune_scanner_cache(cache, active_keys=None):
    """Prunes old timestamps, missing files, and non-repo entries from the cache."""
    pruned = {}
    active_set = set(active_keys) if active_keys else set()
    latest_by_rel = {}

    # Prioritize active keys for target files in the current run
    for key in sorted(cache.keys()):
        if key in active_set:
            parts = key.split(":")
            if len(parts) >= 4:
                latest_by_rel[parts[0]] = key
                pruned[key] = cache[key]

    for key, val in cache.items():
        if key in pruned:
            continue
        parts = key.split(":")
        if len(parts) < 4:
            continue
        rel = parts[0]
        if rel in latest_by_rel:
            continue
        abs_path = os.path.join(REPO_DIR, rel)
        if not os.path.isfile(abs_path):
            continue
        try:
            st = os.stat(abs_path)
            if str(st.st_mtime_ns) == parts[1] and str(st.st_size) == parts[2]:
                latest_by_rel[rel] = key
                pruned[key] = val
        except OSError:
            pass
    return pruned


def load_scanner_cache():
    """Loads cached scan results from .scanner_cache.json with in-memory caching."""
    global _MEMORY_SCANNER_CACHE, _MEMORY_SCANNER_CACHE_MTIME
    if not os.path.exists(CACHE_FILE):
        _MEMORY_SCANNER_CACHE = {}
        _MEMORY_SCANNER_CACHE_MTIME = None
        return {}
    try:
        st = os.stat(CACHE_FILE)
        cache_key = (st.st_mtime_ns, st.st_size)
        if _MEMORY_SCANNER_CACHE is not None and _MEMORY_SCANNER_CACHE_MTIME == cache_key:
            return dict(_MEMORY_SCANNER_CACHE)
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            _MEMORY_SCANNER_CACHE = data
            _MEMORY_SCANNER_CACHE_MTIME = cache_key
            return dict(data)
    except Exception:
        return {}


def save_scanner_cache(cache, active_keys=None):
    """Persists pruned scanner cache using atomic rename and updates in-memory cache."""
    global _MEMORY_SCANNER_CACHE, _MEMORY_SCANNER_CACHE_MTIME
    pruned = prune_scanner_cache(cache, active_keys)
    try:
        tmp = f"{CACHE_FILE}.tmp.{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(pruned, f)
        os.replace(tmp, CACHE_FILE)
        st = os.stat(CACHE_FILE)
        _MEMORY_SCANNER_CACHE = pruned
        _MEMORY_SCANNER_CACHE_MTIME = (st.st_mtime_ns, st.st_size)
    except Exception:
        pass


def build_terms(data_dir):
    """Loads data/*.json and returns a de-duplicated list of searchable terms with precompiled patterns."""
    data_digest = compute_data_digest(data_dir)
    cache_key = (os.path.abspath(data_dir), data_digest)
    if cache_key in _TERMS_CACHE:
        return _TERMS_CACHE[cache_key]

    terms = {}
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".json") or fname not in DOMAIN_BY_FILE:
            continue
        with open(os.path.join(data_dir, fname), "r", encoding="utf-8") as f:
            tree = json.load(f)
        domain = DOMAIN_BY_FILE[fname]
        for path, key, value in flatten(tree):
            if isinstance(value, bool) or value is None or key in SKIP_KEYS:
                continue
            if isinstance(value, str):
                v = value.strip()
                if len(v) < MIN_STRING_LEN or v.lower() in STOP_STRINGS or v.isdigit():
                    continue
                pats = None
                tkey = ("s", v)
            else:
                pats = numeric_patterns(key, value)
                if not pats:
                    continue
                tkey = ("n", pats[0])
                v = _fmt_number(value)
            entry = terms.get(tkey)
            src = f"data/{fname}#{path}"
            if entry is None:
                terms[tkey] = {
                    "kind": tkey[0], "value": v, "patterns": pats, "domain": domain,
                    "field_key": f"{domain}:{path}", "data_source": src, "aliases": [],
                }
            else:
                entry["aliases"].append(src)

    # Precompile regexes and normalized tokens for fast lookups
    term_list = list(terms.values())
    for t in term_list:
        val = t["value"]
        if t["kind"] == "s":
            needle = WS_RE.sub(" ", html.unescape(val)).strip()
            needle_f = needle.casefold()
            url_like = "://" in val or "@" in val
            short = len(needle) <= 20 and not url_like
            t["needle"] = needle
            t["needle_f"] = needle_f
            t["url_like"] = url_like
            t["short"] = short
            t["rx_s"] = re.compile(r"(?<![\w-])" + re.escape(needle_f) + r"(?![\w-])") if short else None
            t["single_word"] = " " not in needle and len(needle) <= 12
            if " " in needle and not url_like:
                words = [re.escape(w) for w in needle.split()]
                t["rx_multiline"] = re.compile(r"(?<![\w-])" + r"(?:[\s]|<[^>]+>)+".join(words) + r"(?![\w-])", re.IGNORECASE)
            else:
                t["rx_multiline"] = None
        else:
            t["rx"] = re.compile(t["patterns"][0])

    _TERMS_CACHE[cache_key] = term_list
    return term_list


def page_index(raw_lines):
    """Returns (page_title, [(line_no, heading_text)]) for section lookup."""
    full = "\n".join(raw_lines)
    m = H1_RE.search(full) or TITLE_RE.search(full)
    page = visible(m.group(1)) if m else ""
    tm = TITLE_RE.search(full)
    if tm:
        page = visible(tm.group(1)) or page
    headings = []
    for hm in HEADING_RE.finditer(full):
        headings.append((full.count("\n", 0, hm.start()) + 1, visible(hm.group(1))))
    return page, headings


def scan_file(path, terms):
    """Scans a single file against the terms using full-document fast filters and line-level mapping."""
    rel = os.path.relpath(path, REPO_DIR)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    raw_lines = content.split("\n")
    page, headings = page_index(raw_lines)
    heading_lines = [h[0] for h in headings]
    vis_lines = [visible(l) for l in raw_lines]
    fold_lines = [l.casefold() for l in vis_lines]
    full_fold = "\n".join(fold_lines)
    full_vis = "\n".join(vis_lines)
    line_offsets = None

    seen = set()
    records = []
    for t in terms:
        val = t["value"]
        if t["kind"] == "s":
            needle = t.get("needle") or WS_RE.sub(" ", html.unescape(val)).strip()
            needle_f = t.get("needle_f") or needle.casefold()
            url_like = t.get("url_like") if "url_like" in t else ("://" in val or "@" in val)

            # Fast document-level rejection check
            if not url_like:
                if needle_f not in full_fold:
                    continue
            else:
                if needle_f not in full_fold and needle not in content:
                    continue

            short = t.get("short") if "short" in t else (len(needle) <= 20 and not url_like)
            rx_s = t.get("rx_s")
            if rx_s is None and short:
                rx_s = re.compile(r"(?<![\w-])" + re.escape(needle_f) + r"(?![\w-])")
            single_word = t.get("single_word") if "single_word" in t else (" " not in needle and len(needle) <= 12)

            for i, fl in enumerate(fold_lines):
                hit = None
                if needle_f in fl:
                    m = rx_s.search(fl) if short else None
                    if short and not m:
                        continue
                    idx = m.start() if m else fl.find(needle_f)
                    shown = vis_lines[i][idx:idx + len(needle)]
                    is_exact = shown == needle
                    if single_word and not is_exact:
                        continue
                    hit = ("exact" if is_exact else "fuzzy", shown)
                elif url_like and needle in raw_lines[i]:
                    hit = ("exact", needle)
                if hit:
                    records.append((i, t, hit[0], hit[1]))

            # Check multiline/split occurrences across tags or line breaks for terms with spaces
            rx_multi = t.get("rx_multiline")
            if rx_multi:
                # Calculate line offsets on demand
                if line_offsets is None:
                    line_offsets = []
                    curr = 0
                    for l in raw_lines:
                        line_offsets.append(curr)
                        curr += len(l) + 1
                for m in rx_multi.finditer(content):
                    matched = m.group(0)
                    if "\n" in matched or "<" in matched:
                        start_char = m.start()
                        line_idx = bisect.bisect_right(line_offsets, start_char) - 1
                        vis_match = visible(matched)
                        is_exact = vis_match == needle
                        records.append((line_idx, t, "exact" if is_exact else "fuzzy", vis_match))
        else:
            rx = t.get("rx")
            if rx is None:
                rx = re.compile(t["patterns"][0])
            if not rx.search(full_vis):
                continue
            for i, vl in enumerate(vis_lines):
                m = rx.search(vl)
                if m:
                    records.append((i, t, "exact", m.group(0)))

    out = []
    for i, t, mtype, shown in records:
        key = (i, t["field_key"])
        if key in seen:
            continue
        seen.add(key)
        j = bisect.bisect_right(heading_lines, i + 1) - 1
        section = headings[j][1] if j >= 0 else "Page Header"
        ctx = vis_lines[i]
        if len(ctx) > CONTEXT_MAX:
            ctx = ctx[:CONTEXT_MAX - 3] + "..."
        rec = {
            "file": rel,
            "page": page,
            "section": section,
            "line_number": i + 1,
            "field_key": t["field_key"],
            "old_value": shown,
            "new_value": t["value"],
            "context": ctx,
            "match_type": mtype,
            "data_source": t["data_source"],
            "_domain": t["domain"],
        }
        if t["aliases"]:
            rec["also_in_data"] = list(t["aliases"])
        out.append(rec)
    return out


def scan_repository(target_files, data_dir=None):
    """Scans target_files for every tracked term with parallel execution and caching.
    Returns {domain: [records]}.
    """
    actual_data_dir = data_dir or DATA_DIR
    terms = build_terms(actual_data_dir)
    terms_by_field = {t["field_key"]: str(t["value"]) for t in terms}
    data_digest = compute_data_digest(actual_data_dir)

    cache = load_scanner_cache()
    cached_records = {}
    uncached_files = []
    active_keys = set()

    # Pre-index existing cache by relative path
    cache_by_rel = {}
    for ck, cval in cache.items():
        parts = ck.split(":")
        if len(parts) >= 3:
            cache_by_rel[parts[0]] = (parts[1], parts[2], ck, cval)

    for path in sorted(target_files):
        rel = os.path.relpath(path, REPO_DIR)
        try:
            st = os.stat(path)
            key = f"{rel}:{st.st_mtime_ns}:{st.st_size}:{data_digest}"
            active_keys.add(key)
            if key in cache:
                cached_records[path] = cache[key]
            elif rel in cache_by_rel:
                c_mtime, c_size, old_key, old_val = cache_by_rel[rel]
                if c_mtime == str(st.st_mtime_ns) and c_size == str(st.st_size):
                    # File on disk is unchanged. Check if any previously matched term changed value.
                    terms_changed = False
                    for r in old_val:
                        fk = r.get("field_key")
                        if fk in terms_by_field and terms_by_field[fk] != str(r.get("new_value")):
                            terms_changed = True
                            break
                    if not terms_changed:
                        cached_records[path] = old_val
                        cache[key] = old_val
                        continue
                uncached_files.append((path, key))
            else:
                uncached_files.append((path, key))
        except OSError:
            uncached_files.append((path, None))

    if uncached_files:
        if len(uncached_files) > 12:
            try:
                with ProcessPoolExecutor() as executor:
                    futures = {executor.submit(scan_file, p, terms): (p, k) for p, k in uncached_files}
                    for fut in futures:
                        p, k = futures[fut]
                        recs = fut.result()
                        cached_records[p] = recs
                        if k:
                            cache[k] = recs
            except Exception:
                for p, k in uncached_files:
                    recs = scan_file(p, terms)
                    cached_records[p] = recs
                    if k:
                        cache[k] = recs
        else:
            for p, k in uncached_files:
                recs = scan_file(p, terms)
                cached_records[p] = recs
                if k:
                    cache[k] = recs

        save_scanner_cache(cache, active_keys=active_keys)

    by_domain = {d: [] for d in DOMAIN_BY_FILE.values()}
    for path in sorted(target_files):
        for rec in cached_records.get(path, []):
            dom = rec.get("_domain")
            clean_rec = {k: v for k, v in rec.items() if k != "_domain"}
            if dom and dom in by_domain:
                by_domain[dom].append(clean_rec)

    for recs in by_domain.values():
        recs.sort(key=lambda r: (r["file"], r["line_number"], r["field_key"]))
    return by_domain


def main():
    """Scan-only mode: refreshes *_references.json without touching pages or data/."""
    sys.path.insert(0, SCRIPT_DIR)
    from synchronizer import ExpertSynchronizer
    sync = ExpertSynchronizer()
    sync.refresh_references_only()


if __name__ == "__main__":
    main()
