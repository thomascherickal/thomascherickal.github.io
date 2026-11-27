#!/usr/bin/env python3
"""
scan_literals.py - Inventory hardcoded protected facts outside <!--ssot:...--> markers.

Outputs docs/ssot-inventory.csv (file, line, literal, suggested_data_key).
Supports --file <path> and --staged for pre-commit enforcement.
"""

import argparse
import csv
import os
import re
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INVENTORY_CSV = os.path.join(REPO_ROOT, "docs", "ssot-inventory.csv")

MARKER_BLOCK_RE = re.compile(r"<!--ssot:[^>]+-->.*?<!--/ssot-->", re.DOTALL)
HTML_TAG_STRIP_RE = re.compile(r"<[^>]+>")

PROTECTED_PATTERNS = [
    (re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?(?:\s*USD)?"), "offerings.<id>.tier1_base_price_usd"),
    (re.compile(r"\b\d+\s*%"), "offerings.<id>.payment_rules.milestone_schedule"),
    (re.compile(r"\b(?:10|20)\s+(?:participants?|seats?|people|persons?)\b", re.I), "offerings.<id>.cohort_capacity"),
    (re.compile(r"\b(?:10|ten)\s+(?:equal\s+)?installments?\b", re.I), "fmt.installment_phrase(<id>)"),
    (re.compile(r"\b(?:2\s+months?|10\s+days?|30\s+days?|60\s+days?|4\s+hours?|4\s+weeks?)\b", re.I), "offerings.<id>.schedule"),
    (re.compile(r"\bwithin\s+(?:one\s+week|24\s+hours)\b", re.I), "fmt.reply_sla_phrase()"),
]

SCAN_DIRS = [".", "pricing", "business-services", "personal-services"]


def collect_target_files(single_file=None, staged=False):
    if single_file:
        p = os.path.join(REPO_ROOT, single_file) if not os.path.isabs(single_file) else single_file
        return [p] if os.path.exists(p) else []
    if staged:
        res = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        files = []
        for rel in res.stdout.splitlines():
            if rel.endswith((".html", ".md")) or rel == "llms.txt":
                p = os.path.join(REPO_ROOT, rel)
                if os.path.exists(p):
                    files.append(p)
        return files
    files = []
    for d in SCAN_DIRS:
        full = os.path.join(REPO_ROOT, d)
        if not os.path.isdir(full):
            continue
        for fn in sorted(os.listdir(full)):
            if fn.endswith(".html") and fn not in ("header.html", "footer.html"):
                files.append(os.path.join(full, fn))
    llms = os.path.join(REPO_ROOT, "llms.txt")
    if os.path.exists(llms):
        files.append(llms)
    return files


def scan_file(path):
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    # Mask ssot marker contents so wrapped facts are not flagged
    masked = MARKER_BLOCK_RE.sub(lambda m: "\n" * m.group(0).count("\n"), raw)
    findings = []
    rel = os.path.relpath(path, REPO_ROOT)
    for idx, line in enumerate(masked.splitlines(), 1):
        text = HTML_TAG_STRIP_RE.sub(" ", line)
        for pat, key in PROTECTED_PATTERNS:
            for m in pat.finditer(text):
                findings.append((rel, idx, m.group(0).strip(), key))
    return findings


def main():
    parser = argparse.ArgumentParser(description="Scan for hardcoded protected facts outside SSOT markers")
    parser.add_argument("--file", help="Scan a single file")
    parser.add_argument("--staged", action="store_true", help="Scan only git-staged files")
    args = parser.parse_args()

    files = collect_target_files(single_file=args.file, staged=args.staged)
    all_findings = []
    for p in files:
        all_findings.extend(scan_file(p))

    if not args.file and not args.staged:
        os.makedirs(os.path.dirname(INVENTORY_CSV), exist_ok=True)
        with open(INVENTORY_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "line", "literal", "suggested_data_key"])
            writer.writerows(all_findings)
        print(f"Wrote {len(all_findings)} literal findings to {os.path.relpath(INVENTORY_CSV, REPO_ROOT)}")
    else:
        print(f"Found {len(all_findings)} unprotected literal(s).")
        for rel, ln, lit, key in all_findings[:30]:
            print(f"  {rel}:{ln}: {lit!r} -> {key}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
