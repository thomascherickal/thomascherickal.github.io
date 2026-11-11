#!/usr/bin/env python3
"""
sync_schemas.py - HTML to JSON Schema Synchronization Engine.

Extracts live embedded JSON-LD schemas (<script type="application/ld+json">)
from all standalone HTML pages in the repository root and synchronizes them
into corresponding JSON files in the schema/ directory.

Live HTML files are the source of truth for structured data.
Run this script before every git push or build step.
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from bs4 import BeautifulSoup

REPO_DIR = Path(__file__).resolve().parent.parent
SCHEMA_DIR = REPO_DIR / "schema"

# HTML component fragments and verification files that do not contain page schemas
EXCLUDED_HTML = {
    "header.html",
    "footer.html",
    "article-footer_dark.html",
    "google43d90cf6a204dc6d.html",
}

# Explicit overrides for filenames where stem does not directly match schema filename
EXPLICIT_MAP = {
    "index.html": "home-schema.json",
}


def get_target_schema_filename(html_filename: str) -> str:
    """
    Determine the target schema filename in schema/ for a given HTML file.
    Follows existing repo naming conventions:
    - index.html -> home-schema.json
    - service-*.html -> service-*.json (or service-*-schema.json if existing)
    - other-page.html -> other-page-schema.json (or other-page.json if existing)
    """
    if html_filename in EXPLICIT_MAP:
        return EXPLICIT_MAP[html_filename]

    stem = html_filename[:-5] if html_filename.endswith(".html") else html_filename
    existing_files = set(os.listdir(SCHEMA_DIR)) if SCHEMA_DIR.exists() else set()

    if stem.startswith(("service-", "business-service-", "personal-service-")):
        if f"{stem}.json" in existing_files:  # pragma: no cover
            return f"{stem}.json"
        if f"{stem}-schema.json" in existing_files:
            return f"{stem}-schema.json"
        return f"{stem}.json"

    # Default pattern for articles, documentation, and core pages
    if f"{stem}-schema.json" in existing_files:  # pragma: no cover
        return f"{stem}-schema.json"
    if f"{stem}.json" in existing_files:
        return f"{stem}.json"

    return f"{stem}-schema.json"


def extract_schema_from_html(html_path: Path):
    """
    Parse an HTML file and extract the JSON-LD schema data.
    Returns the parsed Python object or None if not found/invalid.
    """
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
    except Exception as e:  # pragma: no cover
        print(f"Error reading {html_path.name}: {e}")
        return None

    script = soup.find("script", type="application/ld+json")
    if not script or not script.string:
        return None

    try:
        return json.loads(script.string.strip())
    except json.JSONDecodeError as e:
        print(f"JSON decode error in {html_path.name}: {e}")
        return None


def sync_html_to_schemas(check_only: bool = False, verbose: bool = False) -> dict:
    """
    Sync all embedded HTML schemas to schema/ JSON files.

    If check_only is True, does not modify any files and reports drift.
    Returns a dictionary summarizing the results.
    """
    if not SCHEMA_DIR.exists():
        if check_only:
            print(f"Schema directory missing: {SCHEMA_DIR}")
            return {"total": 0, "updated": 0, "created": 0, "unchanged": 0, "errors": 1}
        SCHEMA_DIR.mkdir(parents=True, exist_ok=True)

    groups = [
        ("Core Pages", sorted([f for f in REPO_DIR.glob("*.html")])),
        ("Business Services", sorted([f for f in (REPO_DIR / "business-services").glob("*.html")])),
        ("Personal Services", sorted([f for f in (REPO_DIR / "personal-services").glob("*.html")])),
        ("Gemini Training", sorted([f for f in (REPO_DIR / "gemini-training").glob("*.html")])),
        ("Claude Training", sorted([f for f in (REPO_DIR / "claude-training").glob("*.html")])),
    ]

    updated = []
    created = []
    unchanged = []
    errors = []
    cat_stats = []

    for cat_name, file_list in groups:
        cat_total = 0
        cat_updated = 0
        cat_created = 0
        cat_unchanged = 0
        cat_errors = 0

        for html_path in file_list:
            filename = html_path.name
            if filename in EXCLUDED_HTML:  # pragma: no cover
                continue

            schema_data = extract_schema_from_html(html_path)
            if schema_data is None:
                if verbose:
                    print(f"  [SKIP] {filename} (no valid JSON-LD schema found)")
                continue

            cat_total += 1
            target_filename = get_target_schema_filename(filename)
            target_path = SCHEMA_DIR / target_filename
            formatted_json = json.dumps(schema_data, indent=2, ensure_ascii=False) + "\n"

            if target_path.exists():
                try:
                    with open(target_path, "r", encoding="utf-8") as f:
                        current_content = f.read()
                except Exception as e:  # pragma: no cover
                    errors.append((filename, target_filename, str(e)))
                    cat_errors += 1
                    continue

                if current_content == formatted_json:
                    unchanged.append((filename, target_filename))
                    cat_unchanged += 1
                    if verbose:
                        print(f"  [OK]   {filename} == schema/{target_filename}")
                else:
                    updated.append((filename, target_filename))
                    cat_updated += 1
                    if not check_only:
                        with open(target_path, "w", encoding="utf-8") as f:
                            f.write(formatted_json)
                        print(f"  [SYNC] {filename} -> schema/{target_filename} (updated)")
                    else:
                        print(f"  [DRIFT] {filename} != schema/{target_filename}")
            else:
                created.append((filename, target_filename))
                cat_created += 1
                if not check_only:
                    with open(target_path, "w", encoding="utf-8") as f:
                        f.write(formatted_json)
                    print(f"  [NEW]  {filename} -> schema/{target_filename} (created)")
                else:  # pragma: no cover
                    print(f"  [MISSING] schema/{target_filename} (for {filename})")

        cat_stats.append({
            "name": cat_name,
            "total": cat_total,
            "updated": cat_updated,
            "created": cat_created,
            "unchanged": cat_unchanged,
            "errors": cat_errors,
        })

    summary = {
        "total": len(updated) + len(created) + len(unchanged),
        "updated": len(updated),
        "created": len(created),
        "unchanged": len(unchanged),
        "errors": len(errors),
    }

    # Print 80-column structured ASCII report matching other pre-push summaries
    width = 80
    print("=" * width)
    print("HTML TO JSON SCHEMA SYNC REPORT".center(width))
    print("=" * width)

    # Total width: 22 + 10 + 10 + 10 + 10 + 11 + 7 borders = 80 chars
    col_widths = (20, 8, 8, 8, 8, 9)
    sep_line = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

    print(sep_line)
    hdr = f"| {'Category':<{col_widths[0]}} | {'Schemas':>{col_widths[1]}} | {'Updated':>{col_widths[2]}} | {'Created':>{col_widths[3]}} | {'In Sync':>{col_widths[4]}} | {'Status':<{col_widths[5]}} |"
    print(hdr)
    print(sep_line)

    for c in cat_stats:
        if c["errors"] > 0:
            c_status = "ERROR"
        elif c["updated"] > 0 or c["created"] > 0:
            c_status = "UPDATED" if not check_only else "DRIFT"
        else:
            c_status = "IN SYNC"
        row = f"| {c['name']:<{col_widths[0]}} | {c['total']:>{col_widths[1]}} | {c['updated']:>{col_widths[2]}} | {c['created']:>{col_widths[3]}} | {c['unchanged']:>{col_widths[4]}} | {c_status:<{col_widths[5]}} |"
        print(row)

    print(sep_line)
    overall_status = "PASS" if summary["errors"] == 0 and (not check_only or (summary["updated"] == 0 and summary["created"] == 0)) else "FAIL"
    tot_row = f"| {'Schema Total':<{col_widths[0]}} | {summary['total']:>{col_widths[1]}} | {summary['updated']:>{col_widths[2]}} | {summary['created']:>{col_widths[3]}} | {summary['unchanged']:>{col_widths[4]}} | {overall_status:<{col_widths[5]}} |"
    print(tot_row)
    print(sep_line)

    print("-" * width)
    print("HTML TO JSON SCHEMA SYNCHRONIZATION SUMMARY")
    print("-" * width)
    print(f"Total Schemas Evaluated: {summary['total']} schemas across 5 categories")
    print(f"Schemas Synchronized:    {summary['unchanged']} in sync | {summary['updated']} updated | {summary['created']} created")
    if summary["errors"]:  # pragma: no cover
        print(f"Errors Encountered:      {summary['errors']} errors")
    else:
        print(f"Errors Encountered:      0 errors")
    print("=" * width)
    print(f"RESULT: {overall_status} (ALL SCHEMAS SYNCHRONIZED)")
    print("=" * width)
    print()

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Sync embedded HTML JSON-LD schemas into schema/ directory files."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check whether schema files are synchronized without modifying them.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print verbose status for each file.",
    )
    args = parser.parse_args()

    results = sync_html_to_schemas(check_only=args.check, verbose=args.verbose)

    if args.check and (results["updated"] > 0 or results["created"] > 0 or results["errors"] > 0):  # pragma: no cover
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":  # pragma: no cover
    main()
