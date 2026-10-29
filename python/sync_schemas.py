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
        if f"{stem}.json" in existing_files:
            return f"{stem}.json"
        if f"{stem}-schema.json" in existing_files:
            return f"{stem}-schema.json"
        return f"{stem}.json"

    # Default pattern for articles, documentation, and core pages
    if f"{stem}-schema.json" in existing_files:
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
    except Exception as e:
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

    html_files = sorted(
        list(REPO_DIR.glob("*.html")) +
        list((REPO_DIR / "business-services").glob("*.html")) +
        list((REPO_DIR / "personal-services").glob("*.html")) +
        list((REPO_DIR / "google").glob("*.html")) +
        list((REPO_DIR / "claude").glob("*.html"))
    )
    updated = []
    created = []
    unchanged = []
    errors = []

    for html_path in html_files:
        filename = html_path.name
        if filename in EXCLUDED_HTML:
            continue

        schema_data = extract_schema_from_html(html_path)
        if schema_data is None:
            if verbose:
                print(f"  [SKIP] {filename} (no valid JSON-LD schema found)")
            continue

        target_filename = get_target_schema_filename(filename)
        target_path = SCHEMA_DIR / target_filename
        formatted_json = json.dumps(schema_data, indent=2, ensure_ascii=False) + "\n"

        if target_path.exists():
            try:
                with open(target_path, "r", encoding="utf-8") as f:
                    current_content = f.read()
            except Exception as e:
                errors.append((filename, target_filename, str(e)))
                continue

            if current_content == formatted_json:
                unchanged.append((filename, target_filename))
                if verbose:
                    print(f"  [OK]   {filename} == schema/{target_filename}")
            else:
                updated.append((filename, target_filename))
                if not check_only:
                    with open(target_path, "w", encoding="utf-8") as f:
                        f.write(formatted_json)
                    print(f"  [SYNC] {filename} -> schema/{target_filename} (updated)")
                else:
                    print(f"  [DRIFT] {filename} != schema/{target_filename}")
        else:
            created.append((filename, target_filename))
            if not check_only:
                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(formatted_json)
                print(f"  [NEW]  {filename} -> schema/{target_filename} (created)")
            else:
                print(f"  [MISSING] schema/{target_filename} (for {filename})")

    summary = {
        "total": len(updated) + len(created) + len(unchanged),
        "updated": len(updated),
        "created": len(created),
        "unchanged": len(unchanged),
        "errors": len(errors),
    }

    print("\n--- HTML to JSON Schema Sync Summary ---")
    print(f"Total schemas checked : {summary['total']}")
    print(f"Updated in schema/    : {summary['updated']}")
    print(f"Created in schema/    : {summary['created']}")
    print(f"Already in sync       : {summary['unchanged']}")
    if summary["errors"]:
        print(f"Errors encountered    : {summary['errors']}")
    print("----------------------------------------\n")

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

    if args.check and (results["updated"] > 0 or results["created"] > 0 or results["errors"] > 0):
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
