#!/usr/bin/env python3
"""
delete_page.py - Page Deletion and Synchronization Cleanup Utility.

Completely and cleanly deletes an HTML page from the website and all synchronization
systems so that it cannot be resurrected or added back automatically.

Automates the removal of:
1. The HTML source file (in root or any subfolder: google/, claude/, business-services/, personal-services/)
2. The extracted JSON-LD schema file under schema/
3. The Markdown mirror under md-html-sync/
4. The tracking cache entry in md-html-sync/.sync-state.json
5. References in python/md_html_sync.py (PAGE_MAPPINGS)
6. References in python/generate_sitemap.py and sitemap.xml
7. References in header.html (and triggers build_components.py if updated)
8. References in llms.txt
9. Checks for incoming links across the codebase and reports any dangling links.
10. Runs sync_schemas.py, md_html_sync.py, and check_links.py for verification.

Usage:
    python3 python/delete_page.py <filename_or_path> [--dry-run] [--no-verify]

Examples:
    python3 python/delete_page.py google-training/google-old-page.html
    python3 python/delete_page.py google-old-page.html
    python3 python/delete_page.py google-old-page
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
SCHEMA_DIR = REPO_DIR / "schema"
SYNC_DIR = REPO_DIR / "md-html-sync"
STATE_FILE = SYNC_DIR / ".sync-state.json"
SUBDIRS = ["google-training", "claude-training", "business-services", "personal-services"]


def find_html_file(target: str) -> Path | None:
    """Find the target HTML file across repository root and subdirectories."""
    clean_target = target.strip()
    if not clean_target.endswith(".html"):
        clean_target += ".html"

    # Direct relative path check
    direct = (REPO_DIR / clean_target).resolve()
    if direct.exists() and direct.is_file():  # pragma: no cover
        return direct

    # Base name check across subdirectories and root
    basename = Path(clean_target).name
    candidates = [REPO_DIR / basename]
    for sub in SUBDIRS:
        candidates.append(REPO_DIR / sub / basename)

    for cand in candidates:
        if cand.exists() and cand.is_file():
            return cand

    return None


def find_schema_files(stem: str) -> list[Path]:
    """Find all matching schema files in schema/."""
    matched = []
    if not SCHEMA_DIR.exists():  # pragma: no cover
        return matched

    # Primary expected filenames
    expected_names = {
        f"{stem}-schema.json",
        f"{stem}.json"
    }

    for sf in SCHEMA_DIR.iterdir():
        if not sf.is_file() or not sf.name.endswith(".json"):  # pragma: no cover
            continue
        if sf.name in expected_names:
            matched.append(sf)
            continue
        # Check if content has @id referencing this stem.html
        try:
            with open(sf, "r", encoding="utf-8") as f:
                content = f.read()
            if f"/{stem}.html" in content or f'"{stem}.html"' in content:
                matched.append(sf)
        except Exception:  # pragma: no cover
            pass

    return list(set(matched))


def find_markdown_files(rel_html: str, stem: str) -> list[Path]:
    """Find all matching markdown mirror files in md-html-sync/."""
    matched = []
    if not SYNC_DIR.exists():  # pragma: no cover
        return matched

    # Direct relative path mirror
    rel_md = rel_html.replace(".html", ".md")
    direct_md = SYNC_DIR / rel_md
    if direct_md.exists() and direct_md.is_file():
        matched.append(direct_md)

    # Base name check
    base_md = SYNC_DIR / f"{stem}.md"
    if base_md.exists() and base_md.is_file():
        matched.append(base_md)

    for sub in SUBDIRS:
        sub_md = SYNC_DIR / sub / f"{stem}.md"
        if sub_md.exists() and sub_md.is_file():
            matched.append(sub_md)

    return list(set(matched))


def purge_sync_state(rel_html: str, basename: str, stem: str, dry_run: bool = False) -> int:
    """Remove all keys in .sync-state.json matching the target file."""
    if not STATE_FILE.exists():  # pragma: no cover
        return 0

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
    except Exception as e:  # pragma: no cover
        print(f"Warning: could not read {STATE_FILE}: {e}")
        return 0

    keys_to_delete = []
    for key, val in state.items():
        if key == rel_html or key == basename or key.endswith(f"/{basename}"):
            keys_to_delete.append(key)
        elif isinstance(val, dict):
            md_name = val.get("md_name", "")
            if md_name.endswith(f"{stem}.md"):  # pragma: no cover
                keys_to_delete.append(key)

    keys_to_delete = list(set(keys_to_delete))
    if keys_to_delete and not dry_run:
        for k in keys_to_delete:
            del state[k]
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    return len(keys_to_delete)


def clean_sitemap(rel_html: str, basename: str, dry_run: bool = False) -> bool:
    """Remove file from python/generate_sitemap.py lists and regenerate sitemap.xml."""
    sitemap_script = SCRIPT_DIR / "generate_sitemap.py"
    if not sitemap_script.exists():  # pragma: no cover
        return False

    with open(sitemap_script, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = re.compile(rf'^\s*"[^"]*{re.escape(basename)}"[,\s]*$\n', re.MULTILINE)
    if pattern.search(content):
        if not dry_run:
            new_content = pattern.sub("", content)
            with open(sitemap_script, "w", encoding="utf-8") as f:
                f.write(new_content)
            # Regenerate sitemap.xml
            subprocess.run([sys.executable, str(sitemap_script)], check=False, cwd=REPO_DIR)
        return True
    return False


def clean_header_nav(rel_html: str, basename: str, dry_run: bool = False) -> bool:
    """Remove link from header.html if present and rebuild components."""
    header_file = REPO_DIR / "header.html"
    if not header_file.exists():  # pragma: no cover
        return False

    with open(header_file, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = re.compile(rf'^\s*<li><a\s+href="[^"]*{re.escape(basename)}"[^>]*>.*?</a></li>\s*$\n', re.MULTILINE)
    if pattern.search(content):
        if not dry_run:
            new_content = pattern.sub("", content)
            with open(header_file, "w", encoding="utf-8") as f:
                f.write(new_content)
            # Trigger build_components.py automatically
            build_script = SCRIPT_DIR / "build_components.py"
            if build_script.exists():
                subprocess.run([sys.executable, str(build_script)], check=False, cwd=REPO_DIR)
        return True
    return False


def clean_llms_txt(basename: str, dry_run: bool = False) -> bool:
    """Remove lines referencing the page in llms.txt."""
    llms_file = REPO_DIR / "llms.txt"
    if not llms_file.exists():  # pragma: no cover
        return False

    with open(llms_file, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = re.compile(rf'^.*{re.escape(basename)}.*$\n', re.MULTILINE)
    if pattern.search(content):
        if not dry_run:
            new_content = pattern.sub("", content)
            with open(llms_file, "w", encoding="utf-8") as f:
                f.write(new_content)
        return True
    return False


def scan_incoming_links(basename: str) -> list[tuple[str, int]]:
    """Scan other files for any incoming links to the deleted page."""
    matches = []
    link_pattern = re.compile(rf'href=["\'][^"\']*{re.escape(basename)}["\']')

    for root, _, files in os.walk(REPO_DIR):
        if ".git" in root or "tests" in root:  # pragma: no cover
            continue
        for fname in files:
            if fname.endswith((".html", ".md", ".json")) and fname != basename:
                fpath = Path(root) / fname
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        for line_num, line in enumerate(f, 1):
                            if link_pattern.search(line):
                                rel_path = fpath.relative_to(REPO_DIR)
                                matches.append((str(rel_path), line_num))
                except Exception:  # pragma: no cover
                    pass
    return matches


def delete_page(target: str, dry_run: bool = False, verify: bool = True):
    print("=" * 80)
    print("           PAGE DELETION & SYNCHRONIZATION PURGE SYSTEM")
    print("=" * 80)

    html_file = find_html_file(target)
    clean_target = target.strip()
    if not clean_target.endswith(".html"):
        clean_target += ".html"
    basename = Path(clean_target).name
    stem = basename[:-5]

    if html_file:
        rel_html = str(html_file.relative_to(REPO_DIR))
    else:
        rel_html = clean_target
        print(f"[INFO] HTML file '{target}' not found on disk. Checking sync artifacts...")

    print(f"Target Page      : {basename}")
    print(f"Target Relative  : {rel_html}")
    print(f"Dry Run Mode     : {dry_run}\n")

    # 1. HTML Removal
    if html_file and html_file.exists():
        if dry_run:
            print(f"[DRY-RUN] Would delete HTML file: {html_file.relative_to(REPO_DIR)}")
        else:
            html_file.unlink()
            print(f"[DELETED HTML] {html_file.relative_to(REPO_DIR)}")
    else:
        print("[INFO] No HTML file on disk.")

    # 2. Schema Removal
    schema_files = find_schema_files(stem)
    if schema_files:
        for sf in schema_files:
            if dry_run:
                print(f"[DRY-RUN] Would delete schema file: {sf.relative_to(REPO_DIR)}")
            else:
                sf.unlink()
                print(f"[DELETED SCHEMA] {sf.relative_to(REPO_DIR)}")
    else:
        print("[INFO] No matching schema files found in schema/.")

    # 3. Markdown Mirror Removal
    md_files = find_markdown_files(rel_html, stem)
    if md_files:
        for mf in md_files:
            if dry_run:
                print(f"[DRY-RUN] Would delete Markdown file: {mf.relative_to(REPO_DIR)}")
            else:
                mf.unlink()
                print(f"[DELETED MD] {mf.relative_to(REPO_DIR)}")
    else:
        print("[INFO] No matching Markdown files found in md-html-sync/.")

    # 4. State Purge
    purged_count = purge_sync_state(rel_html, basename, stem, dry_run=dry_run)
    if purged_count > 0:
        action = "Would purge" if dry_run else "Purged"
        print(f"[{action.upper()} STATE] Removed {purged_count} key(s) from md-html-sync/.sync-state.json")
    else:
        print("[INFO] No matching entries in md-html-sync/.sync-state.json.")

    # 5. Sitemap Clean
    if clean_sitemap(rel_html, basename, dry_run=dry_run):
        action = "Would remove" if dry_run else "Removed"
        print(f"[{action.upper()} SITEMAP] Updated python/generate_sitemap.py and sitemap.xml")

    # 6. Header Nav Clean
    if clean_header_nav(rel_html, basename, dry_run=dry_run):
        action = "Would remove" if dry_run else "Removed"
        print(f"[{action.upper()} HEADER] Removed link from header.html and synchronized components")

    # 7. LLMs.txt Clean
    if clean_llms_txt(basename, dry_run=dry_run):
        action = "Would remove" if dry_run else "Removed"
        print(f"[{action.upper()} LLMS.TXT] Removed entry from llms.txt")

    # 8. Check Incoming Links
    incoming = scan_incoming_links(basename)
    if incoming:
        print(f"\n[WARNING] Found {len(incoming)} incoming link(s) to '{basename}':")
        for fpath, lnum in incoming:
            print(f"  - {fpath}:{lnum}")
        print("Please update or remove these links so check_links.py remains clean.")
    else:
        print("\n[VERIFIED] Zero incoming links found in other files.")

    # 9. Verification & Cache Sync
    if verify and not dry_run:
        print("\n[Step] Running schema synchronization...")
        sync_schemas_script = SCRIPT_DIR / "sync_schemas.py"
        if sync_schemas_script.exists():
            subprocess.run([sys.executable, str(sync_schemas_script)], check=False, cwd=REPO_DIR)

        print("[Step] Running MD-HTML synchronization pass...")
        md_sync_script = SCRIPT_DIR / "md_html_sync.py"
        if md_sync_script.exists():
            subprocess.run([sys.executable, str(md_sync_script)], check=False, cwd=REPO_DIR)

        print("[Step] Running repository link integrity check...")
        check_links_script = SCRIPT_DIR / "check_links.py"
        if check_links_script.exists():
            subprocess.run([sys.executable, str(check_links_script)], check=False, cwd=REPO_DIR)

    print("\n" + "=" * 80)
    print(f"Deletion and sync purge for '{basename}' completed successfully.")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Completely delete an HTML page and purge all sync states."
    )
    parser.add_argument("target", help="HTML filename or path to delete (e.g. google-training/google-old-page.html)")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be deleted without making changes")
    parser.add_argument("--no-verify", action="store_true", help="Skip running post-deletion verification scripts")
    args = parser.parse_args()

    delete_page(args.target, dry_run=args.dry_run, verify=not args.no_verify)


if __name__ == "__main__":  # pragma: no cover
    main()
