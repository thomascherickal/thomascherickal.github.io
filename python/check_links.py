#!/usr/bin/env python3
"""
Internal Link and Anchor Checker
Scans all HTML files for internal links and fragment identifiers (anchors),
verifying that target files exist and target anchor IDs exist in those files.
"""

import os
import re
import sys
from urllib.parse import urlparse, unquote

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Regex to find hrefs in HTML
HREF_PATTERN = re.compile(r'href=[\'"]([^\'"]+)[\'"]', re.IGNORECASE)
ID_PATTERN = re.compile(r'\bid=[\'"]([^\'"]+)[\'"]', re.IGNORECASE)
NAME_PATTERN = re.compile(r'\bname=[\'"]([^\'"]+)[\'"]', re.IGNORECASE)

def get_html_files():
    html_files = []
    for root, dirs, files in os.walk(REPO_DIR):
        dirs[:] = [d for d in dirs if d not in ['.git', 'node_modules', '.agents']]
        for f in files:
            if f.endswith('.html'):
                html_files.append(os.path.join(root, f))
    return sorted(html_files)

def collect_file_targets(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        return set()
    ids = set(ID_PATTERN.findall(content))
    ids.update(NAME_PATTERN.findall(content))
    return ids

def check_links():
    html_files = get_html_files()
    file_ids_cache = {}
    
    for f in html_files:
        file_ids_cache[f] = collect_file_targets(f)
        
    errors = []
    checked_count = 0

    for source_file in html_files:
        try:
            with open(source_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception:
            continue
            
        hrefs = HREF_PATTERN.findall(content)
        for href in hrefs:
            href_clean = href.strip()
            # Ignore external links, mailto, tel, javascript, and JS template interpolation
            if href_clean.startswith(('http://', 'https://', 'mailto:', 'tel:', 'javascript:', 'data:', 'chrome-extension:', '${')):
                continue
            if not href_clean or href_clean.startswith('?'):
                continue
                
            checked_count += 1
            parsed = urlparse(href_clean)
            path = unquote(parsed.path)
            fragment = unquote(parsed.fragment)
            
            # Resolve target file path
            if not path:
                target_file = source_file
            else:
                if path.startswith('/'):
                    target_file = os.path.normpath(os.path.join(REPO_DIR, path.lstrip('/')))
                else:
                    target_file = os.path.normpath(os.path.join(os.path.dirname(source_file), path))
                    
            # Check target file existence
            if not os.path.exists(target_file):
                # Try index.html if it's a directory
                if os.path.isdir(target_file) and os.path.exists(os.path.join(target_file, 'index.html')):
                    target_file = os.path.join(target_file, 'index.html')
                else:
                    errors.append(f"Broken link in {os.path.relpath(source_file, REPO_DIR)}: '{href}' -> Target file {os.path.relpath(target_file, REPO_DIR)} not found")
                    continue
                    
            # Check fragment/anchor if specified
            if fragment:
                # Cache target file ids if not cached
                if target_file not in file_ids_cache:
                    file_ids_cache[target_file] = collect_file_targets(target_file)
                target_ids = file_ids_cache[target_file]
                if fragment not in target_ids:
                    errors.append(f"Broken anchor in {os.path.relpath(source_file, REPO_DIR)}: '{href}' -> ID '#{fragment}' not found in {os.path.relpath(target_file, REPO_DIR)}")

    print(f"Checked {checked_count} internal links across {len(html_files)} HTML files.")
    if errors:
        print(f"FOUND {len(errors)} BROKEN LINKS/ANCHORS:")
        for err in errors:
            print("  -", err)
        return False
    else:
        print("ZERO broken internal links or fragments found! All links verified.")
        return True

if __name__ == '__main__':
    success = check_links()
    if not success:
        sys.exit(1)
