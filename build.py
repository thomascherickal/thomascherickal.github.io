#!/usr/bin/env python3
"""
build.py — Build-time component includer for thomascherickal.github.io.
Extracts <nav> from header.html and <footer> from footer.html,
and replaces iframes or existing build-time include blocks across all HTML pages.
"""

import glob
import os
import re
import sys

REPO_DIR = os.path.dirname(os.path.abspath(__file__))

HEADER_SRC = os.path.join(REPO_DIR, "header.html")
FOOTER_SRC = os.path.join(REPO_DIR, "footer.html")

def extract_nav(header_content: str) -> str:
    match = re.search(r'(<nav\b[\s\S]*?</nav>)', header_content, re.IGNORECASE)
    if not match:
        raise ValueError("Could not find <nav> tag in header.html")
    return match.group(1).strip()

def extract_footer(footer_content: str) -> str:
    match = re.search(r'(<footer\b[\s\S]*?</footer>)', footer_content, re.IGNORECASE)
    if not match:
        raise ValueError("Could not find <footer> tag in footer.html")
    return match.group(1).strip()

def build():
    with open(HEADER_SRC, "r", encoding="utf-8") as f:
        nav_html = extract_nav(f.read())

    with open(FOOTER_SRC, "r", encoding="utf-8") as f:
        footer_html = extract_footer(f.read())

    footer_styles = """  <style id="footer-styles">
    .social-icon {
      font-size: 1rem;
      flex-shrink: 0;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 18px;
      height: 18px;
    }
    .social-icon svg,
    .social-svg {
      width: 18px;
      height: 18px;
      display: block;
      flex-shrink: 0;
      transition: transform 0.28s ease, filter 0.28s ease;
    }
    .social-link:hover .social-icon svg,
    .social-link:hover .social-svg {
      transform: scale(1.22);
      filter: drop-shadow(0 0 6px rgba(255, 255, 255, 0.45));
    }
    .social-link {
      border: 2px solid rgba(56, 189, 248, 0.55) !important;
      box-shadow: 0 0 14px rgba(56, 189, 248, 0.35), 0 0 28px rgba(56, 189, 248, 0.18), 0 2px 8px rgba(0, 0, 0, 0.4) !important;
    }
    .social-link:hover {
      border-color: #38bdf8 !important;
      box-shadow: 0 0 26px rgba(56, 189, 248, 0.9), 0 0 45px rgba(56, 189, 248, 0.5), 0 4px 14px rgba(0, 0, 0, 0.5) !important;
    }
    .social-link.highlight {
      border: 2px solid rgba(251, 191, 36, 0.75) !important;
      box-shadow: 0 0 14px rgba(251, 191, 36, 0.3), 0 2px 8px rgba(0, 0, 0, 0.4) !important;
    }
    .social-link.highlight:hover {
      border-color: #fbbf24 !important;
      box-shadow: 0 0 26px rgba(251, 191, 36, 0.85), 0 4px 14px rgba(0, 0, 0, 0.5) !important;
    }
    .footer-top .newsletter-card {
      margin-top: 0;
    }
    .newsletter-card {
      border: 2px solid rgba(251, 191, 36, 0.5) !important;
      box-shadow: 0 0 16px rgba(251, 191, 36, 0.15) !important;
    }
    .newsletter-name {
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    .newsletter-icon {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 18px;
      height: 18px;
      color: #F87171;
      flex-shrink: 0;
    }
    .newsletter-svg {
      width: 18px;
      height: 18px;
      fill: #F87171;
      display: block;
    }
    .location-svg {
      width: 14px;
      height: 14px;
      display: inline-block;
      vertical-align: -2px;
      fill: var(--cyan);
      margin-right: 4px;
    }
  </style>"""

    header_block = f"  <!-- START:HEADER -->\n  {nav_html}\n  <!-- END:HEADER -->"
    footer_block = f"    <!-- START:FOOTER -->\n{footer_styles}\n    {footer_html}\n    <!-- END:FOOTER -->"

    # Regex patterns for iframe or existing include block
    header_pattern = re.compile(
        r'(<!-- START:HEADER -->[\s\S]*?<!-- END:HEADER -->|<iframe\s+src="header\.html"[\s\S]*?</iframe>)',
        re.IGNORECASE
    )
    footer_pattern = re.compile(
        r'(<!-- START:FOOTER -->[\s\S]*?<!-- END:FOOTER -->|<iframe\s+src="footer\.html"[\s\S]*?</iframe>)',
        re.IGNORECASE
    )

    html_files = sorted(glob.glob(os.path.join(REPO_DIR, "*.html")))
    excluded = {
        os.path.abspath(HEADER_SRC),
        os.path.abspath(FOOTER_SRC)
    }

    updated_count = 0
    for file_path in html_files:
        if os.path.abspath(file_path) in excluded:
            continue
        if os.path.basename(file_path).startswith("google"):
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        new_content = header_pattern.sub(header_block, content)
        new_content = footer_pattern.sub(footer_block, new_content)

        if new_content != content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            updated_count += 1
            print(f"Included components in: {os.path.basename(file_path)}")
        else:
            print(f"No changes needed: {os.path.basename(file_path)}")

    print(f"\nBuild complete. Successfully updated {updated_count} HTML files.")

if __name__ == "__main__":
    build()
