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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

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

    header_styles = """  <style id="header-styles">
    /* Global Root Mobile & Tablet Safeguards */
    html, body {
      max-width: 100%;
      overflow-x: hidden;
      position: relative;
    }

    img, svg, video, iframe {
      max-width: 100%;
      height: auto;
    }

    /* Prevent iOS Safari automatic zoom on form input focus */
    @media (max-width: 768px) {
      input, select, textarea {
        font-size: 16px !important;
      }
    }

    /* Mobile & Tablet Navigation Drawer (<= 1024px) */
    @media (max-width: 1024px) {
      .nav-toggle-label {
        display: flex !important;
        align-items: center;
        justify-content: center;
        min-width: 44px;
        min-height: 44px;
        padding: 0.6rem;
        cursor: pointer;
        z-index: 120;
      }

      .nav-toggle-label span,
      .nav-toggle-label span::before,
      .nav-toggle-label span::after {
        display: block;
        background: var(--text-primary);
        height: 2px;
        width: 22px;
        position: relative;
        transition: all 0.3s ease-in-out;
      }

      .nav-toggle-label span::before,
      .nav-toggle-label span::after {
        content: '';
        position: absolute;
      }

      .nav-toggle-label span::before {
        top: -6px;
      }

      .nav-toggle-label span::after {
        top: 6px;
      }

      .nav-links {
        position: fixed !important;
        top: 0 !important;
        right: 0 !important;
        bottom: 0 !important;
        left: 0 !important;
        width: 100% !important;
        height: 100dvh !important;
        background: rgba(5, 8, 16, 0.98) !important;
        backdrop-filter: blur(24px) !important;
        -webkit-backdrop-filter: blur(24px) !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: flex-start !important;
        gap: 0.85rem !important;
        padding: 5rem 1.25rem 3.5rem !important;
        overflow-y: auto !important;
        -webkit-overflow-scrolling: touch !important;
        overscroll-behavior: contain !important;
        touch-action: pan-y !important;
        z-index: 110 !important;
        transform: translateY(-100%) !important;
        opacity: 0 !important;
        pointer-events: none !important;
        transition: transform 0.35s cubic-bezier(0.16, 1, 0.3, 1), opacity 0.25s ease !important;
      }

      .nav-links li {
        width: 100%;
        max-width: 440px;
        text-align: center;
      }

      .nav-links a {
        font-size: 1.15rem !important;
        min-height: 44px;
        padding: 0.6rem 1rem !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 100%;
        box-sizing: border-box;
      }

      .nav-dropdown {
        width: 100% !important;
        max-width: 440px !important;
        text-align: center !important;
      }

      .nav-dropdown-toggle {
        cursor: pointer;
        user-select: none;
      }

      .dropdown-menu {
        position: static !important;
        transform: none !important;
        opacity: 1 !important;
        visibility: visible !important;
        pointer-events: auto !important;
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: var(--radius) !important;
        box-shadow: none !important;
        margin-top: 0.4rem !important;
        width: 100% !important;
        min-width: 100% !important;
        max-width: 100% !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        max-height: none !important;
        overflow: visible !important;
        padding: 0.5rem 0.25rem !important;
      }

      .dropdown-menu.mobile-hidden {
        display: none !important;
      }

      .dropdown-menu li {
        width: 100% !important;
      }

      .dropdown-menu a {
        min-height: 44px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
        font-size: 0.84rem !important;
        padding: 0.5rem 0.75rem !important;
        white-space: normal !important;
        line-height: 1.35 !important;
        border-radius: 6px;
      }

      .nav-toggle:checked ~ .nav-links {
        transform: translateY(0) !important;
        opacity: 1 !important;
        pointer-events: auto !important;
      }

      .nav-toggle:checked ~ .nav-right-container .nav-toggle-label span {
        background: transparent !important;
      }

      .nav-toggle:checked ~ .nav-right-container .nav-toggle-label span::before {
        transform: rotate(45deg) !important;
        top: 0 !important;
      }

      .nav-toggle:checked ~ .nav-right-container .nav-toggle-label span::after {
        transform: rotate(-45deg) !important;
        top: 0 !important;
      }
    }

    /* Mobile Carousel Card Responsiveness */
    @media (max-width: 640px) {
      .article-carousel-card {
        width: min(340px, calc(100vw - 1.5rem)) !important;
        min-width: min(340px, calc(100vw - 1.5rem)) !important;
        max-width: min(340px, calc(100vw - 1.5rem)) !important;
      }
    }

    @media (max-width: 380px) {
      .article-carousel-card {
        width: min(290px, calc(100vw - 1rem)) !important;
        min-width: min(290px, calc(100vw - 1rem)) !important;
        max-width: min(290px, calc(100vw - 1rem)) !important;
        padding: 1.25rem 1rem !important;
      }
    }
  </style>"""

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
      min-height: 44px;
      display: flex;
      align-items: center;
      gap: 0.6rem;
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
    .social-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(min(100%, 130px), 1fr));
      gap: 0.6rem;
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
    @media (max-width: 600px) {
      .footer-top {
        grid-template-columns: 1fr;
        gap: 1.75rem;
      }
      .newsletter-card {
        flex-direction: column;
        align-items: stretch;
        text-align: center;
        padding: 1.25rem 1rem !important;
        gap: 1rem;
      }
      .newsletter-card .btn {
        width: 100%;
        justify-content: center;
        min-height: 44px;
      }
      .footer-bottom {
        flex-direction: column;
        align-items: center;
        text-align: center;
        gap: 0.75rem;
      }
    }
  </style>"""

    header_block = f"  <!-- START:HEADER -->\n{header_styles}\n  {nav_html}\n  <!-- END:HEADER -->"
    footer_block = f"    <!-- START:FOOTER -->\n{footer_styles}\n    {footer_html}\n    <!-- END:FOOTER -->"

    # Regex patterns for iframe or existing include block
    header_pattern = re.compile(
        r'([ \t]*<!-- START:HEADER -->[\s\S]*?<!-- END:HEADER -->|[ \t]*<iframe\s+src="header\.html"[\s\S]*?</iframe>)',
        re.IGNORECASE
    )
    footer_pattern = re.compile(
        r'([ \t]*<!-- START:FOOTER -->[\s\S]*?<!-- END:FOOTER -->|[ \t]*<iframe\s+src="footer\.html"[\s\S]*?</iframe>)',
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
        if "footer" in os.path.basename(file_path).lower():
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
    
    try:
        import md_html_sync
        md_html_sync.run_sync()
    except Exception as e:
        print(f"Warning: MD-HTML sync step failed: {e}")

if __name__ == "__main__":
    build()
