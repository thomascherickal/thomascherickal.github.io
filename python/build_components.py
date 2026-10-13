#!/usr/bin/env python3
"""
build_components.py — Build-time component includer for thomascherickal.github.io.
Extracts <nav> from header.html and <footer> from footer.html,
and replaces iframes or existing build-time include blocks across all HTML pages.
Supports --watch to monitor header.html and footer.html for automatic updates.
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

    .nav-brand {
      font-family: var(--font-mono);
      font-size: 0.72rem;
      font-weight: 500;
      letter-spacing: 0.08em;
      color: var(--text-secondary);
      text-decoration: none;
      text-transform: uppercase;
      white-space: nowrap;
      display: inline-flex;
      align-items: center;
      gap: 0.55rem;
      transition: opacity 0.2s ease, transform 0.2s ease, color 0.2s ease;
    }

    .nav-brand-logo {
      width: 20px;
      height: 20px;
      border-radius: 4px;
      object-fit: cover;
      display: inline-block;
      flex-shrink: 0;
      border: 1px solid rgba(56, 189, 248, 0.4);
      box-shadow: 0 0 8px rgba(56, 189, 248, 0.25);
      transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .nav-brand:hover {
      color: var(--cyan);
      opacity: 0.95;
      transform: translateY(-1px);
    }

    .nav-brand:hover .nav-brand-logo {
      transform: scale(1.08);
      box-shadow: 0 0 12px rgba(56, 189, 248, 0.5);
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
    .footer-brand {
      display: flex;
      align-items: center;
      gap: 0.85rem;
    }
    @keyframes tcLogoPulse {
      0%, 100% {
        transform: translateY(0) scale(1);
        border-color: rgba(56, 189, 248, 0.45);
        box-shadow: 0 0 16px rgba(56, 189, 248, 0.3), 0 0 0 1px rgba(56, 189, 248, 0.2);
        filter: brightness(1) drop-shadow(0 2px 8px rgba(56, 189, 248, 0.25));
      }
      50% {
        transform: translateY(-3px) scale(1.04);
        border-color: rgba(245, 158, 11, 0.8);
        box-shadow: 0 0 28px rgba(56, 189, 248, 0.55), 0 0 16px rgba(245, 158, 11, 0.4), 0 0 0 1.5px rgba(245, 158, 11, 0.45);
        filter: brightness(1.12) drop-shadow(0 4px 14px rgba(245, 158, 11, 0.35));
      }
    }
    .footer-brand-logo-link {
      display: inline-flex;
      text-decoration: none;
      border-radius: 8px;
    }
    .footer-brand-logo {
      border-radius: 8px;
      object-fit: cover;
      border: 1px solid rgba(56, 189, 248, 0.45);
      box-shadow: 0 0 16px rgba(56, 189, 248, 0.25);
      flex-shrink: 0;
      animation: tcLogoPulse 3.5s ease-in-out infinite;
      transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
      will-change: transform, box-shadow;
    }
    .footer-brand:hover .footer-brand-logo {
      transform: translateY(-4px) scale(1.08);
      box-shadow: 0 0 32px rgba(56, 189, 248, 0.7), 0 0 18px rgba(245, 158, 11, 0.5);
      border-color: #38bdf8;
    }
    @media (prefers-reduced-motion: reduce) {
      .footer-brand-logo {
        animation: none;
      }
    }
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
      padding: 0.6rem 0.85rem !important;
      font-size: 0.82rem !important;
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
      grid-template-columns: repeat(auto-fill, minmax(min(100%, 150px), 1fr));
      gap: 0.6rem;
    }
    @media (min-width: 1750px) {
      .social-grid {
        grid-template-columns: repeat(10, 1fr);
      }
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
    .dv {
      border: none;
      border-top: 1px solid #232d42;
      margin: 1.45rem 0;
    }
    .tc-label {
      font-size: 13.5px;
      letter-spacing: .1em;
      text-transform: uppercase;
      font-weight: 700;
      color: #38bdf8;
      margin-bottom: .75rem;
      display: flex;
      align-items: center;
      gap: 7px;
    }
    .wg {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 10px;
      margin-bottom: 1.5rem;
    }
    .wl {
      border: 1px solid #253147;
      border-radius: 10px;
      padding: 13px 15px;
      text-decoration: none;
      color: #f8fafc;
      display: flex;
      flex-direction: column;
      gap: 4px;
      background: #131926;
      min-width: 0;
      transition: background .14s, border-color .14s, transform .12s, box-shadow .14s;
    }
    .wl:hover {
      background: #1c2539;
      border-color: rgba(56, 189, 248, 0.5);
      transform: translateY(-2px);
      box-shadow: 0 6px 16px rgba(0, 0, 0, 0.35);
    }
    .wl-head {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .wl-ic {
      width: 32px;
      height: 32px;
      border-radius: 8px;
      background: #1c2436;
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
    }
    .wl-ic svg {
      display: block;
    }
    .wl-name {
      font-size: 15.5px;
      font-weight: 700;
      color: #ffffff;
      line-height: 1.25;
    }
    .wl-desc {
      font-size: 13.5px;
      color: #94a3b8;
      line-height: 1.45;
      margin: 0;
    }
    .wl-url {
      font-size: 12.5px;
      color: #38bdf8;
      font-family: var(--font-mono, monospace);
      opacity: .85;
      margin-top: auto;
      word-break: break-all;
    }
    .wl:hover .wl-url {
      opacity: 1;
      text-decoration: underline;
    }
    .tc-copy {
      text-align: center;
      font-size: 14px;
      color: #94a3b8;
      line-height: 1.8;
      margin-top: 10px;
      padding: 0 10px;
    }
    .tc-copy a {
      color: #cbd5e1;
      text-decoration: none;
      font-weight: 600;
      transition: color .14s;
    }
    .tc-copy a:hover {
      color: #38bdf8;
      text-decoration: underline;
    }
    .tc-copy span.tc-sep {
      display: inline-block;
      width: 18px;
      height: 2px;
      background: linear-gradient(90deg, #6366f1, #10b981);
      vertical-align: middle;
      border-radius: 2px;
      margin: 0 8px;
    }
    @media (max-width: 820px) {
      .wg {
        grid-template-columns: repeat(2, 1fr);
      }
    }
    @media (max-width: 640px) {
      .wg {
        grid-template-columns: 1fr;
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

def watch():
    import time
    print(f"Watching {os.path.basename(HEADER_SRC)} and {os.path.basename(FOOTER_SRC)} for changes... (Press Ctrl+C to stop)")
    last_mtimes = {
        HEADER_SRC: os.path.getmtime(HEADER_SRC) if os.path.exists(HEADER_SRC) else 0,
        FOOTER_SRC: os.path.getmtime(FOOTER_SRC) if os.path.exists(FOOTER_SRC) else 0,
    }
    try:
        while True:
            time.sleep(1)
            changed = False
            for path in [HEADER_SRC, FOOTER_SRC]:
                if os.path.exists(path):
                    mtime = os.path.getmtime(path)
                    if mtime != last_mtimes.get(path, 0):
                        last_mtimes[path] = mtime
                        changed = True
            if changed:
                print(f"\n[{time.strftime('%X')}] Change detected in header or footer! Rebuilding...")
                build()
    except KeyboardInterrupt:
        print("\nComponent watcher stopped.")

if __name__ == "__main__":
    if "--watch" in sys.argv:
        build()
        watch()
    else:
        build()
