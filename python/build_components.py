#!/usr/bin/env python3
"""
build_components.py - Build-time component includer for thomascherickal.github.io.
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
    header_block = f"  <!-- START:HEADER -->\n  {nav_html}\n  <!-- END:HEADER -->"
    footer_block = f"    <!-- START:FOOTER -->\n    {footer_html}\n    <!-- END:FOOTER -->"

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
        if os.path.basename(file_path).startswith("google43d"):
            continue
        if "footer" in os.path.basename(file_path).lower():
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        if header_pattern.search(content):
            new_content = header_pattern.sub(header_block, content)
        else:
            if "<body" in content:
                skip_match = re.search(r'(<a\s+[^>]*class="skip-link"[^>]*>[\s\S]*?</a>)', content, re.IGNORECASE)
                if skip_match:
                    new_content = content[:skip_match.end()] + "\n" + header_block + content[skip_match.end():]
                else:
                    body_match = re.search(r'(<body\b[^>]*>)', content, re.IGNORECASE)
                    new_content = content[:body_match.end()] + "\n" + header_block + content[body_match.end():]
            else:
                new_content = content

        if footer_pattern.search(new_content):
            footer_match = footer_pattern.search(new_content)
            main_close_pos = new_content.rfind("</main>")
            if main_close_pos != -1 and footer_match.start() < main_close_pos:
                # Footer was inside <main>! Remove it from inside <main> and re-place outside </main>
                new_content = footer_pattern.sub("", new_content)
                main_close = new_content.rfind("</main>") + len("</main>")
                new_content = new_content[:main_close] + "\n" + footer_block + "\n" + new_content[main_close:]
            else:
                new_content = footer_pattern.sub(footer_block, new_content)
        else:
            if "</main>" in new_content:
                main_close = new_content.rfind("</main>") + len("</main>")
                new_content = new_content[:main_close] + "\n" + footer_block + "\n" + new_content[main_close:]
            elif "</body>" in new_content:
                body_close = new_content.rfind("</body>")
                new_content = new_content[:body_close] + "\n" + footer_block + "\n" + new_content[body_close:]

        if "components.js" not in new_content and "</body>" in new_content:
            body_close = new_content.rfind("</body>")
            new_content = new_content[:body_close] + '<script src="components.js"></script>\n' + new_content[body_close:]

        if new_content != content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            updated_count += 1
            print(f"Included components in: {os.path.basename(file_path)}")
        else:
            print(f"No changes needed: {os.path.basename(file_path)}")

    print(f"\nBuild complete. Successfully updated {updated_count} HTML files.")
    
    try:
        import sync_pricing
        sync_pricing.sync_all_pricing()
    except Exception as e:
        print(f"Warning: Pricing sync step failed: {e}")

    try:
        import sync_schemas
        sync_schemas.sync_html_to_schemas()
    except Exception as e:
        print(f"Warning: HTML to schema sync step failed: {e}")

    try:
        import md_html_sync
        md_html_sync.run_sync()
    except Exception as e:
        print(f"Warning: MD-HTML sync step failed: {e}")

    try:
        import generate_metrics
        generate_metrics.generate_metrics()
    except Exception as e:
        print(f"Warning: Metrics generation step failed: {e}")

    try:
        import sync_og_card
        sync_og_card.sync_og_card()
    except Exception as e:
        print(f"Warning: OG card sync step failed: {e}")

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
