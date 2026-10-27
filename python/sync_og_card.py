#!/usr/bin/env python3
"""
sync_og_card.py - Synchronizes assets/og-card.png with the factual stats in index.html.

Extracts the 4 stat numbers and labels from index.html's .stats-strip,
updates only the stats fields in python/generate_og_card.py,
and executes generate_og_card.py to regenerate assets/og-card.png.
"""

import os
import re
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
INDEX_HTML = os.path.join(REPO_DIR, "index.html")
GENERATE_SCRIPT = os.path.join(SCRIPT_DIR, "generate_og_card.py")
OG_CARD_PATH = os.path.join(REPO_DIR, "assets", "og-card.png")


def extract_index_stats():
    """Extracts the 4 stat items (number and label) from index.html's stats-strip."""
    with open(INDEX_HTML, "r", encoding="utf-8") as f:
        content = f.read()

    match = re.search(r'<div class="stats-strip">([\s\S]*?)</div>\s*<!--', content)
    if not match:
        raise ValueError("Could not find <div class=\"stats-strip\"> in index.html")

    strip_html = match.group(1)
    items = re.findall(
        r'<div class="stat-num">([\s\S]*?)</div>\s*<div class="stat-label">([\s\S]*?)</div>',
        strip_html
    )

    stats = [(num.strip(), lbl.strip()) for num, lbl in items]
    if len(stats) < 4:
        raise ValueError(f"Expected at least 4 stats in index.html, found {len(stats)}")

    return stats[:4]


def update_generate_script(stats):
    """Updates only the metrics-grid stats fields in python/generate_og_card.py."""
    with open(GENERATE_SCRIPT, "r", encoding="utf-8") as f:
        code = f.read()

    card_classes = ["card-gold", "card-cyan", "card-emerald", "card-purple"]
    val_classes = ["val-gold", "val-cyan", "val-emerald", "val-purple"]

    lines = ['      <div class="metrics-grid">']
    for idx, (num, lbl) in enumerate(stats):
        c_cls = card_classes[idx]
        v_cls = val_classes[idx]
        lines.append(f'        <div class="metric-card {c_cls}">')
        lines.append(f'          <div class="metric-val {v_cls}">{num}</div>')
        lines.append(f'          <div class="metric-lbl">{lbl}</div>')
        lines.append('        </div>')
    lines.append('      </div>')

    new_grid = "\n".join(lines)

    grid_pattern = re.compile(
        r'[ \t]*<div class="metrics-grid">[\s\S]*?</div>\s*</div>\s*</div>\s*</body>',
        re.DOTALL
    )

    # Replace only the metrics-grid block
    replacement = f"{new_grid}\n    </div>\n  </div>\n</body>"
    new_code, count = grid_pattern.subn(replacement, code)
    if count == 0:
        # Fallback to direct replacement of metrics-grid
        grid_exact = re.compile(r'[ \t]*<div class="metrics-grid">[\s\S]*?</div>\n    </div>', re.DOTALL)
        new_code, count = grid_exact.subn(f"{new_grid}\n    </div>", code)

    if count == 0:
        raise ValueError("Could not locate metrics-grid in python/generate_og_card.py")

    changed = (new_code != code)
    if changed:
        with open(GENERATE_SCRIPT, "w", encoding="utf-8") as f:
            f.write(new_code)
        print("Updated stats fields in python/generate_og_card.py.")
    else:
        print("Stats fields in python/generate_og_card.py are already up to date.")

    return changed


def regenerate_og_card():
    """Runs generate_og_card.py to produce the new assets/og-card.png."""
    cmd = [sys.executable, GENERATE_SCRIPT]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"generate_og_card.py failed:\n{res.stderr}")
    print(res.stdout.strip())


def sync_og_card(force_render=False):
    """Main synchronization routine."""
    stats = extract_index_stats()
    print("Found stats in index.html:")
    for idx, (num, lbl) in enumerate(stats, 1):
        print(f"  {idx}. {num} -> {lbl}")

    script_changed = update_generate_script(stats)

    # Regenerate image if script changed, force_render is True, or image missing
    if script_changed or force_render or not os.path.exists(OG_CARD_PATH):
        print(f"Regenerating {OG_CARD_PATH}...")
        regenerate_og_card()
        print(f"Successfully synced {OG_CARD_PATH} with index.html stats.")
    else:
        print(f"{OG_CARD_PATH} is already in sync with index.html stats.")


if __name__ == "__main__":
    force = "--force" in sys.argv
    sync_og_card(force_render=force)
