#!/usr/bin/env python3
"""
render.py - Deterministic SSOT marker & block renderer.

Replaces only the content between <!--ssot:...--> and <!--/ssot-->.
Supports --check mode for CI and pre-commit hooks.
"""

import argparse
import json
import os
import re
import sys
from bs4 import BeautifulSoup, Comment
from jinja2 import Environment, FileSystemLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from python.ssot.load import SSOTData
from python.ssot.format import call_formatter

TEMPLATES_DIR = os.path.join(REPO_ROOT, "templates", "blocks")
TARGET_DIRS = [
    ".",
    "business-services",
    "personal-services",
    "pricing",
    "claude-training",
    "gemini-training",
]

MARKER_RE = re.compile(r"(<!--ssot:(.+?)-->)(.*?)(<!--/ssot-->)", re.DOTALL)
JSONLD_SCRIPT_RE = re.compile(
    r'(<script type="application/ld\+json">)\s*(\{[\s\S]*?"@type"\s*:\s*"FAQPage"[\s\S]*?\})\s*(</script>)'
)


def _clean_node_text(node) -> str:
    frag = BeautifulSoup(str(node), "html.parser")
    for c in frag.find_all(string=lambda s: isinstance(s, Comment)):
        c.extract()
    text = frag.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,;:?!)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    return text


def sync_faq_jsonld(html_text: str) -> str:
    """Derive FAQPage JSON-LD deterministically from rendered .faq-card elements."""
    if '"FAQPage"' not in html_text or 'class="faq-card' not in html_text:
        return html_text
    soup = BeautifulSoup(html_text, "html.parser")
    cards = soup.select("article.faq-card")
    if not cards:
        return html_text
    entities = []
    for card in cards:
        q_el = card.select_one(".faq-question")
        a_el = card.select_one(".faq-answer")
        if not q_el or not a_el:
            continue
        entities.append(
            {
                "@type": "Question",
                "name": _clean_node_text(q_el),
                "acceptedAnswer": {
                    "@type": "Answer",
                    "text": _clean_node_text(a_el),
                },
            }
        )
    if not entities:
        return html_text
    payload = {
        "@context": "https://schema.org",
        "@type": "FAQPage",
        "mainEntity": entities,
    }
    formatted = json.dumps(payload, indent=2, ensure_ascii=False)
    indented = "\n".join("  " + line for line in formatted.splitlines())
    return JSONLD_SCRIPT_RE.sub(
        lambda m: f"{m.group(1)}\n{indented}\n  {m.group(3)}",
        html_text,
        count=1,
    )


def discover_targets(repo_root=REPO_ROOT):
    targets = []
    for d in TARGET_DIRS:
        full = os.path.join(repo_root, d)
        if not os.path.isdir(full):
            continue
        for fn in sorted(os.listdir(full)):
            if fn.endswith(".html"):
                targets.append(os.path.join(full, fn))
    for root_doc in ("llms.txt", "README.md"):
        doc_path = os.path.join(repo_root, root_doc)
        if os.path.exists(doc_path):
            targets.append(doc_path)
    return targets


def resolve_marker(data: SSOTData, jinja_env, expr: str) -> str:
    expr = expr.strip()
    if expr.startswith("fmt."):
        return call_formatter(data, expr[4:])
    if expr.startswith("block."):
        block_name = expr[6:].strip()
        tpl_name = f"{block_name}.html.j2"
        tpl = jinja_env.get_template(tpl_name)
        return tpl.render(data=data.raw, offerings=data.offerings, tiers=data.tiers_list)
    val = data.get(expr)
    return str(val)


def render_file(path: str, data: SSOTData, jinja_env, check_only: bool = False) -> bool:
    with open(path, encoding="utf-8") as f:
        raw = f.read()

    if "<!--ssot:" not in raw and '"FAQPage"' not in raw:
        return False

    def _replace(match):
        open_tag, expr, _old_body, close_tag = match.groups()
        line_no = raw[: match.start()].count("\n") + 1
        try:
            rendered = resolve_marker(data, jinja_env, expr)
        except Exception as exc:
            raise RuntimeError(f"{path}:{line_no}: failed to render ssot marker '{expr}': {exc}") from exc
        return f"{open_tag}{rendered}{close_tag}"

    updated = MARKER_RE.sub(_replace, raw) if "<!--ssot:" in raw else raw
    updated = sync_faq_jsonld(updated)
    if updated == raw:
        return False

    if not check_only:
        with open(path, "w", encoding="utf-8") as f:
            f.write(updated)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description="Render SSOT markers across HTML and llms.txt")
    parser.add_argument("--check", action="store_true", help="Exit non-zero if any file would change")
    args = parser.parse_args(argv)

    data = SSOTData()
    os.makedirs(TEMPLATES_DIR, exist_ok=True)
    jinja_env = Environment(loader=FileSystemLoader(TEMPLATES_DIR), autoescape=False)

    changed = []
    for path in discover_targets():
        if render_file(path, data, jinja_env, check_only=args.check):
            changed.append(os.path.relpath(path, REPO_ROOT))

    if args.check:
        if changed:
            print(f"SSOT CHECK FAILED: {len(changed)} file(s) out of sync:")
            for c in changed:
                print(f"  - {c}")
            return 1
        print("SSOT CHECK PASSED: All markers in sync.")
        return 0

    print(f"SSOT RENDER COMPLETE: Updated {len(changed)} file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
