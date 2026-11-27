#!/usr/bin/env python3
"""
render.py - Deterministic SSOT marker & block renderer.

Replaces only the content between <!--ssot:...--> and <!--/ssot-->.
Supports --check mode for CI and pre-commit hooks.
"""

import argparse
import os
import re
import sys
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


def discover_targets(repo_root=REPO_ROOT):
    targets = []
    for d in TARGET_DIRS:
        full = os.path.join(repo_root, d)
        if not os.path.isdir(full):
            continue
        for fn in sorted(os.listdir(full)):
            if fn.endswith(".html"):
                targets.append(os.path.join(full, fn))
    llms = os.path.join(repo_root, "llms.txt")
    if os.path.exists(llms):
        targets.append(llms)
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

    if "<!--ssot:" not in raw:
        return False

    def _replace(match):
        open_tag, expr, _old_body, close_tag = match.groups()
        line_no = raw[: match.start()].count("\n") + 1
        try:
            rendered = resolve_marker(data, jinja_env, expr)
        except Exception as exc:
            raise RuntimeError(f"{path}:{line_no}: failed to render ssot marker '{expr}': {exc}") from exc
        return f"{open_tag}{rendered}{close_tag}"

    updated = MARKER_RE.sub(_replace, raw)
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
