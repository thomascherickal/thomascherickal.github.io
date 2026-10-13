# Project Automation & Agent Rules

## Component Inlining (`header.html` & `footer.html`)
Whenever `header.html` or `footer.html` is modified, created, or updated:
- **ALWAYS automatically run** `python3 python/build_components.py` immediately in the same turn.
- Do not wait for the user to request it.
- This ensures all 20+ standalone HTML pages stay synchronized with the latest navigation bar, footer elements, styles, and Markdown sync files (`md-html-sync/`).
