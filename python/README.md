# Python Automation & Build Directory

This directory contains the core build, verification, synchronization, and testing scripts powering the repository.

Strict Repo Compliance:
- Zero emojis.
- Zero em-dashes (spaced hyphens ' - ' or plain hyphens '-' used).

---

## Primary Scripts

### 1. `run_tests.py`
Executes the comprehensive unit test suite in `tests/` across 11 test modules. Sets `ANTIGRAVITY_IN_TEST=1` to prevent recursive test triggering, outputs a formatted test summary, and exits with code 0 on complete pass or code 1 on failure.
```bash
python3 python/run_tests.py
# or using standard unittest:
python3 -m unittest discover -s tests -v
```

### 2. `check_links.py`
High-speed internal link and anchor fragment verification engine. Scans all 64+ HTML files to ensure 100% link integrity (5,190+ links verified, 0 broken links). Run before committing.
```bash
python3 python/check_links.py
```

### 3. `build_components.py` (alias: `build.py`)
Build-time component includer for repository pages. Inlines `<nav>` from `header.html` and `<footer>` from `footer.html` across all standalone HTML pages, ensuring navigation synchronization.
```bash
# Single pass:
python3 python/build_components.py

# Continuous watch mode:
python3 python/build_components.py --watch
```

### 4. `sync_schemas.py`
Extracts embedded JSON-LD schemas (`<script type="application/ld+json">`) from all HTML pages and synchronizes the copy into corresponding JSON files in `schema/`. Live HTML files are the definitive source of truth.
```bash
python3 python/sync_schemas.py
```

### 5. `md_html_sync.py`
Bidirectional synchronization engine between HTML pages and `md-html-sync/*.md` files, preserving Markdown mirrors.
```bash
python3 python/md_html_sync.py
```

### 6. `build_md_files.py`
Full document AST compiler converting HTML pages into structured GitHub Flavored Markdown files.
```bash
python3 python/build_md_files.py
```

### 7. `generate_metrics.py`
Computes repository metrics, file counts, line numbers, and architectural telemetry into `METRICS.md`.
```bash
python3 python/generate_metrics.py
```

### 8. `generate_sitemap.py`
Constructs the complete XML sitemap (`sitemap.xml`) covering all published HTML pages.
```bash
python3 python/generate_sitemap.py
```

### 9. `delete_page.py`
Completely and cleanly deletes an HTML page from the website, removing its schema (`schema/*.json`), Markdown mirror (`md-html-sync/*.md`), cache entry (`md-html-sync/.sync-state.json`), sitemap entry, and header links to prevent it from ever being resurrected by the sync engine.
```bash
python3 python/delete_page.py <path_or_filename>
```

---

## Related Systems

- **`python/ssot/`**: Deterministic Single Source of Truth (SSOT) build and synchronization pipeline. Validates `data/*.json` against `data/schema/*.schema.json` and cross-file invariants, renders `<!--ssot:...-->` markers and Jinja2 blocks deterministically across HTML pages and `llms.txt`, exports `assets/data/site-data.json` for client-side JavaScript, and audits unprotected literal facts.
  ```bash
  # Validate data/*.json schemas and cross-file invariants:
  python3 python/ssot/validate.py

  # Render SSOT markers across HTML and llms.txt:
  python3 python/ssot/render.py

  # Verify zero unrendered drift (CI / hook check):
  python3 python/ssot/render.py --check

  # Export client-side JS data bundle:
  python3 python/ssot/export_site_data.py

  # Scan for hardcoded facts outside SSOT markers:
  python3 python/ssot/scan_literals.py
  ```

- **`failed-tests/`**: Centralized directory storing all failed-test reports from `python/run_tests.py`, `python/run_html_tests.py`, and `pricing-tests.js`. Reports are automatically removed when their respective test suite passes 100%.

- **`.git/hooks/pre-commit`** & **`python/hooks/pre-commit`**:
  Pre-commit hook that automatically runs `python/ssot/validate.py`, `python/build_components.py`, `python/ssot/render.py`, `python/ssot/export_site_data.py`, `python/md_html_sync.py`, `python/sync_schemas.py`, `python/ssot/render.py --check`, and Python/JS/HTML tests before permitting a commit.
