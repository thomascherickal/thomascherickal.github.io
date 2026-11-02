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
Computes repository metrics, file counts, line numbers, and architectural telemetry into `metrics.md`.
```bash
python3 python/generate_metrics.py
```

### 8. `generate_sitemap.py`
Constructs the complete XML sitemap (`sitemap.xml`) covering all published HTML pages.
```bash
python3 python/generate_sitemap.py
```

---

## Related Systems

- **`nlp-fuzzy-expert/`**: Forward-chaining expert system with NLP entity recognition and fuzzy string matching. Synchronizes authoritative facts from `data/*.json` to HTML/JS files repository-wide.
  ```bash
  # Single sync pass:
  python3 nlp-fuzzy-expert/synchronizer.py

  # Autonomous background watcher:
  python3 nlp-fuzzy-expert/watcher.py
  ```

- **`.git/hooks/pre-commit`** & **`python/hooks/pre-commit`**:
  Pre-commit hook that automatically runs `run_tests.py`, `build_components.py`, `sync_schemas.py`, and `md_html_sync.py` before permitting a commit.
