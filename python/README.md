# Python Scripts Directory

All Python build, sync, and automation scripts for this repository are stored in this directory.

## Scripts

- **`build.py`**:
  Build-time component includer for `thomascherickal.github.io`. Inlines `<nav>` from `header.html` and `<footer>` from `footer.html` across all HTML pages, and triggers `md_html_sync.run_sync()`.
  ```bash
  python3 python/build.py
  ```

- **`md_html_sync.py`**:
  Bidirectional synchronization engine between HTML pages and `md-html-sync/*.md` files.
  ```bash
  python3 python/md_html_sync.py
  ```

- **`build_md_files.py`**:
  Extracts structured, high-fidelity Markdown files from all HTML pages and writes them to `md-html-sync/`.
  ```bash
  python3 python/build_md_files.py
  ```

## Guideline
Save all future Python scripts in this `python/` directory.
