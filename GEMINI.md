# Project Automation & Agent Rules

## Component Inlining (`header.html` & `footer.html`)
Whenever `header.html` or `footer.html` is modified, created, or updated:
- **ALWAYS automatically run** `python3 python/build_components.py` immediately in the same turn.
- Do not wait for the user to request it.
- This ensures all 20+ standalone HTML pages stay synchronized with the latest navigation bar, footer elements, styles, and Markdown sync files (`md-html-sync/`).

## Navigation Architecture & No New Navbars
- **DO NOT create or add new navbars** (`<nav>`, sub-navbars, in-page guide/topic navigation strips, secondary navbars) when creating or updating content or pages.
- The website relies strictly and exclusively on the centralized primary navigation bar defined in `header.html` (which is inlined across all pages via `python3 python/build_components.py`).
- Navigation to new content must only be handled via:
  1. The existing global navigation dropdowns or links in `header.html`.
  2. Standard in-content hyperlinks (cards, lists, or buttons) within the page body.
- Never add bespoke `<nav>` elements, sticky topic strips, or secondary horizontal/vertical navigation bars to content pages.

## Canonical Profile Categories & Business Website
- **Business Website**: `https://thedigitalfuturist.com`
- Always categorize and list `https://thedigitalfuturist.com` under the **Business Website** category across all profile registries, documentation, schemas, and links.

## The Digital Futurist Tagline Rule
- **Canonical Tagline**: `The Future Is Now`
- Always use `The Future Is Now` as the tagline wherever text appears next to, above, or alongside The Digital Futurist logo (such as the hero eyebrow on `index.html` and markdown documentation).
- Do not use `The Digital Futurist · Remote Worldwide`.

## Canonical Site & Profile Description
- **Canonical Description**: `"description": "Thomas Cherickal - Generative AI Consultant and The Digital Futurist."`
- Always use `"Thomas Cherickal - Generative AI Consultant and The Digital Futurist."` as the primary site, author, and profile description across all HTML files, metadata, and schemas.

## Call to Action (CTA) & Free LinkedIn Consultation Rule
- **ALWAYS include the Free LinkedIn Consultation** link/button in **every Call to Action (CTA)** block, banner, or section created or updated across any page:
  `<a class="btn btn-linkedin" href="https://linkedin.com/in/thomascherickal" rel="noopener" target="_blank">Free LinkedIn Consultation ↗</a>`
- Ensure it is prominently featured alongside other primary actions (such as `pricing.html` or `contact.html`).

## Model Version Numbers Rule (Claude, Gemini, and all LLMs)
- **DO NOT include version numbers for any LLM models** (e.g., avoid Claude 3, Claude 3.5, Gemini 1.5, Gemini 2.5, Llama 3, Qwen 2.5, DeepSeek V3, etc.) anywhere across general website prose, documentation, or schemas.
- Refer to models exclusively by their model family names without version numbers (e.g. Claude, Claude Opus, Claude Sonnet, Claude Haiku, Gemini, Gemini Pro, Gemini Flash, Gemini Argon, Qwen, DeepSeek, Mistral, Gemma).
- **STRICT EXCEPTIONS (Model version numbers are permitted ONLY in these two cases)**:
  1. **Endpoints and Model Strings in Code Examples**: Do NOT change model numbers in API endpoints, parameters, or model identifiers within code blocks, script samples, configuration snippets, or test payloads.
  2. **Pricing Mentions**: When quoting specific pricing, token rates, cost calculations, pricing tables, or break-even analyses, always include the specific model version numbers tied to those prices.

## No Emojis Rule
- **DO NOT use emojis** in any further generation of any file in the future in this repo.
- Ensure all generated or modified files (including HTML, CSS, JavaScript, Markdown, Python, and configuration files) remain completely free of emojis.

## No Em-Dash Rule
- **DO NOT use em-dashes** (the em-dash character U+2014 or the &amp;mdash; / &amp;#8212; entities) anywhere in any file in this repo.
- Use a spaced hyphen ` - ` in prose, headings, and titles.
- En-dashes in numeric ranges (e.g., `Days 1-15`, `2020-2026`) must also be written as plain hyphens `-`.

## Platform Rules
- Dev.to is an allowed platform. Include `https://dev.to/thomascherickal` in platform lists, `sameAs` arrays, and footers wherever the other publishing platforms appear.
- Never remove existing Dev.to links.
- Mastodon is excluded.

## Mermaid Diagrams Layout & Vertical Orientation Rule
- **Vertical or Square Layouts by Default**: All Mermaid diagrams in HTML files (`<div class="mermaid">`) or Markdown files (````mermaid`) must be designed in a vertical (`TD`, `TB`) or square orientation to ensure minimal horizontal scrolling across viewport sizes.
- **Rotate Horizontal Diagrams by 90 Degrees**: Do NOT use long horizontal flowcharts or graphs (`flowchart LR`, `graph LR`, `RL`). Long horizontal chains must be rotated by 90 degrees to vertical top-down orientation (`flowchart TD` or `graph TD`).
- **Preserve Readability**: Never shrink diagram scales or compress node labels to unreadable font sizes to circumvent layout boundaries. Use vertical flow with standard, legible label typography.

## Article Footers
- Keep `article-footer.md`, `article-footer_dark.html`, and `article-footer-plain.docx` in the repo; do not delete or move them.
- Never insert an article footer into any Claude article (`claude-*.html` or `md-html-sync/claude-*.md`).

## Canonical Email ID
- **Canonical Email ID**: `thomascherickal@gmail.com`
- **ALWAYS** save, render, and use `thomascherickal@gmail.com` as the primary contact email address across all files, footer links, mailto references, documentation, schemas, and configurations.

## HTML to JSON Schema Synchronization (`schema/`)
- **Run `python3 python/sync_schemas.py` only when committing**, not after every change.
- Live HTML files are the definitive source of truth for all structured data and schema markup.
- `python/sync_schemas.py` extracts embedded JSON-LD schemas from all HTML pages and synchronizes the copy into the corresponding JSON file under the `schema/` directory (e.g., `index.html` -> `schema/home-schema.json`, `service-*.html` -> `schema/service-*.json`, and other pages -> `schema/<page>-schema.json`).
- Ensure all updated schema files in `schema/` are staged and included in the commit.

## Markdown Synchronization (`md-html-sync/`)
- **Run `python3 python/md_html_sync.py` only when committing**, not after every change.
- Keeps Markdown files in `md-html-sync/` synchronized with their corresponding HTML pages.
- When full Markdown generation is required across all pages, run `python3 python/build_md_files.py`.

## Link Verification (`python/check_links.py`)
- **DO NOT check for broken links at every change or edit.**
- **Run `python3 python/check_links.py` ONLY before committing**, not after every individual change.
- If any broken links or missing anchor targets are detected, fix them immediately before finalizing and committing.

## No Underlined Text in Buttons Rule
- **NEVER underline text in buttons** (`<button>`, `input[type="button"]`, `input[type="submit"]`, `.btn`, `a.btn`, `a.tier-btn`, or any button element/class) across any page, stylesheet, or inline style.
- Buttons must always maintain `text-decoration: none !important;` across all pseudo-classes (`:hover`, `:focus`, `:active`, `:visited`).
- When applying parent link styles (such as `a { text-decoration: underline; }`), always exclude buttons using `:not(.btn)` or explicitly enforce `text-decoration: none !important;` on button selectors.

## Gold Hyperlinks by Default Rule
- **ALWAYS use gold colors for hyperlinks by default** across all HTML pages, markdown documentation, components, stylesheets, and content blocks in this repo.
- Hyperlinks must default to gold (`var(--gold, #fbbf24)` or `var(--bs-gold, #fbbf24)` / `#fbbf24`), with hover states transitioning to gold-dim (`var(--gold-dim, #d97706)` or `#d97706`).
- When styling prose, inline references, tables, resource lists, or guide directories, always use gold styling (e.g. `color: var(--gold);` or `class="gold-link"`).
- Buttons (`.btn`, `a.btn`, etc.) remain unaffected and must always maintain their button styling with `text-decoration: none !important;`.

## Uniform Button Formatting Rule
- **Keep all buttons in one HTML page in the same format, sizing, and styling structure.**
- Across any given page and site-wide, buttons within the same section or action group (e.g. Hero action buttons, bottom CTA blocks, or card action rows) must share identical dimensions, padding, font-size, line-height, letter-spacing, text-transform, min-height, and flex alignment.
- Never use mismatched inline padding or font sizes (such as mixing `padding:0.75rem 1.6rem;font-size:0.875rem;` on one button with `padding:0.75rem 1.5rem;font-size:0.85rem;` on another).
- In CTA groups, all peer buttons must use identical sizing modifiers (e.g., all `.btn-lg` or all standard `.btn`, never mixing sizes).
- Always use standardized CSS classes (`.btn`, `.btn-commission`, `.btn-linkedin`, `.btn-primary`, `.btn-gold`, `.btn-outline`, `.btn-lg`, `.btn-full`) defined in `css/buttons.css` rather than ad-hoc inline dimensional overrides.
- All buttons must strictly maintain `text-decoration: none !important;` across all states (`:hover`, `:focus`, `:active`, `:visited`).

## NLP-Fuzzy-Expert Data Synchronization & Watcher Rule (`nlp-fuzzy-expert/`)
- **Authoritative Data Directory (`data/*.json`)**: All core data (pricing, services, config, publications, tiers, FAQs, Claude/Gemini guides) is mastered in `data/*.json`.
- **Knowledge Synchronization Engine**: Changes to `data/*.json` are synchronized to target HTML and JS files via `python3 nlp-fuzzy-expert/synchronizer.py`.
- **Working Steps & Line Diff Reporting**: The synchronizer prints all working steps (Steps 1-6) in real-time, reports exact 1-based changed line numbers (e.g., `394-405, 428-449`) for each modified document, and displays a clean ASCII summary table upon completion.
- **Continuous File Watcher Daemon**: Run `python3 nlp-fuzzy-expert/watcher.py` (or `python3 nlp-fuzzy-expert/watcher.py --once` for single pass) to monitor `data/*.json` and trigger automatic repository-wide synchronization on file save.
- **Artifacts Generated**: Domain references (`*_references.json`), master reference dictionary (`references_dictionary.json`), and domain synchronization markdown reports (`*.md`) in `nlp-fuzzy-expert/`.

## Unit Testing Suite, Pre-Commit Test Enforcement & Commit/Push Workflow (`tests/` & `python/run_tests.py`)
- **Test Command**: Run all unit tests with `python3 python/run_tests.py` or `python3 -m unittest discover -s tests -v`.
- **Run Tests After HTML Changes**: When changes are made to repository HTML or JS files, always run the test suite to verify 100% test pass rate across all 36+ test cases.
- **Pre-Commit Enforcement**: The pre-commit hook automatically executes `python3 python/run_tests.py` and `node pricing-tests.js`. If any unit test fails, the commit is aborted. Never bypass or force commits with broken tests.
- **No Automatic Commit and Push on Every Change**: Do NOT commit or push automatically on every individual edit or change. Only commit or push when the user explicitly requests it or when a complete multi-step task is fully concluded and ready for version control. Unrequested incremental commits and pushes must not be executed.

## Practice Pricing, Payment Structure & Licensing Rules
- **Business AI Training Flagship**:
  - Base price: **$3000 USD** for **60 days total** (30 days live engineering training + 30 days dedicated support).
  - Includes up to 10 participants. Extra seats are **$300 USD per seat** after 10 participants, up to a strict maximum of 20 seats.
- **Personal AI Training Flagship**:
  - Base price: **$3000 USD** for **60 days total** (30 days live 1-on-1 training + 30 days dedicated support).
  - Strictly **one person only** (dedicated private acceleration).
- **Individual Service Picks**:
  - **Business Individual Service**: **$300 USD** base for 1 seat, with additional seats at **$300 USD per seat** up to a maximum of 10 seats (5 days live training + 5 days support).
  - **Personal Individual Service**: **$300 USD** strictly for **only one seat** (5 days live training + 5 days support).
- **Delivery Schedule & Mode**:
  - **4 hours per day**, Monday through Friday.
  - **Saturdays and Sundays are holidays**.
  - Delivered **100% remote worldwide**.
- **Payment Structure & Trust Guarantees**:
  - **$0 Upfront**: Strictly zero advance payment or deposit required before commencing technical discovery and training.
  - **20% in Five Installments**: Invoiced in five equal installments (20% each) due strictly after completed milestone blocks of training progress.
  - **Walkaway Guarantee**: Walk away owing nothing if not satisfied after the first milestone training phase.
- **Purchasing Power Parity (PPP) Explanation**:
  - Every page mentioning pricing must provide a short explanation of the six 195-country Purchasing Power Parity (PPP) tiers and include a prominent link to `pricing.html`.
- **Global Parity Pricing Display Everywhere Rule**:
  - Wherever pricing is presented across the website (including Hero banners, package summaries, guarantee blocks, service pages, and guides), always display the full Purchasing Power Parity range to prominently highlight global parity access:
    - **Flagship Tiers**: `$3,000 / $2,250 / $1,500 / $1,500 / $750 / Custom USD` (representing Tier 1 through Tier 6).
    - **Individual Service Tiers**: `$300 / $225 / $150 / $150 / $75 / Custom USD` (representing Tier 1 through Tier 6).
    - **Additional Seats (Flagship & Individual)**: Billed at the identical calibrated tier rates (`+$300 / +$225 / +$150 / +$150 / +$75 / Custom USD per extra seat`).
  - Follow front-end styling and responsive design guidelines to ensure text wraps smoothly on mobile screens without horizontal blowout.
- **High-Quality Training Material & Apache 2.0 Licensing**:
  - All high-quality training materials, curriculum documents, architectural guides, and custom client code are licensed under the **Apache 2.0 license**, permitting commercial use, internal deployment, resale, and relicensing with attribution.
- **Support Period Payments**:
  - **Flagship Support (Business & Personal)**: An additional **$500 USD** for the 30 days support period, paid in two installments: **$250 USD after 15 days** of support and **$250 USD after 30 days** of support.
  - **Individual Services Support (Business & Personal)**: An additional **$50 USD** for the 5 days support period, paid in a single installment **after 5 days** of support.
  - **Zero Upfront for Support**: Support payments are strictly due only after verifying support delivery (after 15 days and 30 days of support for flagships; after 5 days of support for individual services).
  - **Parity Scaled**: Support payments scale across the six Purchasing Power Parity (PPP) tiers.

## Page Deletion & Clean Purge Rule (`python/delete_page.py`)
- **ALWAYS use `python3 python/delete_page.py <path_or_filename>`** whenever deleting an HTML page or guide from the website.
- **Never simply delete the `.html` file with `rm`**: Doing so leaves behind orphaned schemas in `schema/`, markdown files in `md-html-sync/`, tracking entries in `md-html-sync/.sync-state.json`, and links in `header.html` or `sitemap.xml`, which can cause sync conflicts or resurrection.
- The `delete_page.py` utility automatically cleans:
  1. The HTML page from its folder (`google/`, `claude/`, `business-services/`, `personal-services/`, or root).
  2. The corresponding schema file under `schema/`.
  3. The Markdown mirror file under `md-html-sync/`.
  4. The tracking cache entry in `md-html-sync/.sync-state.json`.
  5. Any references in `python/generate_sitemap.py`, `sitemap.xml`, `header.html`, and `llms.txt`.
  6. Runs verification passes (`sync_schemas.py`, `md_html_sync.py`, `check_links.py`) to confirm zero broken links or orphaned sync states.

## Data Directory Single Source of Truth (SSOT) & Source Control Read-Only Rule (`data/*.json`)
- **GEMINI.md as Ultimate Source of Truth**: `GEMINI.md` is the authoritative project specification and the ultimate source of truth across the entire website. The files in `data/*.json` reflect this specification for automated synchronization and data modeling. `GEMINI.md` is the only source whose definitions can override files in the `data/` directory. No other code, runtime script, or test suite should modify the files under the `data/` directory.
- **Tracked in Source Control (Included in Git)**: The `data/` directory is tracked in Git (never listed in `.gitignore`). All files within `data/*.json` are included in commits, pushes, and clones, ensuring authoritative project knowledge is versioned and preserved across repository clones.
- **Strict Read-Only to Source Control Operations**: The `data/` directory is strictly read-only against all source control commands, scripts, and automated tooling. Commands from source control do not touch the `data/` directory. The `data/` directory is included in commits, pushes, and clones, but NEVER in reverts, rebase, resets, merges, checkouts, or other source control commands that could alter, undo, or overwrite user edits.
- **Direct User Editing Only**: Only direct manual editing by the user works. No file in any other folder, no script, and no source control command is permitted to undo, overwrite, or modify the contents of the `data/` directory.
- **Authoritative Data Directory (`data/*.json`)**: All JSON files residing in `data/*.json` (`pricing.json`, `tiers.json`, `business-services.json`, `personal-services.json`, `config.json`, `publications.json`, `faqs.json`, `gemini.json`, `claude.json`) model the authoritative data specified by `GEMINI.md`.
- **Strict One-Way Data Propagation**: Data flows strictly outward in one direction: `data/*.json` -> HTML / JS / Markdown / Schema files. Reverse writes, automatic modifications, or overwriting of `data/*.json` files by sync scripts, inference engines, or testing suites are strictly forbidden.
- **Programmatic Mutation Prevention**: The core file-saving utility `save_json()` in `nlp-fuzzy-expert/expert_system.py` strictly blocks writing to any path within `data/` by raising a `PermissionError`.
- **Test Isolation & Sandbox Execution**: Tests that simulate or verify changes to data (e.g. `tests/test_continuous_updates.py`) must never mutate repository files under `data/*.json`. Such tests must operate within isolated temporary sandboxes (e.g. `tempfile.mkdtemp()`) and instantiate `ExpertSynchronizer(data_dir=temp_dir)`.
- **Pre-Test and Post-Test SHA-256 Integrity Verification**: `python/run_tests.py` computes cryptographic SHA-256 hashes of all `data/*.json` files prior to test execution and validates them immediately following execution. If any file in `data/` is altered, mutated, or created, the test suite aborts with exit code 1.
- **Manual Authoring Only**: Changes to `data/*.json` may only be made by deliberate manual updates by the author or explicitly requested data configuration tasks aligned with `GEMINI.md`.

## New Files & Source Control Integrity Rule
- **Mandatory Pricing & PPP Parity Audit**: Whenever any new file (HTML, Markdown, script, or documentation) is created, updated, or added to source control:
  - All pricing mentions, rates, and Purchasing Power Parity (PPP) numbers across all 6 tiers must strictly conform to the authoritative figures specified in `GEMINI.md` and `data/pricing.json`.
  - Flagships must be specified at **$3000 USD** for **60 days total** (30 days live + 30 days dedicated support), with additional seats at **$300 USD per seat** (up to 20 seats max for business; strictly 1 person only for personal).
  - Individual service picks must be specified at **$300 USD** base rate (covering 5 days live + 5 days dedicated support).
  - Advance deposit must strictly be **$0 upfront** (never corrupted strings or omitted dollar signs).
  - The 6 PPP tiers must explicitly reflect: Tier 1 ($3000), Tier 2 ($2250), Tier 3 ($1500), Tier 4 ($1500), Tier 5 ($750), Tier 6 (Custom).
  - Must strictly preserve zero emojis and zero em-dashes.
## User Clarification & Decision Rule
- **Mandatory Clarification for Ambiguities**: Whenever there are ambiguities, doubts, clarifications required, or questions about an implementation, check whether there is a single clear recommended plan:
  - **Single Recommended Action**: If there is a single Recommended action, do NOT ask the user questions but proceed immediately with the Recommended plan.
  - **Two or More Recommended Actions**: Ask the user a question only if there are two or more Recommended actions or competing implementation paths with comparable merit, providing clear options so the user can choose.
- **Never Assume or Guess Unclear Divergent Paths**: Do not make unilateral assumptions when multiple equally viable divergent paths exist without a clear recommended path. Always pause and verify with the user when two or more distinct recommended options are present.

## Organic Reference Links Rule (No Superscript Numbers)
- **NO Superscript Numbered Reference Links**: Do NOT use superscript or bracketed numbered citation markers (e.g., `<sup>[1]</sup>`, `<sup><a href="#ref-1">[1]</a></sup>`, `[^1]`, or `[1]`) anywhere across repository files.
- **Organic In-Content Anchor Links**: Anchor all references organically within 2 - 4 descriptive words of the actual prose text (e.g., `<a class="gold-link" href="#references">official quickstart guide</a>` or `<a class="gold-link" href="#references">API pricing documentation</a>`).
- **Enforced Repository-Wide**: This standard applies strictly across all content files added to or updated in the repository, including all HTML files and Markdown files (`md-html-sync/`, documentation, and articles).

## Unnumbered Headings & Code Block Sharp Boundaries Rule
- **Unnumbered Headings in Content**: Do NOT use numeric prefixes (such as `1. `, `01. `, `1) `, or `<span class="sec-num">01</span>`) in headings (`<h2>`, `<h3>`, `<h4>`, etc.) across HTML content pages. Numbering belongs exclusively in the Table of Contents (ToC) to maintain a clean reading experience in the body while preserving structured outline numbering in the navigation index.
- **Sharp Boundaries Between Code and Text**: All code samples and preformatted code blocks (`<pre>`, `.code-window`, `.codewrap`, `.code`, `.code-box`) must have sharp, visually distinct boundaries separating them from surrounding prose:      
  - High-contrast dark background (`#040711` or `#060913`).
  - Crisp borders (`1px solid rgba(56, 189, 248, 0.32)` or `rgba(251, 191, 36, 0.32)`) and distinct left accent border (`border-left: 3px solid var(--cyan)` or `var(--gold)`).
  - Explicit vertical margins (`margin: 1.5rem 0;`) ensuring clear visual separation from surrounding text.
  - A structured header bar (`.code-bar`, `.code-head`, `.cap`, `.code-header`) containing a file/language label and window dot accents where applicable.
- **Copy Button on Every Code Sample**: Every code sample across all HTML files on the website must include an accessible, interactive Copy button (`button.copy` or `.copy-btn`) that copies the snippet text to the clipboard and provides visual feedback (`Copied!`). Buttons must strictly maintain `text-decoration: none !important;` across all states.

## Last Commit Output Report Rule (`LAST-COMMIT-REPORT.md` & `README.md`)
- **Last Commit Output Recording**: The full commit output report of the last commit made must always be stored and recorded at `LAST-COMMIT-REPORT.md` in the repository root directory.
- **README Commit & Metrics Summary Synchronization (`python/update_readme_commit_summary.py`)**: Immediately after `LAST-COMMIT-REPORT.md`, its test coverage section, and `METRICS.md` are generated, `python3 python/update_readme_commit_summary.py` extracts the summary of updates (Commit Overview, Quality Gate & Verification Summary, Coverage Executive Summary, and the Repository Code & Engineering Metrics Summary from `METRICS.md`) and appends/updates it in `README.md` just before the commit is finalized.
- **Strictly Pre-Commit Only (Zero Post-Commit Modifications)**: All code synchronization, component building, schema extraction, Markdown mirroring, metrics calculation, test execution, `LAST-COMMIT-REPORT.md` generation, test coverage appending, and `README.md` commit summary updating MUST happen strictly **before** the commit is finalized (inside `.git/hooks/pre-commit` / `python/hooks/pre-commit`, or prior to `git commit` in automated handlers).
- **Clean Working Tree After Every Commit**: Never run post-commit hooks or post-commit scripts that modify tracked or untracked files after `git commit` completes. Immediately after any commit, `git status` must be 100% clean with zero modified or untracked files.
- **Zero Emojis & Zero Em-Dashes**: The generated report strictly enforces zero emojis and zero em-dashes (spaced hyphens ` - ` or plain hyphens `-`).

## Test Coverage Report Rule (Appended to LAST-COMMIT-REPORT.md)
- **Automated Test Coverage Generation**: The repository-wide test coverage report for Python and JavaScript must be automatically generated and appended to `LAST-COMMIT-REPORT.md` before every commit, and staged as part of that commit.
- **Comprehensive Dual-Stack Assessment**: Evaluates Python code coverage (using coverage tracer across `nlp-fuzzy-expert/` and `python/`) and JavaScript code coverage (using native Node.js V8 code coverage `NODE_V8_COVERAGE` across `pricing-calculator.js`, `components.js`, and `mermaid.js`).
- **100% Coverage Target**: Enforces 100% test coverage across both Python and JavaScript codebases.

## Repository Metrics Rule (`METRICS.md`)
- **Automated Update Before Every Commit**: `METRICS.md` must always be generated and updated via `python3 python/generate_metrics.py` before every commit, and staged as part of that commit.
- **Pre-Commit Enforcement**: The pre-commit hook automatically executes `python3 python/generate_metrics.py` before finalizing each commit to ensure all tracked file counts, lines of code (SLOC), asset sizes, and category distributions are completely up-to-date in `METRICS.md`.
- **Zero Emojis & Zero Em-Dashes**: `METRICS.md` strictly adheres to the repository compliance standards with zero emojis and zero em-dashes.

## Image Generation, Asset Compression & Organization Rule (`assets/`)
- **Strict Size Limit (< 400 KB)**: Whenever image generators are used or images are generated, added, or modified in the repository, compress every image to less than 400 KB without exception.
- **Automatic Compression & Old File Cleanup**: When compressing images to under 400 KB, delete the old larger uncompressed files so only the optimized versions exist.
- **Assets Organization in Article Folders**: Organize article-specific images inside dedicated subfolders within `assets/` named after the topic or service (e.g. `assets/local-ai-revolution/`, `assets/claude-orchestration/`, `assets/ai-developer-training/`, `assets/ai-agents-orchestration/`, `assets/advanced-ai-agents/`, `assets/generative-ai-transformation/`, `assets/guarantees/`, `assets/antigravity/`).
- **New Folders for New Articles**: When creating new articles or guides that introduce new images, always create dedicated subfolders in `assets/` and reference images using the organized subfolder path.
- **Reference Integrity**: Always update and maintain image references across all HTML files, Markdown documents (`md-html-sync/`), JSON schemas, and scripts whenever assets are added, moved, or reorganized.




