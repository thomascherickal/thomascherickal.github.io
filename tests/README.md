# Unit Test Suite (`tests/`)

Comprehensive unit test suite validating the NLP-Fuzzy-Expert knowledge synchronization system, data consistency, rule transformations, and continuous update cycles.

Strict Repo Compliance:
- Zero emojis.
- Zero em-dashes (spaced hyphens ' - ' or plain hyphens '-' used).
- 100% pass rate required prior to commit.

---

## Test Execution

### Recommended Command
```bash
python3 python/run_tests.py
```

### Standard Unittest Discovery
```bash
python3 -m unittest discover -s tests -v
```

### JavaScript Test Suite Execution
```bash
# Run all 78 tests via master runner
node pricing-tests.js

# Or run via modular runner
node tests/js/runner.js

# Or run individual modules directly
node tests/js/test_dom_controller.js
node tests/js/test_data_integrity.js
```

---

## Python Test Modules Overview

The Python test suite in `tests/` verifies the deterministic SSOT pipeline, pricing/policy invariants, HTML UI structure across all 85+ pages, link integrity, and Python build/reporting tools. Any failing test suite writes its failure report exclusively to `failed-tests/` and deletes it automatically once passing.

| Module | Focus Area | Key Verifications |
| :--- | :--- | :--- |
| `test_ssot.py` | Deterministic SSOT Pipeline | `python/ssot/load.py`, `format.py`, `validate.py`, `render.py` idempotency and `--check` verification. |
| `test_pricing_and_policy_invariants.py` | Pricing & Policy Invariants | Flagship and Service Pick seat caps (10 included, max 20), 10x10% milestones, Zero Charge Training Materials, Apache 2.0 terms. |
| `test_html_ui.py` | HTML UI & Accessibility Suite | 101 tests across all 85+ standalone HTML pages (SEO meta, single navbar, footer, accessibility, back-to-top button, CTAs, schemas, fault injection). |
| `test_check_links.py` | Link & Anchor Integrity | Internal file link and fragment target verification across all HTML pages. |
| `test_python_tools.py` | Build & Sync Tooling | `build_components.py`, `md_html_sync.py`, `sync_schemas.py`, `delete_page.py`, `generate_metrics.py`, `run_tests.py`. |
| `test_coverage_report.py` | Coverage Reporting | Python and Node.js V8 coverage parsing and `LAST-COMMIT-REPORT.md` appending. |
| `test_last_commit_report.py` | Commit Reporting | Structured Python, JS, and HTML UI test report generation in `LAST-COMMIT-REPORT.md`. |

---

## JavaScript Test Modules Overview (`tests/js/`)

The modular JavaScript test suite contains 9 dedicated modules covering 78 individual test cases:

| Module | Focus Area | Tests | Key Verifications |
| :--- | :--- | :--- | :--- |
| `test_data_integrity.js` | Data Integrity & Metadata | 5 | 6 parity tiers, multipliers, discounts, Tier 6 custom flag, 12 unique service slugs. |
| `test_zero_selection.js` | Zero Selection ($0 USD) | 10 | $0 price on empty selection across modes, participants, and all 6 parity tiers. |
| `test_individual_services.js` | Individual Pick-Services | 12 | Tier discounts, seat enforcement (strictly 1 person), flagship nudge thresholds. |
| `test_business_services.js` | Business Pick-Services | 9 | 10 included seats, parity discounts across tiers, flagship nudge threshold ($1,000). |
| `test_extra_participants.js` | Extra Seat Scaling | 5 | Scaling over 10 included seats (+ $100/seat scaled across parity tiers). |
| `test_flagship_packages.js` | Flagship Packages | 8 | Business and personal flagship rates and savings across all 6 parity tiers. |
| `test_custom_package.js` | Custom Amount Packages | 5 | Custom scope pricing, negative input clamping to $0, parity percentage scaling. |
| `test_edge_cases.js` | Defensive Edge Cases | 3 | Null/undefined parameters, invalid/out-of-range tier IDs, NaN participant normalization. |
| `test_dom_controller.js` | DOM Controller & UI | 21 | Mock DOM events, interactive mode switching, service checkboxes, live pane updates. |

---

## Pre-Commit Verification

The test suite is integrated into `.git/hooks/pre-commit`. Any commit attempt will run `python3 python/run_tests.py` and block the commit if any test fails.

