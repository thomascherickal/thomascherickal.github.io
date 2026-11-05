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

The Python test suite contains 15 dedicated test modules covering 63 individual test cases:

| Module | Focus Area | Key Verifications |
| :--- | :--- | :--- |
| `test_fuzzy_engine.py` | Fuzzy Logic Engine | Triangular & trapezoidal membership, Levenshtein distance, Token Sort & Set ratios, Jaccard similarity, linguistic classifications. |
| `test_nlp_engine.py` | NLP Normalization | HTML normalization, entity extraction (MONEY, DURATION, SEATS), context extraction, domain classification. |
| `test_expert_system.py` | Expert System Core | KnowledgeBase reloading, atomic proposition storage, rule salience ordering, occurrence tracking. |
| `test_master_orchestrator.py` | Synchronizer Execution | Target discovery across root and subdirectories, master dictionary aggregation, markdown report generation, single file syncing. |
| `test_sync_pricing.py` | Pricing Rules | Authoritative facts extraction, pricing-calculator.js parameter updates, HTML card prices. |
| `test_sync_business_services.py` | Business Services | Business catalog names, 12 business service cards, business flagship curriculum sync. |
| `test_sync_personal_services.py` | Personal Services | Personal catalog names, 12 personal service cards, personal flagship curriculum sync. |
| `test_sync_tiers.py` | Parity Tiers | 6-tier multiplier consistency, country parity tables in pricing.html and pricing-calculator.js. |
| `test_sync_claude_and_gemini.py` | Architecture Guides | Claude and Gemini guide titles, file listings, and cross-references. |
| `test_sync_publications_config_faqs.py`| Publications & FAQs | Publications directory sync, site configuration branding, FAQ policies. |
| `test_continuous_updates.py` | Cyclic & Mutation Tests | Mutates JSON data in data/ back and forth across cycles, verifying exact propagation and idempotency. |

---

## JavaScript Test Modules Overview (`tests/js/`)

The modular JavaScript test suite contains 9 dedicated modules covering 78 individual test cases:

| Module | Focus Area | Tests | Key Verifications |
| :--- | :--- | :--- | :--- |
| `test_data_integrity.js` | Data Integrity & Metadata | 5 | 6 parity tiers, multipliers, discounts, Tier 6 custom flag, 12 unique service slugs. |
| `test_zero_selection.js` | Zero Selection ($0 USD) | 10 | $0 price on empty selection across modes, participants, and all 6 parity tiers. |
| `test_individual_services.js` | Individual Pick-Services | 12 | Tier discounts, seat enforcement (strictly 1 person), flagship nudge thresholds. |
| `test_business_services.js` | Business Pick-Services | 9 | 10 included seats, parity discounts across tiers, flagship nudge threshold ($3,000). |
| `test_extra_participants.js` | Extra Seat Scaling | 5 | Scaling over 10 included seats (+ $300/seat scaled across parity tiers). |
| `test_flagship_packages.js` | Flagship Packages | 8 | Business and personal flagship rates and savings across all 6 parity tiers. |
| `test_custom_package.js` | Custom Amount Packages | 5 | Custom scope pricing, negative input clamping to $0, parity percentage scaling. |
| `test_edge_cases.js` | Defensive Edge Cases | 3 | Null/undefined parameters, invalid/out-of-range tier IDs, NaN participant normalization. |
| `test_dom_controller.js` | DOM Controller & UI | 21 | Mock DOM events, interactive mode switching, service checkboxes, live pane updates. |

---

## Pre-Commit Verification

The test suite is integrated into `.git/hooks/pre-commit`. Any commit attempt will run `python3 python/run_tests.py` and block the commit if any test fails.

