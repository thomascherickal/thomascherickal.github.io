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

---

## Test Modules Overview

The test suite contains 11 dedicated test modules covering 36 individual test cases:

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

## Pre-Commit Verification

The test suite is integrated into `.git/hooks/pre-commit`. Any commit attempt will run `python3 python/run_tests.py` and block the commit if any test fails.

