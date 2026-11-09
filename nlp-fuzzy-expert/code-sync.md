# Source Code and Test Suite Knowledge Synchronization Report

Generated on: 2026-11-09 13:54:35

## Summary
- **Target Files Evaluated**: 46
- **Files Updated**: 0
- **Tracked References**: 22

## File Status Details

| File | Status | Changed Lines |
| :--- | :---: | :--- |
| `pricing-calculator.js` | **IN SYNC** | - |
| `payment-schedule.js` | **IN SYNC** | - |
| `components.js` | **IN SYNC** | - |
| `python/generate_sitemap.py` | **IN SYNC** | - |
| `python/build_components.py` | **IN SYNC** | - |
| `python/delete_page.py` | **IN SYNC** | - |
| `python/generate_og_card.py` | **IN SYNC** | - |
| `python/check_links.py` | **IN SYNC** | - |
| `pricing-calculator/pricing-calculator-tier-1.js` | **IN SYNC** | - |
| `pricing-calculator/pricing-calculator-tier-2.js` | **IN SYNC** | - |
| `pricing-calculator/pricing-calculator-tier-3.js` | **IN SYNC** | - |
| `pricing-calculator/pricing-calculator-tier-4.js` | **IN SYNC** | - |
| `pricing-calculator/pricing-calculator-tier-5.js` | **IN SYNC** | - |
| `pricing-calculator/pricing-calculator-tier-6.js` | **IN SYNC** | - |
| `pricing-tests.js` | **IN SYNC** | - |
| `tests/js/test_payment_schedule.js` | **IN SYNC** | - |
| `pricing-calculator/pricing-calculator-tier-1-tests.js` | **IN SYNC** | - |
| `pricing-calculator/pricing-calculator-tier-2-tests.js` | **IN SYNC** | - |
| `pricing-calculator/pricing-calculator-tier-3-tests.js` | **IN SYNC** | - |
| `pricing-calculator/pricing-calculator-tier-4-tests.js` | **IN SYNC** | - |
| `pricing-calculator/pricing-calculator-tier-5-tests.js` | **IN SYNC** | - |
| `pricing-calculator/pricing-calculator-tier-6-tests.js` | **IN SYNC** | - |
| `tests/test_auto_healer.py` | **IN SYNC** | - |
| `tests/test_check_links.py` | **IN SYNC** | - |
| `tests/test_code_test_syncer.py` | **IN SYNC** | - |
| `tests/test_continuous_updates.py` | **IN SYNC** | - |
| `tests/test_core_target_documents.py` | **IN SYNC** | - |
| `tests/test_coverage_report.py` | **IN SYNC** | - |
| `tests/test_expert_system.py` | **IN SYNC** | - |
| `tests/test_fuzzy_engine.py` | **IN SYNC** | - |
| `tests/test_last_commit_report.py` | **IN SYNC** | - |
| `tests/test_master_orchestrator.py` | **IN SYNC** | - |
| `tests/test_nlp_engine.py` | **IN SYNC** | - |
| `tests/test_python_tools.py` | **IN SYNC** | - |
| `tests/test_reference_scanner.py` | **IN SYNC** | - |
| `tests/test_report_generator.py` | **IN SYNC** | - |
| `tests/test_sync_business_services.py` | **IN SYNC** | - |
| `tests/test_sync_claude_and_gemini.py` | **IN SYNC** | - |
| `tests/test_sync_guarantees.py` | **IN SYNC** | - |
| `tests/test_sync_personal_services.py` | **IN SYNC** | - |
| `tests/test_sync_pricing.py` | **IN SYNC** | - |
| `tests/test_sync_pricing_tiers_and_calculators.py` | **IN SYNC** | - |
| `tests/test_sync_publications_config_faqs.py` | **IN SYNC** | - |
| `tests/test_sync_tiers.py` | **IN SYNC** | - |
| `tests/test_unidirectional_sync.py` | **IN SYNC** | - |
| `tests/test_watcher.py` | **IN SYNC** | - |

## Synchronized Entities and Facts

| Target | Category | Domain | Field | Detail |
| :--- | :--- | :--- | :--- | :--- |
| `pricing-calculator.js` | source_code | pricing | `calculator_constants` | Synced ratePerSvc=300, basePrice=3000, extraSeat=300, seats=10 |
| `payment-schedule.js` | source_code | pricing | `payment_schedule_terms` | Verified PARITY_TIERS and trust terms with flagship=3000, indiv=300 |
| `components.js` | source_code | claude | `orchestration_pages` | Synced 12 orchestration guide page names |
| `python/generate_sitemap.py` | source_code | sitemap | `url_catalog` | Synced 12 business, 12 Claude, 12 Gemini URLs |
| `python/build_components.py` | source_code | components | `subdirs` | Synced active subdirectory folder paths |
| `python/delete_page.py` | source_code | page_cleaner | `subdirs` | Synced active subdirectory folder paths |
| `python/generate_og_card.py` | source_code | config | `og_card_author` | Synced author=Thomas Cherickal, title=Generative AI Consultant, brand=The Digital Futurist |
| `pricing-calculator/pricing-calculator-tier-1.js` | source_code | pricing | `tier_calculator_engine` | Verified tier calculator engine in pricing-calculator-tier-1.js |
| `pricing-calculator/pricing-calculator-tier-2.js` | source_code | pricing | `tier_calculator_engine` | Verified tier calculator engine in pricing-calculator-tier-2.js |
| `pricing-calculator/pricing-calculator-tier-3.js` | source_code | pricing | `tier_calculator_engine` | Verified tier calculator engine in pricing-calculator-tier-3.js |
| `pricing-calculator/pricing-calculator-tier-4.js` | source_code | pricing | `tier_calculator_engine` | Verified tier calculator engine in pricing-calculator-tier-4.js |
| `pricing-calculator/pricing-calculator-tier-5.js` | source_code | pricing | `tier_calculator_engine` | Verified tier calculator engine in pricing-calculator-tier-5.js |
| `pricing-calculator/pricing-calculator-tier-6.js` | source_code | pricing | `tier_calculator_engine` | Verified tier calculator engine in pricing-calculator-tier-6.js |
| `pricing-tests.js` | test | pricing | `test_constants` | Synced B_FLAG=3000, P_FLAG=3000, EXTRA=300, IND=300, SEATS=10 |
| `tests/js/test_payment_schedule.js` | test | pricing | `payment_schedule_tests` | Verified payment schedule test suite |
| `pricing-calculator/pricing-calculator-tier-1-tests.js` | test | pricing | `tier_calculator_tests` | Verified tier test suite in pricing-calculator-tier-1-tests.js |
| `pricing-calculator/pricing-calculator-tier-2-tests.js` | test | pricing | `tier_calculator_tests` | Verified tier test suite in pricing-calculator-tier-2-tests.js |
| `pricing-calculator/pricing-calculator-tier-3-tests.js` | test | pricing | `tier_calculator_tests` | Verified tier test suite in pricing-calculator-tier-3-tests.js |
| `pricing-calculator/pricing-calculator-tier-4-tests.js` | test | pricing | `tier_calculator_tests` | Verified tier test suite in pricing-calculator-tier-4-tests.js |
| `pricing-calculator/pricing-calculator-tier-5-tests.js` | test | pricing | `tier_calculator_tests` | Verified tier test suite in pricing-calculator-tier-5-tests.js |
| `pricing-calculator/pricing-calculator-tier-6-tests.js` | test | pricing | `tier_calculator_tests` | Verified tier test suite in pricing-calculator-tier-6-tests.js |
| `tests/test_sync_claude_and_gemini.py` | test | guides | `guide_paths` | Verified claude-training/ and google-training/ target paths in test |
