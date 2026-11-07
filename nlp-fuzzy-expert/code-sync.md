# Source Code and Test Suite Knowledge Synchronization Report

Generated on: 2026-11-07 11:28:33

## Summary
- **Target Files Evaluated**: 31
- **Files Updated**: 0
- **Tracked References**: 8

## File Status Details

| File | Status | Changed Lines |
| :--- | :---: | :--- |
| `pricing-calculator.js` | **IN SYNC** | - |
| `components.js` | **IN SYNC** | - |
| `python/generate_sitemap.py` | **IN SYNC** | - |
| `python/build_components.py` | **IN SYNC** | - |
| `python/delete_page.py` | **IN SYNC** | - |
| `python/generate_og_card.py` | **IN SYNC** | - |
| `python/check_links.py` | **IN SYNC** | - |
| `pricing-tests.js` | **IN SYNC** | - |
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
| `tests/test_sync_publications_config_faqs.py` | **IN SYNC** | - |
| `tests/test_sync_tiers.py` | **IN SYNC** | - |
| `tests/test_unidirectional_sync.py` | **IN SYNC** | - |
| `tests/test_watcher.py` | **IN SYNC** | - |

## Synchronized Entities and Facts

| Target | Category | Domain | Field | Detail |
| :--- | :--- | :--- | :--- | :--- |
| `pricing-calculator.js` | source_code | pricing | `calculator_constants` | Synced ratePerSvc=300, basePrice=3000, extraSeat=300, seats=10 |
| `components.js` | source_code | claude | `orchestration_pages` | Synced 12 orchestration guide page names |
| `python/generate_sitemap.py` | source_code | sitemap | `url_catalog` | Synced 12 business, 12 Claude, 12 Gemini URLs |
| `python/build_components.py` | source_code | components | `subdirs` | Synced active subdirectory folder paths |
| `python/delete_page.py` | source_code | page_cleaner | `subdirs` | Synced active subdirectory folder paths |
| `python/generate_og_card.py` | source_code | config | `og_card_author` | Synced author=Thomas Cherickal, title=Generative AI Consultant, brand=The Digital Futurist |
| `pricing-tests.js` | test | pricing | `test_constants` | Synced B_FLAG=3000, P_FLAG=3000, EXTRA=300, IND=300, SEATS=10 |
| `tests/test_sync_claude_and_gemini.py` | test | guides | `guide_paths` | Verified claude-training/ and google-training/ target paths in test |
