# Last Git Commit Report

> **Automated Commit Record** | Recorded on: `2026-11-06 17:48:27 UTC`

## Commit Overview

| Property | Value |
| :--- | :--- |
| **Commit Hash (Full)** | `dc4a59cb445ea3a45f6ddbb544be5405a4501a6d` |
| **Commit Hash (Short)** | `dc4a59c` |
| **Branch** | `main` |
| **Subject** | docs: record final commit output report and test coverage metrics for 4b44c38 |
| **Author** | Thomas Cherickal `<thomascherickal@gmail.com>` |
| **Author Date** | `2026-11-06T23:17:12+05:30` |
| **Committer** | Thomas Cherickal `<thomascherickal@gmail.com>` |
| **Committer Date** | `2026-11-06T23:17:12+05:30` |
| **Total Commits in Repo** | 491 |

## Quality Gate & Verification Summary

| Quality Gate / Test Dimension | Total Scope | Passed | Failed | Status | Runtime |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Python Unit Test Suite** | 126 tests across 23 modules | 126 | 0 | PASSED | 70.10s |
| **JavaScript Unit Test Suite** | 97 tests across 11 sections | 97 | 0 | PASSED | 0.03s |
| **NLP-Fuzzy-Expert Sync Engine** | 66 target documents | 66 | 0 | PASSED | 1.50s |
| **SSOT Data Immutability Guard** | 11 data files SHA-256 hash | 11 | 0 | PASSED | 0.001s |
| **Internal Link & Anchor Integrity** | 5655 links across 66 HTML files | 5655 | 0 | PASSED | 0.33s |
| **JSON-LD Schema Synchronization** | 62 schema files in `schema/` | 62 | 0 | PASSED | 0.15s |
| **Markdown Sync Verification** | 13 paired markdown files | 13 | 0 | PASSED | 0.10s |
| **Overall Quality Gate Status** | **7 of 7 Quality Gates Passed** | **100%** | **0** | **GREEN** | **PASS** |

## Python Unit Test Suite Details (`tests/`)

| Test Module | Total Tests | Passed | Failed | Errors | Skipped | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `test_auto_healer` | 13 | 13 | 0 | 0 | 0 | PASSED |
| `test_check_links` | 5 | 5 | 0 | 0 | 0 | PASSED |
| `test_code_test_syncer` | 7 | 7 | 0 | 0 | 0 | PASSED |
| `test_continuous_updates` | 2 | 2 | 0 | 0 | 0 | PASSED |
| `test_core_target_documents` | 11 | 11 | 0 | 0 | 0 | PASSED |
| `test_coverage_report` | 3 | 3 | 0 | 0 | 0 | PASSED |
| `test_expert_system` | 6 | 6 | 0 | 0 | 0 | PASSED |
| `test_fuzzy_engine` | 7 | 7 | 0 | 0 | 0 | PASSED |
| `test_last_commit_report` | 8 | 8 | 0 | 0 | 0 | PASSED |
| `test_master_orchestrator` | 4 | 4 | 0 | 0 | 0 | PASSED |
| `test_nlp_engine` | 6 | 6 | 0 | 0 | 0 | PASSED |
| `test_python_tools` | 13 | 13 | 0 | 0 | 0 | PASSED |
| `test_reference_scanner` | 3 | 3 | 0 | 0 | 0 | PASSED |
| `test_report_generator` | 2 | 2 | 0 | 0 | 0 | PASSED |
| `test_sync_business_services` | 3 | 3 | 0 | 0 | 0 | PASSED |
| `test_sync_claude_and_gemini` | 5 | 5 | 0 | 0 | 0 | PASSED |
| `test_sync_guarantees` | 5 | 5 | 0 | 0 | 0 | PASSED |
| `test_sync_personal_services` | 3 | 3 | 0 | 0 | 0 | PASSED |
| `test_sync_pricing` | 5 | 5 | 0 | 0 | 0 | PASSED |
| `test_sync_publications_config_faqs` | 2 | 2 | 0 | 0 | 0 | PASSED |
| `test_sync_tiers` | 3 | 3 | 0 | 0 | 0 | PASSED |
| `test_unidirectional_sync` | 7 | 7 | 0 | 0 | 0 | PASSED |
| `test_watcher` | 3 | 3 | 0 | 0 | 0 | PASSED |
| **Total (23 modules)** | **126** | **126** | **0** | **0** | **0** | **PASSED** |

<details>
<summary><strong>Itemized Python Test Cases (126 test cases)</strong></summary>

| Module | Test Method | Description | Status | Time |
| :--- | :--- | :--- | :---: | :---: |
| `test_auto_healer` | `test_commit_verified_state_clean` | Verifies commit_verified_state does nothing when working tree is clean. | PASSED | 0.0006s |
| `test_auto_healer` | `test_commit_verified_state_dirty` | Verifies commit_verified_state stages and commits when working tree has changes. | PASSED | 0.0010s |
| `test_auto_healer` | `test_compute_data_fingerprint` | Verifies that fingerprints are computed for all data/*.json files. | PASSED | 0.0003s |
| `test_auto_healer` | `test_compute_data_fingerprint_missing_directory` | Verifies compute_data_fingerprint handles non-existent directory. | PASSED | 0.0000s |
| `test_auto_healer` | `test_generate_failed_module_tests_report` | Verifies that failed_[module_name]_tests.report.md is created for failing test suites. | PASSED | 0.0010s |
| `test_auto_healer` | `test_generate_failed_tests_report_omits_dangerously_skip_permissions` | Verifies that failed_tests_report.md generates properly without --dangerously-skip-permissions. | PASSED | 0.0004s |
| `test_auto_healer` | `test_handle_test_verification_all_pass` | Verifies handle_test_verification_and_commit when all tests pass. | PASSED | 0.0010s |
| `test_auto_healer` | `test_handle_test_verification_few_failures` | Verifies handle_test_verification_and_commit when < 5 tests fail. | PASSED | 0.0008s |
| `test_auto_healer` | `test_invoke_agy_healer` | Verifies invoke_agy_healer constructs correct prompt and command. | PASSED | 0.0005s |
| `test_auto_healer` | `test_invoke_agy_healer_not_found` | Verifies invoke_agy_healer raises FileNotFoundError when agy is absent. | PASSED | 0.0002s |
| `test_auto_healer` | `test_parse_test_failures_variations` | Verifies parse_test_failures extracts modules and errors accurately. | PASSED | 0.0000s |
| `test_auto_healer` | `test_verify_data_untouched_detects_mutation` | Verifies that modified fingerprints raise a PermissionError. | PASSED | 0.0002s |
| `test_auto_healer` | `test_verify_data_untouched_success` | Verifies that identical fingerprints pass verification without raising. | PASSED | 0.0003s |
| `test_check_links` | `test_check_links_detects_broken_file_and_fragment` | Verifies check_links detects broken file references and missing fragments. | PASSED | 0.0012s |
| `test_check_links` | `test_check_links_resolves_root_and_directory` | Verifies check_links resolves root-relative paths and directory index.html. | PASSED | 0.0005s |
| `test_check_links` | `test_check_links_success` | Verifies check_links reports success on valid internal links. | PASSED | 0.0005s |
| `test_check_links` | `test_collect_file_targets` | Verifies extracting id and name attributes from HTML content. | PASSED | 0.0002s |
| `test_check_links` | `test_collect_file_targets_nonexistent` | Verifies nonexistent file returns empty set gracefully. | PASSED | 0.0001s |
| `test_code_test_syncer` | `test_compute_changed_lines` | Verifies line diff calculation for identical and modified content. | PASSED | 0.0010s |
| `test_code_test_syncer` | `test_discover_source_files` | Verifies that discover_source_files returns key Python and JS source scripts. | PASSED | 0.0002s |
| `test_code_test_syncer` | `test_discover_test_files` | Verifies that discover_test_files discovers pricing-tests.js and python unit tests. | PASSED | 0.0003s |
| `test_code_test_syncer` | `test_sync_all_generates_artifacts` | Verifies that running sync_all creates code_references.json and code-sync.md. | PASSED | 0.0040s |
| `test_code_test_syncer` | `test_sync_components_js` | Verifies that orchestration pages in components.js match Claude guide slugs. | PASSED | 0.0002s |
| `test_code_test_syncer` | `test_sync_pricing_calculator` | Verifies that sync_pricing_calculator inserts authoritative rates and formulas. | PASSED | 0.0002s |
| `test_code_test_syncer` | `test_sync_pricing_tests_js_constants` | Verifies that sync_pricing_tests_js injects or updates the authoritative facts block. | PASSED | 0.0002s |
| `test_continuous_updates` | `test_personal_services_continuous_name_updates` | Tests changing service name back and forth across cycles. | PASSED | 14.7805s |
| `test_continuous_updates` | `test_pricing_continuous_seat_and_duration_updates` | Tests changing included seats and duration back and forth across 2 cycles. | PASSED | 24.5285s |
| `test_core_target_documents` | `test_compliance_zero_emojis_and_zero_em_dashes` | Verifies zero emojis and zero em-dashes in manifest, contact.json, guarantees.json, and llms.txt. | PASSED | 0.0009s |
| `test_core_target_documents` | `test_contact_json_no_duplication` | Verifies data/contact.json avoids duplication by referencing config and pricing. | PASSED | 0.0004s |
| `test_core_target_documents` | `test_contact_json_schema_and_facts` | Verifies that data/contact.json exists and registers facts in knowledge base. | PASSED | 0.0003s |
| `test_core_target_documents` | `test_contact_page_rule_execution` | Verifies that ContactPagePricingRule matches contact.html and generates occurrences. | PASSED | 0.0007s |
| `test_core_target_documents` | `test_discover_target_files_includes_contact_and_llms` | Verifies that discover_target_files includes contact.html and llms.txt. | PASSED | 0.0007s |
| `test_core_target_documents` | `test_guarantees_json_no_duplication` | Verifies data/guarantees.json avoids duplication by referencing config, faqs, and pricing. | PASSED | 0.0003s |
| `test_core_target_documents` | `test_guarantees_json_schema_and_facts` | Verifies that data/guarantees.json exists and registers facts in knowledge base. | PASSED | 0.0003s |
| `test_core_target_documents` | `test_llms_txt_rule_execution` | Verifies that LLMsTxtRule matches llms.txt and generates occurrences. | PASSED | 0.0006s |
| `test_core_target_documents` | `test_references_dictionary_preserved_and_intact` | Verifies that master references_dictionary.json is preserved and intact. | PASSED | 0.0003s |
| `test_core_target_documents` | `test_references_manifest_structure_and_counts` | Verifies that references_manifest.json contains lightweight indexing. | PASSED | 0.0003s |
| `test_core_target_documents` | `test_rules_include_llms_txt_and_contact_rules` | Verifies that LLMsTxtRule and ContactPagePricingRule are registered. | PASSED | 0.0004s |
| `test_coverage_report` | `test_format_line_ranges` | Verifies formatting line numbers into concise readable ranges. | PASSED | 0.0004s |
| `test_coverage_report` | `test_generate_coverage_report_mocked` | Verifies report generation with mocked coverage engines. | PASSED | 0.0051s |
| `test_coverage_report` | `test_get_git_info` | Verifies retrieving git commit and branch info. | PASSED | 0.0025s |
| `test_expert_system` | `test_expert_system_edge_cases` | Verifies KnowledgeBase and InferenceEngine error handling and edge cases. | PASSED | 0.0008s |
| `test_expert_system` | `test_knowledge_base_loading` | Verifies KnowledgeBase loads facts from all data/*.json files. | PASSED | 0.0002s |
| `test_expert_system` | `test_record_occurrence` | Verifies occurrence tracking records file, page, section, and line. | PASSED | 0.0002s |
| `test_expert_system` | `test_rule_salience_ordering` | Verifies production rules are resolved and fired in salience order. | PASSED | 0.0002s |
| `test_expert_system` | `test_save_and_load_json_atomic` | Verifies atomic json writing and corruption-free reading. | PASSED | 0.0013s |
| `test_expert_system` | `test_save_json_blocks_data_dir_mutation` | Verifies save_json strictly raises PermissionError when targeting data/. | PASSED | 0.0002s |
| `test_fuzzy_engine` | `test_fuzzy_edge_cases` | Verifies edge cases: empty strings, single chars, identical whitespaces, unicode, and extreme mismatch. | PASSED | 0.0004s |
| `test_fuzzy_engine` | `test_fuzzy_matcher_classification` | Verifies holistic composite score and linguistic term classification. | PASSED | 0.0003s |
| `test_fuzzy_engine` | `test_membership_boundary_cases` | Verifies boundary clamping on triangular and trapezoidal membership functions. | PASSED | 0.0000s |
| `test_fuzzy_engine` | `test_membership_functions` | Verifies triangular and trapezoidal membership evaluations. | PASSED | 0.0000s |
| `test_fuzzy_engine` | `test_ngram_similarity` | Verifies character 3-gram Jaccard overlap similarity. | PASSED | 0.0000s |
| `test_fuzzy_engine` | `test_string_distance_and_ratios` | Verifies exact Levenshtein distance and normalized similarity ratios. | PASSED | 0.0000s |
| `test_fuzzy_engine` | `test_token_sort_and_set_ratios` | Verifies token sort and token set robust fuzzy comparisons. | PASSED | 0.0002s |
| `test_last_commit_report` | `test_generate_report_creates_file` | Verifies report is generated and contains accurate HEAD information. | PASSED | 0.5487s |
| `test_last_commit_report` | `test_post_commit_hook_exists_and_executable` | Verifies that post-commit hooks exist in python/hooks and .git/hooks and are executable. | PASSED | 0.5370s |
| `test_last_commit_report` | `test_report_includes_javascript_test_suite_details` | Verifies report includes the JavaScript Unit Test Suite Details table. | PASSED | 0.5383s |
| `test_last_commit_report` | `test_report_includes_link_schema_verification` | Verifies report includes the Link, Schema & Markdown Verification Details table. | PASSED | 0.5554s |
| `test_last_commit_report` | `test_report_includes_nlp_fuzzy_expert_report` | Verifies report includes the NLP-Fuzzy-Expert Knowledge Synchronization Report. | PASSED | 0.5467s |
| `test_last_commit_report` | `test_report_includes_python_test_suite_details` | Verifies report includes the Python Unit Test Suite Details table and itemized list. | PASSED | 0.5023s |
| `test_last_commit_report` | `test_report_includes_quality_gate_summary` | Verifies report includes the Quality Gate & Verification Summary table. | PASSED | 0.5480s |
| `test_last_commit_report` | `test_report_zero_emojis_and_zero_em_dashes` | Strictly verifies zero emojis and zero em-dashes throughout report. | PASSED | 0.5801s |
| `test_master_orchestrator` | `test_discover_target_files` | Verifies discover_target_files finds root HTMLs, JS, and subdirectories. | PASSED | 0.0007s |
| `test_master_orchestrator` | `test_markdown_reports_generated` | Verifies that all 11 domain markdown reports exist under nlp-fuzzy-expert/. | PASSED | 0.0014s |
| `test_master_orchestrator` | `test_sync_all_and_aggregate_dictionary` | Verifies full sync produces references_dictionary.json with 200+ references. | PASSED | 0.1844s |
| `test_master_orchestrator` | `test_sync_file_single` | Verifies sync_file executes cleanly for a target file. | PASSED | 0.2002s |
| `test_nlp_engine` | `test_domain_classification` | Verifies context snippet classification into semantic domains. | PASSED | 0.0002s |
| `test_nlp_engine` | `test_duration_and_seats_edge_cases` | Verifies parsing of various duration and seat expressions. | PASSED | 0.0004s |
| `test_nlp_engine` | `test_entity_extraction` | Verifies semantic entity recognition for MONEY, DURATION, and SEATS. | PASSED | 0.0001s |
| `test_nlp_engine` | `test_highlight_tagging_and_stripping` | Verifies highlight tagging and idempotent clean stripping. | PASSED | 0.0001s |
| `test_nlp_engine` | `test_nlp_edge_cases` | Verifies edge cases for empty strings, punctuation only, and missing numbers. | PASSED | 0.0004s |
| `test_nlp_engine` | `test_text_normalization_and_tokenization` | Verifies HTML unescaping and whitespace normalization. | PASSED | 0.0000s |
| `test_python_tools` | `test_build_components_build_execution` | Verifies build execution runs across repository without raising errors. | PASSED | 4.0878s |
| `test_python_tools` | `test_build_components_extract_nav_and_footer` | Verifies extracting nav and footer tags and handling missing tags. | PASSED | 0.0002s |
| `test_python_tools` | `test_build_md_files_converters` | Verifies build_md_files document converters generate clean markdown. | PASSED | 0.0946s |
| `test_python_tools` | `test_delete_page_detects_nonexistent_target` | Verifies delete_page aborts gracefully when target does not exist. | PASSED | 0.0084s |
| `test_python_tools` | `test_delete_page_dry_run_on_existing_page` | Verifies delete_page dry run evaluates all targets without deleting any files. | PASSED | 0.0084s |
| `test_python_tools` | `test_generate_metrics_execution` | Verifies generate_metrics creates or updates metrics.md successfully. | PASSED | 0.4629s |
| `test_python_tools` | `test_generate_metrics_format_bytes` | Verifies format_bytes human-readable byte size representations. | PASSED | 0.0001s |
| `test_python_tools` | `test_generate_metrics_git_output` | Verifies get_git_output executes git command or returns empty string on error. | PASSED | 0.0063s |
| `test_python_tools` | `test_generate_sitemap_core_pages_exist` | Verifies core navigation pages defined in generate_sitemap exist in repository. | PASSED | 0.0002s |
| `test_python_tools` | `test_generate_sitemap_execution` | Verifies generate_sitemap creates valid XML output. | PASSED | 0.1095s |
| `test_python_tools` | `test_md_html_sync_helpers` | Verifies hash generation, text cleaning, and state serialization. | PASSED | 0.0004s |
| `test_python_tools` | `test_sync_schemas_check_only_mode` | Verifies check_only mode evaluates all pages without modifying schemas. | PASSED | 1.4123s |
| `test_python_tools` | `test_sync_schemas_target_filename_mapping` | Verifies schema target filename mapping conventions. | PASSED | 0.0002s |
| `test_reference_scanner` | `test_reference_files_include_scanned_and_rule_records` |  | PASSED | 0.0818s |
| `test_reference_scanner` | `test_scan_records_schema_and_is_read_only` |  | PASSED | 0.0014s |
| `test_reference_scanner` | `test_terms_cover_all_data_domains` |  | PASSED | 0.0002s |
| `test_report_generator` | `test_classify_rule_field_domains` | Verifies field classification assigns correct domain keys. | PASSED | 0.0000s |
| `test_report_generator` | `test_domain_mappings_completeness` | Verifies that all 11 expected domains have corresponding reference and source files. | PASSED | 0.0000s |
| `test_sync_business_services` | `test_facts_loaded` | Verifies all 12 business services facts are in knowledge base. | PASSED | 0.0003s |
| `test_sync_business_services` | `test_references_and_markdown` | Verifies business_services_references.json and business-services.md exist. | PASSED | 0.1601s |
| `test_sync_business_services` | `test_sync_business_cards_and_curriculum` | Verifies sync_all updates all-business-services.html and business-ai-training.html. | PASSED | 0.3078s |
| `test_sync_claude_and_gemini` | `test_claude_guides_completeness` | Verifies all 12 Claude guides exist on disk and have proper structure. | PASSED | 0.0003s |
| `test_sync_claude_and_gemini` | `test_deepmind_guide_facts_and_sync` | Verifies Google DeepMind guide facts and synchronization in target files. | PASSED | 0.0003s |
| `test_sync_claude_and_gemini` | `test_facts_loaded` | Verifies guides facts exist in knowledge base. | PASSED | 0.0002s |
| `test_sync_claude_and_gemini` | `test_references_and_markdown` | Verifies claude_references.json, gemini_references.json, claude.md, and gemini.md exist. | PASSED | 0.1854s |
| `test_sync_claude_and_gemini` | `test_sync_guides_titles` | Verifies sync_all synchronizes guide titles in HTML files. | PASSED | 0.1887s |
| `test_sync_guarantees` | `test_guarantees_internal_links_across_repo` | Verifies all internal links point to guarantees.html and none to guarantee.html. | PASSED | 0.0011s |
| `test_sync_guarantees` | `test_guarantees_page_content_in_sync` | Verifies guarantees.html content reflects authoritative pricing facts. | PASSED | 0.0005s |
| `test_sync_guarantees` | `test_guarantees_page_exists_and_old_page_removed` | Verifies guarantees.html exists and obsolete guarantee.html is deleted. | PASSED | 0.0004s |
| `test_sync_guarantees` | `test_guarantees_rule_in_expert_system` | Verifies GuaranteesPageRule is registered in the production rules queue. | PASSED | 0.0002s |
| `test_sync_guarantees` | `test_guarantees_tracked_in_references_dictionary` | Verifies guarantees.html occurrences are tracked in references_dictionary.json. | PASSED | 0.0936s |
| `test_sync_personal_services` | `test_facts_loaded` | Verifies all 12 personal services facts are in knowledge base. | PASSED | 0.0003s |
| `test_sync_personal_services` | `test_references_and_markdown` | Verifies personal_services_references.json and personal-services.md exist. | PASSED | 0.1868s |
| `test_sync_personal_services` | `test_sync_personal_cards_and_curriculum` | Verifies sync_all updates all-personal-services.html and personal-ai-training.html. | PASSED | 0.1878s |
| `test_sync_pricing` | `test_pricing_authoritative_facts` | Verifies authoritative values are properly extracted from data/pricing.json. | PASSED | 0.0003s |
| `test_sync_pricing` | `test_pricing_calculator_js_updates` | Verifies pricing-calculator.js contains the synchronized variables. | PASSED | 0.1863s |
| `test_sync_pricing` | `test_pricing_html_updates` | Verifies pricing.html reflects updated pricing and seats. | PASSED | 0.1806s |
| `test_sync_pricing` | `test_pricing_references_and_markdown` | Verifies pricing_references.json and pricing.md exist and are populated. | PASSED | 0.1778s |
| `test_sync_pricing` | `test_pricing_zero_and_edge_multipliers` | Verifies pricing calculations with edge cases and extreme participant volumes. | PASSED | 0.0003s |
| `test_sync_publications_config_faqs` | `test_facts_loaded` | Verifies publications, config, and FAQs facts exist in knowledge base. | PASSED | 0.0002s |
| `test_sync_publications_config_faqs` | `test_sync_execution` | Verifies sync_all generates reference dictionaries and markdown files. | PASSED | 0.1795s |
| `test_sync_tiers` | `test_facts_loaded` | Verifies 6 tiers facts exist in knowledge base. | PASSED | 0.0003s |
| `test_sync_tiers` | `test_references_and_markdown` | Verifies tiers_references.json and tiers.md exist. | PASSED | 0.1984s |
| `test_sync_tiers` | `test_sync_tiers_pricing_and_calc` | Verifies sync_all updates parity tiers in pricing.html and pricing-calculator.js. | PASSED | 0.2897s |
| `test_unidirectional_sync` | `test_code_test_syncer_strictly_excludes_data_dir` | Verifies that CodeTestSynchronizer never targets files inside data/. | PASSED | 0.0008s |
| `test_unidirectional_sync` | `test_data_changes_propagate_outward_to_website_in_sandbox` | Verifies that data changes propagate outward to the website (unidirectional). | PASSED | 15.5487s |
| `test_unidirectional_sync` | `test_discover_target_files_strictly_excludes_data_dir` | Verifies that discover_target_files never includes any file inside data/. | PASSED | 0.0009s |
| `test_unidirectional_sync` | `test_inference_engine_blocks_data_file_processing` | Verifies InferenceEngine.process_file raises PermissionError for any data/*.json file. | PASSED | 0.0005s |
| `test_unidirectional_sync` | `test_save_json_blocks_mutation_of_data_dir` | Verifies save_json raises PermissionError when targeted at data/*.json. | PASSED | 0.0003s |
| `test_unidirectional_sync` | `test_verify_unidirectional_sync_helper` | Verifies verify_unidirectional_sync validates cleanly and flags violations. | PASSED | 0.0003s |
| `test_unidirectional_sync` | `test_website_changes_do_not_propagate_to_data_dir` | Verifies that manual/external edits to HTML files do not alter data/*.json. | PASSED | 1.5838s |
| `test_watcher` | `test_check_for_changes_modified_created_deleted` | Verifies detecting created, modified, and deleted data files. | PASSED | 0.0010s |
| `test_watcher` | `test_get_data_files_and_snapshot` | Verifies retrieving data files and taking snapshots. | PASSED | 0.0005s |
| `test_watcher` | `test_handle_signal` | Verifies handle_signal sets running flag to False. | PASSED | 0.0005s |

</details>

## JavaScript Unit Test Suite Details (`pricing-tests.js`)

| Test Section | Tests | Passed | Failed | Status |
| :--- | :---: | :---: | :---: | :---: |
| Data Integrity & Metadata Tests | 5 | 5 | 0 | PASSED |
| Nothing Selected (0 USD) Tests | 10 | 10 | 0 | PASSED |
| Individual Pick-Services Tests | 12 | 12 | 0 | PASSED |
| Business Pick-Services Tests | 9 | 9 | 0 | PASSED |
| Business Extra Participant Scaling Tests | 5 | 5 | 0 | PASSED |
| Flagship Package Tests | 8 | 8 | 0 | PASSED |
| Custom Amount Package Tests | 5 | 5 | 0 | PASSED |
| Defensive Edge Cases & Malformed Inputs Tests | 3 | 3 | 0 | PASSED |
| DOM Controller Simulated Live-Update Tests | 21 | 21 | 0 | PASSED |
| Components Navigation & UI Tests | 14 | 14 | 0 | PASSED |
| Mermaid Dynamic MD Link Transformation Tests | 5 | 5 | 0 | PASSED |
| **Total (11 Sections)** | **97** | **97** | **0** | **PASSED** |

## NLP-Fuzzy-Expert Knowledge Synchronization Report

- **Engine**: nlp-fuzzy-expert v1.0.0
- **Architecture**: Expert System (Forward Chaining) + Fuzzy Logic + NLP
- **Data Flow Direction**: Strictly Unidirectional (`data/*.json` -> HTML / JS / MD / Code)
- **Total Tracked Occurrences**: 20,341

### Master Reference Occurrences by Domain

| Knowledge Domain | Tracked Occurrences | Primary Data Source | Domain Sync Report |
| :--- | :---: | :--- | :--- |
| `config` | 4,494 | `data/config.json` | [`nlp-fuzzy-expert/config.md`](nlp-fuzzy-expert/config.md) |
| `pricing` | 3,582 | `data/pricing.json` | [`nlp-fuzzy-expert/pricing.md`](nlp-fuzzy-expert/pricing.md) |
| `publications` | 3,316 | `data/publications.json` | [`nlp-fuzzy-expert/publications.md`](nlp-fuzzy-expert/publications.md) |
| `tiers` | 2,252 | `data/tiers.json` | [`nlp-fuzzy-expert/tiers.md`](nlp-fuzzy-expert/tiers.md) |
| `contact` | 2,043 | `data/contact.json` | [`nlp-fuzzy-expert/contact.md`](nlp-fuzzy-expert/contact.md) |
| `business_services` | 1,588 | `data/business_services.json` | [`nlp-fuzzy-expert/business-services.md`](nlp-fuzzy-expert/business-services.md) |
| `claude` | 1,207 | `data/claude.json` | [`nlp-fuzzy-expert/claude.md`](nlp-fuzzy-expert/claude.md) |
| `gemini` | 1,207 | `data/gemini.json` | [`nlp-fuzzy-expert/gemini.md`](nlp-fuzzy-expert/gemini.md) |
| `personal_services` | 422 | `data/personal_services.json` | [`nlp-fuzzy-expert/personal-services.md`](nlp-fuzzy-expert/personal-services.md) |
| `guarantees` | 156 | `data/guarantees.json` | [`nlp-fuzzy-expert/guarantees.md`](nlp-fuzzy-expert/guarantees.md) |
| `faqs` | 74 | `data/faqs.json` | [`nlp-fuzzy-expert/faqs.md`](nlp-fuzzy-expert/faqs.md) |
| **Master Dictionary Total** | **20,341** | `data/*.json` (11 domains) | [`references_dictionary.json`](nlp-fuzzy-expert/references_dictionary.json) |

### Core Target Documents Synchronization Status

| Target Document | Status | Changed Lines | Rules Fired | Synchronized Fields |
| :--- | :---: | :---: | :--- | :--- |
| `all-business-services.html` | IN SYNC | - | BusinessServicesCardsRule | Business service titles, prices, cards |
| `all-personal-services.html` | IN SYNC | - | PersonalServicesCardsRule | Personal service titles, prices, cards |
| `business-ai-training.html` | IN SYNC | - | BusinessAITrainingRule | Flagship duration, seats, pricing |
| `contact.html` | IN SYNC | - | ContactPagePricingRule | Personal training quote, tier base rate |
| `faqs.html` | IN SYNC | - | FAQsRule | Included seats, pricing terms |
| `guarantees.html` | IN SYNC | - | GuaranteesPageRule | Flagship terms, walkaway guarantee |
| `index.html` | IN SYNC | - | IndexPageRule | Hero pricing, flagship packages, stats |
| `llms.txt` | IN SYNC | - | LLMsTxtRule | Flagship training rates, parity tiers, service catalog |
| `personal-ai-training.html` | IN SYNC | - | PersonalAITrainingRule | Flagship duration, 1-on-1 model |
| `pricing-calculator.js` | IN SYNC | - | PricingCalculatorScriptRule | Pricing tiers, seat multipliers, rates |
| `pricing.html` | IN SYNC | - | PricingHeroPackagesRule | Parity tables, pricing cards, formulas |

## Link, Schema & Markdown Verification Details

| Verification Scope | Evaluated Target | Passed / Matched | Broken / Errors | Verification Status |
| :--- | :--- | :---: | :---: | :---: |
| **Internal Links & Anchors** | 5655 links across 66 HTML files | 5655 | 0 | VERIFIED CLEAN |
| **Structured JSON-LD Schemas** | 62 schema files in `schema/` | 62 | 0 | 100% IN SYNC |
| **Markdown Page Mirrors** | 13 paired markdown files in `md-html-sync/` | 13 | 0 | 100% IN SYNC |
| **SSOT Data Immutability** | 11 authoritative JSON files in `data/` | 11 | 0 | SHA-256 VERIFIED |

## Changed Files & Status

```text
M	LAST-COMMIT-REPORT.md
M	TEST_COVERAGE_REPORT.md
M	md-html-sync/.sync-state.json
M	metrics.md
M	nlp-fuzzy-expert/business_services_references.json
M	nlp-fuzzy-expert/claude_references.json
M	nlp-fuzzy-expert/code-sync.md
M	nlp-fuzzy-expert/code_references.json
M	nlp-fuzzy-expert/config_references.json
M	nlp-fuzzy-expert/contact_references.json
M	nlp-fuzzy-expert/faqs_references.json
M	nlp-fuzzy-expert/gemini_references.json
M	nlp-fuzzy-expert/personal-services.md
M	nlp-fuzzy-expert/personal_services_references.json
M	nlp-fuzzy-expert/pricing.md
M	nlp-fuzzy-expert/pricing_references.json
M	nlp-fuzzy-expert/publications_references.json
M	nlp-fuzzy-expert/references_dictionary.json
M	nlp-fuzzy-expert/references_manifest.json
M	nlp-fuzzy-expert/tiers.md
M	nlp-fuzzy-expert/tiers_references.json
```

## Commit Output & Stat Summary

```text
commit dc4a59cb445ea3a45f6ddbb544be5405a4501a6d
Author: Thomas Cherickal <thomascherickal@gmail.com>
Date:   Fri Nov 6 23:17:12 2026 +0530

    docs: record final commit output report and test coverage metrics for 4b44c38

 LAST-COMMIT-REPORT.md                              | 226 ++++++++++++++-------
 TEST_COVERAGE_REPORT.md                            |   2 +-
 md-html-sync/.sync-state.json                      |  58 +++---
 metrics.md                                         |  18 +-
 nlp-fuzzy-expert/business_services_references.json |   2 +-
 nlp-fuzzy-expert/claude_references.json            |   2 +-
 nlp-fuzzy-expert/code-sync.md                      |   2 +-
 nlp-fuzzy-expert/code_references.json              |   2 +-
 nlp-fuzzy-expert/config_references.json            |   2 +-
 nlp-fuzzy-expert/contact_references.json           |   2 +-
 nlp-fuzzy-expert/faqs_references.json              |   2 +-
 nlp-fuzzy-expert/gemini_references.json            |   2 +-
 nlp-fuzzy-expert/personal-services.md              |   2 +-
 nlp-fuzzy-expert/personal_services_references.json |   2 +-
 nlp-fuzzy-expert/pricing.md                        |   2 +-
 nlp-fuzzy-expert/pricing_references.json           |   2 +-
 nlp-fuzzy-expert/publications_references.json      |   2 +-
 nlp-fuzzy-expert/references_dictionary.json        |   2 +-
 nlp-fuzzy-expert/references_manifest.json          |   2 +-
 nlp-fuzzy-expert/tiers.md                          |   2 +-
 nlp-fuzzy-expert/tiers_references.json             |   2 +-
 21 files changed, 207 insertions(+), 131 deletions(-)
```

## Verification & Compliance

- Zero emojis: Strictly enforced.
- Zero em-dashes: Strictly enforced (spaced hyphens ' - ' or plain hyphens '-' used).
- Overwrite mode: Automatically updated on every new commit via Git post-commit hook.
