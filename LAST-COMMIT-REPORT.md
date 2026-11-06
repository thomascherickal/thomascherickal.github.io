# Last Git Commit Report

> **Automated Commit Record** | Recorded on: `2026-11-06 18:43:58 UTC`

## Commit Overview

| Property | Value |
| :--- | :--- |
| **Commit Hash (Full)** | `46b0c7dcd66e9db371b15250845fbe486780593a` |
| **Commit Hash (Short)** | `46b0c7d` |
| **Branch** | `main` |
| **Subject** | perf(sync): optimize synchronizer and commit pipeline with manifest files |
| **Author** | Thomas Cherickal `<thomascherickal@gmail.com>` |
| **Author Date** | `2026-11-07T00:13:32+05:30` |
| **Committer** | Thomas Cherickal `<thomascherickal@gmail.com>` |
| **Committer Date** | `2026-11-07T00:13:32+05:30` |
| **Total Commits in Repo** | 495 |

## Quality Gate & Verification Summary

| Quality Gate / Test Dimension | Total Scope | Passed | Failed | Status | Runtime |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Python Unit Test Suite** | 128 tests across 23 modules | 128 | 0 | PASSED | 21.90s |
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
| `test_core_target_documents` | 12 | 12 | 0 | 0 | 0 | PASSED |
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
| `test_watcher` | 4 | 4 | 0 | 0 | 0 | PASSED |
| **Total (23 modules)** | **128** | **128** | **0** | **0** | **0** | **PASSED** |

<details>
<summary><strong>Itemized Python Test Cases (128 test cases)</strong></summary>

| Module | Test Method | Description | Status | Time |
| :--- | :--- | :--- | :---: | :---: |
| `test_auto_healer` | `test_commit_verified_state_clean` | Verifies commit_verified_state does nothing when working tree is clean. | PASSED | 0.0007s |
| `test_auto_healer` | `test_commit_verified_state_dirty` | Verifies commit_verified_state stages and commits when working tree has changes. | PASSED | 0.0005s |
| `test_auto_healer` | `test_compute_data_fingerprint` | Verifies that fingerprints are computed for all data/*.json files. | PASSED | 0.0002s |
| `test_auto_healer` | `test_compute_data_fingerprint_missing_directory` | Verifies compute_data_fingerprint handles non-existent directory. | PASSED | 0.0000s |
| `test_auto_healer` | `test_generate_failed_module_tests_report` | Verifies that failed_[module_name]_tests.report.md is created for failing test suites. | PASSED | 0.0006s |
| `test_auto_healer` | `test_generate_failed_tests_report_omits_dangerously_skip_permissions` | Verifies that failed_tests_report.md generates properly without --dangerously-skip-permissions. | PASSED | 0.0003s |
| `test_auto_healer` | `test_handle_test_verification_all_pass` | Verifies handle_test_verification_and_commit when all tests pass. | PASSED | 0.0009s |
| `test_auto_healer` | `test_handle_test_verification_few_failures` | Verifies handle_test_verification_and_commit when < 5 tests fail. | PASSED | 0.0009s |
| `test_auto_healer` | `test_invoke_agy_healer` | Verifies invoke_agy_healer constructs correct prompt and command. | PASSED | 0.0004s |
| `test_auto_healer` | `test_invoke_agy_healer_not_found` | Verifies invoke_agy_healer raises FileNotFoundError when agy is absent. | PASSED | 0.0002s |
| `test_auto_healer` | `test_parse_test_failures_variations` | Verifies parse_test_failures extracts modules and errors accurately. | PASSED | 0.0000s |
| `test_auto_healer` | `test_verify_data_untouched_detects_mutation` | Verifies that modified fingerprints raise a PermissionError. | PASSED | 0.0002s |
| `test_auto_healer` | `test_verify_data_untouched_success` | Verifies that identical fingerprints pass verification without raising. | PASSED | 0.0003s |
| `test_check_links` | `test_check_links_detects_broken_file_and_fragment` | Verifies check_links detects broken file references and missing fragments. | PASSED | 0.0010s |
| `test_check_links` | `test_check_links_resolves_root_and_directory` | Verifies check_links resolves root-relative paths and directory index.html. | PASSED | 0.0007s |
| `test_check_links` | `test_check_links_success` | Verifies check_links reports success on valid internal links. | PASSED | 0.0006s |
| `test_check_links` | `test_collect_file_targets` | Verifies extracting id and name attributes from HTML content. | PASSED | 0.0002s |
| `test_check_links` | `test_collect_file_targets_nonexistent` | Verifies nonexistent file returns empty set gracefully. | PASSED | 0.0001s |
| `test_code_test_syncer` | `test_compute_changed_lines` | Verifies line diff calculation for identical and modified content. | PASSED | 0.0014s |
| `test_code_test_syncer` | `test_discover_source_files` | Verifies that discover_source_files returns key Python and JS source scripts. | PASSED | 0.0002s |
| `test_code_test_syncer` | `test_discover_test_files` | Verifies that discover_test_files discovers pricing-tests.js and python unit tests. | PASSED | 0.0003s |
| `test_code_test_syncer` | `test_sync_all_generates_artifacts` | Verifies that running sync_all creates code_references.json and code-sync.md. | PASSED | 0.0041s |
| `test_code_test_syncer` | `test_sync_components_js` | Verifies that orchestration pages in components.js match Claude guide slugs. | PASSED | 0.0002s |
| `test_code_test_syncer` | `test_sync_pricing_calculator` | Verifies that sync_pricing_calculator inserts authoritative rates and formulas. | PASSED | 0.0002s |
| `test_code_test_syncer` | `test_sync_pricing_tests_js_constants` | Verifies that sync_pricing_tests_js injects or updates the authoritative facts block. | PASSED | 0.0003s |
| `test_continuous_updates` | `test_personal_services_continuous_name_updates` | Tests changing service name back and forth across cycles. | PASSED | 1.9595s |
| `test_continuous_updates` | `test_pricing_continuous_seat_and_duration_updates` | Tests changing included seats and duration back and forth across 2 cycles. | PASSED | 3.9990s |
| `test_core_target_documents` | `test_compliance_zero_emojis_and_zero_em_dashes` | Verifies zero emojis and zero em-dashes in manifests, contact.json, guarantees.json, and llms.txt. | PASSED | 0.0009s |
| `test_core_target_documents` | `test_contact_json_no_duplication` | Verifies data/contact.json avoids duplication by referencing config and pricing. | PASSED | 0.0003s |
| `test_core_target_documents` | `test_contact_json_schema_and_facts` | Verifies that data/contact.json exists and registers facts in knowledge base. | PASSED | 0.0003s |
| `test_core_target_documents` | `test_contact_page_rule_execution` | Verifies that ContactPagePricingRule matches contact.html and generates occurrences. | PASSED | 0.0007s |
| `test_core_target_documents` | `test_discover_target_files_includes_contact_and_llms` | Verifies that discover_target_files includes contact.html and llms.txt. | PASSED | 0.0007s |
| `test_core_target_documents` | `test_guarantees_json_no_duplication` | Verifies data/guarantees.json avoids duplication by referencing config, faqs, and pricing. | PASSED | 0.0003s |
| `test_core_target_documents` | `test_guarantees_json_schema_and_facts` | Verifies that data/guarantees.json exists and registers facts in knowledge base. | PASSED | 0.0003s |
| `test_core_target_documents` | `test_llms_txt_rule_execution` | Verifies that LLMsTxtRule matches llms.txt and generates occurrences. | PASSED | 0.0006s |
| `test_core_target_documents` | `test_references_dictionary_preserved_and_intact` | Verifies that master references_dictionary.json is preserved and intact. | PASSED | 0.0003s |
| `test_core_target_documents` | `test_references_manifest_structure_and_counts` | Verifies that references_manifest.json contains lightweight indexing. | PASSED | 0.0003s |
| `test_core_target_documents` | `test_rules_include_llms_txt_and_contact_rules` | Verifies that LLMsTxtRule and ContactPagePricingRule are registered. | PASSED | 0.0003s |
| `test_core_target_documents` | `test_sync_manifest_structure_and_counts` | Verifies that sync_manifest.json exists and tracks data digests and targets. | PASSED | 0.0003s |
| `test_coverage_report` | `test_format_line_ranges` | Verifies formatting line numbers into concise readable ranges. | PASSED | 0.0003s |
| `test_coverage_report` | `test_generate_coverage_report_mocked` | Verifies report generation with mocked coverage engines. | PASSED | 0.0035s |
| `test_coverage_report` | `test_get_git_info` | Verifies retrieving git commit and branch info. | PASSED | 0.0021s |
| `test_expert_system` | `test_expert_system_edge_cases` | Verifies KnowledgeBase and InferenceEngine error handling and edge cases. | PASSED | 0.0008s |
| `test_expert_system` | `test_knowledge_base_loading` | Verifies KnowledgeBase loads facts from all data/*.json files. | PASSED | 0.0002s |
| `test_expert_system` | `test_record_occurrence` | Verifies occurrence tracking records file, page, section, and line. | PASSED | 0.0002s |
| `test_expert_system` | `test_rule_salience_ordering` | Verifies production rules are resolved and fired in salience order. | PASSED | 0.0002s |
| `test_expert_system` | `test_save_and_load_json_atomic` | Verifies atomic json writing and corruption-free reading. | PASSED | 0.0011s |
| `test_expert_system` | `test_save_json_blocks_data_dir_mutation` | Verifies save_json strictly raises PermissionError when targeting data/. | PASSED | 0.0002s |
| `test_fuzzy_engine` | `test_fuzzy_edge_cases` | Verifies edge cases: empty strings, single chars, identical whitespaces, unicode, and extreme mismatch. | PASSED | 0.0004s |
| `test_fuzzy_engine` | `test_fuzzy_matcher_classification` | Verifies holistic composite score and linguistic term classification. | PASSED | 0.0003s |
| `test_fuzzy_engine` | `test_membership_boundary_cases` | Verifies boundary clamping on triangular and trapezoidal membership functions. | PASSED | 0.0000s |
| `test_fuzzy_engine` | `test_membership_functions` | Verifies triangular and trapezoidal membership evaluations. | PASSED | 0.0000s |
| `test_fuzzy_engine` | `test_ngram_similarity` | Verifies character 3-gram Jaccard overlap similarity. | PASSED | 0.0000s |
| `test_fuzzy_engine` | `test_string_distance_and_ratios` | Verifies exact Levenshtein distance and normalized similarity ratios. | PASSED | 0.0000s |
| `test_fuzzy_engine` | `test_token_sort_and_set_ratios` | Verifies token sort and token set robust fuzzy comparisons. | PASSED | 0.0002s |
| `test_last_commit_report` | `test_generate_report_creates_file` | Verifies report is generated and contains accurate HEAD information. | PASSED | 0.5736s |
| `test_last_commit_report` | `test_post_commit_hook_exists_and_executable` | Verifies that post-commit hooks exist in python/hooks and .git/hooks and are executable. | PASSED | 0.5353s |
| `test_last_commit_report` | `test_report_includes_javascript_test_suite_details` | Verifies report includes the JavaScript Unit Test Suite Details table. | PASSED | 0.5335s |
| `test_last_commit_report` | `test_report_includes_link_schema_verification` | Verifies report includes the Link, Schema & Markdown Verification Details table. | PASSED | 0.5812s |
| `test_last_commit_report` | `test_report_includes_nlp_fuzzy_expert_report` | Verifies report includes the NLP-Fuzzy-Expert Knowledge Synchronization Report. | PASSED | 0.5305s |
| `test_last_commit_report` | `test_report_includes_python_test_suite_details` | Verifies report includes the Python Unit Test Suite Details table and itemized list. | PASSED | 0.5679s |
| `test_last_commit_report` | `test_report_includes_quality_gate_summary` | Verifies report includes the Quality Gate & Verification Summary table. | PASSED | 0.5082s |
| `test_last_commit_report` | `test_report_zero_emojis_and_zero_em_dashes` | Strictly verifies zero emojis and zero em-dashes throughout report. | PASSED | 0.5723s |
| `test_master_orchestrator` | `test_discover_target_files` | Verifies discover_target_files finds root HTMLs, JS, and subdirectories. | PASSED | 0.0008s |
| `test_master_orchestrator` | `test_markdown_reports_generated` | Verifies that all 11 domain markdown reports exist under nlp-fuzzy-expert/. | PASSED | 0.0016s |
| `test_master_orchestrator` | `test_sync_all_and_aggregate_dictionary` | Verifies full sync produces references_dictionary.json with 200+ references. | PASSED | 0.0059s |
| `test_master_orchestrator` | `test_sync_file_single` | Verifies sync_file executes cleanly for a target file. | PASSED | 0.0057s |
| `test_nlp_engine` | `test_domain_classification` | Verifies context snippet classification into semantic domains. | PASSED | 0.0001s |
| `test_nlp_engine` | `test_duration_and_seats_edge_cases` | Verifies parsing of various duration and seat expressions. | PASSED | 0.0003s |
| `test_nlp_engine` | `test_entity_extraction` | Verifies semantic entity recognition for MONEY, DURATION, and SEATS. | PASSED | 0.0001s |
| `test_nlp_engine` | `test_highlight_tagging_and_stripping` | Verifies highlight tagging and idempotent clean stripping. | PASSED | 0.0000s |
| `test_nlp_engine` | `test_nlp_edge_cases` | Verifies edge cases for empty strings, punctuation only, and missing numbers. | PASSED | 0.0002s |
| `test_nlp_engine` | `test_text_normalization_and_tokenization` | Verifies HTML unescaping and whitespace normalization. | PASSED | 0.0001s |
| `test_python_tools` | `test_build_components_build_execution` | Verifies build execution runs across repository without raising errors. | PASSED | 4.5526s |
| `test_python_tools` | `test_build_components_extract_nav_and_footer` | Verifies extracting nav and footer tags and handling missing tags. | PASSED | 0.0003s |
| `test_python_tools` | `test_build_md_files_converters` | Verifies build_md_files document converters generate clean markdown. | PASSED | 0.1245s |
| `test_python_tools` | `test_delete_page_detects_nonexistent_target` | Verifies delete_page aborts gracefully when target does not exist. | PASSED | 0.0099s |
| `test_python_tools` | `test_delete_page_dry_run_on_existing_page` | Verifies delete_page dry run evaluates all targets without deleting any files. | PASSED | 0.0092s |
| `test_python_tools` | `test_generate_metrics_execution` | Verifies generate_metrics creates or updates metrics.md successfully. | PASSED | 0.4300s |
| `test_python_tools` | `test_generate_metrics_format_bytes` | Verifies format_bytes human-readable byte size representations. | PASSED | 0.0002s |
| `test_python_tools` | `test_generate_metrics_git_output` | Verifies get_git_output executes git command or returns empty string on error. | PASSED | 0.0050s |
| `test_python_tools` | `test_generate_sitemap_core_pages_exist` | Verifies core navigation pages defined in generate_sitemap exist in repository. | PASSED | 0.0001s |
| `test_python_tools` | `test_generate_sitemap_execution` | Verifies generate_sitemap creates valid XML output. | PASSED | 0.0953s |
| `test_python_tools` | `test_md_html_sync_helpers` | Verifies hash generation, text cleaning, and state serialization. | PASSED | 0.0004s |
| `test_python_tools` | `test_sync_schemas_check_only_mode` | Verifies check_only mode evaluates all pages without modifying schemas. | PASSED | 1.3444s |
| `test_python_tools` | `test_sync_schemas_target_filename_mapping` | Verifies schema target filename mapping conventions. | PASSED | 0.0003s |
| `test_reference_scanner` | `test_reference_files_include_scanned_and_rule_records` |  | PASSED | 0.0866s |
| `test_reference_scanner` | `test_scan_records_schema_and_is_read_only` |  | PASSED | 0.0022s |
| `test_reference_scanner` | `test_terms_cover_all_data_domains` |  | PASSED | 0.0003s |
| `test_report_generator` | `test_classify_rule_field_domains` | Verifies field classification assigns correct domain keys. | PASSED | 0.0000s |
| `test_report_generator` | `test_domain_mappings_completeness` | Verifies that all 11 expected domains have corresponding reference and source files. | PASSED | 0.0000s |
| `test_sync_business_services` | `test_facts_loaded` | Verifies all 12 business services facts are in knowledge base. | PASSED | 0.0003s |
| `test_sync_business_services` | `test_references_and_markdown` | Verifies business_services_references.json and business-services.md exist. | PASSED | 0.0111s |
| `test_sync_business_services` | `test_sync_business_cards_and_curriculum` | Verifies sync_all updates all-business-services.html and business-ai-training.html. | PASSED | 0.0052s |
| `test_sync_claude_and_gemini` | `test_claude_guides_completeness` | Verifies all 12 Claude guides exist on disk and have proper structure. | PASSED | 0.0003s |
| `test_sync_claude_and_gemini` | `test_deepmind_guide_facts_and_sync` | Verifies Google DeepMind guide facts and synchronization in target files. | PASSED | 0.0003s |
| `test_sync_claude_and_gemini` | `test_facts_loaded` | Verifies guides facts exist in knowledge base. | PASSED | 0.0002s |
| `test_sync_claude_and_gemini` | `test_references_and_markdown` | Verifies claude_references.json, gemini_references.json, claude.md, and gemini.md exist. | PASSED | 0.0044s |
| `test_sync_claude_and_gemini` | `test_sync_guides_titles` | Verifies sync_all synchronizes guide titles in HTML files. | PASSED | 0.0054s |
| `test_sync_guarantees` | `test_guarantees_internal_links_across_repo` | Verifies all internal links point to guarantees.html and none to guarantee.html. | PASSED | 0.0008s |
| `test_sync_guarantees` | `test_guarantees_page_content_in_sync` | Verifies guarantees.html content reflects authoritative pricing facts. | PASSED | 0.0004s |
| `test_sync_guarantees` | `test_guarantees_page_exists_and_old_page_removed` | Verifies guarantees.html exists and obsolete guarantee.html is deleted. | PASSED | 0.0002s |
| `test_sync_guarantees` | `test_guarantees_rule_in_expert_system` | Verifies GuaranteesPageRule is registered in the production rules queue. | PASSED | 0.0002s |
| `test_sync_guarantees` | `test_guarantees_tracked_in_references_dictionary` | Verifies guarantees.html occurrences are tracked in references_dictionary.json. | PASSED | 0.0938s |
| `test_sync_personal_services` | `test_facts_loaded` | Verifies all 12 personal services facts are in knowledge base. | PASSED | 0.0004s |
| `test_sync_personal_services` | `test_references_and_markdown` | Verifies personal_services_references.json and personal-services.md exist. | PASSED | 0.0076s |
| `test_sync_personal_services` | `test_sync_personal_cards_and_curriculum` | Verifies sync_all updates all-personal-services.html and personal-ai-training.html. | PASSED | 0.0066s |
| `test_sync_pricing` | `test_pricing_authoritative_facts` | Verifies authoritative values are properly extracted from data/pricing.json. | PASSED | 0.0002s |
| `test_sync_pricing` | `test_pricing_calculator_js_updates` | Verifies pricing-calculator.js contains the synchronized variables. | PASSED | 0.0052s |
| `test_sync_pricing` | `test_pricing_html_updates` | Verifies pricing.html reflects updated pricing and seats. | PASSED | 0.0065s |
| `test_sync_pricing` | `test_pricing_references_and_markdown` | Verifies pricing_references.json and pricing.md exist and are populated. | PASSED | 0.0074s |
| `test_sync_pricing` | `test_pricing_zero_and_edge_multipliers` | Verifies pricing calculations with edge cases and extreme participant volumes. | PASSED | 0.0003s |
| `test_sync_publications_config_faqs` | `test_facts_loaded` | Verifies publications, config, and FAQs facts exist in knowledge base. | PASSED | 0.0002s |
| `test_sync_publications_config_faqs` | `test_sync_execution` | Verifies sync_all generates reference dictionaries and markdown files. | PASSED | 0.0046s |
| `test_sync_tiers` | `test_facts_loaded` | Verifies 6 tiers facts exist in knowledge base. | PASSED | 0.0003s |
| `test_sync_tiers` | `test_references_and_markdown` | Verifies tiers_references.json and tiers.md exist. | PASSED | 0.0053s |
| `test_sync_tiers` | `test_sync_tiers_pricing_and_calc` | Verifies sync_all updates parity tiers in pricing.html and pricing-calculator.js. | PASSED | 0.0056s |
| `test_unidirectional_sync` | `test_code_test_syncer_strictly_excludes_data_dir` | Verifies that CodeTestSynchronizer never targets files inside data/. | PASSED | 0.0007s |
| `test_unidirectional_sync` | `test_data_changes_propagate_outward_to_website_in_sandbox` | Verifies that data changes propagate outward to the website (unidirectional). | PASSED | 3.6313s |
| `test_unidirectional_sync` | `test_discover_target_files_strictly_excludes_data_dir` | Verifies that discover_target_files never includes any file inside data/. | PASSED | 0.0014s |
| `test_unidirectional_sync` | `test_inference_engine_blocks_data_file_processing` | Verifies InferenceEngine.process_file raises PermissionError for any data/*.json file. | PASSED | 0.0005s |
| `test_unidirectional_sync` | `test_save_json_blocks_mutation_of_data_dir` | Verifies save_json raises PermissionError when targeted at data/*.json. | PASSED | 0.0003s |
| `test_unidirectional_sync` | `test_verify_unidirectional_sync_helper` | Verifies verify_unidirectional_sync validates cleanly and flags violations. | PASSED | 0.0004s |
| `test_unidirectional_sync` | `test_website_changes_do_not_propagate_to_data_dir` | Verifies that manual/external edits to HTML files do not alter data/*.json. | PASSED | 1.0155s |
| `test_watcher` | `test_check_for_changes_modified_created_deleted` | Verifies detecting created, modified, and deleted data files. | PASSED | 0.0011s |
| `test_watcher` | `test_default_poll_interval` | Verifies default poll interval is 60.0 seconds. | PASSED | 0.0006s |
| `test_watcher` | `test_get_data_files_and_snapshot` | Verifies retrieving data files and taking snapshots. | PASSED | 0.0005s |
| `test_watcher` | `test_handle_signal` | Verifies handle_signal sets running flag to False. | PASSED | 0.0006s |

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
- **Total Tracked Occurrences**: 20,146

### Master Reference Occurrences by Domain

| Knowledge Domain | Tracked Occurrences | Primary Data Source | Domain Sync Report |
| :--- | :---: | :--- | :--- |
| `config` | 4,491 | `data/config.json` | [`nlp-fuzzy-expert/config.md`](nlp-fuzzy-expert/config.md) |
| `pricing` | 3,522 | `data/pricing.json` | [`nlp-fuzzy-expert/pricing.md`](nlp-fuzzy-expert/pricing.md) |
| `publications` | 3,268 | `data/publications.json` | [`nlp-fuzzy-expert/publications.md`](nlp-fuzzy-expert/publications.md) |
| `tiers` | 2,246 | `data/tiers.json` | [`nlp-fuzzy-expert/tiers.md`](nlp-fuzzy-expert/tiers.md) |
| `contact` | 2,043 | `data/contact.json` | [`nlp-fuzzy-expert/contact.md`](nlp-fuzzy-expert/contact.md) |
| `business_services` | 1,552 | `data/business_services.json` | [`nlp-fuzzy-expert/business-services.md`](nlp-fuzzy-expert/business-services.md) |
| `claude` | 1,207 | `data/claude.json` | [`nlp-fuzzy-expert/claude.md`](nlp-fuzzy-expert/claude.md) |
| `gemini` | 1,207 | `data/gemini.json` | [`nlp-fuzzy-expert/gemini.md`](nlp-fuzzy-expert/gemini.md) |
| `personal_services` | 380 | `data/personal_services.json` | [`nlp-fuzzy-expert/personal-services.md`](nlp-fuzzy-expert/personal-services.md) |
| `guarantees` | 156 | `data/guarantees.json` | [`nlp-fuzzy-expert/guarantees.md`](nlp-fuzzy-expert/guarantees.md) |
| `faqs` | 74 | `data/faqs.json` | [`nlp-fuzzy-expert/faqs.md`](nlp-fuzzy-expert/faqs.md) |
| **Master Dictionary Total** | **20,146** | `data/*.json` (11 domains) | [`references_dictionary.json`](nlp-fuzzy-expert/references_dictionary.json) |

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
M	nlp-fuzzy-expert/business-services.md
M	nlp-fuzzy-expert/business_services_references.json
M	nlp-fuzzy-expert/code-sync.md
M	nlp-fuzzy-expert/code_references.json
M	nlp-fuzzy-expert/config.md
M	nlp-fuzzy-expert/config_references.json
M	nlp-fuzzy-expert/personal-services.md
M	nlp-fuzzy-expert/personal_services_references.json
M	nlp-fuzzy-expert/pricing.md
M	nlp-fuzzy-expert/pricing_references.json
M	nlp-fuzzy-expert/reference_scanner.py
M	nlp-fuzzy-expert/references_dictionary.json
M	nlp-fuzzy-expert/references_manifest.json
M	nlp-fuzzy-expert/sync_manifest.json
M	nlp-fuzzy-expert/synchronizer.py
M	nlp-fuzzy-expert/tiers.md
M	nlp-fuzzy-expert/tiers_references.json
M	python/hooks/pre-commit
M	tests/test_core_target_documents.py
```

## Commit Output & Stat Summary

```text
commit 46b0c7dcd66e9db371b15250845fbe486780593a
Author: Thomas Cherickal <thomascherickal@gmail.com>
Date:   Sat Nov 7 00:13:32 2026 +0530

    perf(sync): optimize synchronizer and commit pipeline with manifest files

 LAST-COMMIT-REPORT.md                              |  108 +-
 TEST_COVERAGE_REPORT.md                            |   18 +-
 md-html-sync/.sync-state.json                      |   58 +-
 metrics.md                                         |   38 +-
 nlp-fuzzy-expert/business-services.md              |    2 +-
 nlp-fuzzy-expert/business_services_references.json |    2 +-
 nlp-fuzzy-expert/code-sync.md                      |    2 +-
 nlp-fuzzy-expert/code_references.json              |    2 +-
 nlp-fuzzy-expert/config.md                         |    2 +-
 nlp-fuzzy-expert/config_references.json            |    2 +-
 nlp-fuzzy-expert/personal-services.md              |   10 +-
 nlp-fuzzy-expert/personal_services_references.json |   76 +-
 nlp-fuzzy-expert/pricing.md                        |  228 +-
 nlp-fuzzy-expert/pricing_references.json           | 2431 +++++++++-----------
 nlp-fuzzy-expert/reference_scanner.py              |   23 +
 nlp-fuzzy-expert/references_dictionary.json        | 2315 ++++++++-----------
 nlp-fuzzy-expert/references_manifest.json          |   18 +-
 nlp-fuzzy-expert/sync_manifest.json                |   64 +-
 nlp-fuzzy-expert/synchronizer.py                   |    9 +-
 nlp-fuzzy-expert/tiers.md                          |    2 +-
 nlp-fuzzy-expert/tiers_references.json             |    2 +-
 python/hooks/pre-commit                            |   10 +-
 tests/test_core_target_documents.py                |   20 +-
 23 files changed, 2335 insertions(+), 3107 deletions(-)
```

## Verification & Compliance

- Zero emojis: Strictly enforced.
- Zero em-dashes: Strictly enforced (spaced hyphens ' - ' or plain hyphens '-' used).
- Overwrite mode: Automatically updated on every new commit via Git post-commit hook.

---

## Test Coverage Report (Python & JavaScript)

Recorded at: **2026-11-06 18:49:18 UTC** | Commit: **46b0c7d** (`main`) | Elapsed: **0.57s**

### Coverage Executive Summary

| Language | Tracked Files | Total Statements / Lines | Covered | Missed | Coverage % | Status |
|:---|---:|---:|---:|---:|---:|:---|
| Python (`nlp-fuzzy-expert/`, `python/`) | 27 | 5,356 | 3,481 | 1,875 | **64.99%** | WARN |
| JavaScript (Root scripts) | 3 | 1,254 | 1,254 | 0 | **100.00%** | PASS |
| **Repository Combined Total** | **30** | **6,610** | **4,735** | **1,875** | **71.63%** | **WARN** |

### Python Test Coverage

| File | Statements | Covered | Missed | Coverage % | Missing Lines |
|:---|---:|---:|---:|---:|:---|
| `nlp-fuzzy-expert/__init__.py` | 7 | 0 | 7 | **0.0%** | `6-9, 16, 23, 25` |
| `nlp-fuzzy-expert/auto_healer.py` | 187 | 151 | 36 | **80.8%** | `54-57, 59, 61-64, 66, 93, 95, 97...` |
| `nlp-fuzzy-expert/code_test_syncer.py` | 235 | 217 | 18 | **92.3%** | `49, 53, 59-60, 62-63, 65-67, 71,...` |
| `nlp-fuzzy-expert/expert_system.py` | 314 | 289 | 25 | **92.0%** | `41, 45, 72, 78-79, 86-87, 93-94,...` |
| `nlp-fuzzy-expert/fuzzy_engine.py` | 121 | 109 | 12 | **90.1%** | `22-24, 36, 64, 88, 92, 94, 123, ...` |
| `nlp-fuzzy-expert/nlp_engine.py` | 74 | 69 | 5 | **93.2%** | `42, 77-78, 113, 119` |
| `nlp-fuzzy-expert/reference_scanner.py` | 364 | 316 | 48 | **86.8%** | `74, 91, 117, 150, 154-163, 171-1...` |
| `nlp-fuzzy-expert/report_generator.py` | 114 | 109 | 5 | **95.6%** | `103-104, 122, 124, 219` |
| `nlp-fuzzy-expert/rules.py` | 546 | 519 | 27 | **95.0%** | `1318-1319, 1322-1328, 1335-1341,...` |
| `nlp-fuzzy-expert/synchronizer.py` | 298 | 256 | 42 | **85.9%** | `95, 102-103, 159, 216-217, 261-2...` |
| `nlp-fuzzy-expert/watcher.py` | 92 | 49 | 43 | **53.3%** | `41, 55-56, 73-74, 90-91, 93-95, ...` |
| `python/build.py` | 10 | 0 | 10 | **0.0%** | `5-6, 8-9, 11, 13-16, 18` |
| `python/build_components.py` | 147 | 95 | 52 | **64.6%** | `71, 104-107, 109-110, 112, 119-1...` |
| `python/build_md_files.py` | 781 | 119 | 662 | **15.2%** | `19, 25-27, 29, 40-43, 45-46, 48-...` |
| `python/check_links.py` | 77 | 71 | 6 | **92.2%** | `53-54, 90, 106-108` |
| `python/delete_page.py` | 237 | 137 | 100 | **57.8%** | `50, 65, 74, 84, 86-87, 93-95, 10...` |
| `python/generate_favicons.py` | 40 | 0 | 40 | **0.0%** | `16-19, 21-26, 28-30, 32-33, 36-3...` |
| `python/generate_last_commit_report.py` | 317 | 251 | 66 | **79.2%** | `53-56, 65-66, 69-71, 73-79, 81-8...` |
| `python/generate_metrics.py` | 267 | 264 | 3 | **98.9%** | `180-181, 345` |
| `python/generate_og_card.py` | 28 | 0 | 28 | **0.0%** | `6-9, 11-15, 17-18, 20, 418-420, ...` |
| `python/generate_sitemap.py` | 48 | 44 | 4 | **91.7%** | `92-94, 136` |
| `python/generate_test_coverage_report.py` | 241 | 100 | 141 | **41.5%** | `27, 30, 39-40, 49-50, 52, 55, 62...` |
| `python/md_html_sync.py` | 318 | 97 | 221 | **30.5%** | `59, 63, 70, 78, 86-88, 94-95, 99...` |
| `python/run_tests.py` | 150 | 100 | 50 | **66.7%** | `58-61, 73-76, 88-89, 104, 118-11...` |
| `python/sync_og_card.py` | 71 | 53 | 18 | **74.7%** | `29, 39, 74-75, 78, 82-84, 93-97,...` |
| `python/sync_pricing.py` | 163 | 0 | 163 | **0.0%** | `27-30, 32-34, 37, 116, 132-134, ...` |
| `python/sync_schemas.py` | 109 | 66 | 43 | **60.5%** | `55-57, 62-63, 65, 76-78, 86-88, ...` |

### JavaScript Test Coverage

| File | Executable Lines | Covered | Missed | Coverage % | Missing Lines |
|:---|---:|---:|---:|---:|:---|
| `components.js` | 510 | 510 | 0 | **100.0%** | `-` |
| `mermaid.js` | 96 | 96 | 0 | **100.0%** | `-` |
| `pricing-calculator.js` | 648 | 648 | 0 | **100.0%** | `-` |

### Detailed Uncovered Lines Breakdown

The following files contain uncovered branches and line numbers identified for ongoing test expansion:

- **`nlp-fuzzy-expert/__init__.py`** (0.0% covered):
  - Missing Line(s): `6-9, 16, 23, 25`
- **`nlp-fuzzy-expert/auto_healer.py`** (80.8% covered):
  - Missing Line(s): `54-57, 59, 61-64, 66, 93, 95, 97-99, 118, 137-138, 350-353, 355-356, 358-362, 364-369, 371`
- **`nlp-fuzzy-expert/code_test_syncer.py`** (92.3% covered):
  - Missing Line(s): `49, 53, 59-60, 62-63, 65-67, 71, 294, 348, 459, 462-463, 557-559`
- **`nlp-fuzzy-expert/expert_system.py`** (92.0% covered):
  - Missing Line(s): `41, 45, 72, 78-79, 86-87, 93-94, 96-100, 117, 130-131, 134-137, 154, 168, 408, 412`
- **`nlp-fuzzy-expert/fuzzy_engine.py`** (90.1% covered):
  - Missing Line(s): `22-24, 36, 64, 88, 92, 94, 123, 136, 157, 159`
- **`nlp-fuzzy-expert/nlp_engine.py`** (93.2% covered):
  - Missing Line(s): `42, 77-78, 113, 119`
- **`nlp-fuzzy-expert/reference_scanner.py`** (86.8% covered):
  - Missing Line(s): `74, 91, 117, 150, 154-163, 171-173, 184-185, 200-201, 214, 318, 359, 430-439, 442-444, 457-462, 487-490, 494`
- **`nlp-fuzzy-expert/report_generator.py`** (95.6% covered):
  - Missing Line(s): `103-104, 122, 124, 219`
- **`nlp-fuzzy-expert/rules.py`** (95.0% covered):
  - Missing Line(s): `1318-1319, 1322-1328, 1335-1341, 1347, 1419-1427, 1432`
- **`nlp-fuzzy-expert/synchronizer.py`** (85.9% covered):
  - Missing Line(s): `95, 102-103, 159, 216-217, 261-262, 286-287, 319, 328-329, 375-376, 392-393, 456-458, 460-462, 517-522, 524-530, 539-541, 549-550, 554`
- **`nlp-fuzzy-expert/watcher.py`** (53.3% covered):
  - Missing Line(s): `41, 55-56, 73-74, 90-91, 93-95, 97-98, 100-110, 113-119, 121, 124-127, 129-133, 135-136, 139`
- **`python/build.py`** (0.0% covered):
  - Missing Line(s): `5-6, 8-9, 11, 13-16, 18`
- **`python/build_components.py`** (64.6% covered):
  - Missing Line(s): `71, 104-107, 109-110, 112, 119-121, 125-130, 138, 147-148, 166-167, 172-173, 178-179, 184-185, 190-191, 194-196, 200-214, 217-219, 221`
- **`python/build_md_files.py`** (15.2% covered):
  - Missing Line(s): `19, 25-27, 29, 40-43, 45-46, 48-49, 51-53, 55-56, 58-61, 63-65, 67-77, 79-81, 83, 85-92, 94-96, 98-101, 103-106, 108-112, 114-117, 119-122, 124-125, 127, 130-132, 134, 159-163, 165-170, 172-173, 175-176, 178-179, 181-183, 186-187, 189-191, 193-194, 196-197, 199-205, 207, 209-263, 265-266, 268-269, 271-273, 276-280, 282, 295-298, 300-301, 303-306, 308-312, 314-317, 319-320, 322-332, 334-337, 339-346, 348-359, 361, 383-384, 387-394, 396, 398-403, 405-406, 408-413, 415-418, 420-421, 445, 447, 470, 473-479, 486-488, 490, 503-506, 508-509, 511-514, 516-520, 522-525, 527-528, 530-531, 533-537, 539-542, 544-552, 554, 640, 644, 680-681, 683-684, 686-687, 689-690, 692, 703-708, 710-711, 713-729, 731-733, 735-753, 755-760, 762, 765-775, 777-784, 787-788, 790-793, 795-801, 803, 816-824, 826-833, 835-853, 855-866, 868-870, 872-879, 881-894, 896-908, 910-914, 916, 919-920, 922-923, 925-926, 928-934, 936, 947-963, 965-973, 975-1025, 1027-1028, 1030-1035, 1041, 1044-1045, 1047, 1050, 1060-1069, 1072-1084, 1086-1089, 1092, 1098-1107, 1109, 1112`
- **`python/check_links.py`** (92.2% covered):
  - Missing Line(s): `53-54, 90, 106-108`
- **`python/delete_page.py`** (57.8% covered):
  - Missing Line(s): `50, 65, 74, 84, 86-87, 93-95, 104, 110, 115, 120, 128, 133-135, 140, 144, 148-151, 160, 167-170, 172-173, 181, 188-191, 193-196, 204, 211-215, 227-237, 249, 268-269, 276-278, 280-281, 288-290, 292-293, 300-301, 307-308, 312-313, 317-318, 323-326, 332-335, 337-340, 342-345, 353, 356-359, 361, 365`
- **`python/generate_favicons.py`** (0.0% covered):
  - Missing Line(s): `16-19, 21-26, 28-30, 32-33, 36-39, 42, 51-55, 58-61, 64-67, 69, 73-76, 78-79`
- **`python/generate_last_commit_report.py`** (79.2% covered):
  - Missing Line(s): `53-56, 65-66, 69-71, 73-79, 81-84, 86-88, 90-91, 101-104, 112-113, 115, 134-135, 138-146, 149, 160, 177-185, 215-216, 227-228, 310-315, 478, 508`
- **`python/generate_metrics.py`** (98.9% covered):
  - Missing Line(s): `180-181, 345`
- **`python/generate_og_card.py`** (0.0% covered):
  - Missing Line(s): `6-9, 11-15, 17-18, 20, 418-420, 422-423, 434, 437-443, 445, 447-448`
- **`python/generate_sitemap.py`** (91.7% covered):
  - Missing Line(s): `92-94, 136`
- **`python/generate_test_coverage_report.py`** (41.5% covered):
  - Missing Line(s): `27, 30, 39-40, 49-50, 52, 55, 62-63, 65-73, 75, 77, 88, 91, 100, 102-103, 105-106, 108, 111, 118, 122, 125, 129, 132, 134, 136, 139, 144-146, 148-149, 151-152, 154, 160-161, 163-165, 167-168, 170, 172-187, 189-193, 195-199, 201-202, 204-207, 209, 211-212, 221, 224, 228, 233, 238-239, 261-262, 274, 276, 283, 294, 298, 303-304, 316, 318-319, 324, 331, 333-334, 339, 346, 349, 351-352, 354-355, 360, 364, 367-368, 370, 372, 374-376, 378-383, 385, 387, 389-390, 392-393, 396-397`
- **`python/md_html_sync.py`** (30.5% covered):
  - Missing Line(s): `59, 63, 70, 78, 86-88, 94-95, 99, 111-112, 114-115, 117, 119, 121-122, 124-128, 130, 132-135, 138-144, 146, 159-160, 162, 181-182, 184, 223-232, 234, 247, 311-331, 333, 335-336, 338, 345-346, 348-349, 351, 353-354, 356-357, 359-362, 365-366, 368, 370-377, 380-388, 390-393, 396-400, 402, 404-411, 413-414, 416, 418-428, 430, 432-435, 437-447, 449-453, 455-458, 461-462, 464, 485, 500-501, 503-504, 506-507, 511-512, 519-521, 523-524, 526-527, 529-530, 546, 566-567, 569, 571-575, 577-579, 581-582, 584-585, 587-590, 592-593, 595-596, 603-608, 610`
- **`python/run_tests.py`** (66.7% covered):
  - Missing Line(s): `58-61, 73-76, 88-89, 104, 118-119, 150-155, 173-174, 182-185, 189-190, 203-206, 215-216, 219-233, 238-239`
- **`python/sync_og_card.py`** (74.7% covered):
  - Missing Line(s): `29, 39, 74-75, 78, 82-84, 93-97, 111-113, 119-120`
- **`python/sync_pricing.py`** (0.0% covered):
  - Missing Line(s): `27-30, 32-34, 37, 116, 132-134, 141, 147-149, 151-155, 162, 170-171, 173, 175-190, 192-209, 211-216, 218-222, 224-238, 240-241, 263, 265-281, 283-284, 286, 313-316, 321-322, 324-326, 328-330, 333-345, 348, 352-353, 355-357, 359-364, 366, 377, 381-385, 387-395, 397-398`
- **`python/sync_schemas.py`** (60.5% covered):
  - Missing Line(s): `55-57, 62-63, 65, 76-78, 86-88, 99-102, 124, 135-137, 142, 144-148, 150, 152-156, 158, 174, 181, 184, 189, 195, 197, 199-200, 202, 206`

---
*Automated test coverage generated across Python unittest and Node.js V8 suites.*
