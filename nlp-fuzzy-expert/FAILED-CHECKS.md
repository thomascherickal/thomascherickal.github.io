# Failed Checks Report

- **Status**: 1 failure(s) detected

## Details

```text
================================================================================
           UNIT TEST SUITE EXECUTION (tests/)
================================================================================
Target directory:  /home/thomas/GitHub.io/thomascherickal.github.io/tests
Discovery pattern: test_*.py
--------------------------------------------------------------------------------

[Auto-Healer] Executing commit: 'test commit'...
  -> Working tree is clean. Nothing to commit.

[Auto-Healer] Executing commit: 'feat: update test'...
  -> Git commit output:
[main abc] commit

[Auto-Healer] Executing commit: 'feat: error test'...
  -> Git commit notice: git failed

[Auto-Healer] Wrote failure report to: failed_test_sync_pricing_tests.report.md
[Auto-Healer] Wrote aggregate report to: failed_tests_report.md

[Auto-Healer] Wrote failure report to: failed_test_sample_tests.report.md
[Auto-Healer] Wrote aggregate report to: failed_tests_report.md

[Auto-Healer] Post-sync verification: ALL UNIT TESTS PASSED.

[Auto-Healer] Post-sync verification: 2 UNIT TEST(S) FAILED.
[Auto-Healer] Failure count (2) is less than 5. Continuing with commit without aborting.

================================================================================
 ACTION REQUIRED: UNIT TEST DISCREPANCY DETECTED
================================================================================
Less than 5 tests failed (2 failure(s)). The commit has proceeded.
Comprehensive report(s) created: failed_tests_report.md
To fix the tests automatically with AGY, run:
  agy --print "Fix the unit test failures reported in failed_tests_report.md"
================================================================================


[Auto-Healer] Post-sync verification: 6 UNIT TEST(S) FAILED.
[Auto-Healer] Failure count (6) >= 5. Initiating AGY auto-healing loop...

[Auto-Healer] Test suite is completely GREEN after 1 repair iteration(s)!

[Auto-Healer] Post-sync verification: 6 UNIT TEST(S) FAILED.
[Auto-Healer] Failure count (6) >= 5. Initiating AGY auto-healing loop...
[Auto-Healer] Failures reduced to 2 (< 5). Proceeding with commit and report.

[Auto-Healer] Post-sync verification: 6 UNIT TEST(S) FAILED.
[Auto-Healer] Failure count (6) >= 5. Initiating AGY auto-healing loop...

[Auto-Healer] Invoking agy CLI to diagnose and heal failure (Attempt 1/3)...
Checked 2 internal links across 1 HTML files.
FOUND 2 BROKEN LINKS/ANCHORS:
  - Broken link in ../../../../../../../tmp/tmpkai5c29y/broken.html: 'missing.html' -> Target file ../../../../../../../tmp/tmpkai5c29y/missing.html not found
  - Broken anchor in ../../../../../../../tmp/tmpkai5c29y/broken.html: '#missing-anchor' -> ID '#missing-anchor' not found in ../../../../../../../tmp/tmpkai5c29y/broken.html
Checked 1 internal links across 2 HTML files.
ZERO broken internal links or fragments found! All links verified.
Checked 1 internal links across 2 HTML files.
ZERO broken internal links or fragments found! All links verified.
================================================================================
           NLP-FUZZY-EXPERT KNOWLEDGE SYNCHRONIZATION SYSTEM
================================================================================
  [DATA FLOW] Direction: data/*.json -> HTML/JS/MD/Code (STRICTLY UNIDIRECTIONAL / ONE-WAY)
[Step 1/6] Loading authoritative facts from data/*.json into KnowledgeBase...
  [UPDATE SCOPE] Domain: data/*.json -> Target: KnowledgeBase atomic facts cache
  -> Authoritative sources loaded: 11 data domains
     - data/business_services.json: 3 top-level entries
     - data/claude.json: 2 top-level entries
     - data/config.json: 12 top-level entries
     - data/contact.json: 6 top-level entries
     - data/faqs.json: 2 top-level entries
     - data/gemini.json: 2 top-level entries
     - data/guarantees.json: 5 top-level entries
     - data/personal_services.json: 3 top-level entries
     - data/pricing.json: 6 top-level entries
     - data/publications.json: 48 top-level entries
     - data/tiers.json: 1 top-level entries
  -> Total atomic propositions registered in KnowledgeBase: 310

[Step 2/6] Registering production rules in salience priority order...
  [UPDATE SCOPE] Domain: rules.py -> Target: InferenceEngine priority rule queue
  -> Active production rules registered: 18
     - Salience 100: PricingHeroPackagesRule (domain: pricing)
     - Salience 100: PricingCalculatorScriptRule (domain: pricing)
     - Salience  95: PricingCalculatorSelectOptionsRule (domain: pricing)
     - Salience  92: PricingSection5CardsRule (domain: pricing)
     - Salience  90: BusinessAITrainingRule (domain: business_training)
     ... and 13 additional production rules

[Step 3/6] Discovering target documents across repository...
  [UPDATE SCOPE] Domain: Repository scan -> Target: Target HTML and JS documents list
  -> Discovered 99 target documents (.html, .js) for synchronization.

[Step 4/6] Executing forward-chaining inference & computing line diffs...
  [UPDATE SCOPE] Fields: Authoritative pricing, tiers, curricula & configs -> Target: HTML/JS repository pages
  -> [MODIFIED] all-personal-services.html | Field: personal_service_01_card, personal_service_02_card, personal_service_03_card, personal_service_04_card, personal_service_05_card, personal_service_06_card, personal_service_07_card, personal_service_08_card, personal_service_09_card, personal_service_10_card, personal_service_11_card, personal_service_12_card, personal_service_13_card, personal_service_14_card, personal_service_15_card, personal_service_16_card | Location: Line(s) 154, 350
     Changed Lines: 154, 350
     Rules Fired:   PersonalServicesNamesAndCardsRule
     Fields Synced: personal_service_01_card, personal_service_02_card, personal_service_03_card, personal_service_04_card, personal_service_05_card, personal_service_06_card, personal_service_07_card, personal_service_08_card, personal_service_09_card, personal_service_10_card, personal_service_11_card, personal_service_12_card, personal_service_13_card, personal_service_14_card, personal_service_15_card, personal_service_16_card
  -> [MODIFIED] personal-services/personal-service-agentic-assistants.html | Field: individual_service_duration, individual_service_price, personal_service_04_name | Location: Line(s) 7, 298
     Changed Lines: 7, 298
     Rules Fired:   PersonalServicesNamesAndCardsRule
     Fields Synced: individual_service_duration, individual_service_price, personal_service_04_name
  -> Total documents modified: 2 of 99

[Step 4b] Synchronizing source code and unit test suites...
  [UPDATE SCOPE] Fields: Authoritative facts -> Target: python/*.py, root JS, and tests/*
  -> All 48 source/test files evaluated. Zero changes required (fully in sync).

[Step 5/6] Generating domain references and master dictionary...
  [UPDATE SCOPE] Fields: Tracked occurrences & references -> Target: nlp-fuzzy-expert/*_references.json
  -> Reference dictionaries and manifest already in sync.

[Step 6/6] Generating domain markdown synchronization reports...
  [UPDATE SCOPE] Fields: Domain summaries & occurrence tables -> Target: nlp-fuzzy-expert/*.md reports
  -> Markdown reports already in sync.

================================================================================
                     SYNCHRONIZATION SUMMARY TABLE
================================================================================
+-----------------------------------------------+---------+---------------+--------------------------+--------------------------+
| Target File                                   | Status  | Changed Lines | Rules Fired              | Fields Synced            |
+-----------------------------------------------+---------+---------------+--------------------------+--------------------------+
| all-personal-services.html                    | UPDATED | 154, 350      | PersonalServicesNames... | personal_service_01_c... |
| personal-services/personal-service-agentic... | UPDATED | 7, 298        | PersonalServicesNames... | individual_service_du... |
+-----------------------------------------------+---------+---------------+--------------------------+--------------------------+
Summary: 99 files evaluated | 2 updated | 97 in sync | 64 references tracked
================================================================================
================================================================================
           NLP-FUZZY-EXPERT KNOWLEDGE SYNCHRONIZATION SYSTEM
================================================================================
  [DATA FLOW] Direction: data/*.json -> HTML/JS/MD/Code (STRICTLY UNIDIRECTIONAL / ONE-WAY)
[Step 1/6] Loading authoritative facts from data/*.json into KnowledgeBase...
  [UPDATE SCOPE] Domain: data/*.json -> Target: KnowledgeBase atomic facts cache
  -> Authoritative sources loaded: 11 data domains
     - data/business_services.json: 3 top-level entries
     - data/claude.json: 2 top-level entries
     - data/config.json: 12 top-level entries
     - data/contact.json: 6 top-level entries
     - data/faqs.json: 2 top-level entries
     - data/gemini.json: 2 top-level entries
     - data/guarantees.json: 5 top-level entries
     - data/personal_services.json: 3 top-level entries
     - data/pricing.json: 6 top-level entries
     - data/publications.json: 48 top-level entries
     - data/tiers.json: 1 top-level entries
  -> Total atomic propositions registered in KnowledgeBase: 310

[Step 2/6] Registering production rules in salience priority order...
  [UPDATE SCOPE] Domain: rules.py -> Target: InferenceEngine priority rule queue
  -> Active production rules registered: 18
     - Salience 100: PricingHeroPackagesRule (domain: pricing)
     - Salience 100: PricingCalculatorScriptRule (domain: pricing)
     - Salience  95: PricingCalculatorSelectOptionsRule (domain: pricing)
     - Salience  92: PricingSection5CardsRule (domain: pricing)
     - Salience  90: BusinessAITrainingRule (domain: business_training)
     ... and 13 additional production rules

[Step 3/6] Discovering target documents across repository...
  [UPDATE SCOPE] Domain: Repository scan -> Target: Target HTML and JS documents list
  -> Discovered 99 target documents (.html, .js) for synchronization.

[Step 4/6] Executing forward-chaining inference & computing line diffs...
  [UPDATE SCOPE] Fields: Authoritative pricing, tiers, curricula & configs -> Target: HTML/JS repository pages
  -> [MODIFIED] personal-services/personal-service-agentic-assistants.html | Field: individual_service_duration, individual_service_price, personal_service_04_name | Location: Line(s) 7, 298
     Changed Lines: 7, 298
     Rules Fired:   PersonalServicesNamesAndCardsRule
     Fields Synced: individual_service_duration, individual_service_price, personal_service_04_name
  -> Total documents modified: 1 of 99

[Step 4b] Synchronizing source code and unit test suites...
  [UPDATE SCOPE] Fields: Authoritative facts -> Target: python/*.py, root JS, and tests/*
  -> All 48 source/test files evaluated. Zero changes required (fully in sync).

[Step 5/6] Generating domain references and master dictionary...
  [UPDATE SCOPE] Fields: Tracked occurrences & references -> Target: nlp-fuzzy-expert/*_references.json
  -> Exhaustive scan located 28549 term occurrences across 99 documents
  -> Wrote pricing_references.json (3360 references)
  -> Wrote business_services_references.json (3253 references)
  -> Wrote config_references.json (6434 references)
  -> Master Dictionary generated: references_dictionary.json (28667 total references)
  -> Lightweight Manifest generated: references_manifest.json (108 documents indexed)

[Step 6/6] Generating domain markdown synchronization reports...
  [UPDATE SCOPE] Fields: Domain summaries & occurrence tables -> Target: nlp-fuzzy-expert/*.md reports
  -> Wrote pricing.md
  -> Wrote business-services.md
  -> Wrote config.md

================================================================================
                     SYNCHRONIZATION SUMMARY TABLE
================================================================================
+-----------------------------------------------+---------+---------------+--------------------------+--------------------------+
| Target File                                   | Status  | Changed Lines | Rules Fired              | Fields Synced            |
+-----------------------------------------------+---------+---------------+--------------------------+--------------------------+
| personal-services/personal-service-agentic... | UPDATED | 7, 298        | PersonalServicesNames... | individual_service_du... |
+-----------------------------------------------+---------+---------------+--------------------------+--------------------------+
Summary: 99 files evaluated | 1 updated | 98 in sync | 100 references tracked
================================================================================
================================================================================
           NLP-FUZZY-EXPERT KNOWLEDGE SYNCHRONIZATION SYSTEM
================================================================================
  [DATA FLOW] Direction: data/*.json -> HTML/JS/MD/Code (STRICTLY UNIDIRECTIONAL / ONE-WAY)
[Step 1/6] Loading authoritative facts from data/*.json into KnowledgeBase...
  [UPDATE SCOPE] Domain: data/*.json -> Target: KnowledgeBase atomic facts cache
  -> Authoritative sources loaded: 11 data domains
     - data/business_services.json: 3 top-level entries
     - data/claude.json: 2 top-level entries
     - data/config.json: 12 top-level entries
     - data/contact.json: 6 top-level entries
     - data/faqs.json: 2 top-level entries
     - data/gemini.json: 2 top-level entries
     - data/guarantees.json: 5 top-level entries
     - data/personal_services.json: 3 top-level entries
     - data/pricing.json: 6 top-level entries
     - data/publications.json: 48 top-level entries
     - data/tiers.json: 1 top-level entries
  -> Total atomic propositions registered in KnowledgeBase: 310

[Step 2/6] Registering production rules in salience priority order...
  [UPDATE SCOPE] Domain: rules.py -> Target: InferenceEngine priority rule queue
  -> Active production rules registered: 18
     - Salience 100: PricingHeroPackagesRule (domain: pricing)
     - Salience 100: PricingCalculatorScriptRule (domain: pricing)
     - Salience  95: PricingCalculatorSelectOptionsRule (domain: pricing)
     - Salience  92: PricingSection5CardsRule (domain: pricing)
     - Salience  90: BusinessAITrainingRule (domain: business_training)
     ... and 13 additional production rules

[Step 3/6] Discovering target documents across repository...
  [UPDATE SCOPE] Domain: Repository scan -> Target: Target HTML and JS documents list
  -> Discovered 99 target documents (.html, .js) for synchronization.

[Step 4/6] Executing forward-chaining inference & computing line diffs...
  [UPDATE SCOPE] Fields: Authoritative pricing, tiers, curricula & configs -> Target: HTML/JS repository pages
  -> [MODIFIED] faqs.html | Field: business_included_seats | Location: Line(s) 1857, 1876, 1888
     Changed Lines: 1857, 1876, 1888
     Rules Fired:   FAQsPoliciesRule
     Fields Synced: business_included_seats
  -> [MODIFIED] guarantees.html | Field: guarantees_business_flagship_price, guarantees_personal_flagship_price | Location: Line(s) 355, 357, 524-525
     Changed Lines: 355, 357, 524-525
     Rules Fired:   GuaranteesPageRule
     Fields Synced: guarantees_business_flagship_price, guarantees_personal_flagship_price
  -> [MODIFIED] llms.txt | Field: llms_business_flagship_price, llms_individual_service_price, llms_personal_flagship_price | Location: Line(s) 13, 40, 42
     Changed Lines: 13, 40, 42
     Rules Fired:   LLMsTxtRule
     Fields Synced: llms_business_flagship_price, llms_individual_service_price, llms_personal_flagship_price
  -> [MODIFIED] pricing-calculator.js | Field: calculator_seat_threshold, tier_1_calc_multiplier, tier_2_calc_multiplier, tier_3_calc_multiplier, tier_4_calc_multiplier, tier_5_calc_multiplier, tier_6_calc_multiplier | Location: Line(s) 170, 316-317, 422, 442, 541
     Changed Lines: 170, 316-317, 422, 442, 541
     Rules Fired:   PricingCalculatorScriptRule
     Fields Synced: calculator_seat_threshold, tier_1_calc_multiplier, tier_2_calc_multiplier, tier_3_calc_multiplier, tier_4_calc_multiplier, tier_5_calc_multiplier, tier_6_calc_multiplier
  -> [MODIFIED] pricing.html | Field: business_flagship_days, business_flagship_price, business_service_01_pricing_card, business_service_02_pricing_card, business_service_03_pricing_card, business_service_04_pricing_card, business_service_05_pricing_card, business_service_06_pricing_card, business_service_07_pricing_card, business_service_08_pricing_card, business_service_09_pricing_card, business_service_10_pricing_card, business_service_11_pricing_card, business_service_12_pricing_card, business_service_13_pricing_card, business_service_14_pricing_card, business_service_15_pricing_card, business_service_16_pricing_card, calculator_options, personal_flagship_days, personal_flagship_price, tier_1_parity, tier_2_parity, tier_3_parity, tier_4_parity, tier_5_parity, tier_6_parity | Location: Line(s) 60, 94, 183, 453, 1002, 1053, 1101, 1245, 1250
     Changed Lines: 60, 94, 183, 453, 1002, 1053, 1101, 1245, 1250
     Rules Fired:   PricingHeroPackagesRule, PricingCalculatorSelectOptionsRule
     Fields Synced: business_flagship_days, business_flagship_price, business_service_01_pricing_card, business_service_02_pricing_card, business_service_03_pricing_card, business_service_04_pricing_card, business_service_05_pricing_card, business_service_06_pricing_card, business_service_07_pricing_card, business_service_08_pricing_card, business_service_09_pricing_card, business_service_10_pricing_card, business_service_11_pricing_card, business_service_12_pricing_card, business_service_13_pricing_card, business_service_14_pricing_card, business_service_15_pricing_card, business_service_16_pricing_card, calculator_options, personal_flagship_days, personal_flagship_price, tier_1_parity, tier_2_parity, tier_3_parity, tier_4_parity, tier_5_parity, tier_6_parity
  -> Total documents modified: 5 of 99

[Step 4b] Synchronizing source code and unit test suites...
  [UPDATE SCOPE] Fields: Authoritative facts -> Target: python/*.py, root JS, and tests/*
  -> [MODIFIED] pricing-tests.js | Status: UPDATED | Changed Lines: 21
  -> Total source/test files updated: 1 of 48

[Step 5/6] Generating domain references and master dictionary...
  [UPDATE SCOPE] Fields: Tracked occurrences & references -> Target: nlp-fuzzy-expert/*_references.json
  -> Reference dictionaries and manifest already in sync.

[Step 6/6] Generating domain markdown synchronization reports...
  [UPDATE SCOPE] Fields: Domain summaries & occurrence tables -> Target: nlp-fuzzy-expert/*.md reports
  -> Markdown reports already in sync.

================================================================================
                     SYNCHRONIZATION SUMMARY TABLE
================================================================================
+-----------------------+---------+------------------------+--------------------------+--------------------------+
| Target File           | Status  | Changed Lines          | Rules Fired              | Fields Synced            |
+-----------------------+---------+------------------------+--------------------------+--------------------------+
| faqs.html             | UPDATED | 1857, 1876, 1888       | FAQsPoliciesRule         | business_included_seats  |
| guarantees.html       | UPDATED | 355, 357, 524-525      | GuaranteesPageRule       | guarantees_business_f... |
| llms.txt              | UPDATED | 13, 40, 42             | LLMsTxtRule              | llms_business_flagshi... |
| pricing-calculator.js | UPDATED | 170, 316-317, 422, ... | PricingCalculatorScri... | calculator_seat_thres... |
| pricing.html          | UPDATED | 60, 94, 183, 453, 1... | PricingHeroPackagesRu... | business_flagship_day... |
| pricing-tests.js      | UPDATED | 21                     | CodeTestSynchronizer     | source_and_tests         |
+-----------------------+---------+------------------------+--------------------------+--------------------------+
Summary: 99 files evaluated | 6 updated | 93 in sync | 172 references tracked
================================================================================
================================================================================
           NLP-FUZZY-EXPERT KNOWLEDGE SYNCHRONIZATION SYSTEM
================================================================================
  [DATA FLOW] Direction: data/*.json -> HTML/JS/MD/Code (STRICTLY UNIDIRECTIONAL / ONE-WAY)
[Step 1/6] Loading authoritative facts from data/*.json into KnowledgeBase...
  [UPDATE SCOPE] Domain: data/*.json -> Target: KnowledgeBase atomic facts cache
  -> Authoritative sources loaded: 11 data domains
     - data/business_services.json: 3 top-level entries
     - data/claude.json: 2 top-level entries
     - data/config.json: 12 top-level entries
     - data/contact.json: 6 top-level entries
     - data/faqs.json: 2 top-level entries
     - data/gemini.json: 2 top-level entries
     - data/guarantees.json: 5 top-level entries
     - data/personal_services.json: 3 top-level entries
     - data/pricing.json: 6 top-level entries
     - data/publications.json: 48 top-level entries
     - data/tiers.json: 1 top-level entries
  -> Total atomic propositions registered in KnowledgeBase: 310

[Step 2/6] Registering production rules in salience priority order...
  [UPDATE SCOPE] Domain: rules.py -> Target: InferenceEngine priority rule queue
  -> Active production rules registered: 18
     - Salience 100: PricingHeroPackagesRule (domain: pricing)
     - Salience 100: PricingCalculatorScriptRule (domain: pricing)
     - Salience  95: PricingCalculatorSelectOptionsRule (domain: pricing)
     - Salience  92: PricingSection5CardsRule (domain: pricing)
     - Salience  90: BusinessAITrainingRule (domain: business_training)
     ... and 13 additional production rules

[Step 3/6] Discovering target documents across repository...
  [UPDATE SCOPE] Domain: Repository scan -> Target: Target HTML and JS documents list
  -> Discovered 99 target documents (.html, .js) for synchronization.

[Step 4/6] Executing forward-chaining inference & computing line diffs...
  [UPDATE SCOPE] Fields: Authoritative pricing, tiers, curricula & configs -> Target: HTML/JS repository pages
  -> [MODIFIED] faqs.html | Field: business_included_seats | Location: Line(s) 1857, 1876, 1888
     Changed Lines: 1857, 1876, 1888
     Rules Fired:   FAQsPoliciesRule
     Fields Synced: business_included_seats
  -> [MODIFIED] guarantees.html | Field: guarantees_business_flagship_price, guarantees_personal_flagship_price | Location: Line(s) 355, 357, 524-525
     Changed Lines: 355, 357, 524-525
     Rules Fired:   GuaranteesPageRule
     Fields Synced: guarantees_business_flagship_price, guarantees_personal_flagship_price
  -> [MODIFIED] llms.txt | Field: llms_business_flagship_price, llms_individual_service_price, llms_personal_flagship_price | Location: Line(s) 13, 40, 42
     Changed Lines: 13, 40, 42
     Rules Fired:   LLMsTxtRule
     Fields Synced: llms_business_flagship_price, llms_individual_service_price, llms_personal_flagship_price
  -> [MODIFIED] pricing-calculator.js | Field: calculator_seat_threshold, tier_1_calc_multiplier, tier_2_calc_multiplier, tier_3_calc_multiplier, tier_4_calc_multiplier, tier_5_calc_multiplier, tier_6_calc_multiplier | Location: Line(s) 170, 316-317, 422, 442, 541
     Changed Lines: 170, 316-317, 422, 442, 541
     Rules Fired:   PricingCalculatorScriptRule
     Fields Synced: calculator_seat_threshold, tier_1_calc_multiplier, tier_2_calc_multiplier, tier_3_calc_multiplier, tier_4_calc_multiplier, tier_5_calc_multiplier, tier_6_calc_multiplier
  -> [MODIFIED] pricing.html | Field: business_flagship_days, business_flagship_price, business_service_01_pricing_card, business_service_02_pricing_card, business_service_03_pricing_card, business_service_04_pricing_card, business_service_05_pricing_card, business_service_06_pricing_card, business_service_07_pricing_card, business_service_08_pricing_card, business_service_09_pricing_card, business_service_10_pricing_card, business_service_11_pricing_card, business_service_12_pricing_card, business_service_13_pricing_card, business_service_14_pricing_card, business_service_15_pricing_card, business_service_16_pricing_card, calculator_options, personal_flagship_days, personal_flagship_price, tier_1_parity, tier_2_parity, tier_3_parity, tier_4_parity, tier_5_parity, tier_6_parity | Location: Line(s) 60, 94, 183, 453, 1002, 1053, 1101, 1245, 1250
     Changed Lines: 60, 94, 183, 453, 1002, 1053, 1101, 1245, 1250
     Rules Fired:   PricingHeroPackagesRule, PricingCalculatorSelectOptionsRule
     Fields Synced: business_flagship_days, business_flagship_price, business_service_01_pricing_card, business_service_02_pricing_card, business_service_03_pricing_card, business_service_04_pricing_card, business_service_05_pricing_card, business_service_06_pricing_card, business_service_07_pricing_card, business_service_08_pricing_card, business_service_09_pricing_card, business_service_10_pricing_card, business_service_11_pricing_card, business_service_12_pricing_card, business_service_13_pricing_card, business_service_14_pricing_card, business_service_15_pricing_card, business_service_16_pricing_card, calculator_options, personal_flagship_days, personal_flagship_price, tier_1_parity, tier_2_parity, tier_3_parity, tier_4_parity, tier_5_parity, tier_6_parity
  -> Total documents modified: 5 of 99

[Step 4b] Synchronizing source code and unit test suites...
  [UPDATE SCOPE] Fields: Authoritative facts -> Target: python/*.py, root JS, and tests/*
  -> [MODIFIED] pricing-tests.js | Status: UPDATED | Changed Lines: 21
  -> Total source/test files updated: 1 of 48

[Step 5/6] Generating domain references and master dictionary...
  [UPDATE SCOPE] Fields: Tracked occurrences & references -> Target: nlp-fuzzy-expert/*_references.json
  -> Reference dictionaries and manifest already in sync.

[Step 6/6] Generating domain markdown synchronization reports...
  [UPDATE SCOPE] Fields: Domain summaries & occurrence tables -> Target: nlp-fuzzy-expert/*.md reports
  -> Markdown reports already in sync.

================================================================================
                     SYNCHRONIZATION SUMMARY TABLE
================================================================================
+-----------------------+---------+------------------------+--------------------------+--------------------------+
| Target File           | Status  | Changed Lines          | Rules Fired              | Fields Synced            |
+-----------------------+---------+------------------------+--------------------------+--------------------------+
| faqs.html             | UPDATED | 1857, 1876, 1888       | FAQsPoliciesRule         | business_included_seats  |
| guarantees.html       | UPDATED | 355, 357, 524-525      | GuaranteesPageRule       | guarantees_business_f... |
| llms.txt              | UPDATED | 13, 40, 42             | LLMsTxtRule              | llms_business_flagshi... |
| pricing-calculator.js | UPDATED | 170, 316-317, 422, ... | PricingCalculatorScri... | calculator_seat_thres... |
| pricing.html          | UPDATED | 60, 94, 183, 453, 1... | PricingHeroPackagesRu... | business_flagship_day... |
| pricing-tests.js      | UPDATED | 21                     | CodeTestSynchronizer     | source_and_tests         |
+-----------------------+---------+------------------------+--------------------------+--------------------------+
Summary: 99 files evaluated | 6 updated | 93 in sync | 172 references tracked
================================================================================
================================================================================
           NLP-FUZZY-EXPERT KNOWLEDGE SYNCHRONIZATION SYSTEM
================================================================================
  [DATA FLOW] Direction: data/*.json -> HTML/JS/MD/Code (STRICTLY UNIDIRECTIONAL / ONE-WAY)
[Step 1/6] Loading authoritative facts from data/*.json into KnowledgeBase...
  [UPDATE SCOPE] Domain: data/*.json -> Target: KnowledgeBase atomic facts cache
  -> Authoritative sources loaded: 11 data domains
     - data/business_services.json: 3 top-level entries
     - data/claude.json: 2 top-level entries
     - data/config.json: 12 top-level entries
     - data/contact.json: 6 top-level entries
     - data/faqs.json: 2 top-level entries
     - data/gemini.json: 2 top-level entries
     - data/guarantees.json: 5 top-level entries
     - data/personal_services.json: 3 top-level entries
     - data/pricing.json: 6 top-level entries
     - data/publications.json: 48 top-level entries
     - data/tiers.json: 1 top-level entries
  -> Total atomic propositions registered in KnowledgeBase: 310

[Step 2/6] Registering production rules in salience priority order...
  [UPDATE SCOPE] Domain: rules.py -> Target: InferenceEngine priority rule queue
  -> Active production rules registered: 18
     - Salience 100: PricingHeroPackagesRule (domain: pricing)
     - Salience 100: PricingCalculatorScriptRule (domain: pricing)
     - Salience  95: PricingCalculatorSelectOptionsRule (domain: pricing)
     - Salience  92: PricingSection5CardsRule (domain: pricing)
     - Salience  90: BusinessAITrainingRule (domain: business_training)
     ... and 13 additional production rules

[Step 3/6] Discovering target documents across repository...
  [UPDATE SCOPE] Domain: Repository scan -> Target: Target HTML and JS documents list
  -> Discovered 99 target documents (.html, .js) for synchronization.

[Step 4/6] Executing forward-chaining inference & computing line diffs...
  [UPDATE SCOPE] Fields: Authoritative pricing, tiers, curricula & configs -> Target: HTML/JS repository pages
  -> [MODIFIED] faqs.html | Field: business_included_seats | Location: Line(s) 1857, 1876, 1888
     Changed Lines: 1857, 1876, 1888
     Rules Fired:   FAQsPoliciesRule
     Fields Synced: business_included_seats
  -> [MODIFIED] llms.txt | Field: llms_business_flagship_price, llms_individual_service_price, llms_personal_flagship_price | Location: Line(s) 13, 40, 42
     Changed Lines: 13, 40, 42
     Rules Fired:   LLMsTxtRule
     Fields Synced: llms_business_flagship_price, llms_individual_service_price, llms_personal_flagship_price
  -> Total documents modified: 2 of 99

[Step 4b] Synchronizing source code and unit test suites...
  [UPDATE SCOPE] Fields: Authoritative facts -> Target: python/*.py, root JS, and tests/*
  -> [MODIFIED] pricing-tests.js | Status: UPDATED | Changed Lines: 21
  -> Total source/test files updated: 1 of 48

[Step 5/6] Generating domain references and master dictionary...
  [UPDATE SCOPE] Fields: Tracked occurrences & references -> Target: nlp-fuzzy-expert/*_references.json
  -> Exhaustive scan located 28549 term occurrences across 99 documents
  -> Wrote pricing_references.json (3399 references)
  -> Wrote business_services_references.json (3285 references)
  -> Wrote config_references.json (6435 references)
  -> Master Dictionary generated: references_dictionary.json (28739 total references)
  -> Lightweight Manifest generated: references_manifest.json (108 documents indexed)

[Step 6/6] Generating domain markdown synchronization reports...
  [UPDATE SCOPE] Fields: Domain summaries & occurrence tables -> Target: nlp-fuzzy-expert/*.md reports
  -> Wrote pricing.md
  -> Wrote business-services.md
  -> Wrote config.md

================================================================================
                     SYNCHRONIZATION SUMMARY TABLE
================================================================================
+------------------+---------+------------------+----------------------+--------------------------+
| Target File      | Status  | Changed Lines    | Rules Fired          | Fields Synced            |
+------------------+---------+------------------+----------------------+--------------------------+
| faqs.html        | UPDATED | 1857, 1876, 1888 | FAQsPoliciesRule     | business_included_seats  |
| llms.txt         | UPDATED | 13, 40, 42       | LLMsTxtRule          | llms_business_flagshi... |
| pricing-tests.js | UPDATED | 21               | CodeTestSynchronizer | source_and_tests         |
+------------------+---------+------------------+----------------------+--------------------------+
Summary: 99 files evaluated | 3 updated | 96 in sync | 172 references tracked
================================================================================
[Coverage Report] Successfully appended coverage report to /tmp/tmpf1aaq3k7/LAST-COMMIT-REPORT.md (Combined: 97.86% | Elapsed: 0.0s)
[Coverage Report] Successfully appended coverage report to /tmp/tmpf1aaq3k7/LAST-COMMIT-REPORT.md (Combined: 97.86% | Elapsed: 0.0s)
[Coverage Report] Successfully appended coverage report to /tmp/tmpt0lh7ade/LAST-COMMIT-REPORT.md (Combined: 100.00% | Elapsed: 0.0s)
[Pre-Commit Report] Successfully recorded report (parent 7d06e87) to /tmp/tmpm8ldgy5p/LAST-COMMIT-REPORT.md
[Pre-Commit Report] Successfully recorded report (parent M	README.md) to /tmp/tmpm8ldgy5p/LAST-COMMIT-REPORT.md
[Pre-Commit Report] Successfully recorded report (parent ) to /tmp/tmpm8ldgy5p/LAST-COMMIT-REPORT.md
[README Commit Summary] Successfully updated /tmp/tmpwb_tjsmg/README.md from /tmp/tmpwb_tjsmg/rep.md and /tmp/tmpwb_tjsmg/METRICS.md
[README Commit Summary] Successfully updated /tmp/tmpwb_tjsmg/README.md from /tmp/tmpwb_tjsmg/rep.md and /tmp/tmpwb_tjsmg/METRICS.md
[README Commit Summary] Successfully updated /tmp/tmpwb_tjsmg/README_plain.md from /tmp/tmpwb_tjsmg/rep_min.md and /tmp/tmpwb_tjsmg/missing_METRICS.md
[README Commit Summary] Successfully updated /tmp/tmpwb_tjsmg/README_plain.md from /tmp/tmpwb_tjsmg/rep_min.md and /tmp/tmpwb_tjsmg/missing_METRICS.md
================================================================================
           NLP-FUZZY-EXPERT KNOWLEDGE SYNCHRONIZATION SYSTEM
================================================================================
  [DATA FLOW] Direction: data/*.json -> HTML/JS/MD/Code (STRICTLY UNIDIRECTIONAL / ONE-WAY)
[Step 1/6] Loading authoritative facts from data/*.json into KnowledgeBase...
  [UPDATE SCOPE] Domain: data/*.json -> Target: KnowledgeBase atomic facts cache
  -> Authoritative sources loaded: 11 data domains
     - data/business_services.json: 3 top-level entries
     - data/claude.json: 2 top-level entries
     - data/config.json: 12 top-level entries
     - data/contact.json: 6 top-level entries
     - data/faqs.json: 2 top-level entries
     - data/gemini.json: 2 top-level entries
     - data/guarantees.json: 5 top-level entries
     - data/personal_services.json: 3 top-level entries
     - data/pricing.json: 6 top-level entries
     - data/publications.json: 48 top-level entries
     - data/tiers.json: 1 top-level entries
  -> Total atomic propositions registered in KnowledgeBase: 310

[Step 2/6] Registering production rules in salience priority order...
  [UPDATE SCOPE] Domain: rules.py -> Target: InferenceEngine priority rule queue
  -> Active production rules registered: 18
     - Salience 100: PricingHeroPackagesRule (domain: pricing)
     - Salience 100: PricingCalculatorScriptRule (domain: pricing)
     - Salience  95: PricingCalculatorSelectOptionsRule (domain: pricing)
     - Salience  92: PricingSection5CardsRule (domain: pricing)
     - Salience  90: BusinessAITrainingRule (domain: business_training)
     ... and 13 additional production rules

[Step 3/6] Discovering target documents across repository...
  [UPDATE SCOPE] Domain: Repository scan -> Target: Target HTML and JS documents list
  -> Discovered 99 target documents (.html, .js) for synchronization.

[Step 4/6] Executing forward-chaining inference & computing line diffs...
  [UPDATE SCOPE] Fields: Authoritative pricing, tiers, curricula & configs -> Target: HTML/JS repository pages
  -> All 99 target documents evaluated. Zero changes required (fully in sync).

[Step 4b] Synchronizing source code and unit test suites...
  [UPDATE SCOPE] Fields: Authoritative facts -> Target: python/*.py, root JS, and tests/*
  -> All 48 source/test files evaluated. Zero changes required (fully in sync).

[Step 5/6] Generating domain references and master dictionary...
  [UPDATE SCOPE] Fields: Tracked occurrences & references -> Target: nlp-fuzzy-expert/*_references.json
  -> Reference dictionaries and manifest already in sync.

[Step 6/6] Generating domain markdown synchronization reports...
  [UPDATE SCOPE] Fields: Domain summaries & occurrence tables -> Target: nlp-fuzzy-expert/*.md reports
  -> Markdown reports already in sync.

================================================================================
                     SYNCHRONIZATION SUMMARY TABLE
================================================================================
+----------------------------+---------+---------------+-------------+---------------+
| Target File                | Status  | Changed Lines | Rules Fired | Fields Synced |
+----------------------------+---------+---------------+-------------+---------------+
| all-business-services.html | IN SYNC | -             | -           | -             |
| all-personal-services.html | IN SYNC | -             | -           | -             |
| contact.html               | IN SYNC | -             | -           | -             |
| faqs.html                  | IN SYNC | -             | -           | -             |
| guarantees.html            | IN SYNC | -             | -           | -             |
| index.html                 | IN SYNC | -             | -           | -             |
| llms.txt                   | IN SYNC | -             | -           | -             |
| pricing-calculator.js      | IN SYNC | -             | -           | -             |
| pricing.html               | IN SYNC | -             | -           | -             |
+----------------------------+---------+---------------+-------------+---------------+
Summary: 99 files evaluated | 0 updated | 99 in sync | 0 references tracked
================================================================================
================================================================================
           NLP-FUZZY-EXPERT KNOWLEDGE SYNCHRONIZATION SYSTEM
================================================================================
  [DATA FLOW] Direction: data/*.json -> HTML/JS/MD/Code (STRICTLY UNIDIRECTIONAL / ONE-WAY)
[Step 1/6] Loading authoritative facts from data/*.json into KnowledgeBase...
  [UPDATE SCOPE] Domain: data/*.json -> Target: KnowledgeBase atomic facts cache
  -> Authoritative sources loaded: 11 data domains
     - data/business_services.json: 3 top-level entries
     - data/claude.json: 2 top-level entries
     - data/config.json: 12 top-level entries
     - data/contact.json: 6 top-level entries
     - data/faqs.json: 2 top-level entries
     - data/gemini.json: 2 top-level entries
     - data/guarantees.json: 5 top-level entries
     - data/personal_services.json: 3 top-level entries
     - data/pricing.json: 6 top-level entries
     - data/publications.json: 48 top-level entries
     - data/tiers.json: 1 top-level entries
  -> Total atomic propositions registered in KnowledgeBase: 310

[Step 2/6] Registering production rules in salience priority order...
  [UPDATE SCOPE] Domain: rules.py -> Target: InferenceEngine priority rule queue
  -> Active production rules registered: 18
     - Salience 100: PricingHeroPackagesRule (domain: pricing)
     - Salience 100: PricingCalculatorScriptRule (domain: pricing)
     - Salience  95: PricingCalculatorSelectOptionsRule (domain: pricing)
     - Salience  92: PricingSection5CardsRule (domain: pricing)
     - Salience  90: BusinessAITrainingRule (domain: business_training)
     ... and 13 additional production rules

[Step 3/6] Discovering target documents across repository...
  [UPDATE SCOPE] Domain: Repository scan -> Target: Target HTML and JS documents list
  -> Discovered 99 target documents (.html, .js) for synchronization.

[Step 4/6] Executing forward-chaining inference & computing line diffs...
  [UPDATE SCOPE] Fields: Authoritative pricing, tiers, curricula & configs -> Target: HTML/JS repository pages
  -> All 99 target documents evaluated. Zero changes required (fully in sync).

[Step 4b] Synchronizing source code and unit test suites...
  [UPDATE SCOPE] Fields: Authoritative facts -> Target: python/*.py, root JS, and tests/*
  -> All 48 source/test files evaluated. Zero changes required (fully in sync).

[Step 5/6] Generating domain references and master dictionary...
  [UPDATE SCOPE] Fields: Tracked occurrences & references -> Target: nlp-fuzzy-expert/*_references.json
  -> Reference dictionaries and manifest already in sync.

[Step 6/6] Generating domain markdown synchronization reports...
  [UPDATE SCOPE] Fields: Domain summaries & occurrence tables -> Target: nlp-fuzzy-expert/*.md reports
  -> Markdown reports already in sync.

================================================================================
                     SYNCHRONIZATION SUMMARY TABLE
================================================================================
+----------------------------+---------+---------------+-------------+---------------+
| Target File                | Status  | Changed Lines | Rules Fired | Fields Synced |
+----------------------------+---------+---------------+-------------+---------------+
| all-business-services.html | IN SYNC | -             | -           | -             |
| all-personal-services.html | IN SYNC | -             | -           | -             |
| contact.html               | IN SYNC | -             | -           | -             |
| faqs.html                  | IN SYNC | -             | -           | -             |
| guarantees.html            | IN SYNC | -             | -           | -             |
| index.html                 | IN SYNC | -             | -           | -             |
| llms.txt                   | IN SYNC | -             | -           | -             |
| pricing-calculator.js      | IN SYNC | -             | -           | -             |
| pricing.html               | IN SYNC | -             | -           | -             |
+----------------------------+---------+---------------+-------------+---------------+
Summary: 99 files evaluated | 0 updated | 99 in sync | 0 references tracked
================================================================================
  -> Wrote business_services_references.json (0 references)
  -> Wrote personal_services_references.json (0 references)
  -> Wrote tiers_references.json (0 references)
  -> Wrote claude_references.json (0 references)
  -> Wrote gemini_references.json (0 references)
  -> Wrote publications_references.json (0 references)
  -> Wrote config_references.json (0 references)
  -> Wrote faqs_references.json (0 references)
  -> Wrote contact_references.json (0 references)
  -> Wrote guarantees_references.json (0 references)
  -> Wrote pricing_references.json (1 references)
  -> Wrote business_services_references.json (0 references)
  -> Wrote personal_services_references.json (0 references)
  -> Wrote tiers_references.json (0 references)
  -> Wrote claude_references.json (0 references)
  -> Wrote gemini_references.json (0 references)
  -> Wrote publications_references.json (0 references)
  -> Wrote config_references.json (0 references)
  -> Wrote faqs_references.json (0 references)
  -> Wrote contact_references.json (0 references)
  -> Wrote guarantees_references.json (0 references)
  -> Master Dictionary generated: references_dictionary.json (1 total references)
  -> Lightweight Manifest generated: references_manifest.json (1 documents indexed)
  -> Wrote pricing.md
  -> Wrote business-services.md
  -> Wrote personal-services.md
  -> Wrote tiers.md
  -> Wrote claude.md
  -> Wrote gemini.md
  -> Wrote publications.md
  -> Wrote config.md
  -> Wrote faqs.md
  -> Wrote contact.md
  -> Wrote guarantees.md
================================================================================
           NLP-FUZZY-EXPERT KNOWLEDGE SYNCHRONIZATION SYSTEM
================================================================================
  [DATA FLOW] Direction: data/*.json -> HTML/JS/MD/Code (STRICTLY UNIDIRECTIONAL / ONE-WAY)
[Step 1/6] Loading authoritative facts from data/*.json into KnowledgeBase...
  [UPDATE SCOPE] Domain: data/*.json -> Target: KnowledgeBase atomic facts cache
  -> Authoritative sources loaded: 11 data domains
     - data/business_services.json: 3 top-level entries
     - data/claude.json: 2 top-level entries
     - data/config.json: 12 top-level entries
     - data/contact.json: 6 top-level entries
     - data/faqs.json: 2 top-level entries
     - data/gemini.json: 2 top-level entries
     - data/guarantees.json: 5 top-level entries
     - data/personal_services.json: 3 top-level entries
     - data/pricing.json: 6 top-level entries
     - data/publications.json: 48 top-level entries
     - data/tiers.json: 1 top-level entries
  -> Total atomic propositions registered in KnowledgeBase: 310

[Step 2/6] Registering production rules in salience priority order...
  [UPDATE SCOPE] Domain: rules.py -> Target: InferenceEngine priority rule queue
  -> Active production rules registered: 18
     - Salience 100: PricingHeroPackagesRule (domain: pricing)
     - Salience 100: PricingCalculatorScriptRule (domain: pricing)
     - Salience  95: PricingCalculatorSelectOptionsRule (domain: pricing)
     - Salience  92: PricingSection5CardsRule (domain: pricing)
     - Salience  90: BusinessAITrainingRule (domain: business_training)
     ... and 13 additional production rules

[Step 3/6] Discovering target documents across repository...
  [UPDATE SCOPE] Domain: Repository scan -> Target: Target HTML and JS documents list
  -> Discovered 99 target documents (.html, .js) for synchronization.

[Step 4/6] Executing forward-chaining inference & computing line diffs...
  [UPDATE SCOPE] Fields: Authoritative pricing, tiers, curricula & configs -> Target: HTML/JS repository pages
  -> All 99 target documents evaluated. Zero changes required (fully in sync).

[Step 4b] Synchronizing source code and unit test suites...
  [UPDATE SCOPE] Fields: Authoritative facts -> Target: python/*.py, root JS, and tests/*
  -> All 48 source/test files evaluated. Zero changes required (fully in sync).

[Step 5/6] Generating domain references and master dictionary...
  [UPDATE SCOPE] Fields: Tracked occurrences & references -> Target: nlp-fuzzy-expert/*_references.json
  -> Reference dictionaries and manifest already in sync.

[Step 6/6] Generating domain markdown synchronization reports...
  [UPDATE SCOPE] Fields: Domain summaries & occurrence tables -> Target: nlp-fuzzy-expert/*.md reports
  -> Markdown reports already in sync.

================================================================================
                     SYNCHRONIZATION SUMMARY TABLE
================================================================================
+----------------------------+---------+---------------+-------------+---------------+
| Target File                | Status  | Changed Lines | Rules Fired | Fields Synced |
+----------------------------+---------+---------------+-------------+---------------+
| all-business-services.html | IN SYNC | -             | -           | -             |
| all-personal-services.html | IN SYNC | -             | -           | -             |
| contact.html               | IN SYNC | -             | -           | -             |
| faqs.html                  | IN SYNC | -             | -           | -             |
| guarantees.html            | IN SYNC | -             | -           | -             |
| index.html                 | IN SYNC | -             | -           | -             |
| llms.txt                   | IN SYNC | -             | -           | -             |
| pricing-calculator.js      | IN SYNC | -             | -           | -             |
| pricing.html               | IN SYNC | -             | -           | -             |
+----------------------------+---------+---------------+-------------+---------------+
Summary: 99 files evaluated | 0 updated | 99 in sync | 0 references tracked
================================================================================
================================================================================
           NLP-FUZZY-EXPERT KNOWLEDGE SYNCHRONIZATION SYSTEM
================================================================================
  [DATA FLOW] Direction: data/*.json -> HTML/JS/MD/Code (STRICTLY UNIDIRECTIONAL / ONE-WAY)
[Step 1/6] Loading authoritative facts from data/*.json into KnowledgeBase...
  [UPDATE SCOPE] Domain: data/*.json -> Target: KnowledgeBase atomic facts cache
  -> Authoritative sources loaded: 11 data domains
     - data/business_services.json: 3 top-level entries
     - data/claude.json: 2 top-level entries
     - data/config.json: 12 top-level entries
     - data/contact.json: 6 top-level entries
     - data/faqs.json: 2 top-level entries
     - data/gemini.json: 2 top-level entries
     - data/guarantees.json: 5 top-level entries
     - data/personal_services.json: 3 top-level entries
     - data/pricing.json: 6 top-level entries
     - data/publications.json: 48 top-level entries
     - data/tiers.json: 1 top-level entries
  -> Total atomic propositions registered in KnowledgeBase: 310

[Step 2/6] Registering production rules in salience priority order...
  [UPDATE SCOPE] Domain: rules.py -> Target: InferenceEngine priority rule queue
  -> Active production rules registered: 18
     - Salience 100: PricingHeroPackagesRule (domain: pricing)
     - Salience 100: PricingCalculatorScriptRule (domain: pricing)
     - Salience  95: PricingCalculatorSelectOptionsRule (domain: pricing)
     - Salience  92: PricingSection5CardsRule (domain: pricing)
     - Salience  90: BusinessAITrainingRule (domain: business_training)
     ... and 13 additional production rules

[Step 3/6] Discovering target documents across repository...
  [UPDATE SCOPE] Domain: Repository scan -> Target: Target HTML and JS documents list
  -> Discovered 99 target documents (.html, .js) for synchronization.

[Step 4/6] Executing forward-chaining inference & computing line diffs...
  [UPDATE SCOPE] Fields: Authoritative pricing, tiers, curricula & configs -> Target: HTML/JS repository pages
  -> All 99 target documents evaluated. Zero changes required (fully in sync).

[Step 4b] Synchronizing source code and unit test suites...
  [UPDATE SCOPE] Fields: Authoritative facts -> Target: python/*.py, root JS, and tests/*
  -> All 48 source/test files evaluated. Zero changes required (fully in sync).

[Step 5/6] Generating domain references and master dictionary...
  [UPDATE SCOPE] Fields: Tracked occurrences & references -> Target: nlp-fuzzy-expert/*_references.json
  -> Reference dictionaries and manifest already in sync.

[Step 6/6] Generating domain markdown synchronization reports...
  [UPDATE SCOPE] Fields: Domain summaries & occurrence tables -> Target: nlp-fuzzy-expert/*.md reports
  -> Markdown reports already in sync.

================================================================================
                     SYNCHRONIZATION SUMMARY TABLE
================================================================================
+----------------------------+---------+---------------+-------------+---------------+
| Target File                | Status  | Changed Lines | Rules Fired | Fields Synced |
+----------------------------+---------+---------------+-------------+---------------+
| all-business-services.html | IN SYNC | -             | -           | -             |
| all-personal-services.html | IN SYNC | -             | -           | -             |
| contact.html               | IN SYNC | -             | -           | -             |
| faqs.html                  | IN SYNC | -             | -           | -             |
| guarantees.html            | IN SYNC | -             | -           | -             |
| index.html                 | IN SYNC | -             | -           | -             |
| llms.txt                   | IN SYNC | -             | -           | -             |
| pricing-calculator.js      | IN SYNC | -             | -           | -             |
| pricing.html               | IN SYNC | -             | -           | -             |
+----------------------------+---------+---------------+-------------+---------------+
Summary: 99 files evaluated | 0 updated | 99 in sync | 0 references tracked
================================================================================
================================================================================
           NLP-FUZZY-EXPERT KNOWLEDGE SYNCHRONIZATION SYSTEM
================================================================================
  [DATA FLOW] Direction: data/*.json -> HTML/JS/MD/Code (STRICTLY UNIDIRECTIONAL / ONE-WAY)
[Step 1/6] Loading authoritative facts from data/*.json into KnowledgeBase...
  [UPDATE SCOPE] Domain: data/*.json -> Target: KnowledgeBase atomic facts cache
  -> Authoritative sources loaded: 11 data domains
     - data/business_services.json: 3 top-level entries
     - data/claude.json: 2 top-level entries
     - data/config.json: 12 top-level entries
     - data/contact.json: 6 top-level entries
     - data/faqs.json: 2 top-level entries
     - data/gemini.json: 2 top-level entries
     - data/guarantees.json: 5 top-level entries
     - data/personal_services.json: 3 top-level entries
     - data/pricing.json: 6 top-level entries
     - data/publications.json: 48 top-level entries
     - data/tiers.json: 1 top-level entries
  -> Total atomic propositions registered in KnowledgeBase: 310

[Step 2/6] Registering production rules in salience priority order...
  [UPDATE SCOPE] Domain: rules.py -> Target: InferenceEngine priority rule queue
  -> Active production rules registered: 18
     - Salience 100: PricingHeroPackagesRule (domain: pricing)
     - Salience 100: PricingCalculatorScriptRule (domain: pricing)
     - Salience  95: PricingCalculatorSelectOptionsRule (domain: pricing)
     - Salience  92: PricingSection5CardsRule (domain: pricing)
     - Salience  90: BusinessAITrainingRule (domain: business_training)
     ... and 13 additional production rules

[Step 3/6] Discovering target documents across repository...
  [UPDATE SCOPE] Domain: Repository scan -> Target: Target HTML and JS documents list
  -> Discovered 99 target documents (.html, .js) for synchronization.

[Step 4/6] Executing forward-chaining inference & computing line diffs...
  [UPDATE SCOPE] Fields: Authoritative pricing, tiers, curricula & configs -> Target: HTML/JS repository pages
  -> All 99 target documents evaluated. Zero changes required (fully in sync).

[Step 4b] Synchronizing source code and unit test suites...
  [UPDATE SCOPE] Fields: Authoritative facts -> Target: python/*.py, root JS, and tests/*
  -> All 48 source/test files evaluated. Zero changes required (fully in sync).

[Step 5/6] Generating domain references and master dictionary...
  [UPDATE SCOPE] Fields: Tracked occurrences & references -> Target: nlp-fuzzy-expert/*_references.json
  -> Reference dictionaries and manifest already in sync.

[Step 6/6] Generating domain markdown synchronization reports...
  [UPDATE SCOPE] Fields: Domain summaries & occurrence tables -> Target: nlp-fuzzy-expert/*.md reports
  -> Markdown reports already in sync.

================================================================================
                     SYNCHRONIZATION SUMMARY TABLE
================================================================================
+----------------------------+---------+---------------+-------------+---------------+
| Target File                | Status  | Changed Lines | Rules Fired | Fields Synced |
+----------------------------+---------+---------------+-------------+---------------+
| all-business-services.html | IN SYNC | -             | -           | -             |
| all-personal-services.html | IN SYNC | -             | -           | -             |
| contact.html               | IN SYNC | -             | -           | -             |
| faqs.html                  | IN SYNC | -             | -           | -             |
| guarantees.html            | IN SYNC | -             | -           | -             |
| index.html                 | IN SYNC | -             | -           | -             |
| llms.txt                   | IN SYNC | -             | -           | -             |
| pricing-calculator.js      | IN SYNC | -             | -           | -             |
| pricing.html               | IN SYNC | -             | -           | -             |
+----------------------------+---------+---------------+-------------+---------------+
Summary: 99 files evaluated | 0 updated | 99 in sync | 0 references tracked
================================================================================
================================================================================
           NLP-FUZZY-EXPERT KNOWLEDGE SYNCHRONIZATION SYSTEM
================================================================================
  [DATA FLOW] Direction: data/*.json -> HTML/JS/MD/Code (STRICTLY UNIDIRECTIONAL / ONE-WAY)
[Step 1/6] Loading authoritative facts from data/*.json into KnowledgeBase...
  [UPDATE SCOPE] Domain: data/*.json -> Target: KnowledgeBase atomic facts cache
  -> Authoritative sources loaded: 11 data domains
     - data/business_services.json: 3 top-level entries
     - data/claude.json: 2 top-level entries
     - data/config.json: 12 top-level entries
     - data/contact.json: 6 top-level entries
     - data/faqs.json: 2 top-level entries
     - data/gemini.json: 2 top-level entries
     - data/guarantees.json: 5 top-level entries
     - data/personal_services.json: 3 top-level entries
     - data/pricing.json: 6 top-level entries
     - data/publications.json: 48 top-level entries
     - data/tiers.json: 1 top-level entries
  -> Total atomic propositions registered in KnowledgeBase: 310

[Step 2/6] Registering production rules in salience priority order...
  [UPDATE SCOPE] Domain: rules.py -> Target: InferenceEngine priority rule queue
  -> Active production rules registered: 18
     - Salience 100: PricingHeroPackagesRule (domain: pricing)
     - Salience 100: PricingCalculatorScriptRule (domain: pricing)
     - Salience  95: PricingCalculatorSelectOptionsRule (domain: pricing)
     - Salience  92: PricingSection5CardsRule (domain: pricing)
     - Salience  90: BusinessAITrainingRule (domain: business_training)
     ... and 13 additional production rules

[Step 3/6] Discovering target documents across repository...
  [UPDATE SCOPE] Domain: Repository scan -> Target: Target HTML and JS documents list
  -> Discovered 99 target documents (.html, .js) for synchronization.

[Step 4/6] Executing forward-chaining inference & computing line diffs...
  [UPDATE SCOPE] Fields: Authoritative pricing, tiers, curricula & configs -> Target: HTML/JS repository pages
  -> All 99 target documents evaluated. Zero changes required (fully in sync).

[Step 4b] Synchronizing source code and unit test suites...
  [UPDATE SCOPE] Fields: Authoritative facts -> Target: python/*.py, root JS, and tests/*
  -> All 48 source/test files evaluated. Zero changes required (fully in sync).

[Step 5/6] Generating domain references and master dictionary...
  [UPDATE SCOPE] Fields: Tracked occurrences & references -> Target: nlp-fuzzy-expert/*_references.json
  -> Reference dictionaries and manifest already in sync.

[Step 6/6] Generating domain markdown synchronization reports...
  [UPDATE SCOPE] Fields: Domain summaries & occurrence tables -> Target: nlp-fuzzy-expert/*.md reports
  -> Markdown reports already in sync.

================================================================================
                     SYNCHRONIZATION SUMMARY TABLE
================================================================================
+----------------------------+---------+---------------+-------------+---------------+
| Target File                | Status  | Changed Lines | Rules Fired | Fields Synced |
+----------------------------+---------+---------------+-------------+---------------+
| all-business-services.html | IN SYNC | -             | -           | -             |
| all-personal-services.html | IN SYNC | -             | -           | -             |
| contact.html               | IN SYNC | -             | -           | -             |
| faqs.html                  | IN SYNC | -             | -           | -             |
| guarantees.html            | IN SYNC | -             | -           | -             |
| index.html                 | IN SYNC | -             | -           | -             |
| llms.txt                   | IN SYNC | -             | -           | -             |
| pricing-calculator.js      | IN SYNC | -             | -           | -             |
| pricing.html               | IN SYNC | -             | -           | -             |
+----------------------------+---------+---------------+-------------+---------------+
Summary: 99 files evaluated | 0 updated | 99 in sync | 0 references tracked
================================================================================
================================================================================
           NLP-FUZZY-EXPERT KNOWLEDGE SYNCHRONIZATION SYSTEM
================================================================================
  [DATA FLOW] Direction: data/*.json -> HTML/JS/MD/Code (STRICTLY UNIDIRECTIONAL / ONE-WAY)
[Step 1/6] Loading authoritative facts from data/*.json into KnowledgeBase...
  [UPDATE SCOPE] Domain: data/*.json -> Target: KnowledgeBase atomic facts cache
  -> Authoritative sources loaded: 11 data domains
     - data/business_services.json: 3 top-level entries
     - data/claude.json: 2 top-level entries
     - data/config.json: 12 top-level entries
     - data/contact.json: 6 top-level entries
     - data/faqs.json: 2 top-level entries
     - data/gemini.json: 2 top-level entries
     - data/guarantees.json: 5 top-level entries
     - data/personal_services.json: 3 top-level entries
     - data/pricing.json: 6 top-level entries
     - data/publications.json: 48 top-level entries
     - data/tiers.json: 1 top-level entries
  -> Total atomic propositions registered in KnowledgeBase: 310

[Step 2/6] Registering production rules in salience priority order...
  [UPDATE SCOPE] Domain: rules.py -> Target: InferenceEngine priority rule queue
  -> Active production rules registered: 18
     - Salience 100: PricingHeroPackagesRule (domain: pricing)
     - Salience 100: PricingCalculatorScriptRule (domain: pricing)
     - Salience  95: PricingCalculatorSelectOptionsRule (domain: pricing)
     - Salience  92: PricingSection5CardsRule (domain: pricing)
     - Salience  90: BusinessAITrainingRule (domain: business_training)
     ... and 13 additional production rules

[Step 3/6] Discovering target documents across repository...
  [UPDATE SCOPE] Domain: Repository scan -> Target: Target HTML and JS documents list
  -> Discovered 99 target documents (.html, .js) for synchronization.

[Step 4/6] Executing forward-chaining inference & computing line diffs...
  [UPDATE SCOPE] Fields: Authoritative pricing, tiers, curricula & configs -> Target: HTML/JS repository pages
  -> All 99 target documents evaluated. Zero changes required (fully in sync).

[Step 4b] Synchronizing source code and unit test suites...
  [UPDATE SCOPE] Fields: Authoritative facts -> Target: python/*.py, root JS, and tests/*
  -> All 48 source/test files evaluated. Zero changes required (fully in sync).

[Step 5/6] Generating domain references and master dictionary...
  [UPDATE SCOPE] Fields: Tracked occurrences & references -> Target: nlp-fuzzy-expert/*_references.json
  -> Reference dictionaries and manifest already in sync.

[Step 6/6] Generating domain markdown synchronization reports...
  [UPDATE SCOPE] Fields: Domain summaries & occurrence tables -> Target: nlp-fuzzy-expert/*.md reports
  -> Markdown reports already in sync.

================================================================================
                     SYNCHRONIZATION SUMMARY TABLE
================================================================================
+----------------------------+---------+---------------+-------------+---------------+
| Target File                | Status  | Changed Lines | Rules Fired | Fields Synced |
+----------------------------+---------+---------------+-------------+---------------+
| all-business-services.html | IN SYNC | -             | -           | -             |
| all-personal-services.html | IN SYNC | -             | -           | -             |
| contact.html               | IN SYNC | -             | -           | -             |
| faqs.html                  | IN SYNC | -             | -           | -             |
| guarantees.html            | IN SYNC | -             | -           | -             |
| index.html                 | IN SYNC | -             | -           | -             |
| llms.txt                   | IN SYNC | -             | -           | -             |
| pricing-calculator.js      | IN SYNC | -             | -           | -             |
| pricing.html               | IN SYNC | -             | -           | -             |
+----------------------------+---------+---------------+-------------+---------------+
Summary: 99 files evaluated | 0 updated | 99 in sync | 0 references tracked
================================================================================
================================================================================
           NLP-FUZZY-EXPERT KNOWLEDGE SYNCHRONIZATION SYSTEM
================================================================================
  [DATA FLOW] Direction: data/*.json -> HTML/JS/MD/Code (STRICTLY UNIDIRECTIONAL / ONE-WAY)
[Step 1/6] Loading authoritative facts from data/*.json into KnowledgeBase...
  [UPDATE SCOPE] Domain: data/*.json -> Target: KnowledgeBase atomic facts cache
  -> Authoritative sources loaded: 11 data domains
     - data/business_services.json: 3 top-level entries
     - data/claude.json: 2 top-level entries
     - data/config.json: 12 top-level entries
     - data/contact.json: 6 top-level entries
     - data/faqs.json: 2 top-level entries
     - data/gemini.json: 2 top-level entries
     - data/guarantees.json: 5 top-level entries
     - data/personal_services.json: 3 top-level entries
     - data/pricing.json: 6 top-level entries
     - data/publications.json: 48 top-level entries
     - data/tiers.json: 1 top-level entries
  -> Total atomic propositions registered in KnowledgeBase: 310

[Step 2/6] Registering production rules in salience priority order...
  [UPDATE SCOPE] Domain: rules.py -> Target: InferenceEngine priority rule queue
  -> Active production rules registered: 18
     - Salience 100: PricingHeroPackagesRule (domain: pricing)
     - Salience 100: PricingCalculatorScriptRule (domain: pricing)
     - Salience  95: PricingCalculatorSelectOptionsRule (domain: pricing)
     - Salience  92: PricingSection5CardsRule (domain: pricing)
     - Salience  90: BusinessAITrainingRule (domain: business_training)
     ... and 13 additional production rules

[Step 3/6] Discovering target documents across repository...
  [UPDATE SCOPE] Domain: Repository scan -> Target: Target HTML and JS documents list
  -> Discovered 99 target documents (.html, .js) for synchronization.

[Step 4/6] Executing forward-chaining inference & computing line diffs...
  [UPDATE SCOPE] Fields: Authoritative pricing, tiers, curricula & configs -> Target: HTML/JS repository pages
  -> All 99 target documents evaluated. Zero changes required (fully in sync).

[Step 4b] Synchronizing source code and unit test suites...
  [UPDATE SCOPE] Fields: Authoritative facts -> Target: python/*.py, root JS, and tests/*
  -> All 48 source/test files evaluated. Zero changes required (fully in sync).

[Step 5/6] Generating domain references and master dictionary...
  [UPDATE SCOPE] Fields: Tracked occurrences & references -> Target: nlp-fuzzy-expert/*_references.json
  -> Reference dictionaries and manifest already in sync.

[Step 6/6] Generating domain markdown synchronization reports...
  [UPDATE SCOPE] Fields: Domain summaries & occurrence tables -> Target: nlp-fuzzy-expert/*.md reports
  -> Markdown reports already in sync.

================================================================================
                     SYNCHRONIZATION SUMMARY TABLE
================================================================================
+----------------------------+---------+---------------+-------------+---------------+
| Target File                | Status  | Changed Lines | Rules Fired | Fields Synced |
+----------------------------+---------+---------------+-------------+---------------+
| all-business-services.html | IN SYNC | -             | -           | -             |
| all-personal-services.html | IN SYNC | -             | -           | -             |
| contact.html               | IN SYNC | -             | -           | -             |
| faqs.html                  | IN SYNC | -             | -           | -             |
| guarantees.html            | IN SYNC | -             | -           | -             |
| index.html                 | IN SYNC | -             | -           | -             |
| llms.txt                   | IN SYNC | -             | -           | -             |
| pricing-calculator.js      | IN SYNC | -             | -           | -             |
| pricing.html               | IN SYNC | -             | -           | -             |
+----------------------------+---------+---------------+-------------+---------------+
Summary: 99 files evaluated | 0 updated | 99 in sync | 0 references tracked
================================================================================
================================================================================
           NLP-FUZZY-EXPERT KNOWLEDGE SYNCHRONIZATION SYSTEM
================================================================================
  [DATA FLOW] Direction: data/*.json -> HTML/JS/MD/Code (STRICTLY UNIDIRECTIONAL / ONE-WAY)
[Step 1/6] Loading authoritative facts from data/*.json into KnowledgeBase...
  [UPDATE SCOPE] Domain: data/*.json -> Target: KnowledgeBase atomic facts cache
  -> Authoritative sources loaded: 11 data domains
     - data/business_services.json: 3 top-level entries
     - data/claude.json: 2 top-level entries
     - data/config.json: 12 top-level entries
     - data/contact.json: 6 top-level entries
     - data/faqs.json: 2 top-level entries
     - data/gemini.json: 2 top-level entries
     - data/guarantees.json: 5 top-level entries
     - data/personal_services.json: 3 top-level entries
     - data/pricing.json: 6 top-level entries
     - data/publications.json: 48 top-level entries
     - data/tiers.json: 1 top-level entries
  -> Total atomic propositions registered in KnowledgeBase: 310

[Step 2/6] Registering production rules in salience priority order...
  [UPDATE SCOPE] Domain: rules.py -> Target: InferenceEngine priority rule queue
  -> Active production rules registered: 18
     - Salience 100: PricingHeroPackagesRule (domain: pricing)
     - Salience 100: PricingCalculatorScriptRule (domain: pricing)
     - Salience  95: PricingCalculatorSelectOptionsRule (domain: pricing)
     - Salience  92: PricingSection5CardsRule (domain: pricing)
     - Salience  90: BusinessAITrainingRule (domain: business_training)
     ... and 13 additional production rules

[Step 3/6] Discovering target documents across repository...
  [UPDATE SCOPE] Domain: Repository scan -> Target: Target HTML and JS documents list
  -> Discovered 99 target documents (.html, .js) for synchronization.

[Step 4/6] Executing forward-chaining inference & computing line diffs...
  [UPDATE SCOPE] Fields: Authoritative pricing, tiers, curricula & configs -> Target: HTML/JS repository pages
  -> All 99 target documents evaluated. Zero changes required (fully in sync).

[Step 4b] Synchronizing source code and unit test suites...
  [UPDATE SCOPE] Fields: Authoritative facts -> Target: python/*.py, root JS, and tests/*
  -> All 48 source/test files evaluated. Zero changes required (fully in sync).

[Step 5/6] Generating domain references and master dictionary...
  [UPDATE SCOPE] Fields: Tracked occurrences & references -> Target: nlp-fuzzy-expert/*_references.json
  -> Reference dictionaries and manifest already in sync.

[Step 6/6] Generating domain markdown synchronization reports...
  [UPDATE SCOPE] Fields: Domain summaries & occurrence tables -> Target: nlp-fuzzy-expert/*.md reports
  -> Markdown reports already in sync.

================================================================================
                     SYNCHRONIZATION SUMMARY TABLE
================================================================================
+----------------------------+---------+---------------+-------------+---------------+
| Target File                | Status  | Changed Lines | Rules Fired | Fields Synced |
+----------------------------+---------+---------------+-------------+---------------+
| all-business-services.html | IN SYNC | -             | -           | -             |
| all-personal-services.html | IN SYNC | -             | -           | -             |
| contact.html               | IN SYNC | -             | -           | -             |
| faqs.html                  | IN SYNC | -             | -           | -             |
| guarantees.html            | IN SYNC | -             | -           | -             |
| index.html                 | IN SYNC | -             | -           | -             |
| llms.txt                   | IN SYNC | -             | -           | -             |
| pricing-calculator.js      | IN SYNC | -             | -           | -             |
| pricing.html               | IN SYNC | -             | -           | -             |
+----------------------------+---------+---------------+-------------+---------------+
Summary: 99 files evaluated | 0 updated | 99 in sync | 0 references tracked
================================================================================
================================================================================
           NLP-FUZZY-EXPERT KNOWLEDGE SYNCHRONIZATION SYSTEM
================================================================================
  [DATA FLOW] Direction: data/*.json -> HTML/JS/MD/Code (STRICTLY UNIDIRECTIONAL / ONE-WAY)
[Step 1/6] Loading authoritative facts from data/*.json into KnowledgeBase...
  [UPDATE SCOPE] Domain: data/*.json -> Target: KnowledgeBase atomic facts cache
  -> Authoritative sources loaded: 11 data domains
     - data/business_services.json: 3 top-level entries
     - data/claude.json: 2 top-level entries
     - data/config.json: 12 top-level entries
     - data/contact.json: 6 top-level entries
     - data/faqs.json: 2 top-level entries
     - data/gemini.json: 2 top-level entries
     - data/guarantees.json: 5 top-level entries
     - data/personal_services.json: 3 top-level entries
     - data/pricing.json: 6 top-level entries
     - data/publications.json: 48 top-level entries
     - data/tiers.json: 1 top-level entries
  -> Total atomic propositions registered in KnowledgeBase: 310

[Step 2/6] Registering production rules in salience priority order...
  [UPDATE SCOPE] Domain: rules.py -> Target: InferenceEngine priority rule queue
  -> Active production rules registered: 18
     - Salience 100: PricingHeroPackagesRule (domain: pricing)
     - Salience 100: PricingCalculatorScriptRule (domain: pricing)
     - Salience  95: PricingCalculatorSelectOptionsRule (domain: pricing)
     - Salience  92: PricingSection5CardsRule (domain: pricing)
     - Salience  90: BusinessAITrainingRule (domain: business_training)
     ... and 13 additional production rules

[Step 3/6] Discovering target documents across repository...
  [UPDATE SCOPE] Domain: Repository scan -> Target: Target HTML and JS documents list
  -> Discovered 99 target documents (.html, .js) for synchronization.

[Step 4/6] Executing forward-chaining inference & computing line diffs...
  [UPDATE SCOPE] Fields: Authoritative pricing, tiers, curricula & configs -> Target: HTML/JS repository pages
  -> All 99 target documents evaluated. Zero changes required (fully in sync).

[Step 4b] Synchronizing source code and unit test suites...
  [UPDATE SCOPE] Fields: Authoritative facts -> Target: python/*.py, root JS, and tests/*
  -> All 48 source/test files evaluated. Zero changes required (fully in sync).

[Step 5/6] Generating domain references and master dictionary...
  [UPDATE SCOPE] Fields: Tracked occurrences & references -> Target: nlp-fuzzy-expert/*_references.json
  -> Reference dictionaries and manifest already in sync.

[Step 6/6] Generating domain markdown synchronization reports...
  [UPDATE SCOPE] Fields: Domain summaries & occurrence tables -> Target: nlp-fuzzy-expert/*.md reports
  -> Markdown reports already in sync.

================================================================================
                     SYNCHRONIZATION SUMMARY TABLE
================================================================================
+----------------------------+---------+---------------+-------------+---------------+
| Target File                | Status  | Changed Lines | Rules Fired | Fields Synced |
+----------------------------+---------+---------------+-------------+---------------+
| all-business-services.html | IN SYNC | -             | -           | -             |
| all-personal-services.html | IN SYNC | -             | -           | -             |
| contact.html               | IN SYNC | -             | -           | -             |
| faqs.html                  | IN SYNC | -             | -           | -             |
| guarantees.html            | IN SYNC | -             | -           | -             |
| index.html                 | IN SYNC | -             | -           | -             |
| llms.txt                   | IN SYNC | -             | -           | -             |
| pricing-calculator.js      | IN SYNC | -             | -           | -             |
| pricing.html               | IN SYNC | -             | -           | -             |
+----------------------------+---------+---------------+-------------+---------------+
Summary: 99 files evaluated | 0 updated | 99 in sync | 0 references tracked
================================================================================
================================================================================
           NLP-FUZZY-EXPERT KNOWLEDGE SYNCHRONIZATION SYSTEM
================================================================================
  [DATA FLOW] Direction: data/*.json -> HTML/JS/MD/Code (STRICTLY UNIDIRECTIONAL / ONE-WAY)
[Step 1/6] Loading authoritative facts from data/*.json into KnowledgeBase...
  [UPDATE SCOPE] Domain: data/*.json -> Target: KnowledgeBase atomic facts cache
  -> Authoritative sources loaded: 11 data domains
     - data/business_services.json: 3 top-level entries
     - data/claude.json: 2 top-level entries
     - data/config.json: 12 top-level entries
     - data/contact.json: 6 top-level entries
     - data/faqs.json: 2 top-level entries
     - data/gemini.json: 2 top-level entries
     - data/guarantees.json: 5 top-level entries
     - data/personal_services.json: 3 top-level entries
     - data/pricing.json: 6 top-level entries
     - data/publications.json: 48 top-level entries
     - data/tiers.json: 1 top-level entries
  -> Total atomic propositions registered in KnowledgeBase: 310

[Step 2/6] Registering production rules in salience priority order...
  [UPDATE SCOPE] Domain: rules.py -> Target: InferenceEngine priority rule queue
  -> Active production rules registered: 18
     - Salience 100: PricingHeroPackagesRule (domain: pricing)
     - Salience 100: PricingCalculatorScriptRule (domain: pricing)
     - Salience  95: PricingCalculatorSelectOptionsRule (domain: pricing)
     - Salience  92: PricingSection5CardsRule (domain: pricing)
     - Salience  90: BusinessAITrainingRule (domain: business_training)
     ... and 13 additional production rules

[Step 3/6] Discovering target documents across repository...
  [UPDATE SCOPE] Domain: Repository scan -> Target: Target HTML and JS documents list
  -> Discovered 99 target documents (.html, .js) for synchronization.

[Step 4/6] Executing forward-chaining inference & computing line diffs...
  [UPDATE SCOPE] Fields: Authoritative pricing, tiers, curricula & configs -> Target: HTML/JS repository pages
  -> All 99 target documents evaluated. Zero changes required (fully in sync).

[Step 4b] Synchronizing source code and unit test suites...
  [UPDATE SCOPE] Fields: Authoritative facts -> Target: python/*.py, root JS, and tests/*
  -> All 48 source/test files evaluated. Zero changes required (fully in sync).

[Step 5/6] Generating domain references and master dictionary...
  [UPDATE SCOPE] Fields: Tracked occurrences & references -> Target: nlp-fuzzy-expert/*_references.json
  -> Reference dictionaries and manifest already in sync.

[Step 6/6] Generating domain markdown synchronization reports...
  [UPDATE SCOPE] Fields: Domain summaries & occurrence tables -> Target: nlp-fuzzy-expert/*.md reports
  -> Markdown reports already in sync.

================================================================================
                     SYNCHRONIZATION SUMMARY TABLE
================================================================================
+----------------------------+---------+---------------+-------------+---------------+
| Target File                | Status  | Changed Lines | Rules Fired | Fields Synced |
+----------------------------+---------+---------------+-------------+---------------+
| all-business-services.html | IN SYNC | -             | -           | -             |
| all-personal-services.html | IN SYNC | -             | -           | -             |
| contact.html               | IN SYNC | -             | -           | -             |
| faqs.html                  | IN SYNC | -             | -           | -             |
| guarantees.html            | IN SYNC | -             | -           | -             |
| index.html                 | IN SYNC | -             | -           | -             |
| llms.txt                   | IN SYNC | -             | -           | -             |
| pricing-calculator.js      | IN SYNC | -             | -           | -             |
| pricing.html               | IN SYNC | -             | -           | -             |
+----------------------------+---------+---------------+-------------+---------------+
Summary: 99 files evaluated | 0 updated | 99 in sync | 0 references tracked
================================================================================
================================================================================
           NLP-FUZZY-EXPERT KNOWLEDGE SYNCHRONIZATION SYSTEM
================================================================================
  [DATA FLOW] Direction: data/*.json -> HTML/JS/MD/Code (STRICTLY UNIDIRECTIONAL / ONE-WAY)
[Step 1/6] Loading authoritative facts from data/*.json into KnowledgeBase...
  [UPDATE SCOPE] Domain: data/*.json -> Target: KnowledgeBase atomic facts cache
  -> Authoritative sources loaded: 11 data domains
     - data/business_services.json: 3 top-level entries
     - data/claude.json: 2 top-level entries
     - data/config.json: 12 top-level entries
     - data/contact.json: 6 top-level entries
     - data/faqs.json: 2 top-level entries
     - data/gemini.json: 2 top-level entries
     - data/guarantees.json: 5 top-level entries
     - data/personal_services.json: 3 top-level entries
     - data/pricing.json: 6 top-level entries
     - data/publications.json: 48 top-level entries
     - data/tiers.json: 1 top-level entries
  -> Total atomic propositions registered in KnowledgeBase: 310

[Step 2/6] Registering production rules in salience priority order...
  [UPDATE SCOPE] Domain: rules.py -> Target: InferenceEngine priority rule queue
  -> Active production rules registered: 18
     - Salience 100: PricingHeroPackagesRule (domain: pricing)
     - Salience 100: PricingCalculatorScriptRule (domain: pricing)
     - Salience  95: PricingCalculatorSelectOptionsRule (domain: pricing)
     - Salience  92: PricingSection5CardsRule (domain: pricing)
     - Salience  90: BusinessAITrainingRule (domain: business_training)
     ... and 13 additional production rules

[Step 3/6] Discovering target documents across repository...
  [UPDATE SCOPE] Domain: Repository scan -> Target: Target HTML and JS documents list
  -> Discovered 99 target documents (.html, .js) for synchronization.

[Step 4/6] Executing forward-chaining inference & computing line diffs...
  [UPDATE SCOPE] Fields: Authoritative pricing, tiers, curricula & configs -> Target: HTML/JS repository pages
  -> All 99 target documents evaluated. Zero changes required (fully in sync).

[Step 4b] Synchronizing source code and unit test suites...
  [UPDATE SCOPE] Fields: Authoritative facts -> Target: python/*.py, root JS, and tests/*
  -> All 48 source/test files evaluated. Zero changes required (fully in sync).

[Step 5/6] Generating domain references and master dictionary...
  [UPDATE SCOPE] Fields: Tracked occurrences & references -> Target: nlp-fuzzy-expert/*_references.json
  -> Reference dictionaries and manifest already in sync.

[Step 6/6] Generating domain markdown synchronization reports...
  [UPDATE SCOPE] Fields: Domain summaries & occurrence tables -> Target: nlp-fuzzy-expert/*.md reports
  -> Markdown reports already in sync.

================================================================================
                     SYNCHRONIZATION SUMMARY TABLE
================================================================================
+----------------------------+---------+---------------+-------------+---------------+
| Target File                | Status  | Changed Lines | Rules Fired | Fields Synced |
+----------------------------+---------+---------------+-------------+---------------+
| all-business-services.html | IN SYNC | -             | -           | -             |
| all-personal-services.html | IN SYNC | -             | -           | -             |
| contact.html               | IN SYNC | -             | -           | -             |
| faqs.html                  | IN SYNC | -             | -           | -             |
| guarantees.html            | IN SYNC | -             | -           | -             |
| index.html                 | IN SYNC | -             | -           | -             |
| llms.txt                   | IN SYNC | -             | -           | -             |
| pricing-calculator.js      | IN SYNC | -             | -           | -             |
| pricing.html               | IN SYNC | -             | -           | -             |
+----------------------------+---------+---------------+-------------+---------------+
Summary: 99 files evaluated | 0 updated | 99 in sync | 0 references tracked
================================================================================
================================================================================
           NLP-FUZZY-EXPERT KNOWLEDGE SYNCHRONIZATION SYSTEM
================================================================================
  [DATA FLOW] Direction: data/*.json -> HTML/JS/MD/Code (STRICTLY UNIDIRECTIONAL / ONE-WAY)
[Step 1/6] Loading authoritative facts from data/*.json into KnowledgeBase...
  [UPDATE SCOPE] Domain: data/*.json -> Target: KnowledgeBase atomic facts cache
  -> Authoritative sources loaded: 11 data domains
     - data/business_services.json: 3 top-level entries
     - data/claude.json: 2 top-level entries
     - data/config.json: 12 top-level entries
     - data/contact.json: 6 top-level entries
     - data/faqs.json: 2 top-level entries
     - data/gemini.json: 2 top-level entries
     - data/guarantees.json: 5 top-level entries
     - data/personal_services.json: 3 top-level entries
     - data/pricing.json: 6 top-level entries
     - data/publications.json: 48 top-level entries
     - data/tiers.json: 1 top-level entries
  -> Total atomic propositions registered in KnowledgeBase: 310

[Step 2/6] Registering production rules in salience priority order...
  [UPDATE SCOPE] Domain: rules.py -> Target: InferenceEngine priority rule queue
  -> Active production rules registered: 18
     - Salience 100: PricingHeroPackagesRule (domain: pricing)
     - Salience 100: PricingCalculatorScriptRule (domain: pricing)
     - Salience  95: PricingCalculatorSelectOptionsRule (domain: pricing)
     - Salience  92: PricingSection5CardsRule (domain: pricing)
     - Salience  90: BusinessAITrainingRule (domain: business_training)
     ... and 13 additional production rules

[Step 3/6] Discovering target documents across repository...
  [UPDATE SCOPE] Domain: Repository scan -> Target: Target HTML and JS documents list
  -> Discovered 99 target documents (.html, .js) for synchronization.

[Step 4/6] Executing forward-chaining inference & computing line diffs...
  [UPDATE SCOPE] Fields: Authoritative pricing, tiers, curricula & configs -> Target: HTML/JS repository pages
  -> All 99 target documents evaluated. Zero changes required (fully in sync).

[Step 4b] Synchronizing source code and unit test suites...
  [UPDATE SCOPE] Fields: Authoritative facts -> Target: python/*.py, root JS, and tests/*
  -> All 48 source/test files evaluated. Zero changes required (fully in sync).

[Step 5/6] Generating domain references and master dictionary...
  [UPDATE SCOPE] Fields: Tracked occurrences & references -> Target: nlp-fuzzy-expert/*_references.json
  -> Reference dictionaries and manifest already in sync.

[Step 6/6] Generating domain markdown synchronization reports...
  [UPDATE SCOPE] Fields: Domain summaries & occurrence tables -> Target: nlp-fuzzy-expert/*.md reports
  -> Markdown reports already in sync.

================================================================================
                     SYNCHRONIZATION SUMMARY TABLE
================================================================================
+----------------------------+---------+---------------+-------------+---------------+
| Target File                | Status  | Changed Lines | Rules Fired | Fields Synced |
+----------------------------+---------+---------------+-------------+---------------+
| all-business-services.html | IN SYNC | -             | -           | -             |
| all-personal-services.html | IN SYNC | -             | -           | -             |
| contact.html               | IN SYNC | -             | -           | -             |
| faqs.html                  | IN SYNC | -             | -           | -             |
| guarantees.html            | IN SYNC | -             | -           | -             |
| index.html                 | IN SYNC | -             | -           | -             |
| llms.txt                   | IN SYNC | -             | -           | -             |
| pricing-calculator.js      | IN SYNC | -             | -           | -             |
| pricing.html               | IN SYNC | -             | -           | -             |
+----------------------------+---------+---------------+-------------+---------------+
Summary: 99 files evaluated | 0 updated | 99 in sync | 0 references tracked
================================================================================
================================================================================
           NLP-FUZZY-EXPERT KNOWLEDGE SYNCHRONIZATION SYSTEM
================================================================================
  [DATA FLOW] Direction: data/*.json -> HTML/JS/MD/Code (STRICTLY UNIDIRECTIONAL / ONE-WAY)
[Step 1/6] Loading authoritative facts from data/*.json into KnowledgeBase...
  [UPDATE SCOPE] Domain: data/*.json -> Target: KnowledgeBase atomic facts cache
  -> Authoritative sources loaded: 11 data domains
     - data/business_services.json: 3 top-level entries
     - data/claude.json: 2 top-level entries
     - data/config.json: 12 top-level entries
     - data/contact.json: 6 top-level entries
     - data/faqs.json: 2 top-level entries
     - data/gemini.json: 2 top-level entries
     - data/guarantees.json: 5 top-level entries
     - data/personal_services.json: 3 top-level entries
     - data/pricing.json: 6 top-level entries
     - data/publications.json: 48 top-level entries
     - data/tiers.json: 1 top-level entries
  -> Total atomic propositions registered in KnowledgeBase: 310

[Step 2/6] Registering production rules in salience priority order...
  [UPDATE SCOPE] Domain: rules.py -> Target: InferenceEngine priority rule queue
  -> Active production rules registered: 18
     - Salience 100: PricingHeroPackagesRule (domain: pricing)
     - Salience 100: PricingCalculatorScriptRule (domain: pricing)
     - Salience  95: PricingCalculatorSelectOptionsRule (domain: pricing)
     - Salience  92: PricingSection5CardsRule (domain: pricing)
     - Salience  90: BusinessAITrainingRule (domain: business_training)
     ... and 13 additional production rules

[Step 3/6] Discovering target documents across repository...
  [UPDATE SCOPE] Domain: Repository scan -> Target: Target HTML and JS documents list
  -> Discovered 99 target documents (.html, .js) for synchronization.

[Step 4/6] Executing forward-chaining inference & computing line diffs...
  [UPDATE SCOPE] Fields: Authoritative pricing, tiers, curricula & configs -> Target: HTML/JS repository pages
  -> All 99 target documents evaluated. Zero changes required (fully in sync).

[Step 4b] Synchronizing source code and unit test suites...
  [UPDATE SCOPE] Fields: Authoritative facts -> Target: python/*.py, root JS, and tests/*
  -> All 48 source/test files evaluated. Zero changes required (fully in sync).

[Step 5/6] Generating domain references and master dictionary...
  [UPDATE SCOPE] Fields: Tracked occurrences & references -> Target: nlp-fuzzy-expert/*_references.json
  -> Reference dictionaries and manifest already in sync.

[Step 6/6] Generating domain markdown synchronization reports...
  [UPDATE SCOPE] Fields: Domain summaries & occurrence tables -> Target: nlp-fuzzy-expert/*.md reports
  -> Markdown reports already in sync.

================================================================================
                     SYNCHRONIZATION SUMMARY TABLE
================================================================================
+----------------------------+---------+---------------+-------------+---------------+
| Target File                | Status  | Changed Lines | Rules Fired | Fields Synced |
+----------------------------+---------+---------------+-------------+---------------+
| all-business-services.html | IN SYNC | -             | -           | -             |
| all-personal-services.html | IN SYNC | -             | -           | -             |
| contact.html               | IN SYNC | -             | -           | -             |
| faqs.html                  | IN SYNC | -             | -           | -             |
| guarantees.html            | IN SYNC | -             | -           | -             |
| index.html                 | IN SYNC | -             | -           | -             |
| llms.txt                   | IN SYNC | -             | -           | -             |
| pricing-calculator.js      | IN SYNC | -             | -           | -             |
| pricing.html               | IN SYNC | -             | -           | -             |
+----------------------------+---------+---------------+-------------+---------------+
Summary: 99 files evaluated | 0 updated | 99 in sync | 0 references tracked
================================================================================
================================================================================
           NLP-FUZZY-EXPERT KNOWLEDGE SYNCHRONIZATION SYSTEM
================================================================================
  [DATA FLOW] Direction: data/*.json -> HTML/JS/MD/Code (STRICTLY UNIDIRECTIONAL / ONE-WAY)
[Step 1/6] Loading authoritative facts from data/*.json into KnowledgeBase...
  [UPDATE SCOPE] Domain: data/*.json -> Target: KnowledgeBase atomic facts cache
  -> Authoritative sources loaded: 11 data domains
     - data/business_services.json: 3 top-level entries
     - data/claude.json: 2 top-level entries
     - data/config.json: 12 top-level entries
     - data/contact.json: 6 top-level entries
     - data/faqs.json: 2 top-level entries
     - data/gemini.json: 2 top-level entries
     - data/guarantees.json: 5 top-level entries
     - data/personal_services.json: 3 top-level entries
     - data/pricing.json: 6 top-level entries
     - data/publications.json: 48 top-level entries
     - data/tiers.json: 1 top-level entries
  -> Total atomic propositions registered in KnowledgeBase: 310

[Step 2/6] Registering production rules in salience priority order...
  [UPDATE SCOPE] Domain: rules.py -> Target: InferenceEngine priority rule queue
  -> Active production rules registered: 18
     - Salience 100: PricingHeroPackagesRule (domain: pricing)
     - Salience 100: PricingCalculatorScriptRule (domain: pricing)
     - Salience  95: PricingCalculatorSelectOptionsRule (domain: pricing)
     - Salience  92: PricingSection5CardsRule (domain: pricing)
     - Salience  90: BusinessAITrainingRule (domain: business_training)
     ... and 13 additional production rules

[Step 3/6] Discovering target documents across repository...
  [UPDATE SCOPE] Domain: Repository scan -> Target: Target HTML and JS documents list
  -> Discovered 99 target documents (.html, .js) for synchronization.

[Step 4/6] Executing forward-chaining inference & computing line diffs...
  [UPDATE SCOPE] Fields: Authoritative pricing, tiers, curricula & configs -> Target: HTML/JS repository pages
  -> [MODIFIED] faqs.html | Field: business_included_seats | Location: Line(s) 1857, 1876, 1888
     Changed Lines: 1857, 1876, 1888
     Rules Fired:   FAQsPoliciesRule
     Fields Synced: business_included_seats
  -> [MODIFIED] guarantees.html | Field: guarantees_business_flagship_price, guarantees_personal_flagship_price | Location: Line(s) 357, 525
     Changed Lines: 357, 525
     Rules Fired:   GuaranteesPageRule
     Fields Synced: guarantees_business_flagship_price, guarantees_personal_flagship_price
  -> [MODIFIED] llms.txt | Field: llms_business_flagship_price, llms_individual_service_price, llms_personal_flagship_price | Location: Line(s) 40, 42
     Changed Lines: 40, 42
     Rules Fired:   LLMsTxtRule
     Fields Synced: llms_business_flagship_price, llms_individual_service_price, llms_personal_flagship_price
  -> [MODIFIED] pricing-calculator.js | Field: calculator_seat_threshold, tier_1_calc_multiplier, tier_2_calc_multiplier, tier_3_calc_multiplier, tier_4_calc_multiplier, tier_5_calc_multiplier, tier_6_calc_multiplier | Location: Line(s) 170, 316-317, 442, 541
     Changed Lines: 170, 316-317, 442, 541
     Rules Fired:   PricingCalculatorScriptRule
     Fields Synced: calculator_seat_threshold, tier_1_calc_multiplier, tier_2_calc_multiplier, tier_3_calc_multiplier, tier_4_calc_multiplier, tier_5_calc_multiplier, tier_6_calc_multiplier
  -> [MODIFIED] pricing.html | Field: business_flagship_days, business_flagship_price, business_service_01_pricing_card, business_service_02_pricing_card, business_service_03_pricing_card, business_service_04_pricing_card, business_service_05_pricing_card, business_service_06_pricing_card, business_service_07_pricing_card, business_service_08_pricing_card, business_service_09_pricing_card, business_service_10_pricing_card, business_service_11_pricing_card, business_service_12_pricing_card, business_service_13_pricing_card, business_service_14_pricing_card, business_service_15_pricing_card, business_service_16_pricing_card, calculator_options, personal_flagship_days, personal_flagship_price, tier_1_parity, tier_2_parity, tier_3_parity, tier_4_parity, tier_5_parity, tier_6_parity | Location: Line(s) 60, 94, 183, 453, 1002, 1053, 1101, 1250
     Changed Lines: 60, 94, 183, 453, 1002, 1053, 1101, 1250
     Rules Fired:   PricingHeroPackagesRule, PricingCalculatorSelectOptionsRule
     Fields Synced: business_flagship_days, business_flagship_price, business_service_01_pricing_card, business_service_02_pricing_card, business_service_03_pricing_card, business_service_04_pricing_card, business_service_05_pricing_card, business_service_06_pricing_card, business_service_07_pricing_card, business_service_08_pricing_card, business_service_09_pricing_card, business_service_10_pricing_card, business_service_11_pricing_card, business_service_12_pricing_card, business_service_13_pricing_card, business_service_14_pricing_card, business_service_15_pricing_card, business_service_16_pricing_card, calculator_options, personal_flagship_days, personal_flagship_price, tier_1_parity, tier_2_parity, tier_3_parity, tier_4_parity, tier_5_parity, tier_6_parity
  -> Total documents modified: 5 of 99

[Step 4b] Synchronizing source code and unit test suites...
  [UPDATE SCOPE] Fields: Authoritative facts -> Target: python/*.py, root JS, and tests/*
  -> [MODIFIED] pricing-tests.js | Status: UPDATED | Changed Lines: 21
  -> Total source/test files updated: 1 of 48

[Step 5/6] Generating domain references and master dictionary...
  [UPDATE SCOPE] Fields: Tracked occurrences & references -> Target: nlp-fuzzy-expert/*_references.json
  -> Reference dictionaries and manifest already in sync.

[Step 6/6] Generating domain markdown synchronization reports...
  [UPDATE SCOPE] Fields: Domain summaries & occurrence tables -> Target: nlp-fuzzy-expert/*.md reports
  -> Markdown reports already in sync.

================================================================================
                     SYNCHRONIZATION SUMMARY TABLE
================================================================================
+-----------------------+---------+------------------------+--------------------------+--------------------------+
| Target File           | Status  | Changed Lines          | Rules Fired              | Fields Synced            |
+-----------------------+---------+------------------------+--------------------------+--------------------------+
| faqs.html             | UPDATED | 1857, 1876, 1888       | FAQsPoliciesRule         | business_included_seats  |
| guarantees.html       | UPDATED | 357, 525               | GuaranteesPageRule       | guarantees_business_f... |
| llms.txt              | UPDATED | 40, 42                 | LLMsTxtRule              | llms_business_flagshi... |
| pricing-calculator.js | UPDATED | 170, 316-317, 442, 541 | PricingCalculatorScri... | calculator_seat_thres... |
| pricing.html          | UPDATED | 60, 94, 183, 453, 1... | PricingHeroPackagesRu... | business_flagship_day... |
| pricing-tests.js      | UPDATED | 21                     | CodeTestSynchronizer     | source_and_tests         |
+-----------------------+---------+------------------------+--------------------------+--------------------------+
Summary: 99 files evaluated | 6 updated | 93 in sync | 172 references tracked
================================================================================
================================================================================
           NLP-FUZZY-EXPERT KNOWLEDGE SYNCHRONIZATION SYSTEM
================================================================================
  [DATA FLOW] Direction: data/*.json -> HTML/JS/MD/Code (STRICTLY UNIDIRECTIONAL / ONE-WAY)
[Step 1/6] Loading authoritative facts from data/*.json into KnowledgeBase...
  [UPDATE SCOPE] Domain: data/*.json -> Target: KnowledgeBase atomic facts cache
  -> Authoritative sources loaded: 11 data domains
     - data/business_services.json: 3 top-level entries
     - data/claude.json: 2 top-level entries
     - data/config.json: 12 top-level entries
     - data/contact.json: 6 top-level entries
     - data/faqs.json: 2 top-level entries
     - data/gemini.json: 2 top-level entries
     - data/guarantees.json: 5 top-level entries
     - data/personal_services.json: 3 top-level entries
     - data/pricing.json: 6 top-level entries
     - data/publications.json: 48 top-level entries
     - data/tiers.json: 1 top-level entries
  -> Total atomic propositions registered in KnowledgeBase: 310

[Step 2/6] Registering production rules in salience priority order...
  [UPDATE SCOPE] Domain: rules.py -> Target: InferenceEngine priority rule queue
  -> Active production rules registered: 18
     - Salience 100: PricingHeroPackagesRule (domain: pricing)
     - Salience 100: PricingCalculatorScriptRule (domain: pricing)
     - Salience  95: PricingCalculatorSelectOptionsRule (domain: pricing)
     - Salience  92: PricingSection5CardsRule (domain: pricing)
     - Salience  90: BusinessAITrainingRule (domain: business_training)
     ... and 13 additional production rules

[Step 3/6] Discovering target documents across repository...
  [UPDATE SCOPE] Domain: Repository scan -> Target: Target HTML and JS documents list
  -> Discovered 99 target documents (.html, .js) for synchronization.

[Step 4/6] Executing forward-chaining inference & computing line diffs...
  [UPDATE SCOPE] Fields: Authoritative pricing, tiers, curricula & configs -> Target: HTML/JS repository pages
  -> [MODIFIED] faqs.html | Field: business_included_seats | Location: Line(s) 1857, 1876, 1888
     Changed Lines: 1857, 1876, 1888
     Rules Fired:   FAQsPoliciesRule
     Fields Synced: business_included_seats
  -> [MODIFIED] guarantees.html | Field: guarantees_business_flagship_price, guarantees_personal_flagship_price | Location: Line(s) 357, 525
     Changed Lines: 357, 525
     Rules Fired:   GuaranteesPageRule
     Fields Synced: guarantees_business_flagship_price, guarantees_personal_flagship_price
  -> [MODIFIED] llms.txt | Field: llms_business_flagship_price, llms_individual_service_price, llms_personal_flagship_price | Location: Line(s) 40, 42
     Changed Lines: 40, 42
     Rules Fired:   LLMsTxtRule
     Fields Synced: llms_business_flagship_price, llms_individual_service_price, llms_personal_flagship_price
  -> [MODIFIED] pricing-calculator.js | Field: calculator_seat_threshold, tier_1_calc_multiplier, tier_2_calc_multiplier, tier_3_calc_multiplier, tier_4_calc_multiplier, tier_5_calc_multiplier, tier_6_calc_multiplier | Location: Line(s) 170, 316-317, 442, 541
     Changed Lines: 170, 316-317, 442, 541
     Rules Fired:   PricingCalculatorScriptRule
     Fields Synced: calculator_seat_threshold, tier_1_calc_multiplier, tier_2_calc_multiplier, tier_3_calc_multiplier, tier_4_calc_multiplier, tier_5_calc_multiplier, tier_6_calc_multiplier
  -> Total documents modified: 4 of 99

[Step 4b] Synchronizing source code and unit test suites...
  [UPDATE SCOPE] Fields: Authoritative facts -> Target: python/*.py, root JS, and tests/*
  -> [MODIFIED] pricing-tests.js | Status: UPDATED | Changed Lines: 21
  -> Total source/test files updated: 1 of 48

[Step 5/6] Generating domain references and master dictionary...
  [UPDATE SCOPE] Fields: Tracked occurrences & references -> Target: nlp-fuzzy-expert/*_references.json
  -> Exhaustive scan located 28549 term occurrences across 99 documents

[Step 6/6] Generating domain markdown synchronization reports...
  [UPDATE SCOPE] Fields: Domain summaries & occurrence tables -> Target: nlp-fuzzy-expert/*.md reports

================================================================================
                     SYNCHRONIZATION SUMMARY TABLE
================================================================================
+-----------------------+---------+------------------------+--------------------------+--------------------------+
| Target File           | Status  | Changed Lines          | Rules Fired              | Fields Synced            |
+-----------------------+---------+------------------------+--------------------------+--------------------------+
| faqs.html             | UPDATED | 1857, 1876, 1888       | FAQsPoliciesRule         | business_included_seats  |
| guarantees.html       | UPDATED | 357, 525               | GuaranteesPageRule       | guarantees_business_f... |
| llms.txt              | UPDATED | 40, 42                 | LLMsTxtRule              | llms_business_flagshi... |
| pricing-calculator.js | UPDATED | 170, 316-317, 442, 541 | PricingCalculatorScri... | calculator_seat_thres... |
| pricing-tests.js      | UPDATED | 21                     | CodeTestSynchronizer     | source_and_tests         |
+-----------------------+---------+------------------------+--------------------------+--------------------------+
Summary: 99 files evaluated | 5 updated | 94 in sync | 172 references tracked
================================================================================
================================================================================
           NLP-FUZZY-EXPERT KNOWLEDGE SYNCHRONIZATION SYSTEM
================================================================================
  [DATA FLOW] Direction: data/*.json -> HTML/JS/MD/Code (STRICTLY UNIDIRECTIONAL / ONE-WAY)
[Step 1/6] Loading authoritative facts from data/*.json into KnowledgeBase...
  [UPDATE SCOPE] Domain: data/*.json -> Target: KnowledgeBase atomic facts cache
  -> Authoritative sources loaded: 11 data domains
     - data/business_services.json: 3 top-level entries
     - data/claude.json: 2 top-level entries
     - data/config.json: 12 top-level entries
     - data/contact.json: 6 top-level entries
     - data/faqs.json: 2 top-level entries
     - data/gemini.json: 2 top-level entries
     - data/guarantees.json: 5 top-level entries
     - data/personal_services.json: 3 top-level entries
     - data/pricing.json: 6 top-level entries
     - data/publications.json: 48 top-level entries
     - data/tiers.json: 1 top-level entries
  -> Total atomic propositions registered in KnowledgeBase: 310

[Step 2/6] Registering production rules in salience priority order...
  [UPDATE SCOPE] Domain: rules.py -> Target: InferenceEngine priority rule queue
  -> Active production rules registered: 18
     - Salience 100: PricingHeroPackagesRule (domain: pricing)
     - Salience 100: PricingCalculatorScriptRule (domain: pricing)
     - Salience  95: PricingCalculatorSelectOptionsRule (domain: pricing)
     - Salience  92: PricingSection5CardsRule (domain: pricing)
     - Salience  90: BusinessAITrainingRule (domain: business_training)
     ... and 13 additional production rules

[Step 3/6] Discovering target documents across repository...
  [UPDATE SCOPE] Domain: Repository scan -> Target: Target HTML and JS documents list
  -> Discovered 99 target documents (.html, .js) for synchronization.

[Step 4/6] Executing forward-chaining inference & computing line diffs...
  [UPDATE SCOPE] Fields: Authoritative pricing, tiers, curricula & configs -> Target: HTML/JS repository pages
  -> [MODIFIED] pricing.html | Field: business_flagship_days, business_flagship_price, business_service_01_pricing_card, business_service_02_pricing_card, business_service_03_pricing_card, business_service_04_pricing_card, business_service_05_pricing_card, business_service_06_pricing_card, business_service_07_pricing_card, business_service_08_pricing_card, business_service_09_pricing_card, business_service_10_pricing_card, business_service_11_pricing_card, business_service_12_pricing_card, business_service_13_pricing_card, business_service_14_pricing_card, business_service_15_pricing_card, business_service_16_pricing_card, calculator_options, personal_flagship_days, personal_flagship_price, tier_1_parity, tier_2_parity, tier_3_parity, tier_4_parity, tier_5_parity, tier_6_parity | Location: Line(s) 183, 223, 453, 465
     Changed Lines: 183, 223, 453, 465
     Rules Fired:   PricingHeroPackagesRule
     Fields Synced: business_flagship_days, business_flagship_price, business_service_01_pricing_card, business_service_02_pricing_card, business_service_03_pricing_card, business_service_04_pricing_card, business_service_05_pricing_card, business_service_06_pricing_card, business_service_07_pricing_card, business_service_08_pricing_card, business_service_09_pricing_card, business_service_10_pricing_card, business_service_11_pricing_card, business_service_12_pricing_card, business_service_13_pricing_card, business_service_14_pricing_card, business_service_15_pricing_card, business_service_16_pricing_card, calculator_options, personal_flagship_days, personal_flagship_price, tier_1_parity, tier_2_parity, tier_3_parity, tier_4_parity, tier_5_parity, tier_6_parity
  -> Total documents modified: 1 of 99

[Step 4b] Synchronizing source code and unit test suites...
  [UPDATE SCOPE] Fields: Authoritative facts -> Target: python/*.py, root JS, and tests/*
  -> All 48 source/test files evaluated. Zero changes required (fully in sync).

[Step 5/6] Generating domain references and master dictionary...
  [UPDATE SCOPE] Fields: Tracked occurrences & references -> Target: nlp-fuzzy-expert/*_references.json
  -> Exhaustive scan located 28549 term occurrences across 99 documents
  -> Wrote pricing_references.json (3325 references)
  -> Wrote business_services_references.json (3253 references)
  -> Wrote personal_services_references.json (1166 references)
  -> Wrote tiers_references.json (3413 references)
  -> Wrote config_references.json (6434 references)
  -> Master Dictionary generated: references_dictionary.json (28594 total references)
  -> Lightweight Manifest generated: references_manifest.json (108 documents indexed)

[Step 6/6] Generating domain markdown synchronization reports...
  [UPDATE SCOPE] Fields: Domain summaries & occurrence tables -> Target: nlp-fuzzy-expert/*.md reports
  -> Wrote pricing.md
  -> Wrote business-services.md
  -> Wrote personal-services.md
  -> Wrote tiers.md
  -> Wrote config.md

================================================================================
                     SYNCHRONIZATION SUMMARY TABLE
================================================================================
+--------------+---------+--------------------+-------------------------+--------------------------+
| Target File  | Status  | Changed Lines      | Rules Fired             | Fields Synced            |
+--------------+---------+--------------------+-------------------------+--------------------------+
| pricing.html | UPDATED | 183, 223, 453, 465 | PricingHeroPackagesRule | business_flagship_day... |
+--------------+---------+--------------------+-------------------------+--------------------------+
Summary: 99 files evaluated | 1 updated | 98 in sync | 27 references tracked
================================================================================
================================================================================
           NLP-FUZZY-EXPERT KNOWLEDGE SYNCHRONIZATION SYSTEM
================================================================================
  [DATA FLOW] Direction: data/*.json -> HTML/JS/MD/Code (STRICTLY UNIDIRECTIONAL / ONE-WAY)
[Step 1/6] Loading authoritative facts from data/*.json into KnowledgeBase...
  [UPDATE SCOPE] Domain: data/*.json -> Target: KnowledgeBase atomic facts cache
  -> Authoritative sources loaded: 11 data domains
     - data/business_services.json: 3 top-level entries
     - data/claude.json: 2 top-level entries
     - data/config.json: 12 top-level entries
     - data/contact.json: 6 top-level entries
     - data/faqs.json: 2 top-level entries
     - data/gemini.json: 2 top-level entries
     - data/guarantees.json: 5 top-level entries
     - data/personal_services.json: 3 top-level entries
     - data/pricing.json: 6 top-level entries
     - data/publications.json: 48 top-level entries
     - data/tiers.json: 1 top-level entries
  -> Total atomic propositions registered in KnowledgeBase: 310

[Step 2/6] Registering production rules in salience priority order...
  [UPDATE SCOPE] Domain: rules.py -> Target: InferenceEngine priority rule queue
  -> Active production rules registered: 18
     - Salience 100: PricingHeroPackagesRule (domain: pricing)
     - Salience 100: PricingCalculatorScriptRule (domain: pricing)
     - Salience  95: PricingCalculatorSelectOptionsRule (domain: pricing)
     - Salience  92: PricingSection5CardsRule (domain: pricing)
     - Salience  90: BusinessAITrainingRule (domain: business_training)
     ... and 13 additional production rules

[Step 3/6] Discovering target documents across repository...
  [UPDATE SCOPE] Domain: Repository scan -> Target: Target HTML and JS documents list
  -> Discovered 99 target documents (.html, .js) for synchronization.

[Step 4/6] Executing forward-chaining inference & computing line diffs...
  [UPDATE SCOPE] Fields: Authoritative pricing, tiers, curricula & configs -> Target: HTML/JS repository pages
  -> All 99 target documents evaluated. Zero changes required (fully in sync).

[Step 4b] Synchronizing source code and unit test suites...
  [UPDATE SCOPE] Fields: Authoritative facts -> Target: python/*.py, root JS, and tests/*
  -> All 48 source/test files evaluated. Zero changes required (fully in sync).

[Step 5/6] Generating domain references and master dictionary...
  [UPDATE SCOPE] Fields: Tracked occurrences & references -> Target: nlp-fuzzy-expert/*_references.json
  -> Reference dictionaries and manifest already in sync.

[Step 6/6] Generating domain markdown synchronization reports...
  [UPDATE SCOPE] Fields: Domain summaries & occurrence tables -> Target: nlp-fuzzy-expert/*.md reports
  -> Markdown reports already in sync.

================================================================================
                     SYNCHRONIZATION SUMMARY TABLE
================================================================================
+----------------------------+---------+---------------+-------------------------+--------------------------+
| Target File                | Status  | Changed Lines | Rules Fired             | Fields Synced            |
+----------------------------+---------+---------------+-------------------------+--------------------------+
| all-business-services.html | IN SYNC | -             | -                       | -                        |
| all-personal-services.html | IN SYNC | -             | -                       | -                        |
| contact.html               | IN SYNC | -             | -                       | -                        |
| faqs.html                  | IN SYNC | -             | -                       | -                        |
| guarantees.html            | IN SYNC | -             | -                       | -                        |
| index.html                 | IN SYNC | -             | -                       | -                        |
| llms.txt                   | IN SYNC | -             | -                       | -                        |
| pricing-calculator.js      | IN SYNC | -             | -                       | -                        |
| pricing.html               | IN SYNC | -             | PricingHeroPackagesRule | business_flagship_day... |
+----------------------------+---------+---------------+-------------------------+--------------------------+
Summary: 99 files evaluated | 0 updated | 99 in sync | 27 references tracked
================================================================================
--------------------------------------------------------------------------------
TEST SUITE SUMMARY REPORT
--------------------------------------------------------------------------------
Tests Run:   149
Failures:    2
Errors:      0
Skipped:     0
Time Taken:  10.75 seconds
================================================================================
RESULT: TEST SUITE FAILED.
================================================================================

[Auto-Healer] Wrote failure report to: failed_test_sync_guarantees_tests.report.md
[Auto-Healer] Wrote aggregate report to: failed_tests_report.md
```
