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
  - Broken link in ../../../../../../../tmp/tmpqrklslag/broken.html: 'missing.html' -> Target file ../../../../../../../tmp/tmpqrklslag/missing.html not found
  - Broken anchor in ../../../../../../../tmp/tmpqrklslag/broken.html: '#missing-anchor' -> ID '#missing-anchor' not found in ../../../../../../../tmp/tmpqrklslag/broken.html
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
  -> [MODIFIED] personal-services/personal-service-agentic-assistants.html | Field: individual_service_duration, individual_service_price, personal_service_04_name | Location: Line(s) 6, 296
     Changed Lines: 6, 296
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
| personal-services/personal-service-agentic... | UPDATED | 6, 296        | PersonalServicesNames... | individual_service_du... |
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
  -> [MODIFIED] personal-services/personal-service-agentic-assistants.html | Field: individual_service_duration, individual_service_price, personal_service_04_name | Location: Line(s) 6, 296
     Changed Lines: 6, 296
     Rules Fired:   PersonalServicesNamesAndCardsRule
     Fields Synced: individual_service_duration, individual_service_price, personal_service_04_name
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
[Coverage Report] Successfully appended coverage report to /tmp/tmpsubu6ty4/LAST-COMMIT-REPORT.md (Combined: 97.86% | Elapsed: 0.0s)
[Coverage Report] Successfully appended coverage report to /tmp/tmpsubu6ty4/LAST-COMMIT-REPORT.md (Combined: 97.86% | Elapsed: 0.0s)
[Coverage Report] Successfully appended coverage report to /tmp/tmpmdaxwkko/LAST-COMMIT-REPORT.md (Combined: 100.00% | Elapsed: 0.0s)
[Pre-Commit Report] Successfully recorded report (parent c71de11) to /tmp/tmp51ezax5d/LAST-COMMIT-REPORT.md
[Pre-Commit Report] Successfully recorded report (parent M	README.md) to /tmp/tmp51ezax5d/LAST-COMMIT-REPORT.md
[Pre-Commit Report] Successfully recorded report (parent ) to /tmp/tmp51ezax5d/LAST-COMMIT-REPORT.md
[README Commit Summary] Successfully updated /tmp/tmp54v09i3e/README.md from /tmp/tmp54v09i3e/rep.md and /tmp/tmp54v09i3e/METRICS.md
[README Commit Summary] Successfully updated /tmp/tmp54v09i3e/README.md from /tmp/tmp54v09i3e/rep.md and /tmp/tmp54v09i3e/METRICS.md
[README Commit Summary] Successfully updated /tmp/tmp54v09i3e/README_plain.md from /tmp/tmp54v09i3e/rep_min.md and /tmp/tmp54v09i3e/missing_METRICS.md
[README Commit Summary] Successfully updated /tmp/tmp54v09i3e/README_plain.md from /tmp/tmp54v09i3e/rep_min.md and /tmp/tmp54v09i3e/missing_METRICS.md
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
--------------------------------------------------------------------------------
TEST SUITE SUMMARY REPORT
--------------------------------------------------------------------------------
Tests Run:   149
Failures:    0
Errors:      20
Skipped:     0
Time Taken:  2.88 seconds
================================================================================
RESULT: TEST SUITE FAILED.
================================================================================

[Auto-Healer] Wrote failure report to: failed_test_continuous_updates_tests.report.md
[Auto-Healer] Wrote aggregate report to: failed_tests_report.md
```
