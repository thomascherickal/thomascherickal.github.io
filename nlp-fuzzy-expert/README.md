# NLP-Fuzzy-Expert Knowledge Synchronization System

A pure-Python autonomous knowledge synchronization engine combining **Expert System forward-chaining**, **Fuzzy Logic string matching**, and **Natural Language Processing (NLP) entity extraction**.

Strict Repo Compliance:
- Zero emojis.
- Zero em-dashes (spaced hyphens ' - ' or plain hyphens '-' used).
- Pure standard library Python (no third-party pip dependencies).

---

## Architectural Overview

The `nlp-fuzzy-expert` system ensures that data defined in `data/*.json` remains the authoritative single source of truth across all 65 repository documents (.html, .js) without manual editing or synchronization drift.

```
data/*.json (Authoritative Facts)
       │
       ▼
┌────────────────────────────────────────────────────────┐
│             nlp-fuzzy-expert/synchronizer.py           │
│                                                        │
│  1. KnowledgeBase: Loads atomic propositions           │
│  2. NLP Engine: Normalizes text & extracts entities    │
│  3. Fuzzy Matcher: Levenshtein, Jaccard & Token Ratios │
│  4. Inference Engine: Forward chains 16 salience rules │
│  5. Diff Engine: Computes 1-based changed line ranges  │
└────────────────────────────────────────────────────────┘
       │
       ├──► Target Documents (.html, .js)
       ├──► Domain References (*_references.json)
       ├──► Master Dictionary (references_dictionary.json)
       ├──► Markdown Reports (*.md)
       └──► Post-Sync Unit Test Verification (tests/)
```

---

## Core Modules

### 1. `expert_system.py` - Knowledge Base & Forward-Chaining Inference
- **`Fact`**: Atomic proposition in working memory (`domain:predicate = value`).
- **`KnowledgeBase`**: Working memory that parses all 9 JSON files in `data/` (`pricing.json`, `business-services.json`, `personal-services.json`, `tiers.json`, `claude.json`, `gemini.json`, `publications.json`, `config.json`, `faqs.json`) into 230+ atomic propositions.
- **`Rule`**: Production rule containing name, salience priority, condition antecedent, and action consequent.
- **`InferenceEngine`**: Evaluates rules against candidate files in salience priority order, resolves conflicts, and records tracking occurrences.
- **`compute_changed_lines(old_text, new_text)`**: Uses `difflib.SequenceMatcher` opcodes (`replace`, `insert`, `delete`) to calculate formatted 1-based changed line ranges (e.g., `394-405, 428-449`).

### 2. `fuzzy_engine.py` - Fuzzy Logic & Approximate String Matching
- **Membership Functions**: Triangular and trapezoidal fuzzy sets evaluating confidence degrees `[0.0, 1.0]`.
- **String Distance Metrics**: Pure Python implementations of:
  - Levenshtein edit distance and similarity ratio
  - Token Sort Ratio (word permutation invariant)
  - Token Set Ratio (subset and duplicate resilient)
  - Character 3-Gram Jaccard similarity
- **Linguistic Classification**: Maps similarity scores to fuzzy categories: `EXACT`, `HIGH`, `MEDIUM`, `LOW`, `NONE`.

### 3. `nlp_engine.py` - Semantic Parsing & Entity Recognition
- **Text Normalizer**: Standardizes whitespace, collapses HTML entities, and strips formatting tags for semantic comparison.
- **Entity Extractor**:
  - `MONEY`: Pattern matches and normalizes USD currency amounts (`$100`, `$2,000 USD`, `+$200/seat`).
  - `DURATION`: Extracts day, week, and hour training durations (`60 days`, `30 days live`, `4 hours per day`).
  - `SEATS`: Extracts cohort seat capacities and scale-out thresholds (`10 seats`, `20 participants`).
- **Context Classifier**: Scores paragraph relevance to specific domains (pricing, curriculum, tiers, policies).
- **IDE Highlight Utility**: Inlines subtle semi-transparent amber highlights (`rgba(251, 191, 36, 0.25)`) visible only in IDE editors while remaining invisible on live production sites.

### 4. `rules.py` - Production Rule Base
16 domain rules categorized by priority salience:
- **Salience 100**: `PricingHeroPackagesRule`, `PricingCalculatorScriptRule`
- **Salience 95**: `PricingCalculatorSelectOptionsRule`
- **Salience 92**: `PricingSection5CardsRule`
- **Salience 90**: `BusinessAITrainingRule`, `PersonalAITrainingRule`
- **Salience 85**: `BusinessServicesNamesAndCardsRule`, `PersonalServicesNamesAndCardsRule`
- **Salience 80**: `IndividualServicePagesPricingRule`
- **Salience 75**: `CountryParityTiersRule`
- **Salience 70**: `ClaudeGuidesRule`, `GeminiGuidesRule`
- **Salience 65**: `PublicationsListRule`
- **Salience 60**: `SiteConfigBrandRule`
- **Salience 55**: `FAQsPoliciesRule`
- **Salience 50**: `IndexPagePricingRule`

### 5. `synchronizer.py` - Master Orchestrator & CLI Reporter
- Discovers target documents across the repository root and subdirectories (`business-services/`, `personal-services/`, `claude/`, `google/`).
- Prints real-time working steps (Step 1 through Step 6).
- Reports exact changed line numbers for every modified file.
- Formats a clean ASCII summary table upon completion.
- Triggers `python/run_tests.py` automatically whenever target documents are updated.

### 6. `watcher.py` - Autonomous Background Daemon
- Monitors all files in `data/*.json` using non-blocking filesystem polling.
- Detects file creations, modifications, and deletions.
- Debounces file system writes (150ms) and executes synchronization automatically.

---

## CLI Usage

### Run Single Synchronization Pass
```bash
python3 nlp-fuzzy-expert/synchronizer.py
```

### Start Real-Time Watcher Daemon
```bash
# Continuous daemon mode (monitors data/*.json):
python3 nlp-fuzzy-expert/watcher.py

# Run one synchronization pass and exit:
python3 nlp-fuzzy-expert/watcher.py --once

# Adjust polling interval (e.g., 0.5s):
python3 nlp-fuzzy-expert/watcher.py --interval 0.5
```

---

## Output Artifacts

Running synchronization automatically generates and updates the following files in `nlp-fuzzy-expert/`:

1. **Domain Reference Dictionaries (`*_references.json`)**:
   - `pricing_references.json`
   - `business_services_references.json`
   - `personal_services_references.json`
   - `tiers_references.json`
   - `claude_references.json`
   - `gemini_references.json`
   - `publications_references.json`
   - `config_references.json`
   - `faqs_references.json`

2. **Master Aggregated Reference Dictionary**:
   - `references_dictionary.json` (230+ tracked references across the repository).

3. **Domain Markdown Synchronization Reports (`*.md`)**:
   - `pricing.md`, `business-services.md`, `personal-services.md`, `tiers.md`, `claude.md`, `gemini.md`, `publications.md`, `config.md`, `faqs.md`.

