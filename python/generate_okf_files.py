#!/usr/bin/env python3
"""
generate_okf_files.py - Generates Open Knowledge Format (OKF.md) bundles across all 37 directories.
Strictly zero emojis and zero em-dashes compliant.
Provides rich, accurate, directory-specific context and agent guidance for every folder.
"""

import os
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DIRECTORY_METADATA = {
    '.': {
        'title': 'Repository Root & Web Portal Hub',
        'type': 'module',
        'description': 'Root entry point containing core portal pages, global navigation inliners, pricing calculators, and master architectural specifications.',
        'tags': ['root', 'portal', 'entry-points', 'architecture', 'ssot'],
        'guidelines': [
            'Houses core landing pages (`index.html`, `business-ai-training.html`, `personal-ai-training.html`, `pricing.html`, `all-business-services.html`, `all-personal-services.html`).',
            'Master component sources `header.html` and `footer.html` live here. Modifying either requires immediately running `python3 python/build_components.py` to inline across all HTML pages.',
            '`GEMINI.md` is the ultimate authoritative project specification and single source of truth (SSOT).',
            '`METRICS.md` tracks codebase telemetry and is automatically refreshed by `python3 python/generate_metrics.py`.',
            'Strictly enforce zero emojis and zero em-dashes across all files. Buttons must never be underlined.'
        ]
    },
    'article-footers': {
        'title': 'Article Footers Module',
        'type': 'module',
        'description': 'Reusable article footer templates and components in HTML, Markdown, and Word formats for external publication syndication.',
        'tags': ['article-footers', 'templates', 'reusable-components', 'syndication'],
        'guidelines': [
            'Contains `article-footer.md`, `article-footer_dark.html`, and `article-footer-plain.docx`.',
            'Preserve these files in the repository; never delete or move them.',
            'CRITICAL AGENT RULE: Never insert an article footer into any Claude guide (`claude-training/*.html` or `md-html-sync/claude-training/*.md`).',
            'Used when syndicating articles across Dev.to, Medium, Hashnode, Substack, and HackerNoon.'
        ]
    },
    'assets': {
        'title': 'Global Site Assets & Branding Hub',
        'type': 'dataset',
        'description': 'Core site branding assets, logos, favicons, web manifests, OpenGraph social cards, and service-specific visual subdirectories.',
        'tags': ['assets', 'branding', 'media', 'favicons', 'opengraph'],
        'guidelines': [
            'Holds primary branding imagery: `digital-futurist-logo.jpg`, `apple-touch-icon.png`, `favicon.svg`, `og-card.png`, and `Professional Profile Square.webp`.',
            'Distinct from `images/`: `assets/` holds site identity, favicons, OG cards, and service-specific diagram folders, while `images/` holds standalone editorial WebP publication graphics.',
            'Organized into 11 specialized subdirectories for service diagrams and tool screenshots.',
            'Do not edit binary raster or vector images directly with scripts; refer to them via stable relative or root-relative paths.'
        ]
    },
    'assets/advanced-ai-agents': {
        'title': 'Advanced AI Agents Visual Assets',
        'type': 'dataset',
        'description': 'Architecture diagrams, state machine flows, and visual schematics for the Advanced AI Agents service curriculum.',
        'tags': ['assets', 'advanced-agents', 'diagrams', 'swarms'],
        'guidelines': [
            'Provides diagrammatic assets for `business-services/business-service-advanced-ai-agents.html` and `personal-services/personal-service-advanced-ai-agents.html`.',
            'Illustrates hierarchical multi-agent teams, reflection loops, and sandboxed Docker tool execution.'
        ]
    },
    'assets/agentic-ai-systems': {
        'title': 'Agentic AI Systems Visual Assets',
        'type': 'dataset',
        'description': 'Visual assets, schemas, and flowcharts illustrating autonomous agentic loops, Nous Research Agentic Assistants, and memory systems.',
        'tags': ['assets', 'agentic-systems', 'diagrams', 'nous-hermes'],
        'guidelines': [
            'Supplies graphics for `business-services/business-service-agentic-ai-systems.html` and `personal-services/personal-service-agentic-ai-systems.html`.',
            'Visualizes self-hosted Agentic Assistants, persistent vector/SQLite memory, and autonomous skill acquisition.'
        ]
    },
    'assets/ai-agents-orchestration': {
        'title': 'AI Agent Orchestration Visual Assets',
        'type': 'dataset',
        'description': 'Visual schematics illustrating orchestrator-worker hierarchies, parallel fan-out swarms, and MCP routing.',
        'tags': ['assets', 'orchestration', 'swarms', 'mcp-routing'],
        'guidelines': [
            'Supplies diagrams for `business-services/business-service-ai-agents-orchestration.html` and `personal-services/personal-service-ai-agents-orchestration.html`.',
            'Depicts human-in-the-loop gates, token budget governance, and multi-agent coordination topologies.'
        ]
    },
    'assets/ai-developer-training': {
        'title': 'AI Developer Training Visual Assets',
        'type': 'dataset',
        'description': 'Terminal screenshots, TDD workflow charts, and visual assets for AI Developer bootcamps.',
        'tags': ['assets', 'developer-training', 'tdd', 'coding-agents'],
        'guidelines': [
            'Supports `business-services/business-service-ai-developer-training.html` and `personal-services/personal-service-ai-developer-training.html`.',
            'Illustrates AI-assisted Test-Driven Development loops, Claude Code CLI sessions, and custom MCP development in Python/Rust.'
        ]
    },
    'assets/antigravity': {
        'title': 'Google Antigravity Visual Assets',
        'type': 'dataset',
        'description': 'Branding graphics, IDE interface screenshots, and workflow diagrams for Google Antigravity IDE and SDK documentation.',
        'tags': ['assets', 'antigravity', 'google-ide', 'agentic-workflows'],
        'guidelines': [
            'Supplies rich visuals for `google-training/google-antigravity-ide.html`.',
            'Illustrates agentic workflows with browser verification, sidecar orchestration, and multi-agent task execution.'
        ]
    },
    'assets/claude-orchestration': {
        'title': 'Claude Orchestration Visual Assets',
        'type': 'dataset',
        'description': 'Architecture blueprints and sequential pipeline diagrams for Claude Agent Orchestration workflows.',
        'tags': ['assets', 'claude', 'orchestration', 'blueprints'],
        'guidelines': [
            'Supports `claude-training/claude-orchestration.html`.',
            'Illustrates prompt caching economics, dynamic routing, and subagent delegation patterns.'
        ]
    },
    'assets/generative-ai-transformation': {
        'title': 'Generative AI Transformation Visual Assets',
        'type': 'dataset',
        'description': 'Enterprise AI roadmaps, chatbot-to-agent evolution diagrams, and architectural charts.',
        'tags': ['assets', 'transformation', 'enterprise-ai', 'roadmaps'],
        'guidelines': [
            'Supplies visual assets for `business-services/business-service-generative-ai-transformation.html` and `personal-services/personal-service-generative-ai-transformation.html`.',
            'Visualizes the transition from naive prompting to domain-grounded autonomous agent teams.'
        ]
    },
    'assets/guarantees': {
        'title': 'Service Guarantees Visual Assets',
        'type': 'dataset',
        'description': 'Trust badges, guarantee seals, and visual confirmation assets supporting the 100% money-back and walkaway guarantees.',
        'tags': ['assets', 'guarantees', 'trust-badges', 'risk-reversal'],
        'guidelines': [
            'Used in `guarantees.html` and guarantee callouts across the site.',
            'Represents the $0 upfront, 5 equal 20% milestone installments, and walkaway guarantee trust models.'
        ]
    },
    'assets/js': {
        'title': 'Asset JavaScript Libraries',
        'type': 'module',
        'description': 'Vendored clientside script helpers and interactive UI utilities supporting asset galleries and diagrams.',
        'tags': ['assets', 'javascript', 'frontend', 'libraries'],
        'guidelines': [
            'Contains static helper scripts for asset displays.',
            'Keep scripts lightweight and free of external runtime tracking dependencies.'
        ]
    },
    'assets/local-ai-revolution': {
        'title': 'Local AI Revolution Visual Assets',
        'type': 'dataset',
        'description': 'Hardware topologies, private server charts, VRAM allocation matrices, and benchmark diagrams for Local LLMs.',
        'tags': ['assets', 'local-llms', 'vram', 'benchmarks', 'private-ai'],
        'guidelines': [
            'Supplies diagrams for `business-services/business-service-local-llms-cost-slashing.html` and `claude-training/claude-local-llms.html`.',
            'Illustrates 60%-90% cloud API bill reduction via Ollama, vLLM, and GGUF/AWQ local model inference.'
        ]
    },
    'assets/personal-service-local-llms': {
        'title': 'Personal Local LLMs Visual Assets',
        'type': 'dataset',
        'description': 'Diagrams and hardware allocation graphs for personal workstation local LLM deployments.',
        'tags': ['assets', 'personal-services', 'local-llms', 'workstation'],
        'guidelines': [
            'Supports `personal-services/personal-service-local-llms-cost-slashing.html`.',
            'Shows local workstation setups with consumer GPUs, LM Studio, and Ollama integration.'
        ]
    },
    'business-services': {
        'title': 'Enterprise & Business AI Services Catalog',
        'type': 'module',
        'description': 'Dedicated specification pages for 16 specialized enterprise AI training services, each with complete day-by-day syllabi, deliverables, and architecture blueprints.',
        'tags': ['business-services', 'enterprise-ai', 'service-catalog', 'syllabi'],
        'guidelines': [
            'Contains exactly 16 enterprise training services (`business-service-01` through `business-service-16`).',
            'All services are priced at $300 USD base (1 seat included, +$300 per extra seat up to 10 seats max) for 5 days live + 5 days support.',
            'Ensure every page maintains standardized CTAs featuring the Free LinkedIn Consultation link.',
            'Keep buttons styled with `css/buttons.css` and strictly enforce `text-decoration: none !important;`.'
        ]
    },
    'claude-training': {
        'title': 'Claude Architecture & Engineering Guides',
        'type': 'module',
        'description': 'Comprehensive engineering guides covering the Anthropic Claude ecosystem: Claude Code CLI, MCP tool building, prompt caching, orchestration, and local LLM pairing.',
        'tags': ['claude-training', 'claude-code', 'mcp', 'prompt-caching', 'orchestration'],
        'guidelines': [
            'Contains 12 in-depth architectural guides (`claude-code.html`, `claude-custom-mcp.html`, `claude-orchestration.html`, `claude-local-llms.html`, etc.).',
            'Never insert an article footer into any Claude guide.',
            'Model references must be evergreen without version clutter (Claude, Claude Opus, Claude Sonnet, Claude Haiku) except in code examples or specific pricing tables.',
            'Mermaid diagrams must be vertical (`TD`/`TB`) or square for mobile responsiveness.'
        ]
    },
    'css': {
        'title': 'Global Stylesheets & Design System',
        'type': 'module',
        'description': 'Modular CSS stylesheets defining color tokens, typography, radiant buttons, high-contrast dark theme, and responsive layouts.',
        'tags': ['css', 'design-system', 'tokens', 'responsive', 'buttons'],
        'guidelines': [
            '`buttons.css` defines canonical button formats (`.btn`, `.btn-commission`, `.btn-linkedin`, `.btn-gold`, `.btn-lg`). Buttons must never have underlined text.',
            'Hyperlinks default to gold (`var(--gold, #fbbf24)`) transitioning to gold-dim (`#d97706`) on hover.',
            'Code windows must have sharp dark boundaries (`#040711` or `#060913`) and crisp borders.',
            'Responsive design relies on media queries ensuring zero horizontal scroll blowout across mobile viewports.'
        ]
    },
    'data': {
        'title': 'Authoritative Data Directory (SSOT)',
        'type': 'dataset',
        'description': 'Authoritative Single Source of Truth (SSOT) JSON files modeling pricing, services, publications, FAQs, and platform configurations.',
        'tags': ['data', 'ssot', 'read-only', 'pricing', 'services', 'authoritative'],
        'guidelines': [
            'CRITICAL AGENT RULE: The `data/` directory is strictly read-only against all source control operations, scripts, and test suites.',
            '`GEMINI.md` is the ultimate project specification; `data/*.json` models this specification for outward synchronization.',
            'Data flows strictly outward (`data/*.json` -> HTML / JS / Markdown / Schema files). Never mutate programmatically.',
            'Pre-test and post-test SHA-256 verification in `python/run_tests.py` verifies data integrity. Any automated change causes an immediate test failure.',
            'Only direct manual authoring by the user is permitted.'
        ]
    },
    'docs': {
        'title': 'Architecture & Project Documentation',
        'type': 'module',
        'description': 'Operational playbooks, delivery economics, Purchasing Power Parity registries, and consolidated service summaries.',
        'tags': ['docs', 'architecture', 'playbooks', 'pricing-reference', 'delivery-economics'],
        'guidelines': [
            'Houses `services-summary.md` (all 16 business and 16 personal services with canonical $3000 flagship / $300 individual pricing).',
            'Houses `parity pricing.md` (complete 195-country World Bank FY2027 PPP tier assignments).',
            'Houses `hourly-rates.md` (delivery economics breakdown across all 6 parity tiers).',
            'Houses `website-fixes.md` (historical audit and resolution log).',
            'Strictly preserve zero emojis and zero em-dashes across all markdown files.'
        ]
    },
    'google-training': {
        'title': 'Google Gemini & Gemma Engineering Guides',
        'type': 'module',
        'description': 'Comprehensive technical reference guides for Google AI Studio, Google Antigravity IDE, Gemini Deep Think extended reasoning, Gemma local models, and multimodal architectures.',
        'tags': ['google-training', 'gemini', 'gemma', 'antigravity-ide', 'deepthink', 'multimodal'],
        'guidelines': [
            'Contains 12 deep-dive guides (`google-antigravity-ide.html`, `google-ai-studio.html`, `google-gemma-local-models.html`, `google-gemini-4-argon.html`, etc.).',
            'Model references must be evergreen (Gemini, Gemini Pro, Gemini Flash, Gemini Argon, Gemma) except in code strings or pricing calculations.',
            'Every code block must feature a structured header bar with a working Copy button.',
            'Maintain vertical or square Mermaid diagrams and organic in-prose anchor reference links.'
        ]
    },
    'icons': {
        'title': 'Vector SVG Icon & Badge Repository',
        'type': 'dataset',
        'description': 'Optimized scalable SVG platform icons, publishing logos, service badges, and branding glyphs.',
        'tags': ['icons', 'svg', 'vectors', 'publishing-platforms', 'badges'],
        'guidelines': [
            'Contains 42 standalone SVG vector files for platforms (Dev.to, GitHub, GitLab, LinkedIn, Bluesky, Medium, Substack, Reddit, etc.).',
            'Dev.to is an approved platform (`dev.svg`); never remove Dev.to links or icons.',
            'Use SVGs via `<img>` tags or inline markup with gold/cyan accent colors.',
            'Ensure clean vector markup with zero unnecessary metadata or embedded bitmap rasters.'
        ]
    },
    'images': {
        'title': 'Editorial & Article WebP Images',
        'type': 'dataset',
        'description': 'Standalone editorial and publication graphics in modern WebP format, specifically featured across technical blog articles and articles.html.',
        'tags': ['images', 'webp', 'editorial', 'articles', 'publication-graphics'],
        'guidelines': [
            'Distinct from `assets/`: holds 14 high-resolution WebP technical illustrations for articles (e.g., `frontier-llm-benchmarks-2026.webp`, `quantum-threat-bitcoin-cryptography.webp`, `openclaw-autonomous-ai-engineer.webp`).',
            'Used directly in `articles.html` and syndication headers.',
            'Optimized for rapid web loading and high visual fidelity in dark-themed layouts.'
        ]
    },
    'internal-use': {
        'title': 'Internal Trainer Mastery & Operational Playbooks',
        'type': 'module',
        'description': 'Internal trainer evaluations, full-stack website audits, competitive landscape assessments, and live training delivery runbooks.',
        'tags': ['internal-use', 'trainer-guide', 'playbooks', 'competitive-analysis', 'audit'],
        'guidelines': [
            'Contains `trainer-guide.md`: comprehensive 25,000+ word trainer mastery report.',
            'Reflects the full 16 business + 16 personal service catalog and 80 HTML page footprint.',
            'Details the 50-10 Rhythmic Sprint Protocol for 4-hour daily live engineering sessions.',
            'Strictly internal reference; not linked in public site navigation or sitemaps.'
        ]
    },
    'md-html-sync': {
        'title': 'Markdown HTML Synchronization Root',
        'type': 'module',
        'description': 'Markdown mirror representations of live HTML pages, kept synchronized for consumption by LLM coding agents and crawlers.',
        'tags': ['md-html-sync', 'markdown-mirrors', 'llm-friendly', 'state-tracking'],
        'guidelines': [
            'Synchronized via `python3 python/md_html_sync.py` only during commit preparation, not after every single edit.',
            'Contains `.sync-state.json` to track modification hashes and avoid unnecessary rebuilds.',
            'Organized into subdirectories mirroring the site structure (`business-services/`, `personal-services/`, `claude-training/`, `google-training/`).',
            'Never delete markdown mirrors directly with `rm`; use `python3 python/delete_page.py` when deleting pages.'
        ]
    },
    'md-html-sync/business-services': {
        'title': 'Business Services Markdown Mirrors',
        'type': 'module',
        'description': 'Synchronized Markdown mirrors corresponding to the 16 enterprise business service HTML pages.',
        'tags': ['md-html-sync', 'business-services', 'markdown', 'enterprise-ai'],
        'guidelines': [
            'Mirrors `business-services/*.html` into clean Markdown representations for LLM consumption.',
            'Automatically updated by `python3 python/md_html_sync.py`.'
        ]
    },
    'md-html-sync/claude-training': {
        'title': 'Claude Training Markdown Mirrors',
        'type': 'module',
        'description': 'Synchronized Markdown mirrors corresponding to the 12 Anthropic Claude engineering guides.',
        'tags': ['md-html-sync', 'claude-training', 'markdown', 'claude-code'],
        'guidelines': [
            'Mirrors `claude-training/*.html` into clean Markdown representations.',
            'Excludes article footers in accordance with Claude article guidelines.'
        ]
    },
    'md-html-sync/google-training': {
        'title': 'Google Training Markdown Mirrors',
        'type': 'module',
        'description': 'Synchronized Markdown mirrors corresponding to the 12 Google Gemini and Gemma engineering guides.',
        'tags': ['md-html-sync', 'google-training', 'markdown', 'gemini'],
        'guidelines': [
            'Mirrors `google-training/*.html` into clean Markdown representations.',
            'Maintains code block integrity and structured reference lists.'
        ]
    },
    'md-html-sync/personal-services': {
        'title': 'Personal Services Markdown Mirrors',
        'type': 'module',
        'description': 'Synchronized Markdown mirrors corresponding to the 16 personal 1-on-1 service HTML pages.',
        'tags': ['md-html-sync', 'personal-services', 'markdown', 'mentorship'],
        'guidelines': [
            'Mirrors `personal-services/*.html` into clean Markdown representations.',
            'Preserves syllabi, daily training limits, and pricing details.'
        ]
    },
    'nlp-fuzzy-expert': {
        'title': 'NLP Fuzzy Expert Knowledge Synchronization Engine',
        'type': 'module',
        'description': 'Forward-chaining inference engine, fuzzy matcher, continuous file watcher daemon, and auto-healing subsystem synchronizing data/*.json outward.',
        'tags': ['nlp-fuzzy-expert', 'inference-engine', 'synchronization', 'fuzzy-matcher', 'watcher'],
        'guidelines': [
            'Core engine: `expert_system.py`, `synchronizer.py`, `watcher.py`, and `auto_healer.py`.',
            'Run `python3 nlp-fuzzy-expert/synchronizer.py` to synchronize facts from `data/*.json` outward to HTML and JS files.',
            'Prints working steps 1-6 with 1-based changed line numbers and an ASCII summary table.',
            'Optimization: Summaries are printed to terminal; any failed checks are recorded in `FAILED-CHECKS.md`. If all pass, it records `Everything passed.`',
            '`save_json()` strictly blocks writing to any path inside `data/` via `PermissionError`.'
        ]
    },
    'personal-services': {
        'title': 'Personal & Professional 1-on-1 AI Services Catalog',
        'type': 'module',
        'description': 'Dedicated specification pages for 16 personal AI acceleration services, designed strictly for one person with 4 hours/day of live pair-programming.',
        'tags': ['personal-services', '1-on-1', 'private-mentoring', 'career-acceleration'],
        'guidelines': [
            'Contains exactly 16 personal training services (`personal-service-01` through `personal-service-16`).',
            'All individual picks are priced at $300 USD base (strictly 1 person) for 5 days live + 5 days support.',
            'Ensure every page maintains standardized CTAs featuring the Free LinkedIn Consultation link.',
            'Strictly enforce zero emojis and zero em-dashes across all files.'
        ]
    },
    'pricing': {
        'title': 'Purchasing Power Parity (PPP) Regional Tier Pages',
        'type': 'module',
        'description': 'Dedicated regional specification pages for all 6 Purchasing Power Parity tiers, covering 195 countries based on the World Bank FY2027 framework.',
        'tags': ['pricing', 'ppp-tiers', 'global-parity', 'world-bank-fy2027'],
        'guidelines': [
            'Contains 6 tier landing pages (`pricing-tier-1-usa-china-eu-100.html` through `pricing-tier-6-crisis-affected-10.html`).',
            'Authoritative rates: Tier 1 (100% / $3,000 flagship / $300 pick), Tier 2 (75% / $2,250 / $225), Tier 3 & Tier 4 (50% / $1,500 / $150), Tier 5 (25% / $750 / $75), Tier 6 (Custom quote).',
            'Every pricing page must clearly articulate the $0 upfront advance and 5 equal 20% milestone installment structure.'
        ]
    },
    'pricing-calculator': {
        'title': 'Interactive Pricing Calculator Engine Modules',
        'type': 'module',
        'description': 'Modular JavaScript pricing calculation engines, tier calculators, and unit tests verifying live pricing logic.',
        'tags': ['pricing-calculator', 'javascript-modules', 'calculator-engine', 'parity-calc'],
        'guidelines': [
            'Contains modular calculators: `pricing-calculator-tier-1.js` through `pricing-calculator-tier-6.js`.',
            'Validated by JavaScript tests (`pricing-calculator-tier-*-tests.js`).',
            'Run `node pricing-tests.js` to execute 78+ DOM and calculator assertions.'
        ]
    },
    'python': {
        'title': 'Repository Python Automation & Tooling Suite',
        'type': 'module',
        'description': 'Core automation scripts for component inlining, link verification, schema extraction, markdown mirroring, and metrics reporting.',
        'tags': ['python', 'build-system', 'automation', 'tooling', 'linters'],
        'guidelines': [
            '`build_components.py`: Inlines `header.html` and `footer.html` across all HTML pages.',
            '`check_links.py`: Audits internal links and anchor targets across all HTML files (run before committing).',
            '`sync_schemas.py`: Synchronizes embedded JSON-LD schemas into `schema/` (run before committing).',
            '`md_html_sync.py`: Synchronizes Markdown mirror files in `md-html-sync/` (run before committing).',
            '`generate_metrics.py`: Updates codebase telemetry in `METRICS.md`.',
            '`delete_page.py`: Safely purges HTML pages, schemas, markdown mirrors, and navigation references.'
        ]
    },
    'python/hooks': {
        'title': 'Git Repository Lifecycle Hooks',
        'type': 'module',
        'description': 'Pre-commit, pre-push, and editor lifecycle hooks enforcing quality gates, test pass rates, and schema synchronization.',
        'tags': ['python', 'git-hooks', 'pre-commit', 'pre-push', 'quality-gate'],
        'guidelines': [
            '`pre-commit`: Automatically executes `python3 python/run_tests.py` and `node pricing-tests.js`. Aborts commit if any test fails.',
            '`pre-push`: Performs final link and schema verification before pushing to origin.',
            '`post_edit_hook.py`: IDE lifecycle trigger for component inlining.',
            'Never bypass git hooks with `--no-verify`.'
        ]
    },
    'schema': {
        'title': 'Structured Data & Schema.org JSON-LD Definitions',
        'type': 'dataset',
        'description': 'Comprehensive Schema.org JSON-LD definitions extracted from live HTML pages to empower search engine indexing and generative search engines (GEO/AEO).',
        'tags': ['schema', 'json-ld', 'structured-data', 'schema-org', 'geo-aeo'],
        'guidelines': [
            'Contains 71+ JSON-LD files mirroring schemas embedded in live HTML pages.',
            'Live HTML files are the definitive source of truth for schemas; `python/sync_schemas.py` extracts schemas into `schema/`.',
            'Synchronized before committing via `python3 python/sync_schemas.py`.',
            'Ensures rich snippets for Organization, Person, Service, FAQPage, and TechnicalArticle.'
        ]
    },
    'tests': {
        'title': 'Unit Test Suite & Verification Harness',
        'type': 'module',
        'description': 'Comprehensive unit test suite covering expert synchronizer rules, pricing calculations, link integrity, and automation tools.',
        'tags': ['tests', 'unittest', 'verification', 'qa', 'test-suite'],
        'guidelines': [
            'Execute all unit tests with `python3 python/run_tests.py` or `python3 -m unittest discover -s tests -v`.',
            'Pre-test and post-test cryptographic SHA-256 validation guarantees that `data/*.json` is never mutated by test runs.',
            'Always verify a 100% test pass rate across all test cases before committing.',
            'Pre-commit gate strictly blocks commits on any test failure.'
        ]
    },
    'tests/js': {
        'title': 'JavaScript Unit & Integration Tests',
        'type': 'module',
        'description': 'Node.js test suites verifying interactive clientside calculators, payment schedules, and edge-case pricing selections.',
        'tags': ['tests', 'javascript-tests', 'node', 'calculator-tests'],
        'guidelines': [
            'Contains test suites: `test_business_services.js`, `test_payment_schedule.js`, `test_custom_package.js`, `test_extra_participants.js`, etc.',
            'Executed via `node pricing-tests.js`.',
            'Ensures frontend interactive pricing widgets remain 100% bug-free and synchronized with backend rules.'
        ]
    }
}


def build_okf_content(rel_dir, meta, now_iso):
    abs_dir = os.path.join(REPO_ROOT, rel_dir) if rel_dir != '.' else REPO_ROOT
    items = sorted(os.listdir(abs_dir))
    files = [f for f in items if os.path.isfile(os.path.join(abs_dir, f)) and f != 'OKF.md']
    subdirs = [s for s in items if os.path.isdir(os.path.join(abs_dir, s)) and not s.startswith('.') and s != '__pycache__']

    parent_dir = os.path.dirname(rel_dir) if rel_dir != '.' else None
    tags_str = '[' + ', '.join(meta['tags']) + ']'

    lines = [
        '---',
        f'type: {meta["type"]}',
        f'title: "{meta["title"]}"',
        f'description: "{meta["description"]}"',
        f'tags: {tags_str}',
        f'timestamp: {now_iso}',
        '---',
        '',
        f'# {meta["title"]}',
        '',
        meta["description"],
        '',
        '## Directory Role & Agent Guidelines',
        ''
    ]

    for g in meta['guidelines']:
        lines.append(f'- {g}')
    lines.append('')

    lines.append('## Subdirectories')
    lines.append('')
    if subdirs:
        for s in subdirs:
            lines.append(f'- [`{s}/`]({s}/OKF.md): Subdirectory knowledge bundle')
    else:
        lines.append('None (Leaf directory).')
    lines.append('')

    lines.append('## File Inventory')
    lines.append('')
    if files:
        for f in files[:40]:
            lines.append(f'- [`{f}`]({f})')
        if len(files) > 40:
            lines.append(f'- ... and {len(files) - 40} additional files.')
    else:
        lines.append('None.')
    lines.append('')

    lines.append('## Context & Cross-References')
    lines.append('')
    if parent_dir and parent_dir != '':
        lines.append('- Parent Directory: [`../OKF.md`](../OKF.md)')
    elif rel_dir != '.':
        lines.append('- Root Directory: [`/OKF.md`](../OKF.md)')
    lines.append('- Architectural Specification: [`GEMINI.md`](/GEMINI.md)')
    lines.append('- Repository Telemetry: [`METRICS.md`](/METRICS.md)')
    lines.append('')

    return '\n'.join(lines)


def main():
    now_iso = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    count = 0
    for rel_dir, meta in DIRECTORY_METADATA.items():
        okf_path = os.path.join(REPO_ROOT, rel_dir, 'OKF.md')
        content = build_okf_content(rel_dir, meta, now_iso)
        with open(okf_path, 'w', encoding='utf-8') as f:
            f.write(content)
        count += 1
    print(f'Successfully generated {count} detailed OKF.md files.')


if __name__ == '__main__':
    main()
