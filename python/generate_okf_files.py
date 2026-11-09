#!/usr/bin/env python3
"""
generate_okf_files.py - Generates Open Knowledge Format (OKF.md) bundles across all directories.
Strictly zero emojis and zero em-dashes compliant.
"""

import os
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ALL_DIRS_INFO = {
    '.': {
        'title': 'Website Repository Root',
        'type': 'module',
        'description': 'Root directory containing core website entry points, configuration, and top-level documentation.',
        'tags': ['root', 'entry-points', 'configuration']
    },
    'article-footers': {
        'title': 'Article Footers Module',
        'type': 'module',
        'description': 'Reusable article footer templates and components in HTML, Markdown, and Word formats.',
        'tags': ['article-footers', 'templates', 'reusable-components']
    },
    'assets': {
        'title': 'Site Assets Hub',
        'type': 'dataset',
        'description': 'Global static assets including branding logos, OpenGraph cards, media, and diagrams.',
        'tags': ['assets', 'branding', 'media', 'static']
    },
    'assets/advanced-ai-agents': {
        'title': 'Advanced AI Agents Visual Assets',
        'type': 'dataset',
        'description': 'Architecture diagrams and conceptual artwork for Advanced AI Agents service modules.',
        'tags': ['assets', 'advanced-agents', 'diagrams']
    },
    'assets/agentic-ai-systems': {
        'title': 'Agentic AI Systems Visual Assets',
        'type': 'dataset',
        'description': 'Visual assets, schemas, and diagrams illustrating autonomous Agentic AI loops and workflows.',
        'tags': ['assets', 'agentic-systems', 'diagrams']
    },
    'assets/ai-agents-orchestration': {
        'title': 'AI Agent Orchestration Visual Assets',
        'type': 'dataset',
        'description': 'Diagrams and imagery for multi-agent swarm orchestration and hierarchical coordination.',
        'tags': ['assets', 'orchestration', 'swarms']
    },
    'assets/ai-developer-training': {
        'title': 'AI Developer Training Visual Assets',
        'type': 'dataset',
        'description': 'Technical screenshots, architecture charts, and visual assets for AI Developer bootcamps.',
        'tags': ['assets', 'developer-training', 'diagrams']
    },
    'assets/antigravity': {
        'title': 'Google Antigravity Visual Assets',
        'type': 'dataset',
        'description': 'Branding, screenshots, and visual components for Google Antigravity IDE and SDK documentation.',
        'tags': ['assets', 'antigravity', 'google-ecosystem']
    },
    'assets/claude-orchestration': {
        'title': 'Claude Orchestration Visual Assets',
        'type': 'dataset',
        'description': 'Architecture diagrams and execution models for Claude Agent Orchestration workflows.',
        'tags': ['assets', 'claude', 'orchestration']
    },
    'assets/generative-ai-transformation': {
        'title': 'Generative AI Transformation Visual Assets',
        'type': 'dataset',
        'description': 'Enterprise transformation roadmaps and architectural graphics for Generative AI enablement.',
        'tags': ['assets', 'transformation', 'enterprise-ai']
    },
    'assets/guarantees': {
        'title': 'Service Guarantees Visual Assets',
        'type': 'dataset',
        'description': 'Badges, seals, and visual proof assets supporting the 100% money-back and walkaway guarantees.',
        'tags': ['assets', 'guarantees', 'trust']
    },
    'assets/js': {
        'title': 'Asset JavaScript Libraries',
        'type': 'module',
        'description': 'Vendored clientside script helpers and interactive UI utilities.',
        'tags': ['assets', 'javascript', 'frontend']
    },
    'assets/local-ai-revolution': {
        'title': 'Local AI Revolution Visual Assets',
        'type': 'dataset',
        'description': 'Hardware topologies, private server charts, and benchmark diagrams for Local LLMs.',
        'tags': ['assets', 'local-llms', 'private-ai']
    },
    'assets/personal-service-local-llms': {
        'title': 'Personal Local LLMs Visual Assets',
        'type': 'dataset',
        'description': 'Diagrams and hardware allocation graphs for personal workstation local LLM deployments.',
        'tags': ['assets', 'personal-services', 'local-llms']
    },
    'business-services': {
        'title': 'Enterprise & Business AI Services',
        'type': 'module',
        'description': 'Service specifications and interactive pages for 16 specialized enterprise AI training tracks.',
        'tags': ['business-services', 'enterprise-ai', 'service-catalogue']
    },
    'claude-training': {
        'title': 'Claude Architecture & Training Guides',
        'type': 'module',
        'description': 'Comprehensive guides for Claude Code, MCP tools, prompt caching, and multi-agent coordination.',
        'tags': ['claude-training', 'claude-code', 'mcp', 'agentic-ai']
    },
    'css': {
        'title': 'Global Stylesheets & Design System',
        'type': 'module',
        'description': 'Modern CSS styles, responsive layouts, color systems, typography, and button components.',
        'tags': ['css', 'design-system', 'styling', 'responsive']
    },
    'data': {
        'title': 'Authoritative Knowledge Data Hub',
        'type': 'dataset',
        'description': 'Single Source of Truth (SSOT) JSON data files modeling pricing, tiers, services, and configs.',
        'tags': ['data', 'ssot', 'authoritative', 'json']
    },
    'docs': {
        'title': 'Architecture & Project Documentation',
        'type': 'module',
        'description': 'High-level operational playbooks, architecture specifications, and maintenance guidelines.',
        'tags': ['docs', 'architecture', 'playbooks']
    },
    'google-training': {
        'title': 'Google Gemini & Antigravity Training Guides',
        'type': 'module',
        'description': 'In-depth documentation and masterclasses for Google Gemini, Antigravity IDE, and Gemma models.',
        'tags': ['google-training', 'gemini', 'antigravity', 'gemma']
    },
    'icons': {
        'title': 'Vector Icons & Tech Badges',
        'type': 'dataset',
        'description': 'Optimized SVG technology icons, service badges, and branding elements.',
        'tags': ['icons', 'svg', 'vectors']
    },
    'images': {
        'title': 'General Image Assets',
        'type': 'dataset',
        'description': 'General photographs, profile portraits, hero graphics, and editorial illustrations.',
        'tags': ['images', 'photography', 'illustrations']
    },
    'internal-use': {
        'title': 'Internal Use Assets & References',
        'type': 'dataset',
        'description': 'Internal drafts, work-in-progress references, and staging files.',
        'tags': ['internal-use', 'staging']
    },
    'md-html-sync': {
        'title': 'Markdown HTML Synchronization Root',
        'type': 'module',
        'description': 'Markdown mirror representations of live HTML pages kept synchronized for AI agents.',
        'tags': ['md-html-sync', 'markdown-mirrors', 'dual-format']
    },
    'md-html-sync/business-services': {
        'title': 'Business Services Markdown Mirrors',
        'type': 'module',
        'description': 'Synchronized Markdown files corresponding to business service HTML specification pages.',
        'tags': ['md-html-sync', 'business-services', 'markdown']
    },
    'md-html-sync/claude-training': {
        'title': 'Claude Training Markdown Mirrors',
        'type': 'module',
        'description': 'Synchronized Markdown files corresponding to Claude guide HTML pages.',
        'tags': ['md-html-sync', 'claude-training', 'markdown']
    },
    'md-html-sync/google-training': {
        'title': 'Google Training Markdown Mirrors',
        'type': 'module',
        'description': 'Synchronized Markdown files corresponding to Google Gemini guide HTML pages.',
        'tags': ['md-html-sync', 'google-training', 'markdown']
    },
    'md-html-sync/personal-services': {
        'title': 'Personal Services Markdown Mirrors',
        'type': 'module',
        'description': 'Synchronized Markdown files corresponding to personal service HTML specification pages.',
        'tags': ['md-html-sync', 'personal-services', 'markdown']
    },
    'nlp-fuzzy-expert': {
        'title': 'NLP Fuzzy Expert Knowledge Synchronization Engine',
        'type': 'module',
        'description': 'Forward-chaining inference engine, fuzzy matcher, file watcher, and auto-healing subsystem.',
        'tags': ['nlp-fuzzy-expert', 'inference-engine', 'synchronization', 'rules']
    },
    'personal-services': {
        'title': 'Personal & Professional AI Services',
        'type': 'module',
        'description': '1-on-1 private mentoring specifications and standalone pages across 16 core AI disciplines.',
        'tags': ['personal-services', 'mentorship', 'service-catalogue']
    },
    'pricing': {
        'title': 'Purchasing Power Parity (PPP) Tier Documentation',
        'type': 'module',
        'description': 'Dedicated regional pricing specification pages across all 6 Purchasing Power Parity tiers.',
        'tags': ['pricing', 'ppp-tiers', 'global-parity']
    },
    'pricing-calculator': {
        'title': 'Interactive Pricing Calculator Engine',
        'type': 'module',
        'description': 'Modular JavaScript pricing calculation engines, test runners, and configuration modules.',
        'tags': ['pricing-calculator', 'javascript', 'engine', 'parity']
    },
    'python': {
        'title': 'Repository Python Build & Maintenance Automation',
        'type': 'module',
        'description': 'Build scripts, component inliners, markdown synchronizers, link checkers, and metrics generators.',
        'tags': ['python', 'build-system', 'automation', 'tooling']
    },
    'python/hooks': {
        'title': 'Git Repository Lifecycle Hooks',
        'type': 'module',
        'description': 'Pre-commit and git lifecycle hooks enforcing test pass rates, linting, and metrics updates.',
        'tags': ['python', 'git-hooks', 'pre-commit', 'quality-gate']
    },
    'schema': {
        'title': 'Structured Data & Schema.org Definitions',
        'type': 'dataset',
        'description': 'Machine-readable JSON-LD schemas representing entities, services, guides, and breadcrumbs.',
        'tags': ['schema', 'json-ld', 'structured-data', 'seo']
    },
    'tests': {
        'title': 'Unit Test Suite & Verification Harness',
        'type': 'module',
        'description': 'Comprehensive unit tests covering synchronizer rules, pricing logic, link integrity, and tools.',
        'tags': ['tests', 'unittest', 'verification', 'qa']
    },
    'tests/js': {
        'title': 'JavaScript Unit & Integration Tests',
        'type': 'module',
        'description': 'Node.js and headless browser test suites verifying clientside calculators and scripts.',
        'tags': ['tests', 'javascript-tests', 'node']
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

    if rel_dir == 'data':
        lines.extend([
            '- **CRITICAL AGENT RULE**: The `data/` directory is strictly read-only to source control and code scripts.',
            '- All JSON files here are the authoritative Single Source of Truth (SSOT) specified by `GEMINI.md`.',
            '- Data flows strictly outward (`data/*.json` -> HTML / JS / Markdown / Schemas). Never edit or mutate programmatically.',
            ''
        ])
    elif rel_dir.startswith('assets'):
        lines.extend([
            '- This directory contains static assets (images, diagrams, scripts, branding).',
            '- Refer to images via absolute or relative paths from HTML pages; do not modify binary files directly.',
            ''
        ])
    elif rel_dir == 'nlp-fuzzy-expert':
        lines.extend([
            '- Houses the repository knowledge synchronization engine.',
            '- Run `python3 nlp-fuzzy-expert/synchronizer.py` to synchronize facts from `data/*.json` outward.',
            '- Failure reports are recorded in `FAILED-CHECKS.md`.',
            ''
        ])
    elif rel_dir in ('business-services', 'personal-services'):
        lines.extend([
            '- Contains standalone specification pages for the 16 specialized AI training services.',
            '- Always keep service naming, duration, and parity pricing consistent with `data/` and `GEMINI.md`.',
            ''
        ])
    elif rel_dir in ('claude-training', 'google-training'):
        lines.extend([
            '- Comprehensive technical guides and architectural blueprints.',
            '- Maintain clean unnumbered headings, sharp code block boundaries, and interactive copy buttons.',
            ''
        ])
    elif rel_dir == 'pricing':
        lines.extend([
            '- Regional Purchasing Power Parity (PPP) tier specification pages (Tier 1 through Tier 6).',
            '- Flagships: $3,000 / $2,250 / $1,500 / $1,500 / $750 / Custom USD.',
            '- Individual picks: $300 / $225 / $150 / $150 / $75 / Custom USD.',
            ''
        ])
    elif rel_dir == 'pricing-calculator':
        lines.extend([
            '- Interactive clientside pricing calculator engine.',
            '- Run `node pricing-tests.js` to verify calculation accuracy across all parity tiers.',
            ''
        ])
    elif rel_dir == 'python':
        lines.extend([
            '- Automation scripts for component inlining, link integrity verification, markdown mirrors, and metrics.',
            '- Re-run `python3 python/build_components.py` whenever `header.html` or `footer.html` changes.',
            ''
        ])
    elif rel_dir == 'schema':
        lines.extend([
            '- Machine-readable JSON-LD schemas extracted and mirrored from live HTML pages.',
            '- Synchronized via `python3 python/sync_schemas.py` during release verification passes.',
            ''
        ])
    elif rel_dir in ('tests', 'tests/js'):
        lines.extend([
            '- Comprehensive test verification suite.',
            '- Run Python tests with `python3 python/run_tests.py` or `python3 -m unittest discover -s tests -v`.',
            '- Pre-commit enforces 100% test pass rate across all suites.',
            ''
        ])
    else:
        lines.extend([
            f'- Maintained as part of the {meta["title"]} subsystem.',
            '- Strictly preserve zero emojis and zero em-dashes across all files in this directory.',
            ''
        ])

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
    for rel_dir, meta in ALL_DIRS_INFO.items():
        okf_path = os.path.join(REPO_ROOT, rel_dir, 'OKF.md')
        content = build_okf_content(rel_dir, meta, now_iso)
        with open(okf_path, 'w', encoding='utf-8') as f:
            f.write(content)
        count += 1
    print(f'Successfully generated {count} OKF.md files.')


if __name__ == '__main__':
    main()

