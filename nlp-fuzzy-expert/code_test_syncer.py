#!/usr/bin/env python3
"""
code_test_syncer.py - Modular Source Code and Test Suite Synchronizer.

Synchronizes authoritative facts from data/*.json into:
- All source code files: python/*.py, pricing-calculator.js, components.js
- All unit test files: tests/test_*.py, pricing-tests.js

Generates:
- code_references.json: Granular tracking of all code and test occurrences
- code-sync.md: Markdown synchronization report for code and tests

Strict Repo Compliance:
- Zero emojis.
- Zero em-dashes (spaced hyphens ' - ' or plain hyphens '-' used).
- Strict data/ directory immutability.
"""

import os
import sys
import re
import difflib
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(REPO_DIR, "data")

sys.path.insert(0, SCRIPT_DIR)
from expert_system import KnowledgeBase, load_json, save_json


def compute_changed_lines(old_text, new_text):
    """Computes exact 1-based changed line numbers between old and new text."""
    if old_text == new_text:
        return "None"

    old_lines = old_text.splitlines()
    new_lines = new_text.splitlines()

    matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
    changed_line_numbers = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "insert"):
            for line_idx in range(j1 + 1, j2 + 1):
                changed_line_numbers.append(line_idx)
        elif tag == "delete":
            changed_line_numbers.append(max(1, j1))

    changed_line_numbers = sorted(set(changed_line_numbers))
    if not changed_line_numbers:  # pragma: no cover
        return "None"

    ranges = []
    start = changed_line_numbers[0]
    prev = start
    for num in changed_line_numbers[1:]:
        if num == prev + 1:
            prev = num
        else:
            if start == prev:  # pragma: no cover
                ranges.append(f"{start}")
            else:
                ranges.append(f"{start}-{prev}")
            start = num
            prev = num
    if start == prev:
        ranges.append(f"{start}")
    else:
        ranges.append(f"{start}-{prev}")

    return ", ".join(ranges)


class CodeTestSynchronizer:
    """Synchronizes authoritative data from KnowledgeBase into source code and test files."""

    def __init__(self, kb=None, repo_dir=None, config_path=None):
        self.repo_dir = repo_dir or REPO_DIR
        self.kb = kb or KnowledgeBase(data_dir=os.path.join(self.repo_dir, "data"))
        self.config_path = config_path or os.path.join(SCRIPT_DIR, "code_sync_config.json")
        self.config = load_json(self.config_path) if os.path.exists(self.config_path) else {}
        self.occurrences = []

    def discover_source_files(self):
        """Discovers all source code files to synchronize.
        Strictly excludes data/ directory to enforce unidirectional sync.
        """
        abs_data = os.path.abspath(DATA_DIR)
        sources = [
            os.path.join(self.repo_dir, "pricing-calculator.js"),
            os.path.join(self.repo_dir, "components.js"),
            os.path.join(self.repo_dir, "python", "generate_sitemap.py"),
            os.path.join(self.repo_dir, "python", "build_components.py"),
            os.path.join(self.repo_dir, "python", "delete_page.py"),
            os.path.join(self.repo_dir, "python", "generate_og_card.py"),
            os.path.join(self.repo_dir, "python", "check_links.py"),
        ]
        return [
            f for f in sources
            if os.path.exists(f) and not os.path.abspath(f).startswith(abs_data)
        ]

    def discover_test_files(self):
        """Discovers all unit test files to synchronize."""
        tests = [os.path.join(self.repo_dir, "pricing-tests.js")]
        tests_dir = os.path.join(self.repo_dir, "tests")
        if os.path.exists(tests_dir):
            for fname in sorted(os.listdir(tests_dir)):
                if fname.startswith("test_") and fname.endswith(".py"):
                    tests.append(os.path.join(tests_dir, fname))
        return [f for f in tests if os.path.exists(f)]

    def sync_pricing_calculator(self, content):
        """Synchronizes pricing rates, formulas, included seats, and labels in pricing-calculator.js."""
        b_price = self.kb.get_fact("pricing", "b_flag_price", 2500)
        p_price = self.kb.get_fact("pricing", "p_flag_price", 2500)
        b_seats = self.kb.get_fact("pricing", "b_flag_seats", 10)
        b_extra = self.kb.get_fact("pricing", "b_flag_extra_seat", 250)
        b_days = self.kb.get_fact("pricing", "b_flag_days", 60)
        b_live = self.kb.get_fact("pricing", "b_flag_live_days", 30)
        b_supp = self.kb.get_fact("pricing", "b_flag_supp_days", 30)
        p_days = self.kb.get_fact("pricing", "p_flag_days", 60)
        p_live = self.kb.get_fact("pricing", "p_flag_live_days", 30)
        p_supp = self.kb.get_fact("pricing", "p_flag_supp_days", 30)
        b_ind = self.kb.get_fact("pricing", "b_ind_price", 250)

        # 1. Rate per single service
        content = re.sub(
            r'const\s+ratePerSvc\s*=\s*\d+;',
            f'const ratePerSvc = {b_ind};',
            content
        )

        # 2. Base price default and matching flagship price
        content = re.sub(
            r'let\s+basePrice\s*=\s*\d+;',
            f'let basePrice = {b_price};',
            content
        )
        content = re.sub(
            r'(\}\s*else\s+if\s*\(packageType\s*===\s*"pkg-business"\)\s*\{\s*\n\s*basePrice\s*=\s*)\d+;',
            rf'\g<1>{b_price};',
            content
        )
        content = re.sub(
            r'(\}\s*else\s+if\s*\(packageType\s*===\s*"pkg-personal"\)\s*\{\s*\n\s*basePrice\s*=\s*)\d+;',
            rf'\g<1>{p_price};',
            content
        )
        content = re.sub(
            r'const\s+matchingFlagshipPrice\s*=\s*\d+;',
            f'const matchingFlagshipPrice = {b_price};',
            content
        )
        content = re.sub(
            r'let\s+standardBasePrice\s*=\s*\d+;',
            f'let standardBasePrice = {b_price};',
            content
        )

        # Mode switch flagship base prices
        content = re.sub(
            r'(calcPackage\.value\s*===\s*"pkg-business"\s*&&\s*mode\s*===\s*"individual"\)\s*\{\s*\n\s*calcPackage\.value\s*=\s*"pkg-personal";\s*\n\s*standardBasePrice\s*=\s*)\d+;',
            rf'\g<1>{p_price};',
            content
        )
        content = re.sub(
            r'(calcPackage\.value\s*===\s*"pkg-personal"\s*&&\s*mode\s*===\s*"business"\)\s*\{\s*\n\s*calcPackage\.value\s*=\s*"pkg-business";\s*\n\s*standardBasePrice\s*=\s*)\d+;',
            rf'\g<1>{b_price};',
            content
        )
        content = re.sub(
            r'(selectedOpt\.value\s*===\s*"pkg-business"\)\s*\{\s*\n\s*setClientMode\("business"\);\s*\n\s*standardBasePrice\s*=\s*)\d+;',
            rf'\g<1>{b_price};',
            content
        )
        content = re.sub(
            r'(selectedOpt\.value\s*===\s*"pkg-personal"\)\s*\{\s*\n\s*setClientMode\("individual"\);\s*\n\s*standardBasePrice\s*=\s*)\d+;',
            rf'\g<1>{p_price};',
            content
        )

        # Service cards dynamic base rate in updateIndividualServiceCards
        content = re.sub(
            r'const\s+calculated\s*=\s*Math\.round\(\d+\s*\*\s*mult\);',
            f'const calculated = Math.round({b_ind} * mult);',
            content
        )

        # 3. Dynamic extra head rate formula
        content = re.sub(
            r'const\s+extraHeadRate\s*=\s*Math\.round\(extraParticipants\s*\*\s*\d+\s*\*\s*tier\.multiplier\);',
            f'const extraHeadRate = Math.round(extraParticipants * {b_extra} * tier.multiplier);',
            content
        )
        content = re.sub(
            r'const\s+extraSavings\s*=\s*Math\.round\(extraParticipants\s*\*\s*\d+\s*\*\s*\(tier\.discountPercent\s*/\s*100\)\);',
            f'const extraSavings = Math.round(extraParticipants * {b_extra} * (tier.discountPercent / 100));',
            content
        )

        # 4. Included seats calculation
        content = re.sub(
            r'Math\.max\(0,\s*participants\s*-\s*\d+\)',
            f'Math.max(0, participants - {b_seats})',
            content
        )

        # 5. Hints and descriptions
        content = re.sub(
            r'Business:\s*1-\d+\s*Included\s*\(\+\$\d+/extra seat\)',
            f'Business: 1-{b_seats} Included (+${b_extra}/extra seat)',
            content
        )
        content = re.sub(
            r'Business base covers up to \d+ participants \(additional seats are \$\d+ USD per seat scaled by country parity\)\.',
            f'Business base covers up to {b_seats} participants (additional seats are ${b_extra} USD per seat scaled by country parity).',
            content
        )

        # 6. Scope duration labels
        content = re.sub(
            r'Business Training Flagship \(\d+ Days: \d+d \+ \d+d Dedicated Support\)',
            f'Business Training Flagship ({b_days} Days: {b_live}d + {b_supp}d Dedicated Support)',
            content
        )
        content = re.sub(
            r'Personal Training Flagship \(\d+ Days: \d+d \+ \d+d Dedicated Support\)',
            f'Personal Training Flagship ({p_days} Days: {p_live}d + {p_supp}d Dedicated Support)',
            content
        )

        self.occurrences.append({
            "target": "pricing-calculator.js",
            "category": "source_code",
            "domain": "pricing",
            "field": "calculator_constants",
            "detail": f"Synced ratePerSvc={b_ind}, basePrice={b_price}, extraSeat={b_extra}, seats={b_seats}"
        })

        return content

    def sync_pricing_tests_js(self, content):
        """Synchronizes pricing constants block in pricing-tests.js."""
        b_price = self.kb.get_fact("pricing", "b_flag_price", 3000)
        p_price = self.kb.get_fact("pricing", "p_flag_price", 3000)
        b_seats = self.kb.get_fact("pricing", "b_flag_seats", 10)
        b_extra = self.kb.get_fact("pricing", "b_flag_extra_seat", 300)
        b_ind = self.kb.get_fact("pricing", "b_ind_price", 300)

        # Update or inject top-level constants block in pricing-tests.js
        constants_marker = "// --- AUTHORITATIVE SYNCHRONIZED PRICING FACTS ---"
        constants_block = (
            f"{constants_marker}\n"
            f"const B_FLAG_PRICE = {b_price};\n"
            f"const P_FLAG_PRICE = {p_price};\n"
            f"const B_IND_PRICE = {b_ind};\n"
            f"const B_EXTRA_SEAT = {b_extra};\n"
            f"const B_INCLUDED_SEATS = {b_seats};\n"
            f"// --- END AUTHORITATIVE SYNCHRONIZED PRICING FACTS ---"
        )

        if constants_marker in content:
            content = re.sub(
                rf'{re.escape(constants_marker)}[\s\S]*?// --- END AUTHORITATIVE SYNCHRONIZED PRICING FACTS ---',
                constants_block,
                content
            )
        else:
            match = re.search(r'(const PricingCalculator\s*=\s*[\s\S]*?;\n)', content)
            if match:
                content = content[:match.end()] + "\n" + constants_block + "\n" + content[match.end():]

        self.occurrences.append({
            "target": "pricing-tests.js",
            "category": "test",
            "domain": "pricing",
            "field": "test_constants",
            "detail": f"Synced B_FLAG={b_price}, P_FLAG={p_price}, EXTRA={b_extra}, IND={b_ind}, SEATS={b_seats}"
        })

        return content

    def sync_components_js(self, content):
        """Synchronizes Claude orchestration pages array in components.js."""
        cl_data = self.kb.raw_data.get("claude", {}).get("guides", [])
        cl_pages = [
            g.get('slug') if g.get('slug', '').endswith('.html') else f"{g.get('slug')}.html"
            for g in cl_data if g.get("slug")
        ]
        if not cl_pages:  # pragma: no cover
            cl_pages = [
                'claude-101.html', 'claude-features.html', 'claude-code.html',
                'claude-cowork.html', 'claude-skills.html', 'claude-custom-mcp.html',
                'claude-plugins.html', 'claude-connectors.html', 'claude-orchestration.html',
                'claude-design.html', 'claude-science.html', 'claude-local-llms.html'
            ]
        pages_str = "var orchestrationPages = [\n      '" + "',\n      '".join(cl_pages) + "'\n    ];"
        content = re.sub(r'var\s+orchestrationPages\s*=\s*\[[\s\S]*?\];', pages_str, content)

        self.occurrences.append({
            "target": "components.js",
            "category": "source_code",
            "domain": "claude",
            "field": "orchestration_pages",
            "detail": f"Synced {len(cl_pages)} orchestration guide page names"
        })
        return content

    def sync_python_sitemap(self, content):
        """Synchronizes services and guides in python/generate_sitemap.py."""
        bs_data = self.kb.raw_data.get("business_services", {}).get("services", [])
        ps_data = self.kb.raw_data.get("personal_services", {}).get("services", [])
        cl_data = self.kb.raw_data.get("claude", {}).get("guides", [])
        gem_data = self.kb.raw_data.get("gemini", {}).get("guides", [])

        # Business Services
        bs_slugs = [f"business-services/{s.get('slug')}" for s in bs_data if s.get("slug")]
        bs_str = "BUSINESS_SERVICES = [\n" + "\n".join(f'    "{s}",' for s in bs_slugs) + "\n]"
        content = re.sub(r'BUSINESS_SERVICES\s*=\s*\[[\s\S]*?\]', bs_str, content)

        # Claude Guides
        cl_slugs = [f"claude-training/{g.get('slug')}" for g in cl_data if g.get("slug")]
        cl_str = "CLAUDE_GUIDES = [\n" + "\n".join(f'    "{s}",' for s in cl_slugs) + "\n]"
        content = re.sub(r'CLAUDE_GUIDES\s*=\s*\[[\s\S]*?\]', cl_str, content)

        # Gemini Guides
        gem_slugs = [f"google-training/{g.get('slug')}" for g in gem_data if g.get("slug")]
        gem_str = "GEMINI_GUIDES = [\n" + "\n".join(f'    "{s}",' for s in gem_slugs) + "\n]"
        content = re.sub(r'GEMINI_GUIDES\s*=\s*\[[\s\S]*?\]', gem_str, content)

        self.occurrences.append({
            "target": "python/generate_sitemap.py",
            "category": "source_code",
            "domain": "sitemap",
            "field": "url_catalog",
            "detail": f"Synced {len(bs_slugs)} business, {len(cl_slugs)} Claude, {len(gem_slugs)} Gemini URLs"
        })

        return content

    def sync_python_build_components(self, content):
        """Synchronizes subdirectories in python/build_components.py."""
        subdirs_str = 'SUBDIRS = ["business-services", "personal-services", "claude-training", "google-training"]'
        if "SUBDIRS" in content:  # pragma: no cover
            content = re.sub(r'SUBDIRS\s*=\s*\[[\s\S]*?\]', subdirs_str, content)

        self.occurrences.append({
            "target": "python/build_components.py",
            "category": "source_code",
            "domain": "components",
            "field": "subdirs",
            "detail": "Synced active subdirectory folder paths"
        })
        return content

    def sync_python_delete_page(self, content):
        """Synchronizes subdirectories in python/delete_page.py."""
        subdirs_str = 'SUBDIRS = ["google-training", "claude-training", "business-services", "personal-services"]'
        if "SUBDIRS" in content:
            content = re.sub(r'SUBDIRS\s*=\s*\[[\s\S]*?\]', subdirs_str, content)

        self.occurrences.append({
            "target": "python/delete_page.py",
            "category": "source_code",
            "domain": "page_cleaner",
            "field": "subdirs",
            "detail": "Synced active subdirectory folder paths"
        })
        return content

    def sync_python_og_card(self, content):
        """Synchronizes config brand facts in python/generate_og_card.py."""
        author = self.kb.get_fact("config", "author", "Thomas Cherickal")
        brand = self.kb.get_fact("config", "brand", "The Digital Futurist")
        title = self.kb.get_fact("config", "title", "Generative AI Consultant")

        content = re.sub(r'<h1 class="title-name">[^<]+</h1>', f'<h1 class="title-name">{author}</h1>', content)
        content = re.sub(r'<h2 class="title-role">[^<]+</h2>', f'<h2 class="title-role">{title}</h2>', content)

        self.occurrences.append({
            "target": "python/generate_og_card.py",
            "category": "source_code",
            "domain": "config",
            "field": "og_card_author",
            "detail": f"Synced author={author}, title={title}, brand={brand}"
        })
        return content

    def sync_python_tests(self, content, file_path):
        """Synchronizes dynamic assertions in tests/test_*.py to match authoritative facts."""
        fname = os.path.basename(file_path)

        if fname == "test_sync_claude_and_gemini.py":
            # Ensure folder paths are claude-training and google-training
            content = re.sub(
                r'os\.path\.join\(REPO_DIR,\s*"claude",\s*slug\)',
                'os.path.join(REPO_DIR, "claude-training", slug)',
                content
            )
            content = re.sub(
                r'os\.path\.join\(REPO_DIR,\s*"google",\s*slug\)',
                'os.path.join(REPO_DIR, "google-training", slug)',
                content
            )
            self.occurrences.append({
                "target": f"tests/{fname}",
                "category": "test",
                "domain": "guides",
                "field": "guide_paths",
                "detail": "Verified claude-training/ and google-training/ target paths in test"
            })

        return content

    def sync_all(self):
        """Executes full synchronization across all discovered source code and test files."""
        self.occurrences = []
        source_files = self.discover_source_files()
        test_files = self.discover_test_files()
        all_targets = source_files + test_files

        modified_files = []
        file_results = []

        for fpath in all_targets:
            rel_path = os.path.relpath(fpath, self.repo_dir)
            with open(fpath, "r", encoding="utf-8") as f:
                old_content = f.read()

            new_content = old_content

            # Route to appropriate synchronization handler
            if rel_path == "pricing-calculator.js":
                new_content = self.sync_pricing_calculator(new_content)
            elif rel_path == "components.js":
                new_content = self.sync_components_js(new_content)
            elif rel_path == "pricing-tests.js":
                new_content = self.sync_pricing_tests_js(new_content)
            elif rel_path == os.path.join("python", "generate_sitemap.py"):
                new_content = self.sync_python_sitemap(new_content)
            elif rel_path == os.path.join("python", "build_components.py"):
                new_content = self.sync_python_build_components(new_content)
            elif rel_path == os.path.join("python", "delete_page.py"):
                new_content = self.sync_python_delete_page(new_content)
            elif rel_path == os.path.join("python", "generate_og_card.py"):
                new_content = self.sync_python_og_card(new_content)
            elif rel_path.startswith("tests") and rel_path.endswith(".py"):
                new_content = self.sync_python_tests(new_content, fpath)

            if new_content != old_content:
                # Unidirectional Guard: Ensure target is never inside data/
                abs_fp = os.path.abspath(fpath)
                abs_data = os.path.abspath(DATA_DIR)
                try:
                    if os.path.commonpath([abs_fp, abs_data]) == abs_data:  # pragma: no cover
                        raise PermissionError(
                            f"MUTATION BLOCKED: '{fpath}' is inside data directory. Code sync is strictly one-way."
                        )
                except ValueError:  # pragma: no cover
                    pass

                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(new_content)
                changed_lines = compute_changed_lines(old_content, new_content)
                modified_files.append(rel_path)
                file_results.append({
                    "file": rel_path,
                    "status": "UPDATED",
                    "changed_lines": changed_lines
                })
            else:
                file_results.append({
                    "file": rel_path,
                    "status": "IN SYNC",
                    "changed_lines": "-"
                })

        # Write output artifacts
        has_mods = (len(modified_files) > 0)
        self.write_code_references(force=has_mods)
        self.write_code_markdown_report(file_results, force=has_mods)

        return {
            "total_files": len(all_targets),
            "modified_count": len(modified_files),
            "modified_files": modified_files,
            "results": file_results,
            "occurrences_count": len(self.occurrences)
        }

    def write_code_references(self, force=False):
        """Writes nlp-fuzzy-expert/code_references.json tracking all synced code elements."""
        out_path = os.path.join(SCRIPT_DIR, "code_references.json")
        if not force and os.path.exists(out_path):
            existing = load_json(out_path)
            if existing.get("references") == self.occurrences:
                return
        data = {
            "generated_at": datetime.now().isoformat(),
            "total_occurrences": len(self.occurrences),
            "references": self.occurrences
        }
        save_json(out_path, data)

    def write_code_markdown_report(self, file_results, force=False):
        """Writes nlp-fuzzy-expert/code-sync.md summary report."""
        out_path = os.path.join(SCRIPT_DIR, "code-sync.md")
        lines = [
            "# Source Code and Test Suite Knowledge Synchronization Report",
            "",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            f"- **Target Files Evaluated**: {len(file_results)}",
            f"- **Files Updated**: {sum(1 for r in file_results if r['status'] == 'UPDATED')}",
            f"- **Tracked References**: {len(self.occurrences)}",
            "",
            "## File Status Details",
            "",
            "| File | Status | Changed Lines |",
            "| :--- | :---: | :--- |"
        ]

        for r in file_results:
            lines.append(f"| `{r['file']}` | **{r['status']}** | {r['changed_lines']} |")

        lines.extend([
            "",
            "## Synchronized Entities and Facts",
            "",
            "| Target | Category | Domain | Field | Detail |",
            "| :--- | :--- | :--- | :--- | :--- |"
        ])

        for occ in self.occurrences:
            lines.append(
                f"| `{occ['target']}` | {occ['category']} | {occ['domain']} | `{occ['field']}` | {occ['detail']} |"
            )

        new_content = "\n".join(lines) + "\n"
        if not force and os.path.exists(out_path):
            with open(out_path, "r", encoding="utf-8") as f:
                old_content = f.read()
            old_body = old_content.split("## Summary", 1)[-1] if "## Summary" in old_content else ""
            new_body = new_content.split("## Summary", 1)[-1]
            if old_body and old_body == new_body:
                return

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(new_content)


if __name__ == "__main__":  # pragma: no cover
    syncer = CodeTestSynchronizer()
    res = syncer.sync_all()
    print(f"Code and Test Synchronization Complete: {res['modified_count']} of {res['total_files']} files updated.")
