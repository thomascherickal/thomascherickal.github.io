#!/usr/bin/env python3
"""
synchronizer.py - Master Synchronizer for NLP-Fuzzy-Expert System.

Orchestrates KnowledgeBase, NLP Engine, Fuzzy Matcher, and Inference Engine
to scan repository documents, execute rule-driven transformations, report
line-level diffs, and produce per-domain references, markdown reports, and summary tables.

Includes modular auto-healing integration that triggers the Google Antigravity CLI ('agy')
if post-synchronization tests fail, iteratively healing discrepancies until the test suite
is green, then committing the verified repository state.

Strict Repo Compliance:
- Zero emojis.
- Zero em-dashes (spaced hyphens ' - ' or plain hyphens '-' used).
- Strict data/ directory immutability (no file in any other folder may modify or undo data/).
"""

import os
import sys
import subprocess
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(REPO_DIR, "data")

sys.path.insert(0, SCRIPT_DIR)
from expert_system import KnowledgeBase, InferenceEngine, save_json, load_json
from rules import build_rules
from code_test_syncer import CodeTestSynchronizer
from auto_healer import (
    handle_test_verification_and_commit,
    run_unit_tests,
    compute_data_fingerprint,
    verify_data_untouched,
)
from report_generator import (
    DOMAIN_FILES,
    DOMAIN_DATA_SOURCE,
    classify_rule_field,
    write_reference_files,
    generate_markdown_reports,
)


def render_ascii_table(headers, rows, max_widths=None):
    """Renders a pure ASCII formatted table.
    Strictly zero emojis and zero em-dashes.
    """
    if not max_widths:
        max_widths = [45, 10, 22, 24, 24]

    num_cols = len(headers)
    widths = [len(h) for h in headers]
    for row in rows:
        for i in range(min(num_cols, len(row))):
            widths[i] = max(widths[i], len(str(row[i])))

    for i in range(num_cols):
        widths[i] = min(widths[i], max_widths[i])

    def truncate(text, max_len):
        text = str(text).replace("\n", " ").strip()
        if len(text) <= max_len:
            return text
        if max_len > 3:
            return text[: max_len - 3] + "..."
        return text[:max_len]

    def format_row(cols):
        formatted = []
        for i in range(num_cols):
            val = truncate(cols[i] if i < len(cols) else "", widths[i])
            formatted.append(f" {val.ljust(widths[i])} ")
        return "|" + "|".join(formatted) + "|"

    sep = "+" + "+".join(["-" * (w + 2) for w in widths]) + "+"

    lines = [sep, format_row(headers), sep]
    for row in rows:
        lines.append(format_row(row))
    lines.append(sep)
    return "\n".join(lines)


class ExpertSynchronizer:
    """Master orchestrator for NLP-Fuzzy-Expert synchronization."""

    def __init__(self, data_dir=None):
        self.data_dir = data_dir or DATA_DIR
        self.kb = KnowledgeBase(data_dir=self.data_dir)
        self.engine = InferenceEngine(self.kb)
        self.domain_refs = {}
        for r in build_rules():
            self.engine.register_rule(r)

    def discover_target_files(self):
        """Discovers all repository HTML and JS files subject to synchronization."""
        if hasattr(self, "_cached_target_files") and self._cached_target_files:
            return list(self._cached_target_files)

        target_files = []

        # Root files
        for fname in os.listdir(REPO_DIR):
            if fname.endswith(".html") or fname == "pricing-calculator.js":
                target_files.append(os.path.join(REPO_DIR, fname))

        # Subdirectories
        for sub in ["business-services", "personal-services", "claude-training", "google-training"]:
            sdir = os.path.join(REPO_DIR, sub)
            if os.path.exists(sdir):
                for fname in os.listdir(sdir):
                    if fname.endswith(".html"):
                        target_files.append(os.path.join(sdir, fname))

        self._cached_target_files = sorted(target_files)
        return list(self._cached_target_files)

    def sync_all(self, auto_heal=True, verify_tests=True):
        """Runs the expert inference engine across all target files with live terminal reporting."""
        initial_data_fp = compute_data_fingerprint(self.data_dir)

        print("=" * 80)
        print("           NLP-FUZZY-EXPERT KNOWLEDGE SYNCHRONIZATION SYSTEM")
        print("=" * 80)

        # Step 1: Knowledge Base Facts
        print("[Step 1/6] Loading authoritative facts from data/*.json into KnowledgeBase...")
        print("  [UPDATE SCOPE] Domain: data/*.json -> Target: KnowledgeBase atomic facts cache")
        self.kb.reload()
        raw_counts = {k: len(v) if isinstance(v, (list, dict)) else 1 for k, v in self.kb.raw_data.items()}
        print(f"  -> Authoritative sources loaded: {len(self.kb.raw_data)} data domains")
        for domain, count in sorted(raw_counts.items()):
            print(f"     - data/{domain}.json: {count} top-level entries")
        print(f"  -> Total atomic propositions registered in KnowledgeBase: {len(self.kb.facts)}")

        # Step 2: Rules Registration
        print(f"\n[Step 2/6] Registering production rules in salience priority order...")
        print("  [UPDATE SCOPE] Domain: rules.py -> Target: InferenceEngine priority rule queue")
        print(f"  -> Active production rules registered: {len(self.engine.rules)}")
        for r in self.engine.rules[:5]:
            print(f"     - Salience {r.salience:>3}: {r.name} (domain: {r.domain})")
        if len(self.engine.rules) > 5:
            print(f"     ... and {len(self.engine.rules) - 5} additional production rules")

        # Step 3: Discover Targets
        print(f"\n[Step 3/6] Discovering target documents across repository...")
        print("  [UPDATE SCOPE] Domain: Repository scan -> Target: Target HTML and JS documents list")
        targets = self.discover_target_files()
        print(f"  -> Discovered {len(targets)} target documents (.html, .js) for synchronization.")

        # Step 4: Execute Inference & Diffs
        print(f"\n[Step 4/6] Executing forward-chaining inference & computing line diffs...")
        print("  [UPDATE SCOPE] Fields: Authoritative pricing, tiers, curricula & configs -> Target: HTML/JS repository pages")
        self.engine.tracked_occurrences.clear()
        modified_count = 0
        header_footer_modified = False
        file_reports = []

        for fpath in targets:
            rel = os.path.relpath(fpath, REPO_DIR)
            report = self.engine.process_file(fpath)
            file_reports.append(report)

            if report.get("modified"):
                modified_count += 1
                changed_ln = report.get("changed_lines", "None")
                fired = ", ".join(report.get("fired_rules", [])) or "None"
                fields = ", ".join(report.get("fields_synchronized", [])) or "None"
                print(f"  -> [MODIFIED] {rel} | Field: {fields} | Location: Line(s) {changed_ln}")
                print(f"     Changed Lines: {changed_ln}")
                print(f"     Rules Fired:   {fired}")
                print(f"     Fields Synced: {fields}")

                if rel in ["header.html", "footer.html"]:
                    header_footer_modified = True

        if modified_count == 0:
            print(f"  -> All {len(targets)} target documents evaluated. Zero changes required (fully in sync).")
        else:
            print(f"  -> Total documents modified: {modified_count} of {len(targets)}")

        # Component inlining rule: run build_components if header/footer was modified
        if header_footer_modified:
            print("\n[NLP-Fuzzy-Expert] header.html or footer.html modified. Executing python/build_components.py...")
            subprocess.run([sys.executable, os.path.join(REPO_DIR, "python", "build_components.py")], check=True)

        # Step 4b: Synchronize Source Code & Test Suites
        print(f"\n[Step 4b] Synchronizing source code and unit test suites...")
        print("  [UPDATE SCOPE] Fields: Authoritative facts -> Target: python/*.py, root JS, and tests/*")
        code_syncer = CodeTestSynchronizer(kb=self.kb, repo_dir=REPO_DIR)
        code_res = code_syncer.sync_all()
        code_modified = code_res.get("modified_count", 0)
        modified_count += code_modified

        if code_modified > 0:
            for r in code_res.get("results", []):
                if r.get("status") == "UPDATED":
                    print(f"  -> [MODIFIED] {r['file']} | Status: UPDATED | Changed Lines: {r['changed_lines']}")
            print(f"  -> Total source/test files updated: {code_modified} of {code_res['total_files']}")
        else:
            print(f"  -> All {code_res['total_files']} source/test files evaluated. Zero changes required (fully in sync).")

        # Step 5: Reference Dictionaries
        print(f"\n[Step 5/6] Generating domain references and master dictionary...")
        print("  [UPDATE SCOPE] Fields: Tracked occurrences & references -> Target: nlp-fuzzy-expert/*_references.json")
        self.generate_reference_dictionaries()

        # Step 6: Markdown Reports
        print(f"\n[Step 6/6] Generating domain markdown synchronization reports...")
        print("  [UPDATE SCOPE] Fields: Domain summaries & occurrence tables -> Target: nlp-fuzzy-expert/*.md reports")
        self.generate_markdown_reports()

        # Terminal Summary Table
        print("\n" + "=" * 80)
        print("                     SYNCHRONIZATION SUMMARY TABLE")
        print("=" * 80)
        table_headers = ["Target File", "Status", "Changed Lines", "Rules Fired", "Fields Synced"]
        table_rows = []

        if modified_count > 0:
            for rep in file_reports:
                if rep.get("modified"):
                    table_rows.append([
                        rep.get("file", ""),
                        "UPDATED",
                        rep.get("changed_lines", "None"),
                        ", ".join(rep.get("fired_rules", [])) or "-",
                        ", ".join(rep.get("fields_synchronized", [])) or "-",
                    ])
            if code_modified > 0:
                for r in code_res.get("results", []):
                    if r.get("status") == "UPDATED":
                        table_rows.append([
                            r.get("file", ""),
                            "UPDATED",
                            r.get("changed_lines", "None"),
                            "CodeTestSynchronizer",
                            "source_and_tests"
                        ])
        else:
            sample_files = [
                "pricing.html", "index.html", "pricing-calculator.js",
                "all-business-services.html", "all-personal-services.html",
                "faqs.html", "business-ai-training.html", "personal-ai-training.html",
            ]
            for rep in file_reports:
                if rep.get("file") in sample_files:
                    table_rows.append([
                        rep.get("file", ""),
                        "IN SYNC",
                        "-",
                        ", ".join(rep.get("fired_rules", [])) or "-",
                        ", ".join(rep.get("fields_synchronized", [])) or "-",
                    ])

        print(render_ascii_table(table_headers, table_rows))
        print(
            f"Summary: {len(targets)} files evaluated | {modified_count} updated | "
            f"{len(targets) - modified_count} in sync | {len(self.engine.tracked_occurrences)} references tracked"
        )
        print("=" * 80)

        # Enforce that data/ was untouched during synchronization pass
        verify_data_untouched(initial_data_fp, self.data_dir)

        # Step 7: Post-synchronization unit test verification & auto-healing
        is_testing = (
            os.environ.get("ANTIGRAVITY_IN_TEST") == "1"
            or "unittest" in sys.modules
        )

        if modified_count > 0 and not is_testing and verify_tests:
            print("\n[Step 7] Changes detected in target documents. Running post-sync verification & commit handler...")
            if auto_heal:
                handle_test_verification_and_commit(max_iterations=5)
            else:
                passed, stdout, stderr, fail_count = run_unit_tests()
                if not passed:
                    raise RuntimeError(f"Post-synchronization unit test verification failed ({fail_count} failure(s)).")

        return modified_count

    def sync_file(self, target_name):
        """Synchronizes when a specific file is targeted or triggered."""
        self.kb.reload()
        if not target_name.endswith(".json") and not target_name.endswith(".html") and not target_name.endswith(".js"):
            return False
        self.sync_all()
        return True

    def scan_all_locations(self):
        """Runs the read-only exhaustive scanner over every target document."""
        from reference_scanner import scan_repository

        scanned = scan_repository(self.discover_target_files(), self.data_dir)
        print(
            f"  -> Exhaustive scan located {sum(len(v) for v in scanned.values())} term occurrences "
            f"across {len(self.discover_target_files())} documents"
        )
        return scanned

    def generate_reference_dictionaries(self):
        """Generates domain-specific reference JSONs and the master aggregated dictionary."""
        domain_refs = {k: [] for k in DOMAIN_FILES}
        for ref in self.engine.tracked_occurrences:
            domain = classify_rule_field(ref.get("field_key", ""))
            rec = dict(ref)
            rec["match_type"] = "rule"
            rec["data_source"] = DOMAIN_DATA_SOURCE[domain]
            domain_refs[domain].append(rec)
        for domain, recs in self.scan_all_locations().items():
            domain_refs[domain].extend(recs)

        code_refs_path = os.path.join(SCRIPT_DIR, "code_references.json")
        if os.path.exists(code_refs_path):
            code_data = load_json(code_refs_path)
            for ref in code_data.get("references", []):
                dom = ref.get("domain", "pricing")
                if dom in domain_refs:
                    domain_refs[dom].append({
                        "file": ref.get("target"),
                        "term": ref.get("field"),
                        "match_type": "code_sync",
                        "line_number": 1,
                        "context": ref.get("detail"),
                        "data_source": DOMAIN_DATA_SOURCE.get(dom, "data/pricing.json")
                    })

        self.domain_refs = domain_refs
        write_reference_files(domain_refs)

    def refresh_references_only(self):
        """Scan-only mode: re-scans all pages and rewrites reference JSONs without modifying any page."""
        domain_refs = {k: [] for k in DOMAIN_FILES}
        for domain, fname in DOMAIN_FILES.items():
            existing = load_json(os.path.join(SCRIPT_DIR, fname)).get("all_references", [])
            domain_refs[domain] = [r for r in existing if r.get("match_type", "rule") == "rule"]
        for domain, recs in self.scan_all_locations().items():
            domain_refs[domain].extend(recs)

        code_refs_path = os.path.join(SCRIPT_DIR, "code_references.json")
        if os.path.exists(code_refs_path):
            code_data = load_json(code_refs_path)
            for ref in code_data.get("references", []):
                dom = ref.get("domain", "pricing")
                if dom in domain_refs:
                    domain_refs[dom].append({
                        "file": ref.get("target"),
                        "term": ref.get("field"),
                        "match_type": "code_sync",
                        "line_number": 1,
                        "context": ref.get("detail"),
                        "data_source": DOMAIN_DATA_SOURCE.get(dom, "data/pricing.json")
                    })

        self.domain_refs = domain_refs
        write_reference_files(domain_refs)
        self.generate_markdown_reports()

    def generate_markdown_reports(self):
        """Generates domain-specific markdown reports under nlp-fuzzy-expert/."""
        generate_markdown_reports(self.domain_refs)


def main():
    sync = ExpertSynchronizer()
    sync.sync_all()


if __name__ == "__main__":
    main()
