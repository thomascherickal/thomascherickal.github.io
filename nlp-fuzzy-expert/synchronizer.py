#!/usr/bin/env python3
"""
synchronizer.py - Master Synchronizer for NLP-Fuzzy-Expert System.

Orchestrates KnowledgeBase, NLP Engine, Fuzzy Matcher, and Inference Engine
to scan repository documents, execute rule-driven transformations, report
line-level diffs, and produce per-domain references, markdown reports, and summary tables.

Strict Repo Compliance:
- Zero emojis.
- Zero em-dashes (spaced hyphens ' - ' or plain hyphens '-' used).
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
            return text[:max_len - 3] + "..."
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
        for r in build_rules():
            self.engine.register_rule(r)

    def discover_target_files(self):
        """Discovers all repository HTML and JS files subject to synchronization."""
        target_files = []

        # Root files
        for fname in os.listdir(REPO_DIR):
            if fname.endswith(".html") or fname == "pricing-calculator.js":
                target_files.append(os.path.join(REPO_DIR, fname))

        # Subdirectories
        for sub in ["business-services", "personal-services", "claude", "google"]:
            sdir = os.path.join(REPO_DIR, sub)
            if os.path.exists(sdir):
                for fname in os.listdir(sdir):
                    if fname.endswith(".html"):
                        target_files.append(os.path.join(sdir, fname))

        return sorted(target_files)

    def sync_all(self):
        """Runs the expert inference engine across all target files with live terminal reporting."""
        print("=" * 80)
        print("           NLP-FUZZY-EXPERT KNOWLEDGE SYNCHRONIZATION SYSTEM")
        print("=" * 80)

        # Step 1: Knowledge Base Facts
        print("[Step 1/6] Loading authoritative facts from data/*.json into KnowledgeBase...")
        self.kb.reload()
        raw_counts = {k: len(v) if isinstance(v, (list, dict)) else 1 for k, v in self.kb.raw_data.items()}
        print(f"  -> Authoritative sources loaded: {len(self.kb.raw_data)} data domains")
        for domain, count in sorted(raw_counts.items()):
            print(f"     - data/{domain}.json: {count} top-level entries")
        print(f"  -> Total atomic propositions registered in KnowledgeBase: {len(self.kb.facts)}")

        # Step 2: Rules Registration
        print(f"\n[Step 2/6] Registering production rules in salience priority order...")
        print(f"  -> Active production rules registered: {len(self.engine.rules)}")
        for r in self.engine.rules[:5]:
            print(f"     - Salience {r.salience:>3}: {r.name} (domain: {r.domain})")
        if len(self.engine.rules) > 5:
            print(f"     ... and {len(self.engine.rules) - 5} additional production rules")

        # Step 3: Discover Targets
        print(f"\n[Step 3/6] Discovering target documents across repository...")
        targets = self.discover_target_files()
        print(f"  -> Discovered {len(targets)} target documents (.html, .js) for synchronization.")

        # Step 4: Execute Inference & Diffs
        print(f"\n[Step 4/6] Executing forward-chaining inference & computing line diffs...")
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
                print(f"  -> [MODIFIED] {rel}")
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

        # Step 5: Reference Dictionaries
        print(f"\n[Step 5/6] Generating domain references and master dictionary...")
        self.generate_reference_dictionaries()

        # Step 6: Markdown Reports
        print(f"\n[Step 6/6] Generating domain markdown synchronization reports...")
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
                        ", ".join(rep.get("fields_synchronized", [])) or "-"
                    ])
        else:
            # Display representative sample of in-sync files
            sample_files = ["pricing.html", "index.html", "pricing-calculator.js",
                            "all-business-services.html", "all-personal-services.html",
                            "faqs.html", "business-ai-training.html", "personal-ai-training.html"]
            for rep in file_reports:
                if rep.get("file") in sample_files:
                    table_rows.append([
                        rep.get("file", ""),
                        "IN SYNC",
                        "-",
                        ", ".join(rep.get("fired_rules", [])) or "-",
                        ", ".join(rep.get("fields_synchronized", [])) or "-"
                    ])

        print(render_ascii_table(table_headers, table_rows))
        print(f"Summary: {len(targets)} files evaluated | {modified_count} updated | "
              f"{len(targets) - modified_count} in sync | {len(self.engine.tracked_occurrences)} references tracked")
        print("=" * 80)

        # Step 7: Post-synchronization unit test verification
        is_testing = (
            os.environ.get("ANTIGRAVITY_IN_TEST") == "1"
            or "unittest" in sys.modules
        )

        if modified_count > 0 and not is_testing:
            print("\n[Step 7] Changes detected in target documents. Running post-sync unit test verification...")
            test_cmd = [sys.executable, "-m", "unittest", "discover", "-s", os.path.join(REPO_DIR, "tests"), "-v"]
            test_run = subprocess.run(test_cmd, cwd=REPO_DIR, capture_output=True, text=True)
            if test_run.returncode == 0:
                print("[NLP-Fuzzy-Expert] Post-sync verification: ALL UNIT TESTS PASSED.")
            else:
                print("[NLP-Fuzzy-Expert] Post-sync verification: UNIT TESTS FAILED!")
                print(test_run.stdout)
                print(test_run.stderr)
                raise RuntimeError("Post-synchronization unit test verification failed.")

        return modified_count

    def sync_file(self, target_name):
        """Synchronizes when a specific file is targeted or triggered."""
        self.kb.reload()
        if not target_name.endswith(".json") and not target_name.endswith(".html") and not target_name.endswith(".js"):
            return False
        self.sync_all()
        return True

    def generate_reference_dictionaries(self):
        """Generates domain-specific reference JSONs and the master aggregated dictionary."""
        timestamp = datetime.now().isoformat()
        total_tracked = len(self.engine.tracked_occurrences)

        # Domain map
        domain_files = {
            "pricing": "pricing_references.json",
            "business_services": "business_services_references.json",
            "personal_services": "personal_services_references.json",
            "tiers": "tiers_references.json",
            "claude": "claude_references.json",
            "gemini": "gemini_references.json",
            "publications": "publications_references.json",
            "config": "config_references.json",
            "faqs": "faqs_references.json",
        }

        domain_refs = {k: [] for k in domain_files}

        for ref in self.engine.tracked_occurrences:
            field = ref.get("field_key", "")
            if "pricing" in field or "flagship" in field or "seats" in field or "calculator" in field:
                domain_refs["pricing"].append(ref)
            elif "personal_service" in field:
                domain_refs["personal_services"].append(ref)
            elif "business_service" in field:
                domain_refs["business_services"].append(ref)
            elif "tier" in field:
                domain_refs["tiers"].append(ref)
            elif "claude" in field:
                domain_refs["claude"].append(ref)
            elif "gemini" in field:
                domain_refs["gemini"].append(ref)
            elif "publication" in field:
                domain_refs["publications"].append(ref)
            elif "email" in field or "config" in field:
                domain_refs["config"].append(ref)
            elif "faq" in field:
                domain_refs["faqs"].append(ref)
            else:
                domain_refs["pricing"].append(ref)

        self.domain_refs = domain_refs
        for domain, fname in domain_files.items():
            refs = domain_refs[domain]
            out_dict = {
                "metadata": {
                    "domain": domain,
                    "engine": "nlp-fuzzy-expert",
                    "total_references_tracked": len(refs),
                    "last_synchronized": timestamp
                },
                "all_references": refs
            }
            out_path = os.path.join(SCRIPT_DIR, fname)
            save_json(out_path, out_dict)
            print(f"  -> Wrote {fname} ({len(refs)} references)")

        # Master Dictionary
        master_dict = {
            "metadata": {
                "engine": "nlp-fuzzy-expert",
                "version": "1.0.0",
                "architecture": "Expert System (Forward Chaining) + Fuzzy Logic + NLP",
                "total_references_tracked": total_tracked,
                "last_synchronized": timestamp
            },
            "by_domain": {k: len(v) for k, v in domain_refs.items()},
            "all_references": self.engine.tracked_occurrences
        }
        master_path = os.path.join(SCRIPT_DIR, "references_dictionary.json")
        save_json(master_path, master_dict)
        print(f"  -> Master Dictionary generated: references_dictionary.json ({total_tracked} total references)")

    def generate_markdown_reports(self):
        """Generates domain-specific markdown reports under nlp-fuzzy-expert/."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        reports = [
            ("pricing.md", "pricing", "Pricing & Engagement Models"),
            ("business-services.md", "business_services", "Enterprise & Business AI Services"),
            ("personal-services.md", "personal_services", "Personal & Professional AI Services"),
            ("tiers.md", "tiers", "Country Parity Pricing Tiers"),
            ("claude.md", "claude", "Anthropic Claude Engineering Guides"),
            ("gemini.md", "gemini", "Google Gemini Architecture Guides"),
            ("publications.md", "publications", "Technical Publications"),
            ("config.md", "config", "Site Configuration & Brand Identity"),
            ("faqs.md", "faqs", "Frequently Asked Questions & Policies"),
        ]

        for md_name, domain, title in reports:
            md_path = os.path.join(SCRIPT_DIR, md_name)
            filtered = getattr(self, "domain_refs", {}).get(domain, [])

            lines = [
                f"# NLP-Fuzzy-Expert Synchronization Report - {title}",
                "",
                f"Generated by `nlp-fuzzy-expert/synchronizer.py` via Expert System forward chaining.",
                "",
                "## Summary Metadata",
                "",
                f"- **Domain**: `{domain}`",
                f"- **Tracked Records / Fields**: {len(filtered)}",
                "- **Inference Status**: Rule Fired & Synchronized",
                "- **Highlight Specification**: Visible in IDE (`rgba(251, 191, 36, 0.25)`), invisible on live production sites",
                f"- **Last Synchronized**: {timestamp}",
                "",
                "## Tracked Occurrences Table",
                "",
                "| Field Key | Tracked Value | File Location | Section | Line |",
                "| :--- | :--- | :--- | :--- | :--- |"
            ]

            if not filtered:
                lines.append(f"| {domain}_rules | Authoritative | Target documents | Global Inference | Centralized |")
            else:
                for ref in filtered:
                    fk = ref.get("field_key", "")
                    val = ref.get("new_value", "")
                    fl = ref.get("file", "")
                    sec = ref.get("section", "")
                    ln = str(ref.get("line_number", 1))
                    lines.append(f"| {fk} | {val} | `{fl}` | {sec} | Line {ln} |")

            lines.append("")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            print(f"  -> Wrote {md_name}")

def main():
    sync = ExpertSynchronizer()
    sync.sync_all()

if __name__ == "__main__":
    main()
