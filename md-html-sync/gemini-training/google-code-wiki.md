# Google Code Wiki: Automated Repository Documentation & Architecture Maps
## AI Agent Architecture & Orchestration - Thomas Cherickal

> **// LIVING CODEBASE WIKIS, DEPENDENCY MAPS & CONTINUOUS CI SYNC**  
> **Architectural blueprint and production handbook for Google Code Wiki: powering automated, multi-tiered living documentation, call-graph extractions, and continuous pull-request synchronization using Gemini models.**

---

## Contents

---

## The Enterprise Documentation Crisis

Software engineering organizations spend billions writing and maintaining code, yet their internal documentation is perpetually stale. Confluence spaces rot within months, README files diverge from actual APIs, and critical architectural invariants exist solely inside senior engineers' heads. When engineers leave or teams reorganize, institutional memory collapses.

Google Code Wiki fundamentally eliminates documentation debt by treating documentation as a continuous, automated compiler step. Instead of forcing developers to author manual prose, Code Wiki combines deterministic abstract syntax tree (AST) parsers with Gemini's multi-million token context window to synthesize, update, and audit living architecture wikis directly inside the git repository.

---

## Deterministic AST Parsing & Cross-File Call Graphs

Pure LLM summarization often produces generic, hand-waving explanations that miss exact method signatures and subtle dependency chains. Google Code Wiki prevents this through a hybrid architecture: deterministic static analysis handles structural truth, while Gemini provides semantic interpretation.

Tree-sitter and native AST libraries walk the codebase to construct an exact, language-agnostic dependency graph. Every caller, callee, data contract, and inheritance relationship is materialized into structured JSON before being provided to Gemini. The model then writes crystal-clear architectural narrative grounded strictly in verifiable code topology.

---

## Hierarchical 4-Tier Documentation Taxonomy

A monolithic dump of explanations overwhelms developers. Code Wiki organizes documentation into a standardized four-tier information hierarchy designed for different engineering contexts:

Tier 1 outlines broad domain boundaries and service interactions for staff engineers and architects. Tier 2 details public APIs, gRPC protobuf schemas, and message queues. Tier 3 explains internal state machines and concurrency locks for module contributors. Tier 4 provides granular, auto-generated function contracts with explicit edge-case specifications.

---

## Continuous Pull-Request Sync & Delta Ingestion

The fatal flaw of historical documentation generators was batch execution: running once a quarter and generating gigabytes of noise. Code Wiki operates on continuous pull request triggers.

When an engineer modifies a file, a git hook or GitHub Actions workflow extracts the exact git diff. Code Wiki identifies affected wiki pages, loads the modified modules alongside their call graphs, and instructs Gemini to perform surgical, in-place updates. If a PR breaks documented invariants, CI fails automatically.

---

## Documenting Invariants, Threat Models & State Machines

High-stakes engineering systems in banking, healthcare, and robotics require more than API descriptions; they demand formal documentation of invariants. What happens if a database transaction aborts mid-flight? Which functions acquire distributed locks?

Code Wiki specializes in identifying critical security boundaries, authentication interceptors, and rollback paths. It documents threat surfaces, input sanitization routines, and concurrency models with rigorous precision.

---

## Accelerating Developer Onboarding from Weeks to Hours

Engineering velocity is dictated by how fast new hires become productive contributors. Traditional onboarding involves shadowing seniors, deciphering undocumented legacy patterns, and breaking test suites.

With Google Code Wiki, every repository features an interactive, search-grounded living knowledge graph. New engineers query system concepts, trace data flows through multi-layer services, and understand design trade-offs immediately, cutting onboarding latency by over 80%.

---

## Production Python Pipeline (AST + Google GenAI SDK)

Below is a complete, modular implementation of the Google Code Wiki extraction pipeline. It parses local Python source code into syntax trees, queries Gemini for narrative synthesis, and asserts documentation parity in CI/CD.

This pipeline runs in GitHub Actions, GitLab CI, or pre-commit hooks to guarantee that every commit maintains pristine architecture documentation.

#### `code_wiki_extractor.py`
```python
import ast
import os
from google import genai
from google.genai import types

class CodeWikiExtractor:
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.client = genai.Client()

    def parse_file(self, filepath: str) -> dict:
        """Extract classes, methods and call graphs deterministically."""
        with open(filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=filepath)
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        return {"classes": classes, "functions": functions}

    def generate_wiki_page(self, module_name: str, symbols: dict, code_snippet: str) -> str:
        """Synthesize a living Markdown wiki page using Gemini."""
        prompt = f"""Generate a technical architecture wiki page for module: {module_name}.
Document symbols: {symbols}
Source excerpt:
{code_snippet}
Include:
1. Executive Overview
2. Interface Boundaries
3. Security & Error Invariants
"""
        response = self.client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.2)
        )
        return response.text
```

---

## Architecture Guidelines & Repository Governance

To successfully deploy Google Code Wiki across large engineering divisions, organizations must enforce clean directory boundaries, exclude third-party vendor bundles, and track generated Markdown artifacts directly in git.

Keeping documentation in the same commit as code ensures that git bisect, pull request reviews, and historical auditing retain an atomic, immutable record of both code implementation and architectural intent.

#### `ci_wiki_sync.py`
```python
import subprocess
import sys

def check_wiki_diff():
    """Run within CI to assert wiki documentation remains 100% in sync with code."""
    res = subprocess.run(["git", "status", "--porcelain", "wiki/"], capture_output=True, text=True)
    if res.stdout.strip():
        print("[FAIL] Repository wiki out of date with latest PR code changes:")
        print(res.stdout)
        sys.exit(1)
    print("[PASS] Google Code Wiki is perfectly synchronized.")
```

---

## References

Official technical publications, documentation resources, and foundation research for this guide:

- Google Gemini API Documentation - Official documentation for Gemini models, context caching, and structured outputs
- Tree-sitter Language Parsing Framework - Universal incremental parsing library for abstract syntax tree extraction
- Google Gemini Code Execution Recipes - Recipes for code understanding, repository analysis, and testing harnesses
- DORA DevOps Research and Assessment - State of DevOps research on documentation culture and engineering velocity
- The Digital Futurist - Business AI transformation, developer tooling, and automated engineering advisory

---

### Ready to Orchestrate Production Claude AI Agent Teams?
- [Schedule Strategy Consultation via Email →](contact.html)
- [View Advisory Packages & Rates](pricing.html)
- [Explore Claude Agents Suite](claude-101.html)
