# Claude Skills & Custom Tools: Extensibility Architecture
## AI Agent Architecture & Orchestration — Thomas Cherickal

> **// CLAUDE AGENTS · EXTENSIBILITY**  
> **Learn how Claude Skills provide on-demand modular capabilities to agents—loading specialized domain instructions, custom scripts, and validation schemas only when needed to optimize token efficiency.**

---

## 1. The Philosophy of Modular Skills

Declaring dozens of heavy tool schemas in every system prompt quickly bloats the context window, causing instruction degradation and unnecessary token expense. Claude Skills solve this through progressive disclosure: skills remain idle metadata until the agent triggers them based on intent:

---

## 2. The SKILL.md Specification Standard

Each skill lives in its own directory containing a SKILL.md definition file with structured YAML frontmatter and operational documentation:

#### `skills/database-auditor/SKILL.md`
```python
---
name: database-auditor
description: Inspects PostgreSQL database schema, identifies unindexed foreign keys, and analyzes slow queries.
version: 1.2.0
author: Thomas Cherickal
entrypoint: scripts/audit.py
dependencies:
  - psycopg2-binary
  - tabulate
permissions:
  network: read-only
  filesystem: local-read
---

# Database Auditor Skill

## Purpose
Enables Claude agents to analyze database performance bottlenecks and propose indexing strategies without modifying production schemas.

## Invocation Protocol
When the user mentions "database performance", "slow queries", or "missing indexes":
1. Execute `scripts/audit.py --inspect-slow-queries`.
2. Inspect the JSON telemetry returned.
3. Synthesize an RFC explaining the missing B-tree indexes and anticipated query speedup.
```

---

## 3. Tutorial: Building a Production Skill Script

Here is the corresponding Python entrypoint for the database auditing skill, designed with structured JSON output for deterministic agent consumption:

#### `skills/database-auditor/scripts/audit.py`
```python
import sys
import json
import argparse

def analyze_indexes(db_url: str):
    """Mock analysis of database indexes for demonstration"""
    findings = [
        {
            "table": "orders",
            "column": "customer_id",
            "issue": "Foreign key missing index",
            "estimated_cost_reduction": "68%",
            "suggested_sql": "CREATE INDEX idx_orders_customer_id ON orders(customer_id);"
        },
        {
            "table": "audit_logs",
            "column": "created_at",
            "issue": "Sequential scan on timestamp range filter",
            "estimated_cost_reduction": "84%",
            "suggested_sql": "CREATE INDEX idx_audit_logs_created_at ON audit_logs(created_at DESC);"
        }
    ]
    return findings

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", action="store_true", help="Inspect foreign keys")
    args = parser.parse_args()

    results = analyze_indexes("mock-url")
    # Output pure JSON so Claude agent can parse without regex
    print(json.dumps({"status": "success", "count": len(results), "findings": results}, indent=2))
```

---

### Ready to Orchestrate Production Claude AI Agent Teams?
- [Schedule Strategy Consultation via Email →](collaboration.html)
- [View Advisory Packages & Rates](pricing.html)
- [Explore Claude Agents Suite](agents-claude-101.html)
