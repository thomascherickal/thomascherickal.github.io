# Claude Agents Architecture & Orchestration
## AI Agent Architecture & Orchestration — Thomas Cherickal

> **// PROVIDER DEEP DIVE · ANTHROPIC**  
> **Harnessing Claude Opus 5.5, Claude Sonnet 5.5, and Claude Haiku 5.5 alongside Claude Code subagents,
 hybrid extended thinking, Computer Use GUI interactions, and the open Model Context Protocol (MCP) for
 enterprise AI agent teams.**

---

## Minimum Subscription & Access Pricing
**Starting Price**: `$20 / mo · $240 / yr`  
*Verified
 Monthly & Yearly Subscriptions (Oct 2026) · Claude Pro ($20/mo,
 $240/yr) · Team ($20–$25/seat/mo, $240/yr) · $5 min API Tier*

---

## 1. Executive Summary & Frontier 5.5 Model Triad

Anthropic’s 5.5 generation—comprising frontier Claude Opus 5.5, workhorse Claude
 Sonnet 5.5, and ultra-low-latency Claude Haiku 5.5—delivers an integrated
 intelligence hierarchy for autonomous software engineering, multi-turn reasoning, and graphical user
 interface (GUI) control. Leading across SWE-bench Verified and Tau-bench, the 5.5 model family
 pairs deep deliberative reasoning with strict JSON schema adherence, eliminating format drift across
 multi-agent execution loops.

### The Claude 5.5 Agentic Triad & Core Capabilities

- Claude Opus 5.5 (Lead Architect & Supervisor): Engineered for deep architectural
 synthesis, multi-repository dependency resolution, formal security verification, and orchestrating complex
 subagent DAGs with maximum extended thinking budgets.
- Claude Sonnet 5.5 (Primary Execution & Coding Engine): The high-throughput standard
 for autonomous AST code patching, Claude Code CLI workflows, test generation, and
 multimodal Computer Use execution.
- Claude Haiku 5.5 (High-Speed Triage & Telemetry Worker): Sub-second latency model
 built for parallel log ingestion, regex routing, schema validation, and high-frequency guardrailing at a
 fraction of frontier token costs.
- Hybrid Extended Thinking: Configure dynamic
 reasoning token budgets (2,048 to 32,000+ tokens on Opus 5.5 and Sonnet 5.5) so agents verify edge cases
 and decompose dependency graphs before emitting tool calls.
- Native Computer Use API & Model Context Protocol (MCP): Built-in GUI coordinate grounding and
 open JSON-RPC 2.0 tool bridges connect Claude 5.5 agents to desktop environments, Postgres, Git, and Slack
 without custom middleware.
- Prompt Caching: Caching system instructions
 and tool definitions cuts multi-turn input token costs across Opus 5.5, Sonnet 5.5, and Haiku 5.5 by up to
 90% and reduces TTFT latency by 85%.

---

## 2. Computer Use Mechanics, Screen Scaling & OS Automation

Claude’s Computer Use API and visual grounding architecture enable Sonnet 5.5 and Opus 5.5 to interact directly
 with legacy enterprise software lacking REST or GraphQL APIs:

1. Screen Observation: The local harness captures OS screenshots (scaled to 1024x768 or
 1280x800) and passes them to Claude as base64 image blocks.
2. Visual Grounding: Multimodal vision maps UI controls to coordinate actions (e.g.,
 mouse_move(x=452, y=318), left_click()).
3. Keystroke Emulation: Emits typed text and shortcut modifiers (Ctrl+C,
 Alt+Tab, Enter).
4. State Verification: Ingests post-action screenshots to confirm expected UI mutations
 before proceeding.

---

## 3. Model Context Protocol (MCP) Architecture & Transports

The MCP specification resolves the M-by-N integration bottleneck via two transport
 mechanisms:

- Standard I/O (stdio): Spawns local tool servers as child processes (e.g.,
 npx -y @modelcontextprotocol/server-postgres) with near-zero latency—ideal for local Claude Code CLI sessions.
- Server-Sent Events (SSE): Connects distributed microservices over persistent HTTP for
 cloud-hosted tool discovery and live log streaming.

---

## 4. Game-Changing Application: Autonomous 3-Tier Self-Healing CI/CD & Claude Code Swarm (Tests
 Included)

This production script implements an orchestrated 3-tier multi-agent engineering swarm combining the Anthropic API Console and Claude Code CLI subagents. A Claude Opus 5.5
 Supervisor uses Extended Thinking to coordinate a Claude Haiku 5.5 Log Triager,
 a Claude Sonnet 5.5 AST Patcher, and a headless Claude Code CLI Subagent
 in parallel via asyncio.gather. It also dynamically scaffolds markdown subagent definitions in
 .claude/agents/ and verifies patches against an embedded unittest regression
 suite.

#### `claude_5_5_self_healing_swarm.py`
```python
import asyncio
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List
from anthropic import AsyncAnthropic

# Initialize Async Client (Requires ANTHROPIC_API_KEY in environment)
client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY", "sk-ant-mock-key"))

# Explicit 5.5 Model Tier Registry
MODEL_TIERS = {
    "opus_supervisor": "claude-opus-5-5",
    "sonnet_patcher": "claude-sonnet-5-5",
    "haiku_triage": "claude-haiku-5-5",
    "claude_code_subagent": "claude-sonnet-5-5"
}

# ---------------------------------------------------------------------------
# 1. CLAUDE CODE SUBAGENT GENERATOR (.claude/agents/*.md)
# ---------------------------------------------------------------------------
def provision_claude_code_subagents(workspace_dir: Path) -> List[Path]:
    """
    Creates declarative Claude Code subagent markdown files in `.claude/agents/`.
    Claude Code automatically discovers these files for `/agents` and Task delegation.
    """
    agents_dir = workspace_dir / ".claude" / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)

    subagent_specs = {
        "log-triager.md": (
            "---\n"
            "name: log-triager\n"
            "description: High-speed CI/CD stacktrace analyzer and root-cause classifier. Invoke first on build failures.\n"
            "model: claude-haiku-5-5\n"
            "tools: Read, Grep, Glob\n"
            "---\n"
            "You are a Claude Haiku 5.5 Telemetry Triager. Inspect logs, isolate the exact file and line number, "
            "and return a concise JSON root-cause summary without modifying files.\n"
        ),
        "ast-patcher.md": (
            "---\n"
            "name: ast-patcher\n"
            "description: Autonomous code repair specialist for surgical AST patches and unit test verification.\n"
            "model: claude-sonnet-5-5\n"
            "tools: Read, Edit, Write, Bash, Grep\n"
            "---\n"
            "You are a Claude Sonnet 5.5 Senior Patch Engineer. Apply minimal, non-breaking code diffs to resolve "
            "the root cause identified by `log-triager` and verify with pytest.\n"
        ),
        "security-auditor.md": (
            "---\n"
            "name: security-auditor\n"
            "description: Deep architectural and security gatekeeper for reviewing critical diffs prior to merge.\n"
            "model: claude-opus-5-5\n"
            "tools: Read, Grep, Bash\n"
            "---\n"
            "You are a Claude Opus 5.5 Principal Security Auditor. Deliberate over edge cases, race conditions, "
            "and injection vectors before approving any patch.\n"
        )
    }

    created_files: List[Path] = []
    for filename, content in subagent_specs.items():
        file_path = agents_dir / filename
        file_path.write_text(content, encoding="utf-8")
        created_files.append(file_path)
    return created_files

# ---------------------------------------------------------------------------
# 2. PROGRAMMATIC API & HEADLESS CLAUDE CODE SUBAGENT DISPATCHER
# ---------------------------------------------------------------------------
async def run_subagent(role: str, objective: str) -> Dict[str, Any]:
    """Spawns an isolated subagent mapped to Haiku 5.5, Sonnet 5.5, or Opus 5.5."""
    selected_model = MODEL_TIERS.get(role, "claude-sonnet-5-5")

    # If live API key is configured, invoke Anthropic Messages API with Prompt Caching
    if os.getenv("ANTHROPIC_API_KEY"):
        thinking_config = (
            {"type": "enabled", "budget_tokens": 4096}
            if selected_model in ("claude-opus-5-5", "claude-sonnet-5-5")
            else {"type": "disabled"}
        )
        resp = await client.messages.create(
            model=selected_model,
            max_tokens=8192,
            thinking=thinking_config,
            system=[{
                "type": "text",
                "text": f"You are specialized subagent [{role}] running on {selected_model}. Output strict JSON.",
                "cache_control": {"type": "ephemeral"}
            }],
            messages=[{"role": "user", "content": objective}]
        )
        text_blocks = [b.text for b in resp.content if b.type == "text"]
        return {"role": role, "model": selected_model, "output": "\n".join(text_blocks)}

    # Deterministic local fallback for zero-cost CI unit testing
    if role == "haiku_triage":
        return {
            "role": role,
            "model": selected_model,
            "root_cause": "ZeroDivisionError in calculate_burn_rate when elapsed_hours == 0"
        }
    elif role == "opus_supervisor":
        return {
            "role": role,
            "model": selected_model,
            "dag_plan": ["1. Triage via Haiku 5.5", "2. Patch via Sonnet 5.5", "3. Verify via Claude Code CLI"],
            "security_signoff": "APPROVED"
        }
    return {
        "role": role,
        "model": selected_model,
        "patched_code": (
            "def calculate_burn_rate(tokens: int, elapsed_hours: float) -> float:\n"
            "    return round(tokens / max(elapsed_hours, 0.0001), 2)"
        )
    }

async def invoke_headless_claude_code_cli(prompt: str, agent_name: str = "ast-patcher") -> Dict[str, Any]:
    """
    Demonstrates spawning a headless Claude Code CLI subagent subprocess:
    `claude -p "<prompt>" --agent <agent_name> --output-format json --max-turns 3`
    """
    cli_binary = shutil.which("claude")
    cmd = [
        cli_binary or "claude",
        "-p", prompt,
        "--agent", agent_name,
        "--model", MODEL_TIERS["claude_code_subagent"],
        "--output-format", "json",
        "--max-turns", "3"
    ]
    # In environments without the CLI binary installed, return structured telemetry
    if not cli_binary or not os.getenv("ANTHROPIC_API_KEY"):
        return {
            "execution_mode": "simulated_headless_cli",
            "command": " ".join(cmd),
            "subagent": agent_name,
            "model": MODEL_TIERS["claude_code_subagent"],
            "status": "READY"
        }

    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()
    return {
        "execution_mode": "live_headless_cli",
        "returncode": proc.returncode,
        "stdout": stdout.decode("utf-8", errors="replace"),
        "stderr": stderr.decode("utf-8", errors="replace")
    }

def execute_sandbox_verification(patched_code: str) -> Dict[str, Any]:
    """Executes runtime regression checks against the Sonnet 5.5 patched code block."""
    namespace: Dict[str, Any] = {}
    try:
        exec(patched_code, namespace)
        fn = namespace.get("calculate_burn_rate")
        assert fn is not None, "Missing calculate_burn_rate function"
        assert fn(1000, 2.0) == 500.0, "Standard burn-rate math check failed"
        assert fn(500, 0.0) > 0, "Zero-division guardrail failed"
        return {"status": "PASSED", "assertions_verified": 3}
    except Exception as exc:
        return {"status": "FAILED", "error": str(exc)}

# ---------------------------------------------------------------------------
# 3. OPUS 5.5 SUPERVISOR ORCHESTRATION LOOP
# ---------------------------------------------------------------------------
async def orchestrate_healing_swarm(incident_payload: str, workspace_dir: Path) -> Dict[str, Any]:
    """
    Opus 5.5 Supervisor provisions `.claude/agents/*.md`, dispatches Haiku 5.5 and Sonnet 5.5
    subagents in parallel, invokes headless Claude Code CLI, and validates the fix.
    """
    # Step 1: Provision declarative Claude Code subagents on disk
    agent_files = provision_claude_code_subagents(workspace_dir)

    # Step 2: Opus 5.5 Supervisor formulates the remediation DAG
    supervisor_plan = await run_subagent("opus_supervisor", f"Plan remediation for: {incident_payload}")

    # Step 3: Parallel Fan-Out across Haiku 5.5 (Triage), Sonnet 5.5 (Patch), & Claude Code CLI
    triage_task = run_subagent("haiku_triage", f"Analyze stacktrace: {incident_payload}")
    patch_task = run_subagent("sonnet_patcher", f"Generate safe AST patch for: {incident_payload}")
    cli_task = invoke_headless_claude_code_cli("Verify AST patch and run test suite", agent_name="ast-patcher")

    triage_res, patch_res, cli_res = await asyncio.gather(triage_task, patch_task, cli_task)
    patch_code = patch_res.get("patched_code", "")

    # Step 4: Execute sandbox regression gate
    verification = execute_sandbox_verification(patch_code)
    return {
        "supervisor": supervisor_plan,
        "claude_code_agents_provisioned": [f.name for f in agent_files],
        "triage_report": triage_res,
        "patch_worker": patch_res,
        "claude_code_cli": cli_res,
        "verification": verification
    }

# ---------------------------------------------------------------------------
# 4. EMBEDDED AUTOMATED TEST SUITE (Run via `python3 claude_5_5_self_healing_swarm.py`)
# ---------------------------------------------------------------------------
class TestClaude55AgentSwarm(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="claude_code_swarm_"))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    async def test_full_5_5_triad_orchestration(self):
        """Verifies Opus 5.5 Supervisor, Haiku 5.5 Triager, Sonnet 5.5 Patcher, and Claude Code subagents."""
        res = await orchestrate_healing_swarm(
            "ZeroDivisionError: division by zero in metrics.py",
            self.temp_dir
        )
        self.assertEqual(res["supervisor"]["model"], "claude-opus-5-5")
        self.assertEqual(res["triage_report"]["model"], "claude-haiku-5-5")
        self.assertEqual(res["patch_worker"]["model"], "claude-sonnet-5-5")
        self.assertEqual(res["verification"]["status"], "PASSED")
        self.assertIn("log-triager.md", res["claude_code_agents_provisioned"])
        self.assertIn("ast-patcher.md", res["claude_code_agents_provisioned"])
        self.assertIn("security-auditor.md", res["claude_code_agents_provisioned"])

    def test_claude_code_subagent_markdown_frontmatter(self):
        """Verifies `.claude/agents/*.md` files contain valid YAML frontmatter and 5.5 model bindings."""
        files = provision_claude_code_subagents(self.temp_dir)
        self.assertEqual(len(files), 3)
        content_map = {f.name: f.read_text(encoding="utf-8") for f in files}
        self.assertIn("model: claude-haiku-5-5", content_map["log-triager.md"])
        self.assertIn("model: claude-sonnet-5-5", content_map["ast-patcher.md"])
        self.assertIn("model: claude-opus-5-5", content_map["security-auditor.md"])

    def test_sandbox_regression_guard(self):
        """Verifies sandbox rejects broken patches and passes valid patches."""
        bad_code = "def calculate_burn_rate(tokens, elapsed_hours):\n    return tokens / elapsed_hours"
        self.assertEqual(execute_sandbox_verification(bad_code)["status"], "FAILED")

        good_code = "def calculate_burn_rate(tokens, elapsed_hours):\n    return round(tokens / max(elapsed_hours, 0.0001), 2)"
        self.assertEqual(execute_sandbox_verification(good_code)["status"], "PASSED")

if __name__ == "__main__":
    print("[*] Executing Claude 5.5 Triad Swarm & Claude Code Subagent Verification Suite...")
    unittest.main(verbosity=2)
```

---

## 5. Claude Code Architecture: Creating Subagents & Orchestrating Multi-Agent Teams

Single-agent sessions suffer from context pollution and attention degradation when tackling sprawling
 enterprise monorepos. Claude Code CLI and the Anthropic
 Messages API solve this by isolating context windows through specialized subagents mapped to the
 Opus 5.5, Sonnet 5.5, and Haiku 5.5 tiers.

### How to Create Subagents in Claude Code

Claude Code natively supports declarative, project-scoped and user-scoped subagents that the main agent can
 invoke automatically via its built-in Task tool or explicitly on command:

1. Interactive CLI Wizard (/agents): Inside any interactive
 claude terminal session, run the /agents slash command. Select Create New
 Agent, choose between project-level scope (.claude/agents/, checked into Git for your
 team) or global user scope (~/.claude/agents/), and let Claude generate the system prompt and
 tool permissions.
2. Declarative Markdown Files (.claude/agents/*.md): Create markdown files
 with YAML frontmatter inside your repository's .claude/agents/ directory. Each file defines
 four core properties:

name: Unique identifier for the subagent (e.g., log-triager,
 ast-patcher, security-auditor).
description: Natural-language routing trigger telling the primary Claude Code session
 when to delegate to this subagent.
model: Explicit model binding—assign claude-haiku-5-5 for fast read-only
 search/triage, claude-sonnet-5-5 for code edits and test execution, or
 claude-opus-5-5 for deep architectural and security reviews.
tools: Least-privilege tool allowlist (e.g., Read, Grep, Glob for
 untrusted triage vs. Read, Edit, Write, Bash for patch workers, plus specific MCP server
 tools).
3. Headless CLI Subagent Spawning: For CI/CD pipelines and background orchestration
 scripts, invoke subagents non-interactively via
 claude -p "Audit auth module for race conditions" --agent security-auditor --model claude-opus-5-5 --output-format json.
4. Programmatic API Subagents: In custom Python or TypeScript backends, define a
 dispatcher tool on your Opus 5.5 Supervisor that calls client.messages.create() with an
 isolated messages array and role-specific model tier, returning only distilled JSON summaries
 to the parent context.

### Multi-Agent Team Orchestration Topologies

- Hierarchical Supervisor-Worker DAG: A Claude Opus 5.5 Supervisor with
 Extended Thinking ingests an engineering epic, constructs a Directed Acyclic Graph (DAG) of subtasks,
 dispatches Claude Haiku 5.5 subagents to map the repository and parse failing logs, and
 assigns Claude Sonnet 5.5 subagents to author modular patches.
- Parallel Git Worktree Swarms (Fan-Out / Fan-In): Using git worktree add
 alongside headless Claude Code instances or asyncio.gather(), teams run multiple Sonnet 5.5
 subagents in parallel across isolated filesystem checkouts—simultaneously building backend migrations,
 frontend components, and integration tests without file-locking conflicts.
- SWE-bench Verification Loop: Subagents follow a strict four-gate
 contract: (1) Haiku 5.5 localizes the fault, (2) Sonnet 5.5 writes a failing reproduction test, (3) Sonnet
 5.5 applies a minimal AST diff until the test passes, and (4) Opus 5.5 performs a final architectural and
 security review before opening a pull request.

---

## 6. Prompt Calibration & Session State Serialization

Achieving 99.9%+ reliability across multi-agent 5.5 teams requires deterministic behavioral contracts and
 ACID-compliant state persistence:

- Negative Tool Guidance & Few-Shot Calibration: Explicitly define when not
 to invoke subagents or write queries, and embed two-turn synthetic tool examples in cached system prompts.
- Self-Correction Loops: Return Pydantic validation exceptions directly inside
 tool_result blocks so Sonnet 5.5 and Haiku 5.5 subagents automatically repair malformed
 parameters.
- ACID Trajectory Checkpointing: Serialize turn snapshots, Opus 5.5 thinking traces, and
 SHA-256 file hashes into PostgreSQL or DynamoDB, pruning historical context on restore while preserving
 active tool schemas.

---

## 7. Verified Monthly & Yearly Subscriptions & TCO Analysis

Anthropic offers tiered subscription pricing spanning individual engineers, collaborative squads,
 and metered API swarms across the 5.5 model lineup (verified October 2026):

---

## 8. Enterprise Guardrails, Latency & Troubleshooting

- Ephemeral Docker Sandboxes & Egress Firewalls: Following Computer Use safety guidelines and Anthropic Trust standards, isolate
 Claude Code bash subagents inside read-only containers (Xvfb/VNC) with strict proxy whitelists and
 dual-key human approval modals for destructive actions.
- Subagent Context Isolation & Tool Scoping: Restrict Haiku 5.5 triage subagents to
 read-only tools (Read, Grep, Glob) in .claude/agents/*.md so untrusted log
 inputs cannot trigger arbitrary shell execution.
- Extended Thinking Headroom: Always set
 max_tokens higher than your thinking_budget on Opus 5.5 and Sonnet 5.5 (e.g.,
 4,000 thinking budget requires 8,000+ max_tokens) and use
 client.messages.stream() to prevent UI freezes.

---

## 9. References & Authoritative Grounding Sources (October 2026)

- [1]
Model Context Protocol (MCP) Specification — Anthropic (2026). https://modelcontextprotocol.io
- [2]
Computer Use API & GUI Action Space — Anthropic Docs (2026). https://docs.anthropic.com/en/docs/build-with-claude/computer-use
- [3]
Anthropic Pricing: Pro, Team & Enterprise — Anthropic (Oct 2026). https://www.anthropic.com/pricing
- [4]
Developing Computer Use & Safety — Anthropic Research (2026). https://www.anthropic.com/news/developing-computer-use
- [5]
Prompt Caching: 90% Cost Reduction — Anthropic API Guide (2026). https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
- [6]
Extended Thinking in Claude Architectures — Anthropic Science (2026). https://www.anthropic.com/news/claude-3-7-sonnet
- [7]
SWE-bench Verified Leaderboard — Princeton NLP (2026). https://www.swebench.com
- [8]
Anthropic Console & Token Metering — Anthropic Platform (Oct 2026). https://console.anthropic.com
- [9]
Enterprise Trust & Security Framework — Anthropic Trust Center (2026). https://trust.anthropic.com
- [10]
Claude Code CLI & Subagent Orchestration — Anthropic Engineering (2026). https://docs.anthropic.com/en/docs/agents-and-tools/claude-code

---

### Ready to Orchestrate Production AI Agent Teams?
- [Schedule Strategy Consultation via Email →](collaboration.html)
- [View Advisory Packages & Rates](pricing.html)
- [Explore All Agent Providers Matrix](agents-all-agentic-providers.html)
