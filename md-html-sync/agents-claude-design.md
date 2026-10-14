# Claude Agent Design Patterns: Production System Architecture
## AI Agent Architecture & Orchestration — Thomas Cherickal

> **// CLAUDE AGENTS · ARCHITECTURAL PATTERNS**  
> **Analyze the five architectural patterns that distinguish resilient enterprise agent systems from fragile demo scripts: Evaluator-Optimizer loops, deterministic state machines, routing trees, and checkpointed memory.**

---

## 1. Canonical Pattern 1: The Evaluator-Optimizer

In this pattern, one model generates an initial implementation while a separate critic model evaluates it against strict criteria, feeding actionable critique back until quality criteria are satisfied:

---

## 2. Canonical Pattern 2: The Orchestrator-Workers

The orchestrator analyzes a high-level task, plans a dependency graph of subtasks, delegates them to specialized workers, and synthesizes the outputs into a coherent result:

#### Step 1 · Planning — Task Graph Analysis
Orchestrator breaks complex user goals into non-overlapping subtasks with explicit inputs and outputs.

#### Step 2 · Fan-Out — Parallel Worker Dispatch
Subtasks are dispatched concurrently to worker agents equipped only with relevant tool schemas.

#### Step 3 · Synthesis — Consolidation & Verification
Orchestrator aggregates worker outputs, checks for contradictions, and delivers unified synthesis.

---

## 3. Tutorial: Implementing an Evaluator-Optimizer in Python

#### `evaluator_optimizer.py`
```python
import os
from anthropic import Anthropic

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

def optimize_code(task: str, max_iterations=3):
    current_code = ""
    feedback = "Initial generation"

    for i in range(max_iterations):
        print(f"--- Iteration {i+1} ---")
        # 1. Generator
        gen_prompt = f"Task: {task}\nPrevious Code: {current_code}\nCritique to fix: {feedback}"
        gen_resp = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=2048,
            messages=[{"role": "user", "content": gen_prompt}]
        )
        current_code = gen_resp.content[0].text

        # 2. Evaluator
        eval_resp = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1024,
            messages=[{"role": "user", "content": f"Critique this code for edge-case bugs and performance. Rate PASS or FAIL with specific feedback:\n\n{current_code}"}]
        )
        critique = eval_resp.content[0].text
        print(f"Critique: {critique[:150]}...\n")

        if "PASS" in critique.upper():
            print("Evaluator passed the candidate code!")
            return current_code
        feedback = critique

    return current_code

if __name__ == "__main__":
    optimize_code("Write an asynchronous LRU cache with TTL expiration in Python.")
```

---

### Ready to Orchestrate Production Claude AI Agent Teams?
- [Schedule Strategy Consultation via Email →](collaboration.html)
- [View Advisory Packages & Rates](pricing.html)
- [Explore Claude Agents Suite](agents-claude-101.html)
