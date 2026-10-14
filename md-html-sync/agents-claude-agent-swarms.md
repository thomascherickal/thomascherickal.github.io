# Claude Agent Swarms: Horizontal Fan-Out & Parallel Execution
## AI Agent Architecture & Orchestration — Thomas Cherickal

> **// CLAUDE AGENTS · PARALLEL SWARMS**  
> **Learn how to orchestrate high-density agent swarms with Claude Haiku 5.5 and Sonnet 5.5: fanning out hundreds of concurrent workers across large file trees, security scanning, and synthetic data pipelines.**

---

## 1. Swarms vs Teams: Horizontal Scaling Mechanics

While an Agent Team operates as a small council of specialized experts, an Agent Swarm is composed of dozens or hundreds of homogenous worker instances executing parallel operations across distributed data shards:

---

## 2. High-Throughput Concurrency Governor

Executing hundreds of concurrent agent requests against the Anthropic API requires a rate-limit token bucket governor to maximize throughput without triggering HTTP 429 backpressure:

#### `swarm_governor.py`
```python
import asyncio
import os
from anthropic import AsyncAnthropic

client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

async def worker_task(file_id: int, semaphore: asyncio.Semaphore):
    async with semaphore:
        # Claude Haiku 5.5 handles high-velocity parallel scanning
        response = await client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": f"Scan mock document #{file_id} for exposed secrets or credentials. Return JSON: {{'clean': bool, 'findings': []}}"
            }]
        )
        print(f"[WORKER {file_id:03d}]: Finished audit")
        return response.content[0].text

async def run_swarm(total_tasks=50, max_concurrent=10):
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [worker_task(i, semaphore) for i in range(1, total_tasks + 1)]
    results = await asyncio.gather(*tasks)
    print(f"\nSwarm complete: Processed {len(results)} tasks concurrently.")

if __name__ == "__main__":
    asyncio.run(run_swarm(30, 8))
```

---

### Ready to Orchestrate Production Claude AI Agent Teams?
- [Schedule Strategy Consultation via Email →](collaboration.html)
- [View Advisory Packages & Rates](pricing.html)
- [Explore Claude Agents Suite](agents-claude-101.html)
