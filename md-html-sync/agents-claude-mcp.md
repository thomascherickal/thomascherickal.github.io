# Model Context Protocol (MCP): Architecture & Servers
## AI Agent Architecture & Orchestration — Thomas Cherickal

> **// CLAUDE AGENTS · INTEGRATION PROTOCOL**  
> **The definitive architectural guide to Model Context Protocol (MCP)—the open standard transforming how AI agents connect to enterprise data, internal microservices, and external tooling without custom API glue.**

---

## 1. What is the Model Context Protocol?

Model Context Protocol (MCP) solves the N-by-M integration problem in AI. Instead of writing custom client connectors for every database, issue tracker, and cloud provider, organizations deploy standardized MCP servers. Any MCP-compliant client (Claude Desktop, Claude Code, custom agents) can dynamically discover and execute resources and tools:

---

## 2. MCP Core Primitives: Resources, Tools & Prompts

MCP standardizes three distinct capabilities across all servers:

- Resources (Read-Only Data): File-like data streams (e.g. database logs, API documentation, metrics) that provide context without side effects.
- Tools (Executable Functions): Operations with side effects (e.g. sending a message, creating a pull request, deploying a container) that require permission controls.
- Prompts (Standardized Workflows): Pre-engineered prompt templates parameterized by the server to guide the model through complex tasks.

---

## 3. Tutorial: Building a FastMCP Server in Python

Build and run an enterprise-ready MCP server exposing real-time system metrics and secure queries in under 50 lines of Python:

#### `mcp_server.py`
```python
from mcp.server.fastmcp import FastMCP
import psutil
import json

# Initialize MCP Server
mcp = FastMCP("Enterprise-Infrastructure-Monitor")

@mcp.resource("system://metrics")
def get_system_metrics() -> str:
    """Read-only resource providing live CPU and RAM telemetry."""
    metrics = {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent
    }
    return json.dumps(metrics, indent=2)

@mcp.tool()
def restart_service(service_name: str, force: bool = False) -> str:
    """Executable tool with side effects to restart an infrastructure daemon."""
    # Enterprise safeguards: check service whitelist
    allowed_services = ["nginx", "redis-cache", "celery-worker"]
    if service_name not in allowed_services:
        return f"ERROR: Service '{service_name}' is not in approved restart whitelist."
    
    # Simulate restart
    return f"SUCCESS: Service '{service_name}' restarted successfully (Force={force})."

if __name__ == "__main__":
    mcp.run()
```

---

### Ready to Orchestrate Production Claude AI Agent Teams?
- [Schedule Strategy Consultation via Email →](collaboration.html)
- [View Advisory Packages & Rates](pricing.html)
- [Explore Claude Agents Suite](agents-claude-101.html)
