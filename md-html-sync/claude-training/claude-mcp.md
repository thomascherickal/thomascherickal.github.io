# Claude MCP: Model Context Protocol & Custom Tool Servers
## AI Agent Architecture & Orchestration - Thomas Cherickal

> **// PROVIDER DEEP DIVE**  
> **Master the Model Context Protocol (MCP) to connect Claude cleanly to internal APIs, databases, and developer tools. Covers transports, tool/resource/prompt primitives, simple TypeScript and Python server examples, authentication, and security hardening against prompt injection.**

---

## Table of Contents

Before the Model Context Protocol (MCP), every AI app needed custom glue code for every database, API, and folder it touched. MCP acts like a USB-C port for AI: write one clean server that exposes your internal tools or data, and Claude Desktop, Claude Code, and your own apps can connect to it immediately with consistent permissions.

1. What MCP Is and Why Teams Standardize on It
2. Tools, Resources, Prompts, and Transports Explained Simply
3. Building a Clean Custom MCP Server in Python
4. TypeScript Servers and OAuth Authentication
5. Security Hardening Against Prompt Injection
6. Authoritative References

---

## What MCP Is and Why Teams Standardize on It

Created by Anthropic and adopted across the industry, the Model Context Protocol specification defines a single open language between AI hosts (like Claude Desktop or Claude Code) and external data sources. Instead of maintaining five fragile scripts for five tools, your team maintains one small, tested MCP server per internal system.

---

## Tools, Resources, Prompts, and Transports Explained Simply

MCP keeps the design clean with three primitives and two main ways to connect:

- Tools: Functions Claude can call to do work (such as running a read-only SQL query or creating a draft ticket).
- Resources: Read-only data items (such as a customer record, log file, or API schema) that Claude can inspect like files.
- Prompts: Reusable workflow templates stored on the server so teammates trigger consistent multi-step tasks.
- Transports: Use stdio for local servers running on your laptop, and Streamable HTTP for shared remote servers hosted inside your company cloud.

---

## Building a Clean Custom MCP Server in Python

Using the official MCP Python SDK (FastMCP), you can expose a safe, read-only internal lookup tool in under 25 lines of code.

#### `inventory_mcp_server.py`
```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("inventory-lookup")

STOCK = {"SKU-101": 42, "SKU-204": 0, "SKU-309": 18}

@mcp.tool()
def check_stock(sku: str) -> str:
    """Return current warehouse stock count for a valid SKU code."""
    code = sku.strip().upper()
    if code not in STOCK:
        return f"SKU {code} not found."
    return f"SKU {code}: {STOCK[code]} units available."

if __name__ == "__main__":
    mcp.run()
```

---

## TypeScript Servers and OAuth Authentication

Teams using Node.js can build the exact same server with the MCP TypeScript SDK and validate inputs with Zod schemas. For remote HTTP MCP servers shared across a company, always require OAuth 2.0 bearer tokens tied to the signed-in user so the server enforces real per-user access control rather than a shared static key.

---

## Security Hardening Against Prompt Injection

Whenever an MCP server reads untrusted external text (like customer emails, web pages, or public GitHub issues), hidden instructions inside that text can try to trick Claude into running dangerous tools. Protect your servers with four practical rules:

- Validate Every Argument on the Server: Never trust raw SQL or shell strings from the model - use parameterized queries and strict allowlists.
- Keep Sensitive Tools Separate: Do not mix untrusted web-reading tools with unconfirmed write/delete tools in the same open session.
- Require Human Approval for Side Effects: Prompt the user before sending messages, transferring money, or modifying production databases.
- Return Minimal Data: Strip secrets, tokens, and unnecessary personal fields before returning tool outputs to the model.

---

## Authoritative References

1. Model Context Protocol Official Documentation - Complete introduction, architecture guide, and transport specifications.
2. MCP Python SDK on GitHub - Official Python library including FastMCP for rapid custom tool servers.
3. MCP TypeScript SDK on GitHub - Official TypeScript SDK with Zod schema validation and HTTP transport support.
4. OWASP Top 10 for LLM Applications - Industry security practices for defending tool-connected agents against prompt injection.

---

### Ready to Orchestrate Production Claude AI Agent Teams?
- [Schedule Strategy Consultation via Email →](contact.html)
- [View Advisory Packages & Rates](pricing.html)
- [Explore Claude Agents Suite](claude-101.html)
