# Claude Code in the Terminal and Your IDE
## AI Agent Architecture & Orchestration — Thomas Cherickal

> **// PROVIDER DEEP DIVE**  
> **Claude Code is an agent, not autocomplete. It reads your repository, edits files, runs your tests and commits the result. It started as a terminal tool and now runs inside VS Code, JetBrains IDEs, a desktop app and the browser, with the same agent behind every surface. This guide covers how it works, how to use it in each place, and how to run agents with it, with tutorials for first-time users and for people who already use it daily.By Thomas Cherickal, Generative AI Consultant · The Digital Futurist · thomascherickal.com · thomascherickal.github.io**

---

## Table of Contents

1. Why Engineers Adopted It
2. The Building Blocks
3. How the Agent Loop Works
4. Claude Code in the Terminal
5. Claude Code in Your IDE
6. Tutorials for Beginners

Install and run your first session
Work from VS Code or JetBrains
Teach Claude your project
7. Tutorials for Experts

Build a subagent and a hook
Run parallel background agents
Automate in CI with headless mode
Connect external tools with MCP
8. Working in Large Repositories
9. What Developers Say
10. Adoption by the Numbers
11. References

---

## Why Engineers Adopted It

Claude Code meets engineers where they already work. The terminal is on every machine, and the IDE extensions bring the same agent into the editor without a new workflow to learn.

It also closes the loop that autocomplete leaves open. It runs a command, reads the error, changes the code and runs the command again, all in one session. In large codebases it searches with grep, glob and targeted file reads instead of relying on a pre-built index. It is scriptable for CI, extensible through the Model Context Protocol (MCP), and governed by a permission system that security teams can configure.

It also sits alongside other tools. The Pragmatic Engineer's 2026 survey found that several teams who were heavy Cursor users added Claude Code and kept Cursor too.

---

## The Building Blocks

---

## How the Agent Loop Works

Each task runs through the same cycle. Claude gathers context from CLAUDE.md, your prompt and the files it searches. It plans, checks its permissions, acts with tools, then verifies the result by running tests, a linter or the build. If verification fails, it reads the output and goes round again.

Permission modes decide how much happens without you. Press Shift+Tab in a session to cycle through them. Recent versions start interactive sessions in auto mode, where a classifier reviews each action, lets safe ones run and blocks risky ones such as destructive git commands.

---

## Claude Code in the Terminal

Inside a session, these commands do most of the work:

Prefix a line with ! to run a shell command directly, and type @ to reference a file.

---

## Claude Code in Your IDE

### VS Code, Cursor and other forks

The extension needs VS Code 1.94 or later. It adds a side panel where you review and edit Claude's plan before accepting it, accept or reject each change as an inline diff, and keep several conversations open in tabs. Claude sees your selected text automatically. Cursor and other VS Code forks such as Kiro install the same extension, through the Open VSX registry where needed.

### JetBrains IDEs

The plugin works with IntelliJ IDEA, PyCharm, Android Studio, WebStorm, PhpStorm and GoLand. It runs the claude CLI in the IDE terminal, opens Claude's changes in the native diff viewer, shares your selection, and lets Claude read the IDE's inspection results such as lint and syntax errors.

### Every other editor

Vim, Neovim, Emacs and Zed users can run claude in any terminal next to the editor. From an external terminal, /ide connects a session to a running VS Code or JetBrains window.

---

## Tutorials for Beginners

### BeginnerTutorial 1: Install and run your first session

1. Install. Pick the command for your system.

2. Check it. Open a new terminal and run claude --version.

3. Log in. Run claude inside a project folder and follow the browser prompt. Any paid Claude plan or a Claude Console account works.

4. Ask before you change anything. Type what does this project do? or explain the folder structure. Claude reads the files it needs.

5. Make a small change. Try add input validation to the signup form. Review the diff, then ask commit my changes with a descriptive message.

6. Learn the escape hatches. Esc stops Claude mid-task, /rewind undoes edits, and Shift+Tab switches to plan mode when you want a proposal first.

### BeginnerTutorial 2: Work from VS Code or JetBrains

VS Code: press Cmd+Shift+X (Mac) or Ctrl+Shift+X (Windows, Linux), search for "Claude Code" and install it. Open the panel from the Command Palette by typing "Claude Code". Select a few lines and press Option+K / Alt+K to add a reference like @app.ts#5-10. Ask for a change, then accept or reject each diff. Cmd+Esc / Ctrl+Esc moves focus between the editor and Claude.

JetBrains: install the CLI from Tutorial 1, then the Claude Code plugin from the JetBrains Marketplace, and restart the IDE. Press Cmd+Esc / Ctrl+Esc to launch Claude. Changes open in the IDE's diff viewer. If you prefer an external terminal, run claude there and type /ide.

### BeginnerTutorial 3: Teach Claude your project

Run /init. Claude scans the repository and writes a CLAUDE.md. Edit it by hand, because short, specific rules work better than long ones:

Commit the file so your team shares it. Run /memory later to edit it from inside a session.

---

## Tutorials for Experts

### ExpertTutorial 4: Build a subagent and a hook

A subagent works in its own context window, so the main session stays focused. Create .claude/agents/security-reviewer.md, or ask Claude to create it:

Then prompt: add rate limiting to /login, then have the security-reviewer subagent check the diff. Add a hook in .claude/settings.json so every edit gets formatted:

### ExpertTutorial 5: Run parallel background agents

Agent view (claude agents) lists every background session, grouped into needs input, working and completed. Each background session moves into its own git worktree under .claude/worktrees/ before it edits, so parallel agents never write to the same checkout.

Inside a session, /bg sends the current conversation to the background and /fork copies it into a new one. Pair a background agent with /goal all tests in tests/api pass and it keeps working until that condition holds.

### ExpertTutorial 6: Automate in CI with headless mode

claude -p runs one task and exits with a status code your scripts can check. Add --bare in CI so the run ignores hooks, plugins and CLAUDE.md on the host machine and behaves the same everywhere. Scope tools tightly:

For pull requests, run /install-github-app in a session. It installs the GitHub app and adds a workflow using anthropics/claude-code-action@v1, after which anyone can mention @claude in an issue or PR.

### ExpertTutorial 7: Connect external tools with MCP

Each MCP server adds tool definitions to the context, so connect only what the project uses. /usage shows which servers, skills and subagents drive your plan limits.

---

## Working in Large Repositories

In a codebase over 100,000 lines, split the work into phases. Curate CLAUDE.md, with smaller CLAUDE.md files in big subfolders. Start in plan mode and ask Claude to map every usage of what you are changing before it edits. Work one module at a time and clear context between phases. Give each module its own background agent and worktree, and let tests and hooks verify every edit before a human reviews the pull requests.

---

## What Developers Say

Survey responses and community threads point to the same strengths: multi-file refactors that update imports, types and tests in one pass, a terminal workflow that fits existing habits, and MCP connections to databases and internal tools. The Pragmatic Engineer survey found that Director-level respondents and above were roughly twice as likely as less senior engineers to name it a tool they love.

The most common complaint is usage limits. Others are context taken up by MCP servers before work starts, and long sessions that lose detail after repeated compaction.

Anthropic's own study of about 400,000 Claude Code sessions from October 2025 to April 2026 found a working split: people made about 70% of planning decisions and Claude made about 80% of execution decisions. Understanding the problem domain predicted success more than coding skill did, and novices abandoned sessions at three times the rate of intermediate or expert users. That gap is why the beginner tutorials above start with questions before changes.

---

## Adoption by the Numbers

In the JetBrains survey's mid-2026 wave, Claude Code led workplace use by a wide margin:

Anthropic has not published a total user count. On 12 February 2026 it said weekly active users had doubled since 1 January. The survey figures are self-reported and use different samples, so read them as direction rather than exact market share.

---

## References

1. Claude Code overview, Anthropic documentation
2. Quickstart, Anthropic documentation
3. Use Claude Code in VS Code, Anthropic documentation
4. JetBrains IDEs, Anthropic documentation
5. Permission modes, Anthropic documentation
6. Agent view, Anthropic documentation
7. Subagents and hooks, Anthropic documentation
8. Run Claude Code programmatically, Anthropic documentation
9. Model Context Protocol, Anthropic documentation
10. What's new, Anthropic documentation
11. AI Tooling 2026, The Pragmatic Engineer
12. Claude Code Statistics 2026, Gradually.ai (JetBrains survey, revenue, npm and GitHub figures)
13. Anthropic research on Claude Code sessions, Blockchain.News, 16 June 2026
14. Claude Code on Reddit: What Developers Actually Say, Morph

---

### Ready to Orchestrate Production Claude AI Agent Teams?
- [Schedule Strategy Consultation via Email →](collaboration.html)
- [View Advisory Packages & Rates](pricing.html)
- [Explore Claude Agents Suite](claude-101.html)
