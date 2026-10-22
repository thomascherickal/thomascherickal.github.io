# Claude Skills: Build Your Own, Use What Exists, and Plan for What Comes Next
## AI Agent Architecture & Orchestration — Thomas Cherickal

> **// PROVIDER DEEP DIVE**  
> **How Claude skills work, how to write your own SKILL.md, the skills that already exist across Claude and the open Agent Skills standard, and what this extensibility means for automation and work.**

---

## Table of Contents

1. 1. What A Claude Skill Is
2. 2. Where Skills Run
3. 3. Build Your Own Skill
4. 4. Hands-On Tutorials

Meeting-Notes in Claude Apps
Changelog Script in Claude Code
Skills via Claude API
5. 5. Existing Skills Catalog
6. 6. Skill Security & Safety
7. 7. Automation Economics & Impact
8. 8. References

---

## 1. What A Claude Skill Is

A Claude skill is a folder. Inside it sits a Markdown file called SKILL.md, plus any scripts, templates or reference documents the task needs. Claude opens it only when a request matches what the skill says it does, then follows it like an onboarding guide for one specific job.

Anthropic introduced Agent Skills for Claude on October 16, 2025, and published the format as an open standard on December 18, 2025. Skills now run in the Claude apps, Claude Code, the Claude API and the Agent SDK. This article explains how they work, walks through three hands-on tutorials, lists the skills that already exist, and looks at what this extensibility means for automation and for the people whose work gets automated.

Anthropic's Agent Skills overview in the Claude docs defines skills as modular capabilities that bundle instructions, metadata and optional resources, which Claude uses automatically when relevant. MCP connectors give Claude access to outside systems; a skill describes the procedure to follow once that access exists.

Skills load in three stages, a design Anthropic calls progressive disclosure. The Anthropic engineering post that introduced Agent Skills calls it the principle that lets skills scale.

Scripts run through bash, so only their output enters the context window. A skill can bundle a full API reference at no cost until a task needs it, which is why one installation can carry many skills.

---

## 2. Where Skills Run

- Claude apps: custom skills are uploaded or created under Customize > Skills, on Free, Pro, Max, Team and Enterprise plans with code execution turned on.
- Claude Code: skills are folders in ~/.claude/skills/ for personal use or .claude/skills/ inside a repository, callable as /skill-name.
- Claude API: skills attach to the code execution container by skill_id and are shared across a workspace.
- Agent SDK: skills are discovered from the filesystem and limited with a skills option, per the Agent SDK skills documentation.

---

## 3. Build Your Own Skill

### Step 1: Pick One Repeated Job

The Claude Code guide to extending Claude with skills gives a practical trigger: write a skill when you keep pasting the same instructions, checklist or procedure into chat. Keep each skill to one workflow, because Claude can combine several focused skills in a single task.

### Step 2: Write The SKILL.md File

A skill needs YAML frontmatter between --- markers and Markdown instructions below it. Here is an enterprise release-notes skill directory layout and definition:

#### `Folder Layout — writing-release-notes`
```python
# Folder layout: the folder name doubles as the skill name
writing-release-notes/
├── SKILL.md                     # required: frontmatter plus instructions
├── references/style-guide.md    # optional: read only when SKILL.md links to it
└── scripts/collect_commits.py   # optional: executed, never loaded into context
```

#### `writing-release-notes/SKILL.md`
```python
---
# Required. Lowercase letters, numbers and hyphens, 64 characters max.
name: writing-release-notes
# Required. What the skill does AND when to use it, in the third person.
description: >
  Drafts customer-facing release notes from merged pull requests.
  Use when the user asks for release notes or a changelog summary.
---

# Writing Release Notes

<!-- Step 1 runs a bundled script; only its output reaches Claude -->
1. Run `python scripts/collect_commits.py --since last-tag`.
<!-- Steps 2 and 3 need judgment, so they stay as plain instructions -->
2. Group changes under New, Improved and Fixed.
3. Follow the tone rules in [references/style-guide.md](references/style-guide.md).
```

Anthropic's skill authoring best practices guide sets the rules. Names cannot contain "anthropic" or "claude", descriptions are capped at 1,024 characters, and the body should stay under 500 lines, with supporting files linked one level deep. Use scripts for fragile, sequence-dependent steps and plain instructions where several approaches are valid.

### Step 3: Install The Skill

In Claude Code, copy the folder into a skills directory; edits take effect mid-session without restarting. In the Claude apps, zip the folder with the folder itself at the root, upload it, and enable it under Customize > Skills. On the API, upload through the /v1/skills endpoints.

The Claude Help Center guide to creating skills adds a shortcut: on Pro, Max and Team plans, Cowork in Claude for Mac can build a skill from a screen recording of you doing the task. You can also ask Claude to build one, and the skill-creator skill will interview you and draft SKILL.md.

### Step 4: Test It Against A Baseline

A skill that triggers does not automatically help. Run realistic prompts in fresh sessions with the skill on and then off. The skill-creator plugin for Claude Code automates this: it runs test cases in isolated subagents, grades output, benchmarks pass rate, time and tokens against a no-skill run, and suggests description fixes when the skill fires on the wrong requests.

---

## 4. Hands-On Tutorials

The three tutorials below build systematically on one another. The first requires only the Claude app, the second runs in Claude Code with a bundled Python script and dynamic injection, and the third uploads the custom skill to the Claude API and pairs it with Anthropic's production Word skill.

### Tutorial 1: Meeting-Notes in Claude Apps

1. Create the folder. The folder name has to match the name field you are about to write, so keep it lowercase with hyphens.

#### `meeting-notes/ directory`
```python
meeting-notes/                 # folder name must equal the skill's name field
├── SKILL.md                   # the instructions Claude follows
└── assets/
    └── notes-template.md      # the layout every summary must use
```

2. Write SKILL.md. The description carries the trigger phrases, because it is the only part Claude sees before deciding to load the skill.

#### `meeting-notes/SKILL.md`
```python
---
# Must match the folder name: lowercase letters, numbers and hyphens.
name: meeting-notes
# Claude matches requests against this text, so name the triggers plainly.
description: >
  Turns raw meeting notes or call transcripts into a structured summary
  with decisions, action items, owners and due dates. Use when the user
  pastes meeting notes or a transcript, or asks for minutes or a recap.
---

# Meeting Notes

<!-- The numbered steps are the procedure Claude follows every time. -->
## Steps
1. Read the raw notes in full before writing anything.
2. Fill in assets/notes-template.md, keeping its headings in the same order.
3. Give every action item an owner and a due date. If the notes do not
   say, write "Owner: unassigned" or "Due: not set" and add the item
   to Open questions.
4. Record only decisions the notes state. Never infer a decision from
   discussion.

<!-- One concrete example teaches the format faster than a description. -->
## Example action item
- Send the revised budget to finance. Owner: Priya. Due: 30 Oct.
```

3. Add the template. Save this as assets/notes-template.md. Claude reads it only when step 2 of the instructions sends it there.

#### `meeting-notes/assets/notes-template.md`
```python
<!-- Claude replaces each {placeholder} and keeps the headings in order. -->
# {Meeting title}: {date}

**Attendees:** {names}

## Decisions
- {decision}

## Action items
- {task}. Owner: {name}. Due: {date}.

## Open questions
- {question, or an action item missing an owner or date}
```

4. Zip the folder. Zip the folder itself rather than the loose files inside it, so meeting-notes/ sits at the root of the archive.

#### `Terminal — Zip Archive`
```python
# macOS or Linux: run from the directory that contains meeting-notes/
zip -r meeting-notes.zip meeting-notes

# Windows: right-click the meeting-notes folder, then choose
# Send to > Compressed (zipped) folder
```

5. Upload and enable it. On Free, Pro and Max plans, first turn on "Code execution and file creation" under Settings > Capabilities; on Team and Enterprise plans an owner enables it for the organization. Then open Customize > Skills, click +, choose Create skill, then Upload a skill, select the ZIP and switch the skill on. These steps follow the Help Center guide to using skills in Claude.

6. Test both directions. Paste a messy set of notes and ask for a recap; the reply should follow the template exactly, with gaps listed under Open questions. Then ask for something nearby that should not trigger the skill, such as "draft a meeting invite for Thursday", and confirm the template does not appear. If the skill never fires, sharpen the description or name it directly: "Use my meeting-notes skill on these notes."

### Tutorial 2: Changelog Script in Claude Code

1. Create the folders. A personal skill lives under your home directory and works in every project on that machine.

#### `Terminal — Personal Skills Directory`
```python
# Personal skill: available in all your projects on this machine
mkdir -p ~/.claude/skills/drafting-changelogs/scripts
```

2. Add the script. Save this as scripts/group_commits.py. It uses only the Python standard library and sorts commits written in the Conventional Commits style (feat:, fix:, perf:) into changelog sections.

#### `~/.claude/skills/drafting-changelogs/scripts/group_commits.py`
```python
#!/usr/bin/env python3
"""Group git commits by Conventional Commit type for a changelog draft.

Usage: python3 group_commits.py [since-ref]
If no ref is given, the script uses the most recent git tag.
"""
import re
import subprocess
import sys
from collections import defaultdict

# Map Conventional Commit prefixes to the changelog headings we publish.
SECTIONS = {
    "feat": "New",
    "fix": "Fixed",
    "perf": "Improved",
    "refactor": "Improved",
    "docs": "Documentation",
}
# Print headings in this order; anything unrecognised lands in "Other".
ORDER = ["New", "Improved", "Fixed", "Documentation", "Other"]

# Matches "type(scope)!: summary" and captures the type and the summary.
PATTERN = re.compile(r"^(?P<type>\w+)(\([^)]*\))?!?:\s*(?P<summary>.+)$")


def git(*args):
    """Run a git command and return (exit_code, stdout, stderr)."""
    result = subprocess.run(["git", *args], capture_output=True, text=True)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def main():
    # 1. Work out the starting point: an explicit ref, else the last tag.
    since = sys.argv[1].strip() if len(sys.argv) > 1 else ""
    if not since:
        code, tag, _ = git("describe", "--tags", "--abbrev=0")
        since = tag if code == 0 else ""

    # 2. Read one line per commit: short hash, a tab, then the subject.
    log_range = f"{since}..HEAD" if since else "HEAD"
    code, out, err = git("log", log_range, "--no-merges", "--pretty=format:%h	%s")
    if code != 0:
        # Print a readable error instead of a traceback, so Claude can explain it.
        print(f"Could not read git history: {err}")
        sys.exit(1)

    # 3. Sort each commit into a section based on its prefix.
    groups = defaultdict(list)
    for line in filter(None, out.splitlines()):
        sha, subject = line.split("	", 1)
        match = PATTERN.match(subject)
        if match:
            section = SECTIONS.get(match["type"].lower(), "Other")
            summary = match["summary"]
        else:
            section, summary = "Other", subject
        groups[section].append(f"- {summary} ({sha})")

    # 4. Print Markdown that Claude rewrites into customer-facing notes.
    print(f"Commits since {since or 'the first commit'}:")
    if not groups:
        print("No commits found.")
    for section in ORDER:
        if groups[section]:
            print(f"
## {section}")
            print("
".join(groups[section]))


if __name__ == "__main__":
    main()
```

3. Write SKILL.md. The line that starts with ! is dynamic context injection: Claude Code runs the command before Claude reads the skill and pastes the output in its place, so Claude starts with real commit data instead of guessing.

#### `~/.claude/skills/drafting-changelogs/SKILL.md`
```python
---
name: drafting-changelogs
description: >
  Drafts customer-facing release notes from git history since the last
  tag. Use when the user asks for a changelog, release notes, or what
  changed since the last release.
# Hint shown in the / menu for the optional argument.
argument-hint: "[since-ref]"
# Lets the bundled script run without a permission prompt, for this turn only.
allowed-tools: Bash(python3 *)
---

## Commits since the last tag
<!-- Claude Code runs the next line first and pastes the script's output here. -->
!`python3 ${CLAUDE_SKILL_DIR}/scripts/group_commits.py`

## Steps
<!-- $ARGUMENTS becomes whatever the user typed after /drafting-changelogs. -->
1. If this text is not empty: "$ARGUMENTS", treat it as a git ref and run
   `python3 ${CLAUDE_SKILL_DIR}/scripts/group_commits.py $ARGUMENTS`
   instead of using the list above.
2. Rewrite each entry as one plain sentence a customer would understand.
3. Drop the Other section unless an entry changes what users see.
4. Keep the short commit hashes so reviewers can trace every line.
```

4. Run it. Open a tagged repository, start Claude Code with claude, and type /drafting-changelogs, or /drafting-changelogs v1.0 to pick the starting point. Asking "what changed since the last release?" should trigger it too. On a test repository with five commits after a v1.0 tag, the script hands Claude this before any rewriting happens:

#### `Pre-Injected Context — Commits Output`
```python
Commits since v1.0:

## New
- PDF export from Share menu (2682589)

## Improved
- faster chart rendering (6885426)

## Fixed
- crash when saving empty dashboard (13c4796)

## Documentation
- update README (553b27a)

## Other
- bump deps (9fee148)
```

Claude Code picks up edits to the skill without a restart. To share it with your team, move the folder into the repository's .claude/skills/ directory and commit it. Cloud sessions and Cowork do not read ~/.claude/skills/, so a skill you want there must be enabled on your claude.ai account or committed to the repository.

### Tutorial 3: Skills via Claude API

1. Install the SDK and set your key.

#### `Terminal — Environment Setup`
```python
# Install the official Python SDK
pip install anthropic
# The client reads the key from this environment variable
export ANTHROPIC_API_KEY="your-key-here"
```

2. Save and run this script. It follows the patterns in Anthropic's guide to using Agent Skills with the API, where the Skills API no longer needs a beta header.

#### `run_skills_api.py`
```python
import anthropic
from anthropic.lib import files_from_dir

# The client reads your key from the ANTHROPIC_API_KEY environment variable.
client = anthropic.Anthropic()

# 1. Upload the meeting-notes folder from Tutorial 1.
#    files_from_dir() collects every file and keeps the folder name, so
#    SKILL.md sits at the top of one enclosing folder, as the API requires.
skill = client.skills.create(
    display_name="Meeting Notes",
    files=files_from_dir("meeting-notes"),
)
print("Skill ID:", skill.id)
print("Version:", skill.latest_version_id)

raw_notes = """Planning call, 21 Oct. Attendees: Priya, Sam, Lee.
Agreed to ship PDF export in the November release.
Sam to update the pricing page. Lee to ask legal about the EU terms."""

# 2. Attach both skills to the code execution container. The custom skill
#    decides the layout; Anthropic's docx skill builds the Word file.
response = client.messages.create(
    model="claude-opus",
    max_tokens=4096,
    container={
        "skills": [
            # Pin the version you just uploaded; use "latest" only while testing.
            {"type": "custom", "skill_id": skill.id, "version": skill.latest_version_id},
            {"type": "anthropic", "skill_id": "docx", "version": "latest"},
        ]
    },
    # Skills run inside the code execution container, so this tool is required.
    tools=[{"type": "code_execution_20250825", "name": "code_execution"}],
    messages=[{
        "role": "user",
        "content": f"Turn these notes into meeting minutes as a Word document:

{raw_notes}",
    }],
)

# 3. Print Claude's text reply.
for block in response.content:
    if block.type == "text":
        print(block.text)

# 4. Files created in the container come back as file IDs inside code
#    execution results. Download each one with the Files API.
for block in response.content:
    if block.type == "bash_code_execution_tool_result":
        result = block.content
        if result.type == "bash_code_execution_result":
            for output in result.content:
                meta = client.files.retrieve_metadata(file_id=output.file_id)
                client.files.download(file_id=output.file_id).write_to_file(meta.filename)
                print("Saved", meta.filename)
```

3. Reuse the skill instead of re-uploading it. Running the script twice creates two skills, so store skill.id after the first upload and reference it from then on. When you change the files, upload a new version with client.skills.versions.create(skill_id=..., files=files_from_dir("meeting-notes")); each version is a full snapshot of the folder. Uploaded skills are visible to everyone in the API workspace, and they run in a container with no network access, so any packages a script needs must already be installed there.

---

## 5. Existing Skills Catalog

### Anthropic's Built-In Document Skills

Four pre-built production skills power Claude's file creation capabilities: pptx, xlsx, docx and pdf. They run without setup across the Claude web and desktop apps, the Claude API, Claude Platform on AWS, and Microsoft Foundry.

### Public anthropics/skills Repo

The anthropics/skills repository on GitHub held 19 skills as of October 2026 and had passed 180,000 stars. Most are Apache 2.0 licensed; the document skills are source-available. In Claude Code, run /plugin marketplace add anthropics/skills to install them.

| Skill Name | Core Capability & Scope |
| :--- | :--- |
| academy-guide | Recommends tailored Claude Academy courses based on project objectives |
| algorithmic-art | Interactive generative art and mathematical visualizations in p5.js |
| brand-guidelines | Applies Anthropic's official brand palettes, typography, and spacing |
| canvas-design | Editorial posters, infographics, and visual art exported as high-res PNG or PDF |
| claude-api | Complete Claude API, SDK methods, and streaming endpoints reference |
| discernment-nudge | Adds structured Socratic follow-up questions for critical evaluation |
| doc-coauthoring | Iterative co-writing for technical specs, product proposals, and decision RFCs |
| docx, pdf, pptx, xlsx | Production document parsers, layout engines, and spreadsheet calculators |
| frontend-design | Distinctive UI design, component systems, and responsive typography |
| internal-comms | Internal company memos, announcements, and leadership communications |
| mcp-builder | Scaffolds Model Context Protocol (MCP) servers with schema validation |
| skill-creator | Interactive designer that creates, evaluates, and benchmarks agent skills |
| slack-gif-creator | Generates animated GIFs precisely formatted and compressed for Slack |
| theme-factory | Ten curated design themes for presentation slides, documents, and web pages |
| web-artifacts-builder | Modern React, Tailwind CSS, and shadcn/ui interactive artifacts |
| webapp-testing | Automated Playwright end-to-end testing for local and deployed web apps |

### Skills Bundled With Claude Code

Claude Code ships native skills including /doctor, /code-review, /debug, /batch, /loop and /claude-api. Three work together in an interactive feedback loop: /run launches your app, /verify confirms a code change against the running app, and /run-skill-generator saves the verified build-and-launch recipe as a new skill.

### Partner Skills & Platforms

The Claude blog post on organization skills added a partner directory in December 2025, naming Notion, Canva, Figma, Atlassian, Cloudflare, Sentry, Vercel and Zapier, which Team and Enterprise admins can provision company-wide. The Help Center article on what skills are explains how to browse it under Customize > Skills.

The Agent Skills open standard at agentskills.io lets the same folder work across ecosystem providers. OpenAI's guide to building skills for ChatGPT and Codex builds on it, and GitHub's Copilot documentation on agent skills lists support in Copilot CLI, cloud agent, and VS Code agent mode. For maximum cross-platform portability, stick strictly to the standard's frontmatter fields rather than Claude Code extensions such as context: fork.

---

## 6. Skill Security & Safety

---

## 7. Automation Economics & Impact

A model with a skill library behaves fundamentally differently from a model alone. The foundation model supplies raw reasoning, while each skill supplies one organization's documented, repeatable way of doing one specific job—written once and reused by every human and autonomous agent that accesses it. Anthropic's engineers wrote at launch that they anticipate agents will soon create, edit, benchmark, and evaluate skills entirely autonomously.

Composability gives skills room for automations that nobody had to design in advance. An organization with modular skills for its brand guidelines, its data warehouse schema, and its financial report layout can generate an entirely new report type simply by prompting Claude, which fuses the three skills seamlessly:

The usage data shows the macroeconomic direction. Anthropic's September 2025 Economic Index report found that directive conversations, where users hand Claude a whole autonomous task, rose from 27% to 39% of Claude.ai usage in under a year, and that 77% of business API transcripts exhibited structured automation patterns. The World Economic Forum's Future of Jobs Report 2025 projects 92 million jobs displaced and 170 million created by 2030, with 41% of surveyed employers planning workforce reductions driven by AI automation.

So far, tasks are being automated significantly faster than whole jobs. In the June 2026 Economic Index survey of Claude users, over a third of roughly 9,700 respondents expected AI to handle most of their daily tasks within a year, yet only 10% rated losing their own job as likely. Intriguingly, over a third put a junior colleague's odds of job loss above 60%. While the sample skews technical and is self-reported, the signal is unmistakable.

Skills turn the automation of a documented procedure from a multi-month engineering project into something a domain expert can write on a Friday afternoon. The most exposed roles are those whose steps fit cleanly in a SKILL.md, and the professionals who write, test, and maintain those repositories hold the procedural know-how that enterprise agents depend on. To start, turn the instruction you paste into Claude most often into a skill, and explore skill-creator and pdf in Anthropic's repository today.

---

## 8. References

1. Anthropic Engineering: Equipping agents for the real world with Agent Skills
2. Claude: Introducing Agent Skills
3. Claude Platform Docs: Agent Skills overview
4. Claude Platform Docs: Skill authoring best practices
5. Claude Code Docs: Extend Claude with skills
6. Claude Code Docs: Skills in the Agent SDK
7. Claude Help Center: What are skills?
8. Claude Help Center: How to create custom skills
9. Claude Help Center: Using skills in Claude
10. Claude Platform Docs: Using Agent Skills with the API
11. Claude Blog: Skills for organizations and partners
12. GitHub: anthropics/skills repository
13. GitHub: skill-creator plugin
14. Agent Skills open standard
15. Agent Skills format specification
16. OpenAI: Build skills
17. GitHub Docs: About agent skills
18. Anthropic Economic Index, September 2025
19. Anthropic Economic Index, June 2026
20. World Economic Forum: Future of Jobs Report 2025

---

### Ready to Orchestrate Production Claude AI Agent Teams?
- [Schedule Strategy Consultation via Email →](collaboration.html)
- [View Advisory Packages & Rates](pricing.html)
- [Explore Claude Agents Suite](claude-101.html)
