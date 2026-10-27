# Claude and Microsoft 365: Office Work, Delegated
## AI Agent Architecture & Orchestration - Thomas Cherickal

> **// PROVIDER DEEP DIVE**  
> **Cowork's capabilities became part of Claude on 16 September 2026. Here is how Claude operates in Word, Excel, PowerPoint and Outlook, what it can build, how it plugs into enterprise systems worldwide, and where adoption stands.By Thomas Cherickal, Generative AI Consultant · The Digital Futurist · 22 October 2026 · thomascherickal.com · thomascherickal.github.io**

---

## Table of Contents

Cowork's capabilities became part of Claude on 16 September 2026 (Anthropic merge announcement).

Cowork started as a separate agent mode in January 2026. Since September it lives inside every Claude conversation. Here is what it does in Word, Excel, PowerPoint and Outlook, what it can build, how it plugs into enterprise systems worldwide, and where adoption stands.

1. What Claude Cowork is
2. Microsoft 365 in practice
3. Dashboards, long decks and long documents
4. Enterprise integrations
5. Claude in enterprises around the world
6. Why it suits office automation
7. Adoption in numbers
8. References

---

## What Claude Cowork is

Anthropic released Cowork on 12 January 2026 as a research preview for Max subscribers on macOS. It applies the agent design behind Claude Code to ordinary office tasks. You give Claude access to a folder and describe the outcome you want. Claude plans the steps, carries them out, and asks before significant actions. Anthropic's launch examples included turning receipt photos into an expense spreadsheet and drafting a report from scattered notes.

On 16 September 2026, Anthropic merged Cowork and chat into one Claude. There is no mode to choose anymore: Claude decides whether a request needs a quick answer or a multi-step task, and work keeps running after you close your laptop. Pro and Max plans get the merged experience first, Team and Free follow, and Enterprise admins receive at least 30 days' notice. Connectors, skills and projects carry over, and memory has been shared across chat and Cowork since 25 August.

---

## Microsoft 365 in practice

Claude for Excel, PowerPoint and Word is generally available on paid plans for Windows and macOS, and Claude for Outlook is in beta. One conversation runs across all four apps, so context set in one app carries into the next.

- Excel: ask about any cell, change assumptions without breaking formulas, or build a multi-tab model with working formulas from a Word brief.
- Word: edits arrive as tracked changes, Claude answers comment threads, and new sections follow your heading styles and numbering.
- PowerPoint: slides are built on your template and slide master, with native charts and diagrams.
- Outlook: triage the inbox, draft replies and find meeting times across calendars. Drafts open in the normal compose window and nothing goes out until you press Send.

Once a process works, you can save it as a skill and the whole team runs it the same way in all four apps. The Microsoft 365 connector extends Claude's reach to Outlook, OneDrive, SharePoint and Teams. Microsoft has adopted the approach too: its Copilot Cowork agent, built with Anthropic on the technology behind Claude Cowork, joined Microsoft 365 Copilot in March 2026 for customers in Microsoft's Frontier program.

---

## Dashboards, long decks and long documents

### Dashboards

Claude Dashboards entered beta on 8 October 2026 on Pro, Max, Team and Enterprise plans. You describe the dashboard in a conversation and Claude builds it as a live artifact: a web page plus one or more datasets. Data can come from file extracts or from live connections through MCP connectors, including Microsoft 365, Snowflake and Databricks. On a live dashboard, the refresh button reruns the queries with each viewer's own permissions. Dashboards stay private until shared, and viewers can comment. Single datasets are capped at 8 MB, so very large tables still belong in a warehouse.

### Presentations of 50 slides

Claude has three routes to a deck. Claude Slides, in beta on paid plans, drafts a deck you edit in place, present from Claude, or export to PowerPoint or PDF. Claude for PowerPoint works inside your file using its masters, layouts and fonts. File creation writes a .pptx in code. Anthropic publishes no slide cap for any route, and created files can reach 30 MB. A 50-slide deck is built the way a person would build it: outline first, then slides in batches, with charts drawn from your Excel data.

### Word guides approaching 200 pages

Current Claude models read up to 1 million tokens of context and write up to 128,000 tokens in one response, or 300,000 through the Batch API beta. A 200-page guide runs to roughly 80,000 to 100,000 words. Claude writes it chapter by chapter, holding the outline, heading styles and numbering steady across the whole file, and can deliver it as a Word file, a Claude Docs document or tracked changes in your existing draft.

---

## Enterprise integrations

Connectors built on the open Model Context Protocol let Claude read and act inside other business tools. A February 2026 update added Google Calendar, Drive and Gmail, DocuSign, Apollo, Clay, Outreach, Similarweb, MSCI, LegalZoom, FactSet, WordPress and Harvey. Partners including Slack by Salesforce, LSEG and S&P Global built plugins of their own. Claude can update Salesforce records, and Claude in Chrome covers web tools that have no connector.

For IT teams, Claude for Microsoft 365 can connect through Microsoft Foundry, Amazon Bedrock or Google Cloud Vertex AI credentials, so companies keep their existing cloud contracts and compliance setup. Admins can run private plugin marketplaces and stream usage data through OpenTelemetry into the monitoring tools they already use.

---

## Claude in enterprises around the world

### Regional deployment

Anthropic offers regional processing through the three cloud platforms. Its regional compliance page names Japan, South Korea, Singapore, India and Australia in Asia-Pacific, and says GDPR processing stays within the EU and EEA. Certifications include SOC 2 Type 2, ISO/IEC 27001, 27017 and 27018, and CSA STAR, with HIPAA support and a Data Processing Addendum.

### Offices and regional growth

In Europe, Anthropic added Paris and Munich in January 2026 to London, Dublin and Zurich, and said EMEA run-rate revenue grew more than ninefold in a year. In Asia-Pacific it runs offices in Tokyo, Bengaluru and Seoul. Anthropic reported APAC run-rate revenue up more than tenfold in a year, called India Claude's second-largest market, and placed Korea in the top five for total and per-capita use. SK Telecom built a customer service model with Claude, and the Korean legal firm Law&Company says its Claude-based assistant nearly doubled lawyer efficiency.

### Consulting and IT services partners

Large services firms carry Claude into client companies. Deloitte rolled Claude out to more than 470,000 people across its global network and is certifying 15,000 professionals. Cognizant opened access to about 350,000 associates. Accenture formed an Anthropic Business Group and is training 30,000 professionals. Infosys built Claude into its Topaz platform, and PwC and KPMG have their own collaborations. In March 2026 Anthropic launched the Claude Partner Network with $100 million committed for 2026.

---

## Why it suits office automation

Claude works inside the files people already use instead of asking them to move to a new app. Changes are visible before they land. And it plugs into the cloud platforms enterprises already pay for. Deloitte, Bain, ServiceNow and Citadel appear as customers on Anthropic's Microsoft 365 page; ServiceNow says uptake is fast internally because Claude does the work inside Excel rather than shuttling content between tools.

There are limits. Anthropic warned at launch that Cowork can take destructive actions, such as deleting files, when instructions are unclear, and it flagged prompt injection as a risk. In January, PromptArmor researchers showed that a booby-trapped document could make a Cowork session upload a user's file to an attacker. Keep folder and connector permissions narrow.

---

## Adoption in numbers

Anthropic does not publish Cowork-only user numbers, so the figures below cover all Claude products.

Anthropic's Economic Index tracks Claude usage across 121 countries, and adoption is uneven. In February 2026, the top 20 countries by per-capita use accounted for 48% of population-adjusted usage, up from 45% in November 2025, and usage still tracks GDP per capita. Australia ranked eleventh, with 1.6% of global Claude.ai traffic. Richer countries lean toward work and personal use; lower-income countries lean toward coursework.

For a team already on Microsoft 365, starting is simple: install the add-in from Microsoft Marketplace, connect the tools you use daily, and hand Claude one recurring task.

---

## References

1. Fortune: Anthropic Claude Cowork AI Agent (Jan 2026)
2. ADTmag: Anthropic Expands Claude Computer Agent with Cowork (Jan 2026)
3. Unite.AI: Claude Cowork Evolution (Sep 2026)
4. 9to5Mac: Anthropic Merging Claude Cowork with Chat (Sep 2026)
5. SQ Magazine: Anthropic Merges Claude Chat and Cowork into One Interface
6. Strategized.ai: Cowork is Now Claude
7. Anthropic: Claude for Microsoft 365 Enterprise Hub
8. The New Stack: Claude in Word, Excel, PowerPoint and Outlook
9. Anthropic: Cowork Enterprise Admin Guide
10. NobleProg: Enterprise Claude for Productivity
11. Yahoo Finance: Microsoft and Anthropic Team Up on Claude Cowork in M365 (Mar 2026)
12. Thurrott: Microsoft 365 Copilot Cowork Partnership
13. Tabular Editor: What are Claude Dashboards
14. eesel: Claude Slides Enterprise Review
15. AI Codex: Claude Docs and Slides Workflows
16. Claude Platform Docs: Model Architecture & Context Limits
17. eCommerceNews NZ: Anthropic Boosts Cowork with Plugins and Admin Tools
18. Anthropic: Get Started in Claude Cowork in Three Steps
19. Anthropic: Regional Compliance & Data Residency
20. GCN: Anthropic Expands European Presence with New Offices (Jan 2026)
21. Anthropic: Seoul Becomes Third Office in Asia-Pacific
22. Outlook Business: Anthropic Opens Bengaluru Office & Unveils Major Partnerships
23. Anthropic: Deloitte Network Global Alliance
24. The Next Web: Anthropic Commits $100M to Claude Partner Network
25. HFS Research: Anthropic and IT Services Enterprise Transformations
26. Spotdev: Claude Cowork Merge & Enterprise AI Policy
27. Reuters via Yahoo Finance: Anthropic Enterprise Run Rate Growth
28. Bloomberg via Techleap: Anthropic Revenue & Cloud Partnerships
29. Sovereign Magazine: Anthropic $65 Billion Run Rate Analysis
30. PPC Land: Experienced Claude Users Analysis
31. Anthropic Research: How Australia Uses Claude
32. Anthropic Economic Index: Global Enterprise Adoption (Jan 2026)
33. Morocco World News: Global AI Adoption Patterns (Jul 2026)

---

### Ready to Orchestrate Production Claude AI Agent Teams?
- [Schedule Strategy Consultation via Email →](collaboration.html)
- [View Advisory Packages & Rates](pricing.html)
- [Explore Claude Agents Suite](claude-101.html)
