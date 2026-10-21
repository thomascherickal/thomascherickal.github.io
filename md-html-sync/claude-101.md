# Claude 101: A Detailed Overview
## AI Agent Architecture & Orchestration — Thomas Cherickal

> **// PROVIDER DEEP DIVE**  
> **What each model is for, how the five effort levels work, and what Claude agents have done in recent weeks.By Thomas Cherickal, Generative AI Consultant · The Digital Futurist · thomascherickal.com · thomascherickal.github.io**

---

## Table of contents

Anthropic replaced its entire public model lineup between September 1 and October 7, 2026. Five models are current, and each one can run at five effort levels.

Most office work today happens on a computer: email, spreadsheets, documents, browsers, code editors and business software. Claude can now operate those tools. It drives a web browser, works inside Excel, PowerPoint, Word and Outlook, writes and runs code, and keeps working for hours without a person at the keyboard. That makes this generation of Claude a revolution for the whole working world, not only for programmers. In principle, any task done on a digital computer is now a task an AI agent can attempt, and the share it can finish rises with each release.

1. The five models
2. The five effort levels
3. How Claude agents are changing work
4. Claude at the computer
5. How much office work Claude can already do
6. What it means for people
7. Recent accomplishments
8. Risks and limits
9. References

---

## The five models

Claude Fable 5.1 and Claude Mythos 5.1 are the same model with different safeguards. Fable 5.1 is generally available and blocks or limits some cybersecurity and biology work. Mythos 5.1 relaxes those limits and is open only to vetted cyberdefenders and life scientists.

The Claude 5.5 family covers everyday use:

Opus 5.5 matches Mythos 5.1 in biology and cybersecurity, so Anthropic ships it with safeguards similar to Fable 5.1's.

---

## The five effort levels

Effort sets how many tokens Claude spends on a response, including thinking, text and tool calls. Anthropic's effort documentation calls it a behavioral signal rather than a strict budget, so Claude still thinks on hard problems at low settings, only less.

The levels were recalibrated for this generation, so Anthropic advises testing them on your own evaluations. Developers can also change effort mid-conversation without losing the prompt cache, a beta feature.

---

## How Claude agents are changing work

Anthropic's June 2026 Economic Index report says Claude models now work on their own for hours through Claude Code and the Claude app. In an earlier Anthropic study of Claude Code, 79% of conversations were automation, where Claude did the task itself, against 49% on Claude.ai. Inside Anthropic, Claude wrote more than 80% of the code merged into production in May 2026.

A common setup puts an orchestrator on Opus 5.5 or Fable 5.1 at high or xhigh effort and hands narrow jobs to Haiku 5.5 subagents at low effort. The orchestrator keeps the plan and checks results while the subagents search, run code and read documents in parallel.

On October 21, Anthropic announced Claude Duet, a research preview for US Pro and Max subscribers. Duet runs on its own cloud computer, tracks email and calendars, and keeps working after you close the app. It asks before it sends messages, makes payments, signs documents or deletes anything.

---

## Claude at the computer

The clearest test of whether an AI can do office work is whether it can use a computer the way a person does. OSWorld gives an agent 369 real tasks inside a live virtual machine, in apps such as LibreOffice, GIMP, VS Code, Chrome and Thunderbird, and checks the final state of the system. Humans score 72.36% on it. On the OSWorld-Verified leaderboard, Claude Fable 5 scored 86.0% and Claude Opus 5 scored 83.4%. Anthropic reports 81.8% for Opus 5.5 on its newer OSWorld 2.1 test, and 72.4% for Haiku 5.5 on an offline subset of the same test, up from 15.7% for Haiku 4.5.

Claude reaches everyday software through several products. Claude in Chrome became generally available on August 26, 2026: it reads pages, clicks, types and fills in forms with the user's own logins, and a safety classifier checks each action against the request. Claude for Excel, PowerPoint and Word is generally available and keeps one conversation going across all three. Claude for Outlook, in public beta, triages the inbox, drafts replies and checks calendars, and the user reviews everything before it is sent.

The length of work an agent can carry also keeps growing. The evaluation group METR measured Claude Opus 4.6 at a time horizon of about 14.5 hours: on well-defined software, machine learning and security tasks that take a skilled person that long, it succeeded about half the time. The confidence interval runs from 6 to 98 hours, and the test does not cover the context, judgment and teamwork that fill a real job.

---

## How much office work Claude can already do

GDPval-AA scores models on real work products from 44 occupations across nine industries. Opus 5.5 leads it with an Elo of 1846, ahead of Fable 5.1 at 1735 and OpenAI's GPT-6 Astra at 1542, and Sonnet 5.5 scores nearly level with Opus 5.5. On AutomationBench, a test of business workflows, Opus 5.5 scores 40.0%, just behind GPT-6 Astra at 41.4%. Most of those workflows still fail, but the best models now finish a large minority of them.

Anthropic's January 2026 Economic Index found that 49% of jobs now see Claude used for at least a quarter of their tasks, up from 36% in its earlier analysis. It estimated speedups of about 9x on tasks that need a high-school education and about 12x on tasks that need a college degree. Success rates temper those numbers: Claude completed 67% of tasks on Claude.ai and 49% through the API, and the 1.8-point productivity estimate falls to roughly 1.0 to 1.2 points once failures are counted.

Customer reports show what this looks like inside companies. At Stripe, one Opus 5.5 session coordinated a dozen others through a multi-day rebase of 40 stacked pull requests, and all of them passed CI. Chicago Trading Company says the model diagnosed and fixed a backend bug overnight, with the fix passing the test suite by morning. Hebbia reports that on finance tasks graded against expert rubrics, Opus 5.5 met 86.6% of the criteria, against 60.3% for Opus 5.

Adoption is already broad. Anthropic had more than 300,000 business customers by October 2025, and by April 2026 more than 1,000 customers were spending over $1 million a year, according to Sacra's estimates.

---

## What it means for people

If computers can do much of the work done on computers, the effects reach almost everyone with a desk job. Anthropic CEO Dario Amodei warned in May 2025 that AI could eliminate roughly half of entry-level white-collar jobs within a few years. By May 2026 he was citing the Jevons paradox: if AI automates 90% of a job, people may do the remaining 10%, and that remaining work can grow to fill the role. He also said AI is moving faster than earlier technologies, which could make the transition disruptive.

Early labor data shows the pressure landing on new graduates first. A Stanford Digital Economy Lab update from August 2026 found that workers aged 22 to 25 in the most AI-exposed occupations were about 19% below expected employment, up from a 15% gap a year earlier, mostly because companies hired fewer of them. Experienced workers in the same occupations showed no similar decline.

The same tools reach people outside offices. Anthropic and Teach For All are training educators in 63 countries to use Claude, in a network that serves more than 1.5 million students. The science results in the next section show agents running research projects for days with little supervision, in biology, physics, cryptography and quantum computing.

---

## Recent accomplishments

### Security

- Glasswing. In its first month, Mythos Preview flagged 23,019 vulnerabilities in open-source projects, 6,202 of them rated high or critical. Mozilla fixed 271 Firefox vulnerabilities in one release, ten times its count with Claude Opus 4.6.
- Early October. Anthropic opened a free OSS Scanner for maintainers of widely used projects. As of October 2, human reviewers had confirmed 5,674 of the 6,123 findings they checked, but only 516 had been patched upstream.
- October 16. Claude worked out that the seeds for the NIST elliptic curves P-192 through P-521 came from the C library's lrand48 generator, seeded with a Unix timestamp. It took under half an hour. Anthropic says this is not a weakness and makes backdoor theories less plausible.

### Science

- September 1. Fable 5.1 scored 52.6% on Terminal-Bench-Science, up from 24.7% for Fable 5, and built a sharper elevation map of a third of Venus from NASA Magellan radar data. At launch, Opus 5.5 led the Artificial Analysis and Vals AI indices.
- September 23. About 950 Claude agents searched for 21 hours and flagged an uncharacterized enzyme system the team calls ART. Its function is still unknown. Washington University microbiologist Kevin Blake told Al Jazeera that nothing suggests it rivals CRISPR as a technology.
- September 25. Fable 5.1 computed a nine-loop amplitude in N=4 super Yang-Mills, one loop past the previous record, with little human oversight. Physicist Lance Dixon checked it, and a group using GPT-6 reached most of the same result around then.
- October 16. Claude Mythos 5 designed quantum circuits that beat published versions, including an elliptic-curve point addition circuit with 17x fewer gates and a decoder with 12x fewer gates than Google's.
- October 20. An internal Claude research model spent about 64 hours, mostly unattended, designing proteins against scorpion neurotoxins. Lab tests confirmed 17 of 211 designs as binders. None has been tested yet for neutralizing venom.

---

## Risks and limits

A model that finds vulnerabilities can also help exploit them. That is why Mythos 5.1 is limited to vetted users and Opus 5.5 ships with safeguards. Anthropic says Glasswing is now held back by how fast vulnerabilities can be verified, disclosed and patched. Several science results above are company announcements that outside researchers are still checking.

Benchmarks also overstate readiness for whole jobs. A 40% score on business workflows means most of them still fail, and time-horizon tests cover well-defined tasks rather than the meetings, judgment calls and coordination that fill a working day. The job losses so far fall mainly on entry-level hiring, which raises the question of how people will gain the experience that still protects senior workers.

To get started, Sonnet 5.5 at its default effort handles most work. Move to Opus 5.5 or Fable 5.1 when a task needs more judgment, and use Haiku 5.5 for subagents and high-volume calls.

---

## References

1. Introducing Claude Opus 5.5 - Anthropic
2. Introducing Claude Sonnet 5.5 - Anthropic
3. Introducing Claude Haiku 5.5 - Anthropic
4. Introducing Claude Fable 5.1 and Claude Mythos 5.1 - Anthropic
5. Claude Mythos product page - Anthropic
6. Claude Mythos 5.1 model overview - Claude Platform Docs
7. Effort parameter guide - Claude Platform Docs
8. Anthropic Economic Index Report: Cadences - Anthropic
9. Economic Index: AI's Impact on Software Development - Anthropic
10. Anthropic says 80% of its new production code is authored by Claude - VentureBeat
11. Anthropic takes on ChatGPT Dots and Meta Muse with Claude Duet - 9to5Mac
12. Project Glasswing: An Initial Update - Anthropic
13. Anthropic's OSS Scanner - The New Stack
14. Claude reveals ECDSA curve origins - Anthropic
15. Claude discovers a novel enzyme system - Anthropic
16. AI model Claude discovers CRISPR-like enzyme system - Al Jazeera
17. Claude computes a nine-loop amplitude in N=4 super-Yang-Mills - Anthropic
18. Designing more efficient quantum circuits with Claude - Anthropic
19. How Claude designed protein binders against scorpion neurotoxins - Anthropic
20. Claude Fills Out Its Model Family - The Batch, DeepLearning.AI
21. Anthropic Economic Index Report, January 2026 - Anthropic
22. Models ranked by OSWorld - Developers Digest
23. OSWorld: What the 72% human baseline measures - BuildMVPFast
24. Claude Opus 4.6 METR time horizon - Fanatical Futurist
25. Claude in Chrome is generally available - Claude
26. Collaborate with Claude across Excel, PowerPoint, Word and Outlook - Claude
27. Dario Amodei and the Jevons paradox - Fortune
28. Canaries in the Coal Mine, August 2026 update - summary by Brian Heger
29. Anthropic and Teach For All - Anthropic
30. Anthropic revenue and customers - Sacra

---

### Ready to Orchestrate Production Claude AI Agent Teams?
- [Schedule Strategy Consultation via Email →](collaboration.html)
- [View Advisory Packages & Rates](pricing.html)
- [Explore Claude Agents Suite](claude-101.html)
