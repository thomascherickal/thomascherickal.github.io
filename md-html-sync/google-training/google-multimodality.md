# Google Multimodality: Creative Generative Ecosystem
## AI Agent Architecture & Orchestration - Thomas Cherickal

> **// GEMINI MULTIMODALITY & GENERATIVE CREATION**  
> **Five Google products, ten documented case studies, and the model capability that connects them.**

---

## Contents

---

## Introduction

For most of the last decade, an AI model handled one kind of material at a time. A language model read text, a vision model labeled photos, and a speech model transcribed audio. Gemini was built on a different premise, which is that one model should take in text, images, audio and video together and reason across all of them.

That premise now sits underneath a family of Google products that look quite different on the surface. One tool turns a sentence into an app screen, another builds a marketing campaign from a website, and a third edits video through conversation. Two more produce films and photographs. Each of them is a front end for the same multimodal capability.

This article covers five of them: Stitch, Pomelli, Gemini Omni, Flow and Nano Banana. For each product it explains what the tool does, how it connects to Gemini, and two documented episodes that drew public attention. Some of those episodes went truly viral, and others are the best-documented public examples of the product at work. The text says which is which.

Everything below comes from pages that were open and loading when this article was written, and each claim links to its source. Where a source is a secondary write-up of a Google demo, the text says so. Product details change quickly, so the dates attached to each fact matter.

---

## What multimodality means in Gemini

Multimodality means the model is trained on several kinds of data at once, instead of stitching separate models together afterward. A text-only model that calls an image model has to describe the picture in words before it can think about it. A natively multimodal model skips that translation step and keeps the detail of the original image, sound or frame.

The practical effect shows up in the products. Stitch accepts a sketch or screenshot as readily as a sentence, because the underlying model reads both. Pomelli studies a website and a product photo together to learn a brand. Gemini Omni takes any mix of text, images, audio and video as input and returns video.

There is also a pattern in how Google ships these tools. A research model arrives first, then a Labs product wraps it for a specific job, then the model's improvements flow into every tool that uses it. The image model called Nano Banana is the clearest case, since it now powers features inside several of the other products in this article.

Provenance travels with the output. Several of the tools covered here stamp what they make with SynthID, an invisible watermark that Google's apps can later detect. That matters more as generated video and photographs become hard to tell from recordings.

---

## Google Stitch: interface design from a prompt

Google introduced Stitch on May 20, 2025 as a Google Labs experiment that turns text prompts and image inputs into UI designs and front-end code. It ran on the multimodal strengths of Gemini Pro at launch. A user could describe an app in plain language, including the color palette and the experience wanted, or upload a sketch, whiteboard drawing, screenshot or wireframe.

Stitch generates several variants so a designer can compare layouts, components and styles. Refinement happens through chat and theme selectors. When a design is ready, it can be pasted into Figma or exported as front-end code, which makes the output a starting point for existing workflows instead of a dead end.

The product has grown since launch. A hands-on review from Banani describes text-to-UI and image-to-UI modes, multi-screen flows with a generated design system, interactive prototyping, DESIGN.md files, export to Figma and HTML/CSS, and MCP integration with coding environments. The same review lists a free tier with 400 daily design credits and notes that the service is still in beta.

Stitch suits a particular stage of work. It is useful when an idea exists only as a paragraph or a napkin sketch and a team needs something to react to. It is less suited to the final pass, where a designer still decides spacing, hierarchy and brand detail.

### Case study: the Figma sell-off

The most public moment in Stitch's history came in March 2026, when Google presented the tool as a design agent. According to Invezz's market report, the agent offers real-time design critiques and responds to voice input. Google made the announcement on a Tuesday, and Figma shares fell 8% on Wednesday and more than 4% on Thursday.

The same report says Adobe shares fell about 4% over those two days and that Figma was already down about 35% for the year. It ties the move to a wider worry across software stocks that AI tools will absorb work that used to need a dedicated product. Stitch overlapped with Figma's interface design, prototyping and collaboration features, and it was not even monetized yet.

The episode shows how quickly a free Labs tool can change how investors read an incumbent. It does not show that Stitch replaces Figma. Stitch exports into Figma, and the market reaction reflected a fear about direction more than a measured loss of customers.

### Case study: building a Duolingo-style app in one prompt

The Banani review gives a concrete test. The reviewer asked Stitch, running Gemini Pro, for a Duolingo-style coding app called CodeFlow with four screens: a dashboard with a streak counter and three course cards, a lesson screen with multiple-choice questions, a leaderboard, and a profile page. The prompt also fixed a green primary color and 12px rounded corners.

Stitch broke the prompt down, built a design system and generated all four screens while showing its progress. The reviewer estimated that it met about 85% of the requirements. The criticisms were specific: the result looked like a colored wireframe, the bottom navigation and trophy icons were weak, and the buttons were generic.

That mix of speed and rough finish is the honest summary of the product today. A founder can get a coherent four-screen concept in minutes and then bring it to a designer for polish. The review also found that cohesive flows across more than three related screens are hard to generate, and that imported Figma files can arrive with poorly structured layers.

---

## Google Pomelli: a brand-aware marketing studio

Google Labs and Google DeepMind released Pomelli on October 28, 2025 as a tool for small businesses that need on-brand marketing content without a design team. It launched as a public beta in the United States, Canada, Australia and New Zealand, in English. The workflow has three steps that Google describes plainly.

First, Pomelli reads a business's website and images to build a profile Google calls Business DNA, covering tone, fonts, imagery and colors. Second, it proposes campaign ideas, or the owner can type a prompt. Third, the owner edits the generated text and images and downloads the results.

The multimodal part is the first step. The tool has to read copy, layout, photography and color together and decide what they add up to. Every later asset is generated against that profile, which is why a campaign for a bakery looks different from one for a law firm.

Pomelli is free to try through Google Labs, according to the Photoshoot announcement. That price removes the usual barrier for a one-person shop that has never hired a designer. The cost shows up elsewhere, in the time it takes to check that every generated claim and image matches what the business actually sells.

### Case study: a cookie shop gets a night-out campaign

A MindStudio walkthrough tests Pomelli on a cookie shop concept. The reviewer asked for a campaign aimed at people on a night out, and Pomelli returned concepts including one that pitched fresh cookies delivered across London until the early hours, and another called "Drop the Box," which built a reveal around opening the box.

The review draws a useful lesson about audience framing. A vague brief produced generic ideas, and a more specific audience produced more specific concepts. The example is a hands-on test and not a viral post, but it is one of the clearest public records of what the campaign step produces.

The same review lists limits. Text inside animated videos usually gets garbled, and the suggested workaround is to animate without text and add the words in a video editor. Animation takes no prompts for camera or text movement, and the Animate button appears only on 9:16 story-format images.

### Case study: Photoshoot turns product shots into studio images

In February 2026 Google added a feature called Photoshoot, described in a February 19 launch post. A user picks a product photo, chooses a template or lets Pomelli suggest one, and receives studio or lifestyle images built around the brand's Business DNA. Google says the feature runs on its Nano Banana image generation.

The MindStudio review lists four templates: Studio, Ingredient, In Use and Contextual. Pomelli picks a template automatically based on the product type, and a text instruction such as changing the background to pink works as an edit. Finished images can be downloaded in bulk or added back to Business DNA for later campaigns.

For a shop owner with a phone photo on a kitchen counter, this replaces a studio session. The launch also showed Pomelli borrowing a stronger model from another Google product, a pattern that repeats across this article. The post does not claim a result in sales or engagement, and neither does this article.

---

## Gemini Omni: any input, one video output

Google announced Gemini Omni at Google I/O on May 19, 2026, and the first model is called Omni Flash. Google DeepMind describes it as a unified multimodal generation model. It accepts any combination of text, images, audio and video and produces a single video, with synchronized audio generated alongside it.

Editing is the core idea. A user can change a character, object, environment, camera angle or style over several turns, and Google says the model keeps earlier changes and scene physics intact. According to one summary, Omni Flash makes clips of up to 10 seconds at up to 1080p, and it is available in the Gemini app, Flow, YouTube Shorts and the YouTube Create app.

A later version, Omni Flash, appears on Replicate's model page, which lists text, optional starting image, end frame and reference images as inputs. The page lists outputs from 360p drafts up to 4K in 16:9 or 9:16, with every video carrying an invisible SynthID watermark. Since the two sources describe different versions, the resolution ceilings differ, and readers should check the current spec before building on either.

Access is tied to subscriptions. The Opus summary says Omni is included with Google AI Plus, Pro and Ultra plans and is free for YouTube Shorts and Create users. It also notes that the developer API was not open at the time, that API pricing had not been announced, and that a higher Omni Pro tier has no confirmed date.

### Case study: "Nano Banana, but for video"

The launch framing did much of the work. Digital Camera World reported Google's own description of Omni as "like Nano Banana, but for video," with an aim of keeping a person's likeness intact across edits. Shared demos included a skater video combined with stickers to produce moving comic-book effects, and one avatar rendered in several different styles.

Reaction was mixed, and the article is careful to say the examples were not widely viral. Critics worried about adding to low-quality AI content on social media. One Reddit user said making four videos used an entire Pro plan usage window, and another said the content filters were very tight.

A leak trail came first. WaveSpeed's timeline records a UI string on May 3, new samples pulled from the Gemini mobile app on May 11, and the official launch on May 19 and 20. Early testers reported that two prompts used 86% of a daily AI Pro quota, which is a useful reminder that video generation is expensive for users as well as for Google.

### Case study: editing a clip by talking to it

The strongest demonstrations were edits. An Atlas Cloud review describes a clip of a stone sculpture edited with the instruction to make the sculpture out of bubbles, after which the composition, lighting and shadows stayed the same while the stone became translucent soap. In another example a hand touches a mirror, and the instruction to make the mirror ripple like liquid spreads ripples outward while the arm takes on a reflective surface.

A third example chains three edits on a violinist: a stage, then a transported environment, then an over-the-shoulder camera angle. The review says the face, posture and instrument grip stayed consistent through all three rounds. Follow-up instructions such as golden-hour lighting or a slower camera push were applied one after another.

The caveats matter. The review states that the examples come from Google's launch demos and secondhand reviews, and that independent benchmark verification is still pending. It also reports a physics bug from DataCamp's testing, in which a trebuchet launched its payload backwards.

---

## Google Flow: a filmmaking workspace

Google presented Flow on May 20, 2025 as an AI filmmaking tool built around its Veo, Imagen and Gemini models. It evolved from the earlier VideoFX experiment in Google Labs. Each model has a job: Veo generates video, Imagen creates images used as building blocks, and Gemini interprets prompts written in everyday language.

Flow organizes work around a few ideas. Ingredients are characters, subjects or scenes that can be reused across clips to keep them consistent. Camera controls direct motion, angles and perspective, and Scenebuilder edits and extends an existing shot. Flow TV, a showcase of Veo clips with their prompts, lets users learn from other people's work.

At launch, Flow was available in the United States to Google AI Pro and Ultra subscribers. Pro included 100 generations per month, and Ultra had the highest limits plus early access to Veo with native audio. The Gemini Omni launch later added the Omni models to the same workspace.

Creators should read the fine print on data. The CNET report notes that Google's Labs privacy notice says human reviewers may read and process Labs interactions, and that data is stored for up to 18 months by default. That matters for anyone feeding unreleased scripts, client footage or private photographs into the tool.

### Case study: 100 million videos in 90 days

The clearest adoption figure comes from CNET's August 2025 report, carried by Yahoo Tech. Google confirmed to CNET that Flow users had generated over 100 million AI videos, about 90 days after the tool debuted at I/O. The report adds that Google had counted over 40 million videos around seven weeks after Veo launched.

The article quotes Elias Roman of Google saying the team was "hustling just to keep up with the demand." It also records the price of entry, with Pro at $20 a month and Ultra at $250 a month, discounted to $125 for three months at the time. The report does not say how many people made those videos.

This is a volume number, and it says nothing about quality or revenue. Still, a hundred million generations in three months from a paid tool suggests that creators were willing to pay for video generation once audio arrived with Veo. It also gives context to the later heavy quota usage that Omni testers reported.

### Case study: Ancestra at the Tribeca Festival

The most ambitious public use of Google's video stack was a short film. Ancestra, directed by Eliza McNitt, premiered at the Tribeca Festival on June 13, 2025. It tells the story of a mother whose newborn has a heart defect and draws on McNitt's own birth.

Google DeepMind and Primordial Soup, the studio founded by director Darren Aronofsky, made the film with a team of more than 200 people. Veo generated sequences that were blended with live-action footage, and a generated newborn was added to a live-action birth scene. Imagen produced the concept art that set the film's look, and Gemini helped write prompts based on descriptions of photos taken on the day McNitt was born.

One precision is worth stating. TheWrap's report says Google gave Primordial Soup access to Flow and Veo, but it does not specify which tools were used on Ancestra, and Google's own account credits Veo and Imagen. The film is therefore a showcase for the Veo and Imagen models behind Flow, and it is the first of three planned films from the partnership.

---

## Nano Banana: the image model behind the rest

Nano Banana is the name for Google's Gemini image generation and editing models. It first appeared anonymously on an evaluation arena in August 2025 and was released publicly on August 26, 2025 as Gemini Flash Image. Its core strengths are editing a photo while keeping a person recognizable and following detailed instructions in natural language.

Two upgrades followed. Nano Banana Pro arrived on November 20, 2025, built on Gemini Pro, with legible text in many languages, 2K and 4K output, blending of up to 14 input images, and grounding in Google Search. Nano Banana 2 launched on February 26, 2026 on Gemini Flash Image and replaced Pro in the Gemini app. It also became the default image model in Search's AI Mode, Lens and Flow.

Transparency features travel with the model. Every generated image carries an invisible SynthID watermark, and users can ask the Gemini app whether an image came from Google AI. A visible sparkle watermark remains on images from free and Google AI Pro users.

Pro also added finer editing controls. The Google announcement lists localized edits, camera angle and focus changes, color grading, and lighting changes such as day to night. The model reached NotebookLM, Google Slides, Vids and Google Ads, so a single image engine now sits inside tools that businesses already use every day.

### Case study: the figurine, Polaroid and decades trends

The viral wave began within weeks of release. Tom's Guide documented the trends that spread fastest: turning a photo into a collectible action figure in a toy box with the user's name on the packaging, creating vintage Polaroid-style images with 1980s film grain, and placing yourself into other decades with period clothing and backgrounds. Other trends put users into Seinfeld scenes, famous paintings and world landmarks.

The same article reports that Google's Josh Woodward said the Gemini app gained more than 10 million new users within days of launch. It also cites 200 million images generated or edited in that period. These are Google's figures as relayed by Tom's Guide, and they are the strongest evidence of a consumer feature driving mass app adoption.

The Mona Lisa prompt turned out hardest for the author, who got two Mona Lisas side by side. That detail is a fair reminder that the trends worked because most prompts were easy, and the failures were funny enough to share as well.

### Case study: the Valentine's Day trends of 2026

Interest did not fade by the following winter. A Tom's Guide piece from February 2026 lists five new trends, including an anywhere date night that places a couple in a hot-air balloon or on a city street they cannot physically visit. Another shrinks people to ant size for dates on a laptop keyboard or a coffee mug.

The most personal trend asks the model to combine a current photo with a childhood photo, producing a scene of an adult comforting their younger self. Others make personalized Pixar-style characters that reflect each person's interests and a comic panel for Galentine's Day. The article advises naming specific textures, such as brushed aluminum keys, to make tiny scenes look believable.

These trends rely on features that arrived after the first release, mainly better consistency across several people and stronger prompt following. They also show where the model gets used most, which is personal storytelling around photos people already own. That use case explains why an image model found a mass audience faster than any of the video or design tools here.

---

## How the five tools connect

Read together, the products form a chain rather than five separate bets. Nano Banana sits at the base as the image engine, and Pomelli's Photoshoot and Flow's default image model both use it. Veo and Imagen sit under Flow, and Gemini Omni is the newer single model that brings generation and editing into one conversation.

Stitch sits slightly apart, since its output is code and layout instead of media. It still follows the same rule that shapes the others, which is that a person describes intent in whatever form is convenient and the model supplies the rest. The input can be a sentence, a sketch, a product photo or a clip, and the output is the artifact that person needed.

A small business could see the chain in a single afternoon. Stitch could produce the screens for a booking app, Pomelli could build the launch images from the shop's photographs, and Omni or Flow could cut a short promotional clip. None of that removes the need for review, but it shrinks the first draft from days to minutes.

Cost and access remain real constraints. Stitch is free but credit-limited, Flow and Omni sit behind paid Google AI plans, and Pomelli launched in four countries. Early testers of Omni reported heavy quota use, and reviewers of Stitch and Pomelli found rough edges that still need a human editor.

---

## Conclusion

Gemini's multimodality is easiest to judge by what people did with it. A free design tool moved a public company's share price. A marketing tool turned a website into a brand profile, and an image model drew more than 10 million new users to an app within days. A film that mixed live action with generated sequences premiered at a major festival.

The evidence also has limits. Several of the episodes above are demos, reviews or company-reported figures, and the Omni editing claims still await independent verification. Quality varies by task, and every product here needs a person who knows what good looks like.

Three things are worth watching over the next year. One is whether Omni Pro ships and removes the clip-length cap. Another is whether the developer API opens so that other products can build on Omni, as they already do with Nano Banana. The last is whether independent testing confirms the editing quality that launch demos showed.

The direction is clear enough to plan around. Tools that accept any input and return finished work are replacing chains of single-purpose software. Anyone who designs, markets, edits video or makes images for a living should try these five with a real project and judge the output against their own standards.

---

## Practice Pricing & Global Purchasing Power Parity

I provide live, remote engineering training and advisory consulting covering Google Multimodal architectures, Google Stitch design automation, Google Pomelli brand synthesis, Google Flow video creation, and Nano Banana diffusion pipelines. Engagements enable research organizations, academic departments, enterprises, and individual professionals to master production multimodal systems, source-grounded creative engines, and automated generative workflows.

All services are governed by our transparent 195-country Purchasing Power Parity (PPP) framework, scaling rates fairly across six economic tiers:

### Individual Service Picks (Pick & Choose)

Clients can also avail of individual modular services from our technical catalog (e.g. Gemini Advanced Training, Claude Agent Orchestration, Local LLMs, or Workflow Automations):

- Business Individual Pick: ($300 / $225 / $150 / $150 / $75 / Custom USD) base for 1 seat, with additional seats at ($300 / $225 / $150 / $150 / $75 / Custom USD) per seat up to a maximum of 10 seats, covering 5 days live engineering + 5 days dedicated support (10 days total coverage, 4 hours/day, remote worldwide, Mon - Fri).
- Personal Individual Pick: ($300 / $225 / $150 / $150 / $75 / Custom USD) strictly for only one seat, covering 5 days live training + 5 days dedicated support (10 days total coverage, 4 hours/day, remote worldwide, Mon - Fri).

### Payment Structure & Trust Guarantees

Every client engagement is safeguarded by strict trust guarantees:

- Zero Upfront Payment: Strictly $0 advance deposit required. We never invoice before delivering actual value.
- 20% in Five Installments: Invoiced in 5 equal milestone installments of 20% each, billed strictly after each milestone block is completed.
- Walkaway Guarantee: If you are not satisfied after the initial training phase, you can walk away owing absolutely nothing.
- Commercial Apache 2.0 License: All custom code, architectures, and high-quality training materials are provided under the Apache 2.0 license, allowing full commercial use, internal deployment, resale, and relicensing with attribution.
- Full Pricing Details: Learn more about country tier multipliers, currency choices, and customized curriculum on our Pricing Page.

### The Six Purchasing Power Parity (PPP) Tiers

To ensure fair global access across all 195 countries, fees are adjusted by national economic indicators in accordance with our Purchasing Power Parity framework:

| Tier | Economic Category | Representative Countries | Parity Multiplier | Business Flagship | Personal Flagship |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Tier 1 | High-Income Baseline | United States, UK, Germany, Canada, Japan, Singapore, Australia | 1.00X | $2,500 USD | $2,500 USD |
| Tier 2 | Upper-Middle Economies | Brazil, Mexico, Malaysia, Turkiye, Thailand, Philippines, Sri Lanka | 0.75X | $1,875 USD | $1,875 USD |
| Tier 3 | Lower-Middle & Low-Income | India, Egypt, Kenya, Nigeria, Pakistan, Bangladesh, Vietnam | 0.50X | $1,($300 / $225 / $150 / $150 / $75 / Custom USD) | $1,($300 / $225 / $150 / $150 / $75 / Custom USD) |
| Tier 4 | Early-Stage Startups | Global tech startups with <$1M external funding | 0.50X | $1,($300 / $225 / $150 / $150 / $75 / Custom USD) | $1,($300 / $225 / $150 / $150 / $75 / Custom USD) |
| Tier 5 | Academia & Non-Profits | Universities, accredited labs, registered charities & 501(c)(3) entities | 0.25X | $625 USD | $625 USD |
| Tier 6 | Subsidized Relief | Crisis-affected regions, developing collectives & grassroots builders | Custom | Subsidized | ($300 / $225 / $150 / $150 / $75 / Custom USD) |

---

## References

1. From idea to app: Introducing Stitch, a new way to design UIs Google for Developers Blog, May 20, 2025
2. Figma stock falls as Google AI Stitch sparks design software fears Invezz, March 2026
3. Google Stitch AI review Banani
4. Create on-brand marketing content for your business with Pomelli Google, October 28, 2025
5. Create studio-quality marketing assets with Photoshoot in Pomelli Google, February 19, 2026
6. Google Pomelli review: features and limitations MindStudio
7. Gemini Omni model overview Opus
8. google/gemini-omni-1.1 Replicate
9. Nano Banana went viral, now Google is bringing the same idea to video with Gemini Omni Digital Camera World, May 20, 2026
10. Gemini Omni demos leaked: what we learned WaveSpeed
11. Gemini Omni feature review: edit video through natural conversation Atlas Cloud
12. Meet Flow: AI-powered filmmaking with Veo Google, May 20, 2025
13. Google's AI filmmaker program Flow CNET via Yahoo Tech, August 18, 2025
14. ANCESTRA: behind the scenes Google DeepMind blog
15. Darren Aronofsky's Primordial Soup partners with Google DeepMind TheWrap, May 20, 2025
16. Nano Banana just broke the internet with these viral trends Tom's Guide, updated September 16, 2025
17. Forget flowers: 5 viral Nano Banana trends reshaping Valentine's Day Tom's Guide, February 12, 2026
18. Google's Nano Banana 2 is a faster version of Nano Banana Pro Engadget, February 26, 2026
19. Nano Banana Pro Google, November 20, 2025

---

### Ready to Orchestrate Production Claude AI Agent Teams?
- [Schedule Strategy Consultation via Email →](contact.html)
- [View Advisory Packages & Rates](pricing.html)
- [Explore Claude Agents Suite](claude-101.html)
