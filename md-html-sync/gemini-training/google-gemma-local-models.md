# Google Gemma Local Models: A Practical Guide
## AI Agent Architecture & Orchestration - Thomas Cherickal

> **// OPEN-WEIGHTS & SOVEREIGN LOCAL INFERENCE**  
> **A practical guide to Google's Gemma open-weight small language models: the current lineup, hardware needs, inference tools, the Hugging Face ecosystem, benchmarks against peers, and the Gemmaverse.**

---

## Contents

---

## Introduction

A capable language model used to mean a rented one. You sent your text to a data center, paid per token, and accepted whatever the provider changed next month. Google's Gemma models offer a different deal: you download the weights once and run them on your own machine.

This article covers what small language models are, what they do well, and which Gemma models exist today. It lists the hardware each model needs, the tools that run them, and the Hugging Face ecosystem around them. It also explains why the weights are free, how Gemma benchmarks against similar open models, and what the community has built with them.

---

## What small language models are

A small language model, or SLM, is a language model small enough to run on ordinary hardware such as a phone, a laptop or a single GPU. No official cutoff defines "small," and the line keeps moving upward. In the Gemma family, it covers everything from a 270-million-parameter function-calling model to a 31-billion-parameter dense model.

The research case for SLMs is strong. In an NVIDIA position paper, researchers argue that SLMs are sufficiently powerful, more suitable and more economical for many agentic tasks. Their reasoning is that agents mostly repeat a narrow set of specialized jobs, which do not need the broad conversational range of a giant model.

The paper is a position statement, not a benchmark, so read it as an argument. It still matches what practitioners see: a well-tuned small model handles classification, extraction, routing and short generation at a fraction of the cost. A large model remains the better choice when a task needs wide knowledge or long chains of reasoning.

Gemma describes its models as built from the same research as Gemini, in a size that fits your hardware. Google's launch post for Gemma 4 claims "byte for byte, the most capable open models." Treat that as marketing language, and use the benchmark section below to judge it.

---

## What SLMs are used for

Small models fit tasks where the input is modest, the answer is bounded and speed or privacy matters. The same list appears across Google's documentation and the community projects covered later. Each item below maps to a Gemma capability you can test today.

### On-device assistants and offline work

The Gemma 4 E2B and E4B models target phones, Raspberry Pi boards and NVIDIA Jetson Orin Nano devices, and they run offline. That makes them suitable for field work, travel, and any setting with poor connectivity. Google's AI Edge Gallery app lets you try these two models on a phone.

### Translation and multilingual support

Gemma 4 is pre-trained on more than 140 languages, with out-of-the-box support for more than 35. TranslateGemma, a separate release in sizes of 4B, 12B and 27B, is tuned for translation and covers 55 languages. A model that translates on your own device never sends a private document to a third party.

### Documents, vision and audio

Every Gemma 4 model accepts images, and the models handle video as sequences of frames. The E2B, E4B and 12B models also take audio input of up to 30 seconds. On OmniDocBench 1.5, a document-parsing test where lower is better, the 31B model scores 0.131 and the E2B model scores 0.290, according to the model card.

### Agents and function calling

Gemma 4 supports native function calling, structured JSON output and system instructions. FunctionGemma is a 270-million-parameter model built for function calling, released in December 2025. These features let a small model act as the router or tool-caller inside a larger system.

### Retrieval, coding and medicine

EmbeddingGemma 2 is a 0.7B embedding model for search and retrieval, listed on Google's Hugging Face page. MedGemma covers medical text and images, and the 1.5 release is a 4B model. Code assistance is a common use of the larger Gemma 4 models, which score 80.0 percent on LiveCodeBench v6 at the 31B size.

---

## The current Gemma models

The Gemma project keeps a dated list of every release on its Gemma release page. The current flagship generation is Gemma 4, which arrived in spring 2026 as four models and gained a 12B Unified model on June 3, 2026. A Multi-Token Prediction release on April 16 added drafters for faster decoding.

All Gemma 4 models ship in base and instruction-tuned versions under the Apache 2.0 license. Hugging Face's Welcome Gemma 4 post adds one more member, DiffusionGemma 26B-A4B, a diffusion-based text generator. The table below summarizes the core family using Google's Gemma 4 model card.

The "E" in E2B and E4B stands for effective parameters. Per-Layer Embeddings give these models far more stored parameters than they use per token, which keeps compute low. The 12B Unified model is encoder-free, meaning it projects raw image patches and audio waveforms straight into the language model's embedding space.

The 26B A4B is a mixture-of-experts model that activates about 3.8 billion parameters per token. It runs at roughly the speed of a 4B model, but it must still load all 25.2 billion parameters into memory. Plan your memory around the total, not the active count.

### Specialized Gemma variants

Beyond the core family, Google and its partners publish focused models. The release page lists TranslateGemma (4B, 12B, 27B) and MedGemma 1.5 (4B), both from January 2026. It also lists FunctionGemma (270M) and T5Gemma 2 encoder-decoder models from December 2025, and EmbeddingGemma 2 from October 6, 2026.

Older generations remain available. Gemma 3 spans 270M, 1B, 4B, 12B and 27B sizes, and Gemma 3n introduced the E2B and E4B naming in June 2025. VaultGemma, a 1B model, and ShieldGemma, a safety classifier, round out the list.

---

## Hardware required

Memory decides what you can run, and speed is secondary. Google's Gemma core documentation publishes approximate memory needs for the static weights, including a 20 percent loading overhead. The figures exclude the KV cache and supporting software, so add headroom for long prompts.

Community guides give slightly higher real-world totals, because context and runtime add to the weights. Unsloth's Gemma 4 local guide lists 4-bit totals of about 4 GB for E2B, 5.5 to 6 GB for E4B, 7 to 8 GB for 12B, 16 to 18 GB for 26B A4B and 17 to 20 GB for 31B. On Ollama, the download files for the same models range from about 4.6 GB to 20 GB.

Translating those numbers into machines gives a rough ladder. These tiers are my guidance, not a vendor specification.

- Phone, Raspberry Pi or Jetson: E2B, and E4B on newer devices, through LiteRT-LM.
- 8 GB laptop: E2B comfortably at 4-bit, with E4B possible only with a short context.
- 16 GB laptop or 12 GB GPU: E4B and the 12B Unified model at 4-bit.
- 24 GB GPU or 32 GB unified memory: the 26B A4B and the 31B at 4-bit, with limited context.
- 80 GB data-center GPU: unquantized bfloat16 26B A4B and 31B, which Google says fit on a single H100.

Speed depends on the device. Google's LiteRT-LM Gemma 4 page reports E2B decode speeds of 143 tokens per second on an RTX 4090 and 160 on an M4 Mac. A Galaxy S26 Ultra reaches 52 on its GPU, an iPhone 17 Pro reaches 56, and a 16 GB Raspberry Pi 5 manages 8 on its CPU.

---

## Where Gemma runs locally

Gemma runs on more kinds of hardware than most model families. Google's launch post names NVIDIA GPUs from Jetson Orin Nano to Blackwell, AMD GPUs through ROCm, and Google TPUs. Apple Silicon is a strong fit as well, because unified memory lets a laptop hold models that a discrete GPU of the same price could not.

The browser is a real target too. Transformers.js runs Gemma 4 in the browser, including text, image and audio inference, and the Hugging Face post links a WebGPU demo. Local pages built this way keep user data inside the tab.

At the other end, you can serve the larger models on one server. The 26B A4B and the 31B can fit on a single 80 GB H100 in bfloat16, and quantized versions run on consumer GPUs. For a small team, that is one box that replaces a metered API.

---

## Inference tools

Several tools run Gemma locally, and most of them are free. They differ in how much they hide, so pick by how much control you want. The diagram shows how apps, runtimes and hardware relate.

### Ollama

Ollama runs on macOS, Windows and Linux, and it reports more than 9 million installs a month. Its Gemma 4 library page lists tags for every size, such as ollama run gemma4:12b and ollama run gemma4:31b. It also offers MLX builds and cloud tags that run on Ollama's servers instead of your machine.

### LM Studio

LM Studio's model page lists downloads of 4.2 GB for E2B, 5.9 GB for E4B, 7.4 GB for 12B, 15.6 GB for 26B A4B and 19 GB for 31B. It offers GGUF and MLX formats, including quantization-aware training versions. The company states that MLX and llama.cpp run under the hood of its app.

### llama.cpp

llama.cpp is an MIT-licensed C and C++ inference engine with no external dependencies. It supports quantization from 1.5-bit to 8-bit, hybrid CPU and GPU inference for models larger than your VRAM, and backends including CUDA, HIP, Vulkan, Metal, SYCL and WebGPU. It can serve an OpenAI-compatible API, and the ggml-org Gemma collection hosts ready-made GGUF files.

### Others worth knowing

Unsloth Studio gives you a browser interface for running and fine-tuning Gemma 4, and its guide recommends temperature 1.0, top-p 0.95 and top-k 64. MLX, through the mlx-vlm package, gives full multimodal support on Apple Silicon, and Hugging Face reports about 4x less active memory with its TurboQuant KV-cache compression. LiteRT-LM serves the E2B and E4B models on phones and small boards.

The Transformers library offers day-zero support, and it works with bitsandbytes, PEFT and TRL for fine-tuning. Hugging Face also lists mistral.rs, ONNX checkpoints and SGLang, and Google's launch post adds vLLM, NVIDIA NIM and Docker. If you want one default, start with Ollama or LM Studio, and move to llama.cpp when you need control.

---

## The Hugging Face ecosystem

Hugging Face is the main hub where Gemma lives. Its Hub documentation describes a platform that hosts over 2 million models, 1.5 million datasets and 1.5 million Spaces. All of it is open and publicly available, and every repository is version-controlled with commit history, diffs and branches.

Google's own page at Google on Hugging Face lists 1,136 models, 75 datasets, 25 Spaces and 54 collections. The Gemma 4 collection holds the 31B, 26B A4B and 12B models in base and instruction-tuned form. Google also publishes MedGemma, EmbeddingGemma 2 and the older PaliGemma, CodeGemma and ShieldGemma families there.

### Models, cards and storage

Each model repository carries a model card that documents limitations and biases, plus metadata for tasks, languages and evaluation results. Repositories build on Xet, which splits large files into chunks for faster uploads and downloads. That matters for Gemma, where a single 31B checkpoint is tens of gigabytes.

Storage Buckets add S3-like mutable storage for checkpoints and logs that do not need version history. Organizations let companies, universities and non-profits group accounts with roles that control repository access. The security section covers access tokens, GPG-signed commits and malware scanning.

### Libraries for using and tuning Gemma

The Transformers library is the reference implementation, and the Hugging Face post notes day-zero Gemma 4 support through pip install -U transformers. It provides an any-to-any pipeline for multimodal use. TRL handles fine-tuning, including supervised training and a GRPO reinforcement example, and PEFT supports parameter-efficient methods through the same integration.

The Datasets library gives programmatic access to the Hub's datasets and can stream ones too large for your disk. Transformers.js brings models to the browser, and Hugging Face has a guide to Transformers.js in Chrome extensions. Together these libraries cover the path from data to a trained, deployed Gemma variant.

### Spaces and inference services

Spaces host demo apps using Gradio, Streamlit, static pages or Docker, and ZeroGPU supplies GPUs only when a demo needs them. Google's vision token budget Space shows how image resolution changes visual token counts. For programmatic hosted access, the Hub offers Inference Providers as a serverless API, and its navigation lists dedicated Inference Endpoints.

These hosted options are optional. The point of the local route is that you can download the files and never call a hosted service. The same Hub then works as your package registry, your documentation and your community forum.

---

## Why Gemma is free

Gemma 4 is released under the Apache 2.0 license, which Google's launch post calls commercially permissive. Apache 2.0 lets you use, modify and distribute the weights for commercial work without paying Google. That is a change from Gemma 3, which used the custom Gemma Terms of Use, according to Artificial Analysis's review.

The download itself costs nothing. You can fetch Gemma from Hugging Face, Kaggle and Ollama, and Google's launch post lists those three as the download sources. Artificial Analysis also reports that the 26B and 31B models are free to use in Google AI Studio.

"Free" has limits worth stating. The weights are free, but hardware, electricity and your time are not. Ollama's cloud tags and third-party hosts such as Together AI or Fireworks run the model on their own servers, which is a service and not a free download.

My reading is that Google's reasons are strategic more than charitable, since an open family builds a developer ecosystem and spreads its research. Google has not spelled out its motives in the sources I read. Either way, read the license on the specific model you download.

---

## Why open-weight local models help

An open-weight model gives you the trained parameters, so you control where it runs. That sounds technical, but it changes five practical things.

1. Privacy. Prompts and documents stay on your machine, which simplifies work with client files, medical notes or source code.
2. Offline use. The model works on a plane, in a lab without internet or inside an air-gapped network.
3. Cost predictability. After the hardware, each extra prompt costs electricity, so heavy batch jobs have no per-token bill.
4. Stability. You can pin a specific version for years, so a vendor update cannot change your results.
5. Customization. You can fine-tune the weights on your own data with tools such as TRL and Unsloth.

Stability deserves an example. Google's own help page says its Deep Think mode is being replaced by Gemini 4 Argon, and that Deep Think may be discontinued at any time. A local Gemma file does not disappear when a product line changes, as the Deep Think help page shows for hosted services.

Local models have costs too. A local 31B model is weaker than the best hosted frontier models on hard reasoning, and you take on setup, updates and safety review yourself. Google reports safety testing for Gemma 4, but the model card says outputs can still be incorrect, outdated or biased, and developers should add safeguards.

---

## Benchmarks against peers

Benchmarks only help when you know who ran them. The first table comes from Google's model card for instruction-tuned models, and it shows how much Gemma 4 improved on Gemma 3 27B. All figures are vendor-reported.

The headline in that table is generational. The E2B model, at 2.3B effective parameters, beats Gemma 3 27B on AIME, GPQA Diamond and LiveCodeBench but trails it on MMLU Pro, and the Gemma 4 technical report says it roughly matches the older model overall. The 31B model scores 89.2 percent on AIME 2026 against 20.8 percent for Gemma 3 27B, though the older model ran without thinking.

### Against other open-weight models

Independent comparisons are fewer and less flattering. Artificial Analysis's review gives its Intelligence Index as 39 for the 31B, 31 for the 26B A4B, 19 for E4B and 15 for E2B. Qwen3.5 27B in reasoning mode scores 42, and Qwen3.5 35B A3B scores 37.

In that comparison, the 31B trails Qwen3.5 27B by 3 points, mainly because of weaker agentic performance. It leads on SciCode and TerminalBench Hard, and it ties on GPQA Diamond, IFBench and HLE. The 26B A4B sits 6 points below Qwen3.5 35B A3B and slightly above GLM-4.7-Flash at 30.

Two more details change the picture. Artificial Analysis reports that Gemma 4 31B used 39 million output tokens to run the index, against 98 million for Qwen3.5 27B, so Gemma reaches its score with far less text. On its AA-Omniscience measure, which rewards avoiding wrong answers, the smaller E4B (-20) and E2B (-24) score better than the 31B (-45).

The Gemma 4 technical report adds Arena data. On June 19, 2026, the 31B had a text Elo of 1451 and ranked as the top open dense model, and the 26B A4B scored 1438. The report notes that Qwen 3.5 397B-A17B scored 1444, and that several larger open models, including GLM 5.1, Kimi K2.6 and DeepSeek V4 Pro, scored higher than the 31B.

Taken together, Gemma 4 is strong for its size and competitive, not dominant. It is near the top among models you can run on one GPU, and it trails the largest open models and the best Qwen at similar sizes on some tests. The only reliable verdict comes from your own tasks, so test two or three candidates on your real prompts.

---

## The Gemmaverse

Google calls the community around Gemma the Gemmaverse. On August 21, 2026, The Next Web reported that the Gemma family had passed one billion downloads and that developers had published more than 100,000 variants. Google has not published how it counts either figure, so treat both as company claims.

The official Awesome Gemma list is the best directory. It sorts projects into models, inference tools, fine-tunes, tutorials, demos, challenges and research. The sections below pull out the entries I could verify, grouped by purpose.

### Health and science variants

The MedGemma collection holds medical text and image models, and the TxGemma collection targets therapeutic development. Google says MedGemma is used for outpatient triage at AIIMS and by frontline health workers in rural Uganda, though The Next Web did not verify those claims. Yale and Google built Cell2Sentence-Scale, a Gemma 2 27B model tuned for single-cell biology.

### Language and translation

The TranslateGemma collection covers 55 languages. INSAIT built BgGPT, a Bulgarian-first model on Gemma, and Google also lists a Japanese-language Gemma 2 variant. For offline use, the Awesome Gemma list includes a translation device built on Gemma 4 E2B.

### Tools, demos and learning

The list includes local chat and coding agents for Apple Silicon, a browser extension agent using WebGPU, and a tool that runs and compares several local Gemma instances at once. For learning, Maarten Grootendorst's visual guide explains the architecture with diagrams. Google's vision token budget Space is a hands-on demo.

### Space, nature and challenges

Google reports that NASA, Satlyt and Starcloud run Gemma in space for onboard image analysis and routing, without saying which models fly on which spacecraft. DolphinGemma, built with Georgia Tech and the Wild Dolphin Project, studies dolphin communication. Developers can join the Gemma 4 Good Challenge on Kaggle, and the earlier Gemma Challenge drew more than 1,600 submissions.

The ecosystem includes commercial and national deployments as well. Google says India's National Health Authority integrated Gemma 4 into the Aarogya Setu 2.0 app, which has over 100 million Android downloads. That figure counts the app, not Gemma, and The Next Web did not verify the deployment.

---

## Choosing a model

Start from your hardware and your task, then pick the smallest model that passes your own test. The guide below turns the earlier tables into a short list. It reflects my reading of the published data, so adjust it after you try each model.

1. Phone or Raspberry Pi: E2B for speed, E4B if you have the memory and need better answers.
2. Everyday laptop assistant: E4B or 12B Unified at 4-bit, which fit in 16 GB with room for context.
3. Coding and reasoning on a workstation: 31B if you have 24 GB or more of GPU memory, since it posts the best scores in the family.
4. Fast responses on a large-memory machine: 26B A4B, which runs near 4B speed but needs 14 to 18 GB at 4-bit.
5. Search and retrieval: EmbeddingGemma 2 instead of a chat model.
6. Translation or medical text: TranslateGemma or MedGemma, then validate carefully.

Then run a small test. Take ten real prompts, run them on two models, and compare the answers blind. Record tokens per second and memory use, because speed on your machine matters more than any leaderboard.

Finally, plan for updates. Gemma releases keep arriving, and the release page shows a new model or variant every few months. Keep your prompts and test set in version control so you can rerun them on the next release.

---

## Conclusion

Gemma shows how far small language models have come. A 2.3B effective model now beats the 27B model of the previous generation on several benchmarks, and the 31B model sits near the top of open dense models on Arena. Independent data places it just behind the best Qwen models at similar sizes, which is strong for a model you can run at home.

The practical case is just as clear. Apache 2.0 licensing, free downloads and mature tools such as Ollama, LM Studio and llama.cpp make local use easy, and Hugging Face supplies the surrounding workshop. A single 24 GB machine now covers most of the family, and a phone covers the smallest members.

The Gemmaverse adds a reason to stay. Over 100,000 variants, partner science projects and open challenges mean you rarely start from zero. Pick a model, run it, test it on your own work, and let the results decide.

---

## References

All links below returned a live page, or were read in full, when I opened them on November 4, 2026.

1. Gemma 4: Byte for byte, the most capable open models, Google BlogLaunch post: sizes, Apache 2.0 license, supported tools, hardware and Gemmaverse figures.
2. Gemma 4 model card, Google AI for DevelopersArchitecture, context windows, languages, benchmark table, safety and limitations.
3. Gemma core models, Google AI for DevelopersModel list and approximate memory requirements by precision.
4. Gemma release notes, gemma.aiDated list of every Gemma release and variant.
5. Gemma 4 Technical Report (arXiv 2607.02770)Gemma Team report with quantization memory and Arena results.
6. Gemma 4: Google on the American open weights frontier, Artificial AnalysisIndependent Intelligence Index comparison with Qwen and GLM models, token use and license notes.
7. Small Language Models are the Future of Agentic AI (arXiv 2506.02153)NVIDIA researchers' position paper on SLMs for agents.
8. Welcome Gemma 4, Hugging Face BlogLibrary support, architecture, fine-tuning examples and quantization tips.
9. Hugging Face Hub documentationHub scale, repositories, Spaces, Storage Buckets, organizations and security.
10. Google on Hugging FaceGoogle's organization page with Gemma 4, MedGemma and EmbeddingGemma collections.
11. How to Use Transformers.js in a Chrome Extension, Hugging Face BlogBrowser deployment of local models.
12. Gemma 4 Vision Token Budget, Hugging Face SpaceInteractive demo of image resolution and visual tokens.
13. gemma4, Ollama libraryTags, file sizes and context windows for every Gemma 4 size.
14. OllamaPlatforms and usage figures.
15. Gemma 4, LM StudioDownload sizes, formats and memory guidance.
16. llama.cpp, GitHubInference engine features, backends and license.
17. Gemma 4 GGUF collection, ggml-orgReady-made GGUF files for llama.cpp.
18. Gemma 4 local guide, UnslothMemory totals by quantization, recommended settings and Unsloth Studio.
19. Gemma 4 on LiteRT-LM, Google AI EdgeOn-device speed and memory benchmarks across phones, laptops and Raspberry Pi.
20. Use Deep Think in Gemini Apps, Gemini Apps HelpExample of a hosted feature being replaced.
21. Google says Gemma has passed a billion downloads, The Next WebDownload and variant counts, and notable projects.
22. Awesome Gemma, GitHubOfficial directory of Gemmaverse projects, tools and tutorials.
23. MedGemma collection, Hugging FaceMedical text and image models.
24. TxGemma collection, Hugging FaceTherapeutic development models.
25. TranslateGemma collection, Hugging FaceTranslation models covering 55 languages.
26. C2S-Scale-Gemma-2-27B, Hugging FaceGemma 2 27B tuned for single-cell biology.
27. A Visual Guide to Gemma 4, Maarten GrootendorstIllustrated explanation of the architecture.
28. Gemma 4 Good Challenge, KaggleCommunity hackathon for social impact projects.
29. LM StudioProduct page stating MLX and llama.cpp as underlying runtimes.

---

### Ready to Orchestrate Production Claude AI Agent Teams?
- [Schedule Strategy Consultation via Email →](contact.html)
- [View Advisory Packages & Rates](pricing.html)
- [Explore Claude Agents Suite](claude-101.html)
