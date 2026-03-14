# PATH 0: AI POWER USERS — The Complete Practical Checklist
### Every Tool. Every Workflow. For Anyone Who Wants to Use AI — Not Build It.

> **Who this is for:** Writers, musicians, filmmakers, designers, marketers, business owners, educators, healthcare workers, lawyers, executives, students — anyone who wants to use AI to solve real problems and 10x their work. No programming required.
>
> **Instructions:** Check off each item as you complete it. Follow the sections relevant to your domain, and cover all of Sections 1–5 regardless of profession.

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## SECTION 1: UNDERSTANDING AI (THE NON-TECHNICAL VERSION)
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### 1.1 What AI Actually Is

📚 **Best Resources to Learn:**
- "Explained: Neural Networks" — MIT News — news.mit.edu (search it)
- "How ChatGPT Works" — 3Blue1Brown (no-math YouTube version)
- "AI Basics" — Google — ai.google/education
- "Elements of AI" (FREE course) — elementsofai.com — **Start here if you're brand new**
- Anthropic's "Claude's Constitution" (readable explanation of modern AI) — anthropic.com/news

- [ ] What is Artificial Intelligence (plain English definition)
- [ ] The difference between AI, Machine Learning, and Deep Learning
- [ ] What is a Large Language Model (LLM)
- [ ] What is Generative AI (GenAI)
- [ ] How LLMs work at a high level (token prediction, not "thinking")
- [ ] What is a neural network (without the math)
- [ ] What is training vs. inference (learning vs. using)
- [ ] What is fine-tuning (customizing an existing model)
- [ ] What is a foundation model vs. a specialized model
- [ ] The difference between open-source and closed/proprietary AI models
- [ ] The difference between cloud AI and locally-run AI

🏋️ **Exercises:**
1. Watch 3Blue1Brown's "But what is a neural network?" — write 5 sentences explaining it in your own words to a non-technical friend
2. Look up the release dates of GPT-2, GPT-3, GPT-4, Claude 1, Claude 3 — draw a simple timeline of AI progress
3. Explain the difference between AI, ML, and DL to someone else (test your understanding)

---

### 1.2 The AI Landscape — Models and Providers

📚 **Best Resources to Learn:**
- Chatbot Arena Leaderboard (live model rankings) — lmarena.ai
- "State of AI Report" — stateofai.com (annual report, free)
- Each company's official blog: openai.com/blog, anthropic.com/news, deepmind.google

- [ ] **Text AI Providers**
  - [ ] OpenAI — ChatGPT, GPT-4o, o3
  - [ ] Anthropic — Claude (Haiku, Sonnet, Opus)
  - [ ] Google — Gemini (Flash, Pro, Ultra)
  - [ ] Meta — Llama (open-source)
  - [ ] Mistral (open-source, European)
  - [ ] Perplexity AI (search-focused AI)
  - [ ] xAI — Grok
- [ ] **Image AI Providers**
  - [ ] OpenAI — DALL-E / GPT Image
  - [ ] Midjourney
  - [ ] Stability AI — Stable Diffusion
  - [ ] Adobe — Firefly
  - [ ] Google — Imagen
  - [ ] Black Forest Labs — FLUX
- [ ] **Video AI Providers**
  - [ ] OpenAI — Sora
  - [ ] Runway
  - [ ] Kling (Kuaishou)
  - [ ] Pika
  - [ ] Google — Veo
- [ ] **Audio/Music AI Providers**
  - [ ] ElevenLabs (voice)
  - [ ] Suno (music)
  - [ ] Udio (music)
  - [ ] AIVA (orchestral)
  - [ ] Meta — AudioCraft
- [ ] Understanding model versioning (what GPT-4o means vs GPT-4 vs o3)
- [ ] Understanding context windows (how much text the AI can "remember")
- [ ] Understanding multimodal AI (text + image + audio in one model)

🏋️ **Exercises:**
1. Create free accounts on: ChatGPT, Claude, Gemini, Perplexity — spend 15 mins with each
2. Ask the exact same complex question to 4 different AI models — compare the answers
3. Find the current AI model leaderboard at lmarena.ai — identify which model leads each category

---

### 1.3 AI Capabilities and Limitations

📚 **Best Resources to Learn:**
- "AI Hallucination" — IBM Research blog
- OpenAI usage policies — openai.com/policies
- Anthropic safety research overview — anthropic.com/research

- [ ] What AI is genuinely good at
  - [ ] Text generation, summarization, translation
  - [ ] Code writing and debugging
  - [ ] Brainstorming and ideation
  - [ ] Data analysis and pattern recognition
  - [ ] Question answering (with verification)
  - [ ] Editing and refining content
- [ ] What AI is bad at or unreliable for
  - [ ] Real-time information (unless connected to search)
  - [ ] Complex math and precise calculations
  - [ ] Reasoning about novel situations it wasn't trained on
  - [ ] Long-term memory across sessions
  - [ ] Guaranteed factual accuracy
- [ ] **AI Hallucinations**
  - [ ] What hallucinations are (confident wrong answers)
  - [ ] Why they happen (pattern matching, not knowledge)
  - [ ] How to detect them (verify with authoritative sources)
  - [ ] How to reduce them (be specific, ask for sources, cross-check)
  - [ ] Current hallucination rates (~0.7–5% depending on model)
- [ ] AI biases and where they come from
- [ ] What "context window" means and why it matters
- [ ] The difference between AI knowledge cutoff and real-time AI
- [ ] When NOT to use AI

🏋️ **Exercises:**
1. Deliberately get an AI to hallucinate — ask about a made-up book, person, or event
2. Ask an AI for 5 medical or legal facts — then verify each one with an authoritative source
3. Find one thing AI is confidently wrong about in 3 different sessions — document the pattern

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## SECTION 2: PROMPT ENGINEERING — THE CORE SKILL
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### 2.1 Prompt Fundamentals

📚 **Best Resources to Learn:**
- Learn Prompting (FREE, comprehensive) — learnprompting.org — **Start here**
- Anthropic Prompt Engineering Guide — docs.anthropic.com/en/docs/build-with-claude/prompt-engineering
- OpenAI Prompt Engineering Guide — platform.openai.com/docs/guides/prompt-engineering
- "Prompt Engineering Guide" — promptingguide.ai

- [ ] What is a prompt
- [ ] The anatomy of a good prompt: Role + Task + Format + Constraints + Context + Audience
- [ ] Zero-shot prompting (asking directly with no examples)
- [ ] One-shot prompting (give one example, then ask)
- [ ] Few-shot prompting (give multiple examples)
- [ ] Specifying output format (bullet points, tables, paragraphs, JSON)
- [ ] Specifying length (word count, number of items)
- [ ] Specifying tone (professional, casual, formal, playful)
- [ ] Specifying audience ("explain this to a 10-year-old" / "explain this to an expert")
- [ ] Iterative refinement (using follow-up prompts to improve output)

🏋️ **Exercises:**
1. Take a vague prompt ("write about climate change") and rewrite it using the full anatomy — compare outputs
2. Write the same request using zero-shot, one-shot, and few-shot — compare quality
3. Get the same content rewritten in 5 different tones (professional, casual, humorous, urgent, academic)

---

### 2.2 Intermediate Prompt Techniques

📚 **Best Resources to Learn:**
- "Chain of Thought" paper (readable summary) — ai.googleblog.com
- Anthropic's guide to complex prompting — docs.anthropic.com

- [ ] Role prompting ("Act as a...", "You are a senior...")
- [ ] Chain-of-thought prompting ("Think step by step...")
- [ ] Break-down prompting (asking AI to plan before executing)
- [ ] Self-consistency (asking AI to answer multiple times, pick best)
- [ ] Comparison prompting ("Compare X and Y in a table")
- [ ] Constraint-based prompting ("Write this without using the word...")
- [ ] Reverse prompting ("What's missing from this?" / "What could go wrong?")
- [ ] Negative prompting ("Don't include...", "Avoid...")
- [ ] Conditional prompting ("If X is true, then do Y, else do Z")
- [ ] Using delimiters to structure complex prompts (triple quotes, XML tags)

🏋️ **Exercises:**
1. Use chain-of-thought prompting to solve a complex decision you actually face at work
2. Use role prompting: ask the same marketing question as "a CMO of a Fortune 500" vs "a solo founder with $500 budget"
3. Write a prompt that forces the AI to show its reasoning before giving an answer

---

### 2.3 Advanced Prompt Techniques

📚 **Best Resources to Learn:**
- "The Art of Asking Claude" — Anthropic blog
- Prompt engineering for specific use cases — learnprompting.org/docs/advanced

- [ ] System prompts (setting persistent behavior/persona)
- [ ] Prompt chaining (output of prompt 1 → input of prompt 2)
- [ ] Prompt templates (reusable prompts with fill-in variables)
- [ ] Meta-prompting ("Write me a prompt that would help me...")
- [ ] Structured output prompting (asking for JSON, tables, specific schemas)
- [ ] Multi-turn conversation strategies (how to maintain context)
- [ ] Personas and custom AI assistants (in ChatGPT, Claude Projects)
- [ ] Tool use / function calling (getting AI to call external tools)
- [ ] RAG awareness (understanding when AI uses retrieval to answer)
- [ ] Evaluating and scoring prompt quality systematically

🏋️ **Exercises:**
1. Build a personal custom AI assistant (in ChatGPT "GPTs" or Claude "Projects") with a detailed system prompt for your specific job
2. Create a 5-prompt chain: Research → Outline → Draft → Edit → Format, each output feeding the next
3. Build a library of 10 reusable prompt templates for your most common tasks

🛠️ **Mini-Project:** Build your **Personal Prompt Library** — 20 reusable prompts covering your most common work tasks. Organized by category, with notes on what works and what doesn't.

---

### 2.4 Image Prompting

📚 **Best Resources to Learn:**
- Midjourney documentation — docs.midjourney.com
- "Stable Diffusion Prompt Guide" — stable-diffusion-art.com/prompt-guide
- DALL-E prompting tips — OpenAI help docs

- [ ] How image prompts differ from text prompts
- [ ] Describing style, mood, lighting, composition
- [ ] Aspect ratios and resolution specifications
- [ ] Negative prompts (what to exclude)
- [ ] Reference images and style transfer
- [ ] Midjourney-specific parameters (--ar, --v, --style, --chaos)
- [ ] Inpainting (editing specific parts of an image)
- [ ] Outpainting (extending images beyond their borders)
- [ ] Iterating on image generations (upscale, vary, remix)

🏋️ **Exercises:**
1. Generate the same scene 10 different ways by changing only the style descriptor
2. Practice inpainting: generate an image, then replace one element without changing the rest
3. Create a visual style guide: 5 prompts that consistently produce your desired aesthetic

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## SECTION 3: THE AI TOOLS LANDSCAPE
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### 3.1 Text & Chat AI

📚 **Best Resources to Learn:**
- Compare at: lmarena.ai (neutral comparison)
- Each tool's own getting-started guide

- [ ] **ChatGPT (OpenAI)**
  - [ ] Using GPT-4o for general tasks
  - [ ] Using o3 for complex reasoning tasks
  - [ ] Custom GPTs (building personalized assistants)
  - [ ] ChatGPT memory features
  - [ ] ChatGPT with browsing (real-time web access)
  - [ ] Advanced Data Analysis (code interpreter)
  - [ ] File and image uploads
- [ ] **Claude (Anthropic)**
  - [ ] When Claude beats ChatGPT (long documents, nuanced writing, coding)
  - [ ] Claude Projects (persistent context workspaces)
  - [ ] Uploading large files (PDFs, data files)
  - [ ] Claude's extended thinking mode
- [ ] **Gemini (Google)**
  - [ ] Gemini integration with Google Workspace (Docs, Sheets, Gmail)
  - [ ] Gemini's long context window (1M+ tokens)
  - [ ] Gemini Advanced for deep reasoning
- [ ] **Perplexity AI**
  - [ ] Using Perplexity for research (with citations)
  - [ ] Academic mode (peer-reviewed sources)
  - [ ] Deep Research feature
  - [ ] Following up and fact-checking answers
- [ ] **Specialized writing tools**
  - [ ] Jasper (marketing copy)
  - [ ] Copy.ai (creative copy)
  - [ ] Grammarly + GrammarlyGO (writing quality + AI rewriting)
  - [ ] Notion AI (knowledge management + writing)

🏋️ **Exercises:**
1. Use Perplexity to research a topic you actually care about — evaluate source quality
2. Set up a Claude Project for your primary work context (e.g., "Marketing Assistant for [Company]")
3. Use ChatGPT Advanced Data Analysis to analyze a real spreadsheet or CSV you have

---

### 3.2 Image Generation

📚 **Best Resources to Learn:**
- Midjourney Discord and documentation — docs.midjourney.com
- Adobe Firefly getting started — helpx.adobe.com/firefly
- Stable Diffusion Automatic1111 guide — stable-diffusion-art.com

- [ ] **Midjourney**
  - [ ] Generating images via web interface
  - [ ] Understanding quality and version settings
  - [ ] Using style references and character consistency
  - [ ] Remix and variation workflows
  - [ ] Midjourney for video (short clips)
- [ ] **DALL-E / GPT Image (OpenAI)**
  - [ ] Generating directly within ChatGPT
  - [ ] Editing and inpainting via ChatGPT
  - [ ] When to use DALL-E vs Midjourney
- [ ] **Adobe Firefly**
  - [ ] Generative fill in Photoshop
  - [ ] Background removal and extension
  - [ ] Object replacement
  - [ ] Commercially-safe AI (trained on licensed content)
  - [ ] Firefly in Illustrator (vector generation)
- [ ] **Stable Diffusion (open-source)**
  - [ ] Running locally vs using Automatic1111 web UI
  - [ ] LoRA models (custom styles)
  - [ ] ControlNet (pose/depth control)
  - [ ] When to use SD over closed tools
- [ ] **Canva AI**
  - [ ] Magic Design (AI-generated layouts)
  - [ ] Magic Edit and Magic Expand
  - [ ] Text-to-image within Canva
- [ ] **Ideogram / Flux / Recraft** (text-in-images, specialized use cases)

🏋️ **Exercises:**
1. Create the same image concept in Midjourney, DALL-E, and Firefly — compare style differences
2. Use Firefly's generative fill to add an object to a photo you own
3. Generate 20 product mockup images for a fictional product in under 30 minutes

🛠️ **Mini-Project:** Create a complete **Visual Brand Kit** — logo concepts, color palette images, social media template images, and a hero banner, using only AI image tools.

---

### 3.3 Video Generation

📚 **Best Resources to Learn:**
- Runway documentation — runwayml.com/docs
- Kling documentation — klingai.com
- LTX Studio — ltx.studio

- [ ] **Text-to-video generation**
  - [ ] Sora (OpenAI) — emotional depth, dialogue, narrative
  - [ ] Kling — cinematic, cost-efficient, reliable
  - [ ] Runway Gen-4 — motion quality, Adobe integration
  - [ ] Pika — short-form, viral-style content
- [ ] **Image-to-video**
  - [ ] Animating a still image (pan, zoom, movement)
  - [ ] Creating video from product photos
  - [ ] Style-consistent video from reference image
- [ ] **AI video editing**
  - [ ] Descript (script-based video editing)
  - [ ] Async / Capsule (AI-powered editing)
  - [ ] Adobe Premiere Pro AI features
- [ ] **AI avatar / presenter videos**
  - [ ] Synthesia (enterprise, 230+ avatars)
  - [ ] HeyGen (personal avatar creation, real-time)
  - [ ] D-ID (conversational AI agents)
  - [ ] Creating your own AI avatar (cloning your likeness)
- [ ] **Upscaling and enhancing existing video**
- [ ] Understanding video generation limitations (duration, consistency, control)

🏋️ **Exercises:**
1. Generate a 10-second product ad video using only a text prompt and Kling or Pika
2. Create an explainer video using a Synthesia or HeyGen avatar with a script you write
3. Use Descript to edit a 3-minute video using only the text transcript (delete words to cut footage)

🛠️ **Mini-Project:** Produce a 60-second **AI-generated promo video** for a product or idea. Script → storyboard → video generation → voiceover → edit. No camera required.

---

### 3.4 Audio, Voice, and Music AI

📚 **Best Resources to Learn:**
- ElevenLabs documentation — elevenlabs.io/docs
- Suno getting started — suno.com
- AIVA tutorials — aiva.ai/tutorials

- [ ] **Voice generation (text-to-speech)**
  - [ ] ElevenLabs — voice cloning, multilingual, ultra-realistic
  - [ ] OpenAI TTS (voices in ChatGPT API)
  - [ ] Choosing voice styles (professional, conversational, narrative)
  - [ ] Cloning your own voice (and ethical use)
- [ ] **Music generation**
  - [ ] Suno — full songs with vocals and lyrics
  - [ ] Udio — high-quality music generation
  - [ ] AIVA — orchestral/cinematic (copyright-owned)
  - [ ] Soundraw — royalty-free background music
  - [ ] ElevenLabs Music — commercially cleared
- [ ] **Audio editing with AI**
  - [ ] Descript (transcript-based audio editing)
  - [ ] Adobe Podcast (enhance audio quality, remove noise)
  - [ ] Meta SAM Audio (source separation — isolate instruments)
- [ ] **Voice changing and translation**
  - [ ] Real-time voice changing
  - [ ] AI dubbing and translation of existing audio
  - [ ] Multilingual content creation
- [ ] Understanding audio copyright in AI-generated music
- [ ] When to use AI voice vs hiring voice talent

🏋️ **Exercises:**
1. Create a full 2-minute song in Suno — experiment with genre, mood, and custom lyrics
2. Use ElevenLabs to turn a blog post into a narrated audio file
3. Use Adobe Podcast Enhance to improve the audio quality of a noisy recording

🛠️ **Mini-Project:** Create a **podcast episode** with AI — script written by Claude, narrated by ElevenLabs, background music from Soundraw, edited in Descript.

---

### 3.5 Productivity and Knowledge Management AI

📚 **Best Resources to Learn:**
- Notion AI documentation — notion.so/help/guides/ai
- Obsidian community guides — obsidian.md/community

- [ ] **AI note-taking and knowledge management**
  - [ ] Notion AI (Q&A over your notes, AI writing, summaries)
  - [ ] Obsidian with AI plugins (local, private knowledge base)
  - [ ] NotebookLM (Google) — upload documents, chat with them
- [ ] **AI for email**
  - [ ] Gmail AI features (Smart Compose, Smart Reply)
  - [ ] Superhuman (AI email client)
  - [ ] Using Claude/ChatGPT to draft email templates
- [ ] **AI for meetings**
  - [ ] Otter.ai (meeting transcription)
  - [ ] Fireflies.ai (meeting notes and action items)
  - [ ] Read.ai (meeting analytics)
- [ ] **AI for presentations**
  - [ ] Gamma (AI-generated presentations)
  - [ ] Beautiful.ai (smart presentation design)
  - [ ] ChatGPT + Google Slides / PowerPoint integration
- [ ] **AI search and research**
  - [ ] Perplexity for research (cited answers)
  - [ ] NotebookLM for document Q&A
  - [ ] Consensus (academic research)
- [ ] **AI writing assistants**
  - [ ] Grammarly (grammar + tone + plagiarism)
  - [ ] Hemingway Editor
  - [ ] ProWritingAid

🏋️ **Exercises:**
1. Upload a 50-page PDF to NotebookLM — extract key insights in under 10 minutes by asking questions
2. Build a Notion workspace with AI enabled — set up AI templates for your weekly planning
3. Use Otter.ai or Fireflies.ai in your next 3 meetings — evaluate the quality of AI-generated action items

---

### 3.6 Design and Visual AI

📚 **Best Resources to Learn:**
- Canva tutorial library — designschool.canva.com
- Figma AI documentation — figma.com/ai

- [ ] **Canva AI**
  - [ ] Magic Design (auto-generate design from prompt)
  - [ ] Magic Edit (change elements in designs)
  - [ ] Magic Write (AI copy in designs)
  - [ ] Background removal and AI object placement
- [ ] **Figma AI (for product/UI designers)**
  - [ ] AI-powered layout suggestions
  - [ ] Auto-naming layers and components
  - [ ] Generating UI designs from description
- [ ] **Adobe Firefly (for creative professionals)**
  - [ ] Generative fill and expand
  - [ ] Content-aware fill
  - [ ] Vector pattern generation in Illustrator
- [ ] **AI for social media assets**
  - [ ] Creating consistent brand visuals
  - [ ] Resizing for different platforms
  - [ ] A/B testing visual variations

🏋️ **Exercises:**
1. Design a full social media content pack (Instagram post, story, LinkedIn banner) for a brand using only Canva AI
2. Use Firefly to extend a photo that's too narrow for your intended use
3. Generate 10 logo concepts using Midjourney or Ideogram, then refine the best one

---

### 3.7 Business and Marketing AI

📚 **Best Resources to Learn:**
- HubSpot AI features guide — knowledge.hubspot.com
- Jasper documentation — jasper.ai/guides

- [ ] **Content marketing AI**
  - [ ] Jasper (long-form marketing content)
  - [ ] Copy.ai (marketing copy, ad creative)
  - [ ] SEO content optimization with AI (Surfer SEO, Clearscope)
- [ ] **Social media AI**
  - [ ] Scheduling and AI caption generation (Later, Hootsuite AI)
  - [ ] Creating content calendars with AI
  - [ ] Hashtag and engagement optimization
- [ ] **Email marketing AI**
  - [ ] Instantly.ai (AI email outreach)
  - [ ] Mailchimp AI (subject line optimization, send time)
  - [ ] Personalization at scale
- [ ] **CRM and sales AI**
  - [ ] HubSpot AI (lead scoring, email drafts, deal summaries)
  - [ ] Salesloft (AI sales agent)
  - [ ] AI for customer support (chatbots, ticket routing)
- [ ] **Analytics and reporting AI**
  - [ ] Using AI to interpret analytics dashboards
  - [ ] Auto-generated insights from Google Analytics / GA4
  - [ ] AI-powered competitive analysis

🏋️ **Exercises:**
1. Create a full 30-day social media content calendar using only AI (prompts, captions, image concepts)
2. Write a complete email marketing sequence (5 emails) for a product launch using Claude
3. Use Perplexity to do competitive research on 3 competitors — generate a strategy memo

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## SECTION 4: WORKFLOW AUTOMATION — NO CODE REQUIRED
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### 4.1 Understanding AI Automation Concepts

📚 **Best Resources to Learn:**
- n8n YouTube tutorials — youtube.com/@n8n-io
- Zapier learn hub — learn.zapier.com
- Make Academy — academy.make.com

- [ ] What is workflow automation
- [ ] What is a "trigger" and an "action" in automation
- [ ] What are AI agents (autonomous AI that takes actions)
- [ ] The difference between automated tasks vs AI agents
- [ ] What is an API (without coding — just the concept)
- [ ] What is a webhook
- [ ] Multi-step workflows (chains of actions)
- [ ] Conditional logic in workflows (if this, then that)
- [ ] Error handling and monitoring in automations

🏋️ **Exercises:**
1. Map out 5 manual tasks you do every week that follow a repeatable pattern
2. Describe your ideal automation for each one in plain English before building it

---

### 4.2 Zapier — Best for Non-Technical Users

📚 **Best Resources to Learn:**
- Zapier quick start — zapier.com/learn/getting-started
- "Zapier AI Actions" guide — zapier.com/ai

- [ ] Creating a Zap (trigger + action)
- [ ] Connecting two apps
- [ ] Multi-step Zaps (more than one action)
- [ ] Zapier filters (conditional logic)
- [ ] Zapier AI Actions (ChatGPT/Claude inside Zapier)
- [ ] Zapier Tables (store and query data)
- [ ] Zapier Chatbots (build no-code AI chatbots)
- [ ] Testing and debugging Zaps
- [ ] Common Zapier workflows:
  - [ ] Gmail → Google Sheets data logging
  - [ ] Form submission → AI-generated response → Email
  - [ ] New tweet/post → Summarize with AI → Slack notification
  - [ ] Calendar event → AI-generated prep notes

🏋️ **Exercises:**
1. Build your first Zap: When you get an email with a keyword → automatically add it to a Google Sheet
2. Build an AI content Zap: New RSS feed item → Claude summary → posted to Slack
3. Build a form-to-response workflow: Form submission → AI-generated personalized email reply

---

### 4.3 Make (Integromat) — For Visual Power Users

📚 **Best Resources to Learn:**
- Make Academy — academy.make.com
- Make YouTube channel — youtube.com/@make_hq

- [ ] Building scenarios (the Make equivalent of Zaps)
- [ ] Using modules (inputs/outputs between apps)
- [ ] Array aggregators and iterators (handling multiple items)
- [ ] Make's AI modules (OpenAI, Claude, Perplexity integrations)
- [ ] Error routes and fallback logic
- [ ] Scheduling scenarios
- [ ] Make data stores (simple database)
- [ ] Webhooks in Make (custom triggers)
- [ ] Common Make workflows:
  - [ ] Content generation pipelines
  - [ ] Document processing at scale
  - [ ] Multi-platform social media posting

🏋️ **Exercises:**
1. Build a content pipeline: topic input → AI draft → Grammarly check → auto-publish to WordPress
2. Build a document summarizer: PDF uploaded to Google Drive → AI summary → emailed to you

---

### 4.4 n8n — For Custom and Advanced Workflows

📚 **Best Resources to Learn:**
- n8n documentation — docs.n8n.io
- n8n community forum — community.n8n.io
- n8n AI starter kit guide — n8n.io/ai

- [ ] Self-hosted vs cloud n8n (understanding the options)
- [ ] n8n nodes (trigger + regular + AI nodes)
- [ ] n8n AI nodes (LangChain integration, 70+ AI-specific nodes)
- [ ] Building an AI agent in n8n (with memory and tools)
- [ ] Building a RAG pipeline (connect your documents to an AI)
- [ ] Subworkflows (modular automation design)
- [ ] n8n + vector databases (building AI with memory)
- [ ] Common n8n workflows:
  - [ ] AI customer support bot
  - [ ] Personal research assistant
  - [ ] Automated content moderation
  - [ ] Lead qualification AI agent

🏋️ **Exercises:**
1. Build an AI agent in n8n that can search the web and summarize results for you
2. Build a personal AI assistant that has memory of your previous questions

---

### 4.5 Building AI Agents Without Code

📚 **Best Resources to Learn:**
- Relevance AI — relevanceai.com (no-code AI agents)
- Lindy — lindy.ai (AI employee automation)
- ChatGPT GPT Builder — chatgpt.com/gpts/editor
- Claude Projects

- [ ] What AI agents are (AI that can take actions, not just respond)
- [ ] Types of agents: research agents, email agents, scheduling agents
- [ ] Tools agents use: web search, file access, code execution, API calls
- [ ] Building a custom GPT (no-code AI assistant)
- [ ] Building a Claude Project (persistent assistant)
- [ ] Using Relevance AI to build no-code agents
- [ ] Using Lindy for email and calendar automation
- [ ] Testing agents: verifying they do what you intend
- [ ] Agent safety: what guardrails to put in place

🏋️ **Exercises:**
1. Build a custom GPT for your specific job function — give it a system prompt, knowledge files, and tools
2. Build a no-code research agent using Relevance AI that finds and summarizes news on topics you care about

🛠️ **Project: Build Your AI-Powered Personal Operating System**
— Custom AI assistant in ChatGPT or Claude with your work context, connected to your email via Zapier, automated daily briefing, and 3 recurring workflow automations. Document every automation step.

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## SECTION 5: AI ETHICS, SAFETY, AND SMART USE
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### 5.1 Privacy and Data Safety

📚 **Best Resources to Learn:**
- "AI Privacy Risks" — Electronic Frontier Foundation — eff.org
- OpenAI privacy controls — help.openai.com/en/articles/7730893
- EU AI Act overview — artificial-intelligence-act.com

- [ ] What data AI companies store from your conversations
- [ ] Opting out of training data collection
- [ ] What NOT to put in a public AI tool
  - [ ] Client PII (names, emails, addresses)
  - [ ] Proprietary business strategies
  - [ ] Medical or legal case details
  - [ ] Passwords or credentials
  - [ ] Financial data
- [ ] Using AI in enterprise settings (data agreements, Business Associates Agreements)
- [ ] Self-hosted AI options for sensitive data (Ollama, local LLMs)
- [ ] The EU AI Act (what it means for users in Europe)
- [ ] How to audit your company's AI usage for compliance
- [ ] AI phishing and scam awareness (AI-crafted emails are now 82% of phishing)

🏋️ **Exercises:**
1. Review ChatGPT privacy settings — disable conversation history if needed for sensitive work
2. Write a 1-page AI usage policy for your team or personal practice
3. Identify 5 tasks at your job where you would NOT use public AI tools and explain why

---

### 5.2 Copyright and Intellectual Property

📚 **Best Resources to Learn:**
- US Copyright Office AI guidance — copyright.gov/ai
- Creative Commons guide to AI — creativecommons.org/2023/03/27/ai-and-the-commons

- [ ] AI-generated content copyright basics
  - [ ] US law: AI-only content is not copyrightable
  - [ ] Human authorship required for copyright protection
  - [ ] "Substantial human contribution" standard
- [ ] Commercially safe vs. legally ambiguous AI tools
  - [ ] Adobe Firefly (trained on licensed stock — safest for commercial use)
  - [ ] ChatGPT image generation (training data lawsuits pending)
  - [ ] Music AI legal landscape (Suno/Udio licensing deals with labels)
- [ ] Using AI outputs in commercial work — best practices
- [ ] Disclosing AI usage in creative work
- [ ] AI and plagiarism (how to avoid unintentional copying)
- [ ] Protecting your own creative work from being scraped
  - [ ] robots.txt and opt-out registries
  - [ ] Spawning.ai opt-out for artists

🏋️ **Exercises:**
1. Research the copyright status of AI images in your country — write a 1-paragraph summary
2. Find the terms of service for 3 AI tools you use — what do they say about commercial rights?

---

### 5.3 Deepfakes and Misinformation

📚 **Best Resources to Learn:**
- "Detecting Deepfakes" — MIT Media Lab
- Sensity AI deepfake detection guide — sensity.ai/resources
- "Verify AI Content" — First Draft (firstdraftnews.com)

- [ ] How deepfake images, audio, and video are created
- [ ] Why deepfakes are increasingly hard to detect (voice cloning threshold crossed)
- [ ] How to spot deepfake images
  - [ ] Unnatural hands, teeth, hair
  - [ ] Lighting inconsistencies
  - [ ] Background anomalies
- [ ] How to spot deepfake audio
  - [ ] Unusual cadence or tone shifts
  - [ ] Verify with a callback to known number
- [ ] Deepfake detection tools (Hive Moderation, Sensity.ai)
- [ ] AI-generated text detection limitations (detectors are unreliable)
- [ ] Media verification workflow: reverse image search, source checking, metadata
- [ ] The role of digital watermarking (C2PA standard)

🏋️ **Exercises:**
1. Use a reverse image search on an AI-generated image — understand what traces it leaves
2. Test an AI text detector (GPTZero, Originality.ai) — understand why they're unreliable
3. Find one viral "shocking" image or story and trace its origin using fact-checking tools

---

### 5.4 Responsible AI Use

📚 **Best Resources to Learn:**
- "Responsible AI Principles" — Microsoft — microsoft.com/en-us/ai/responsible-ai
- Anthropic's model card and usage policies — anthropic.com/model-card

- [ ] Transparency: disclosing when you use AI in your work
- [ ] Attribution: crediting sources when AI references them
- [ ] Accuracy: verifying AI outputs before sharing or acting on them
- [ ] Fairness: reviewing AI outputs for bias, stereotyping, or unfair portrayals
- [ ] Accessibility: making sure AI tools don't exclude people
- [ ] Environmental impact of AI (compute and energy usage)
- [ ] AI in high-stakes decisions (medical, legal, financial — human in the loop)
- [ ] Avoiding AI dependency trap (skills atrophying when overusing AI)
- [ ] Principles for professional AI use in your field

🏋️ **Exercises:**
1. Create a personal "AI Responsibility Checklist" you run through before publishing or sharing AI-generated content
2. Find a case study where AI was used irresponsibly in your industry — analyze what went wrong

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## SECTION 6: DOMAIN-SPECIFIC DEEP DIVES
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

> **Instructions:** Complete ALL of Sections 1–5 first, then go deep on the section(s) most relevant to your work.

---

### 6.1 AI for Musicians and Audio Professionals

- [ ] **Understanding AI music generation models**
  - [ ] Text-to-music vs. reference-guided generation
  - [ ] Understanding prompting for music (genre, BPM, mood, instruments)
  - [ ] Copyright in AI-generated music
- [ ] **Suno Studio (generative audio workstation)**
  - [ ] Generating full songs with custom lyrics
  - [ ] Multi-track editing within Suno
  - [ ] Exporting stems
- [ ] **Udio** — high-quality music generation with voice
- [ ] **AIVA** — orchestral/cinematic AI, full copyright ownership model
- [ ] **Soundraw** — customizable royalty-free background music
- [ ] **ElevenLabs Music** — commercially cleared audio generation
- [ ] **AI for music production**
  - [ ] AI stem separation (separate vocals, drums, bass)
  - [ ] BIAS X — AI tone matching
  - [ ] Adobe Audition / GarageBand AI features
- [ ] **AI for live performance**
  - [ ] Real-time AI accompaniment tools
  - [ ] AI sound design and synthesis
- [ ] Building a full AI music workflow: brief → generate → refine → mix → export
- [ ] Understanding the licensing landscape (Warner deals with Suno)

🏋️ **Exercises:**
1. Generate 5 different songs in Suno — experiment with genre, tempo, mood, and custom lyrics
2. Separate a commercial song into stems using an AI tool — analyze the components
3. Create a 60-second background track for a video using Soundraw — match the mood to the content

🛠️ **Project: Full EP Concept** — Create a 4-track AI-assisted EP. Write lyrics with Claude, generate tracks in Suno, add voice with ElevenLabs, design cover art with Midjourney, produce a Bandcamp-ready release.

---

### 6.2 AI for Filmmakers and Video Creators

- [ ] **AI in the writing and pre-production phase**
  - [ ] Using Claude/ChatGPT for script drafts, scene breakdowns, dialogue
  - [ ] AI-generated storyboards (Midjourney for visual reference)
  - [ ] AI for shot lists and production planning
- [ ] **AI video generation tools and their strengths**
  - [ ] Sora — narrative and emotional depth, best for cinematic storytelling
  - [ ] Kling — cinematic quality, cost-effective, most consistent
  - [ ] Runway Gen-4 — motion quality, creative control
  - [ ] Pika — short-form, effects, fast iteration
  - [ ] LTX Studio — professional script-to-screen workflow
- [ ] **AI in post-production**
  - [ ] Descript (transcript-based editing)
  - [ ] Auto-captioning and subtitle generation
  - [ ] Adobe Premiere Pro AI features
  - [ ] Color grading assistance with AI
  - [ ] AI voice-over and dubbing (ElevenLabs)
- [ ] **AI avatars as on-screen presenters**
  - [ ] HeyGen — personal avatar, multilingual dubbing
  - [ ] Synthesia — enterprise scale, SCORM export
  - [ ] D-ID — conversational AI agents
- [ ] **Consistency challenges in AI video** (and how to manage them)
- [ ] **Building a hybrid AI+live action workflow**

🏋️ **Exercises:**
1. Write a 2-minute short film script with Claude, generate 10 key scenes in Kling, edit in CapCut
2. Create a product video using only AI: script → voiceover (ElevenLabs) → visuals (Sora/Kling) → music (Suno)
3. Dub an existing YouTube video into 2 other languages using HeyGen

🛠️ **Project: AI Short Film** — Produce a 3–5 minute AI-generated short film with a clear narrative. Script → storyboard → video generation → AI voiceover → music → color grade → publish on YouTube.

---

### 6.3 AI for Visual Artists and Designers

- [ ] **AI as a concept and ideation tool**
  - [ ] Rapid concept generation (20 variations in minutes)
  - [ ] Style exploration and reference generation
  - [ ] Client mood board creation
- [ ] **Professional AI image workflows**
  - [ ] Photoshop + Firefly integration (generative fill, extend, replace)
  - [ ] Illustrator AI features (vector generation)
  - [ ] Lightroom AI (masking, subject selection, background removal)
- [ ] **Maintaining artistic identity with AI**
  - [ ] Using AI as starting point vs final output
  - [ ] Style consistency across a project
  - [ ] Ethical use (not copying specific artists' styles)
- [ ] **AI for specific design disciplines**
  - [ ] Brand design (logo, identity system)
  - [ ] Web and UI design (Figma AI)
  - [ ] Print design (packaging, editorial)
  - [ ] Motion design (After Effects AI)
- [ ] **3D and spatial design AI**
  - [ ] Text-to-3D tools (Meshy, Luma Dream Machine)
  - [ ] AI texture and material generation
- [ ] **Protecting your work from AI scraping**

🏋️ **Exercises:**
1. Generate 20 brand identity concepts for a fictional company in under 1 hour using Midjourney
2. Use Photoshop Firefly to enhance a client photo — extend, relight, and modify
3. Build an AI-assisted design workflow for a real project and document time savings

🛠️ **Project: Brand Identity System** — Using only AI tools, create a complete brand identity for a business: name, logo (multiple concepts), color system, typography pairing, 3 marketing materials, and a brand guideline document.

---

### 6.4 AI for Writers and Journalists

- [ ] **AI for research**
  - [ ] Perplexity Academic for peer-reviewed sources
  - [ ] Consensus for academic consensus finding
  - [ ] NotebookLM for analyzing research documents
  - [ ] Verifying AI-provided citations
- [ ] **AI in the writing process**
  - [ ] AI for outlining and structure
  - [ ] AI for first drafts (starting points, not final product)
  - [ ] AI for editing, clarity, and style improvement
  - [ ] Maintaining your voice when using AI
- [ ] **AI for specific writing types**
  - [ ] Long-form articles (Claude's 200K context window)
  - [ ] SEO writing (Surfer SEO + AI)
  - [ ] Technical writing (documentation, manuals)
  - [ ] Fiction writing (plot, dialogue, scene development)
  - [ ] Non-fiction book writing (structure, research synthesis)
- [ ] **Ethical journalism with AI**
  - [ ] AI for transcription (not for quotes)
  - [ ] Disclosure standards for AI use
  - [ ] Fact-checking AI research outputs
  - [ ] Verification workflow for AI-generated statistics

🏋️ **Exercises:**
1. Research and draft a 1,500-word article on a topic you're expert in using AI assistance — how much time saved?
2. Use AI to edit your own writing for clarity — identify patterns in what AI consistently improves
3. Build a research workflow: Perplexity for sources → NotebookLM for synthesis → Claude for outline → writing

---

### 6.5 AI for Educators and Students

- [ ] **AI as a learning tool**
  - [ ] Using AI for personalized explanation ("explain this differently")
  - [ ] Concept mapping and visual learning with AI
  - [ ] AI for practice problems and quizzes
  - [ ] AI tutoring for difficult subjects
- [ ] **AI for educators**
  - [ ] Lesson plan generation
  - [ ] Differentiated instruction (modify content for different levels)
  - [ ] Rubric and assessment creation
  - [ ] Personalized feedback at scale
  - [ ] Reducing administrative burden with AI
- [ ] **Student research skills in the AI age**
  - [ ] Using Perplexity Academic correctly (always read the sources)
  - [ ] Citation verification (AI-generated citations are often wrong)
  - [ ] Academic integrity and AI disclosure policies
- [ ] **AI for students** (ethical use)
  - [ ] AI as a study tool, not a submission tool
  - [ ] Using AI to understand, not to bypass learning
  - [ ] Building skills alongside AI, not instead of them

---

### 6.6 AI for Healthcare Professionals (Awareness)

> ⚠️ **Important:** AI in healthcare is highly regulated. Never use unverified AI for clinical decisions. This section covers AI as an administrative and research tool, not a diagnostic tool.

- [ ] Administrative applications (safe and legal)
  - [ ] Clinical documentation assistants (ambient listening)
  - [ ] EHR note summarization
  - [ ] Medical literature summarization
  - [ ] Administrative scheduling and communication
- [ ] Research applications
  - [ ] Medical literature review with Perplexity/Consensus
  - [ ] Understanding clinical trial summaries
- [ ] Understanding current AI diagnostic tools (awareness only)
  - [ ] Radiology AI (reading scans)
  - [ ] Pathology AI
  - [ ] Drug interaction databases
- [ ] HIPAA compliance and AI tools
  - [ ] What constitutes PHI and what can NOT go in public AI tools
  - [ ] HIPAA-compliant AI alternatives
  - [ ] Business Associate Agreements (BAA) for AI vendors
- [ ] Critical thinking about AI medical claims

---

### 6.7 AI for Legal Professionals (Awareness)

> ⚠️ **Important:** Always verify AI legal outputs with current law. AI cannot provide legal advice. Never cite AI-generated case law without verification (AI fabricates citations).

- [ ] Safe use cases for legal AI
  - [ ] Initial document drafts (reviewed and revised by humans)
  - [ ] Research starting points (always verify in official databases)
  - [ ] Client communication drafts
  - [ ] Summarizing long documents
- [ ] High-risk AI use cases to avoid
  - [ ] Relying on AI for case law citations (frequently hallucinated)
  - [ ] Using AI for jurisdiction-specific tax or regulatory advice
- [ ] Legal-specific AI tools
  - [ ] Harvey AI (legal research assistant)
  - [ ] Clio (practice management with AI)
  - [ ] Westlaw AI features
- [ ] Attorney-client privilege and AI
  - [ ] Data residency and confidentiality concerns
  - [ ] What info can go to which tools

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## SECTION 7: STAYING CURRENT WITH AI
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### 7.1 Essential Newsletters and Media

- [ ] **Subscribe to at least 3:**
  - [ ] The Neuron (550K+ readers, tools + trends)
  - [ ] Superhuman AI (productivity focus)
  - [ ] Prompt Engineering Daily
  - [ ] Ben's Bites (quick AI news daily)
  - [ ] TLDR AI (technical summaries)
  - [ ] MIT Technology Review (deeper analysis)

### 7.2 Essential Follows (YouTube / Social)

- [ ] **YouTube:**
  - [ ] Matt Wolfe (AI tools for creators)
  - [ ] Marques Brownlee (tech and AI reviews)
  - [ ] Fireship (fast-paced AI explainers)
  - [ ] Two Minute Papers (AI research simplified)
  - [ ] Andrej Karpathy (technical but accessible)
- [ ] **Twitter/X:** @sama, @karpathy, @emollick, @rowancheung

### 7.3 Communities

- [ ] **Reddit:** r/artificial, r/singularity, r/ChatGPT, r/StableDiffusion
- [ ] **Discord:** Midjourney, Runway, n8n, AI community servers
- [ ] **LinkedIn:** Follow AI practitioners in your industry

### 7.4 Habits for Staying Current

- [ ] Set aside 20 minutes weekly to try a new AI tool
- [ ] Maintain a personal log of AI tools you've tested (notes on what each does)
- [ ] Evaluate new tools with 3 questions: What does it do uniquely? What does it cost? Does it fit my workflow?
- [ ] Join at least one domain-specific AI community (musicians using AI, filmmakers using AI, etc.)

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## PATH 0 CAPSTONE PROJECTS
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Choose at least ONE of the following based on your domain:

**1. The AI-Powered Content Machine**
Build an end-to-end AI workflow that takes a single topic idea and produces: a blog post, social media captions (5 platforms), an AI video script, a thumbnail image, and a short promo video. Fully automated after initial input. Document the entire workflow.

**2. The AI Business Toolkit**
Create a complete AI toolkit for a real small business: AI customer FAQ chatbot, automated email follow-up sequence, social media content calendar (3 months), brand visual assets, and a pitch deck. Time yourself — target: under 8 hours total.

**3. The AI Creative Project**
Pick your creative medium (music, film, design, writing). Create a complete professional-quality creative work using AI tools. Document what you used, how you used it, and how much time you saved vs. traditional methods.

**4. The AI Power User Portfolio**
Document your personal AI stack: 10 tools you actively use, why, how, and with what results. Create 5 tutorial-style walkthroughs of your most powerful workflows. Publish as a blog, video series, or Notion workspace.

---

> **What comes next:**
> If you've completed this path and want to go deeper, consider **Path 2: AI Application Developers** — you'll learn enough Python and technical knowledge to build your own AI tools instead of just using them.
