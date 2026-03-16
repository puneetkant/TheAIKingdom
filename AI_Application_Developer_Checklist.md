# The Complete AI Application Developer Checklist
### Path 2: Building Production AI Applications with LLMs

> **Instructions:** Check off each item as you complete it. Progress sequentially through Phases 1–8, then extend with Bonus Tracks. Prerequisites: Basic Python (Phase 1 of the main checklist). You don't need to finish Phases 2–4 of the main checklist to start here — but the more math you know, the deeper your understanding.
>
> **Who this is for:** Engineers and developers who want to build AI-powered products, APIs, agents, and systems using LLMs — not necessarily train models from scratch.
>
> **Reorganization rationale:** Phases are ordered so each phase builds on the previous. Evaluation (Phase 2) is placed early so you can measure your work throughout. Safety (Phase 5) comes before production deployment. Fine-tuning (Phase 6) is placed after agents, since you'll understand what behavior you want to instill. Inference Mastery (Phase 8) is the deepest topic and naturally comes last.

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## PHASE 1: LLM FOUNDATIONS & API LITERACY
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

> **Why first:** You must understand what LLMs are, how to access them, and how to communicate with them before you can build anything else. Prompt engineering is included here because it's the primary skill for controlling model behavior before any advanced technique is applied.

---

### 1.1 How Large Language Models Work (Conceptual)

#### 1.1.1 The Transformer Architecture (Intuition)

📚 **Best Resources to Learn:**
- 3Blue1Brown "Attention in Transformers" — youtube.com (search "3b1b transformers") — **THE best visual intro**
- Jay Alammar "The Illustrated Transformer" — jalammar.github.io/illustrated-transformer — **must read**
- Andrej Karpathy "Let's build GPT from scratch" — youtube.com — **builds GPT step by step**
- "Attention is All You Need" paper (Vaswani et al.) — arxiv.org/abs/1706.03762 — **the original**

- [ ] What is a language model — predicting the next token
- [ ] Tokenization
  - [ ] What a token is (not a word, not a character)
  - [ ] Byte-Pair Encoding (BPE)
  - [ ] SentencePiece tokenization
  - [ ] Tokenizer vocabulary size and its implications
  - [ ] Tokenizing different languages, code, numbers
  - [ ] Token count vs word count heuristics (~0.75 words per token)
- [ ] Embeddings
  - [ ] Word/token embeddings
  - [ ] Positional encodings (sinusoidal vs learned)
  - [ ] Rotary Position Embedding (RoPE)
  - [ ] ALiBi positional encoding
  - [ ] Embedding dimension and its role
- [ ] Attention mechanism
  - [ ] Self-attention: Q, K, V matrices (intuition)
  - [ ] Scaled dot-product attention
  - [ ] Multi-head attention
  - [ ] Causal (masked) attention (decoder)
  - [ ] Cross-attention (encoder-decoder models)
  - [ ] Attention scores visualization
  - [ ] Grouped Query Attention (GQA)
  - [ ] Multi-Query Attention (MQA)
- [ ] Transformer blocks
  - [ ] Feed-forward layers (MLP in transformer)
  - [ ] Layer normalization (pre-norm vs post-norm)
  - [ ] Residual connections
  - [ ] Activation functions (GELU, SiLU/Swish)
- [ ] Encoder vs decoder vs encoder-decoder architectures
  - [ ] GPT-style (decoder-only): good for generation
  - [ ] BERT-style (encoder-only): good for understanding/classification
  - [ ] T5-style (encoder-decoder): good for seq2seq
- [ ] Context window (context length)
  - [ ] What it means practically
  - [ ] Short vs long context models
  - [ ] Extending context length (RoPE scaling, LongRoPE, etc.)
- [ ] Scale and emergent capabilities
  - [ ] Scaling laws (Chinchilla, Kaplan)
  - [ ] In-context learning (prompting without fine-tuning)
  - [ ] Chain-of-thought as emergent behavior
- [ ] Popular model families
  - [ ] OpenAI GPT series (GPT-3.5, GPT-4, o1/o3)
  - [ ] Anthropic Claude series
  - [ ] Meta LLaMA series (open weights)
  - [ ] Google Gemini / PaLM series
  - [ ] Mistral series
  - [ ] Qwen, DeepSeek, Falcon series
  - [ ] Phi (Microsoft small models)
  - [ ] Open-source vs closed-source tradeoffs

🏋️ **Exercises:**
1. Use tiktoken to tokenize 20 different strings — compare token counts for English, code, Chinese, numbers, whitespace
2. Watch 3Blue1Brown's transformer video, then write a 1-page explanation in your own words with diagrams
3. Follow Karpathy's "Let's build GPT" video — actually code along, run the notebook
4. Visualize attention weights from a small GPT model using BertViz or similar tool
5. Calculate the approximate parameter count for a GPT-2 model (117M) using the architecture spec

🛠️ **Mini-Project:** Build a **Tokenizer Explorer Tool** — given any text, show: token count, token IDs, token strings, cost estimate at $X/1M tokens, comparison across GPT-4/Claude/LLaMA tokenizers.

---

### 1.2 LLM APIs: The Developer's Toolkit

#### 1.2.1 OpenAI API

📚 **Best Resources to Learn:**
- OpenAI API reference — platform.openai.com/docs/api-reference
- OpenAI Cookbook — github.com/openai/openai-cookbook — **dozens of practical examples**
- OpenAI Python SDK — github.com/openai/openai-python

- [ ] API key management and security
  - [ ] Environment variables (never hardcode keys!)
  - [ ] `.env` files and `python-dotenv`
  - [ ] Key rotation best practices
- [ ] Chat Completions API
  - [ ] `messages` array structure (system, user, assistant)
  - [ ] `model` parameter — choosing the right model
  - [ ] `temperature` — controlling randomness (0 = deterministic, 2 = chaotic)
  - [ ] `max_tokens` — output length limit
  - [ ] `top_p` — nucleus sampling
  - [ ] `frequency_penalty` and `presence_penalty` — repetition control
  - [ ] `stop` sequences
  - [ ] `n` — multiple completions
  - [ ] `stream=True` — streaming responses
  - [ ] `seed` — reproducibility (best effort)
  - [ ] `logprobs` — token probabilities
  - [ ] `response_format` — JSON mode
- [ ] Embeddings API
  - [ ] `text-embedding-3-small` vs `text-embedding-3-large`
  - [ ] Embedding dimensions
  - [ ] Batch embedding
- [ ] Vision API (multimodal)
  - [ ] Sending images (base64 vs URL)
  - [ ] Image detail levels (low, high, auto)
- [ ] Function Calling / Tool Use
  - [ ] Defining tools with JSON schema
  - [ ] Parsing tool calls from response
  - [ ] Parallel tool calls
  - [ ] Forced tool use (`tool_choice`)
- [ ] Structured Outputs
  - [ ] JSON schema enforcement
  - [ ] `response_format` with `json_schema`
  - [ ] Pydantic integration
- [ ] Files API (uploading files)
- [ ] Assistants API (stateful, long conversations)
- [ ] Rate limits and quotas
  - [ ] TPM (tokens per minute) limits
  - [ ] RPM (requests per minute) limits
  - [ ] Exponential backoff with jitter
- [ ] Cost estimation
  - [ ] Input vs output token pricing
  - [ ] Batch API (50% cost reduction)
  - [ ] Caching (OpenAI prompt caching)
- [ ] Error handling
  - [ ] `RateLimitError`, `APIConnectionError`, `APITimeoutError`, `AuthenticationError`
  - [ ] Retry logic

🏋️ **Exercises:**
1. Build a simple CLI chatbot with conversation history using the OpenAI chat completions API
2. Experiment with temperature: generate the same prompt 10 times at temp=0, 0.5, 1.0, 1.5 — compare outputs
3. Implement exponential backoff retry logic from scratch for API calls
4. Use logprobs to compute the perplexity of a text passage
5. Build a cost tracker that logs input/output tokens and running cost for every API call
6. Implement streaming: show tokens appearing in real-time in the terminal

#### 1.2.2 Anthropic Claude API

📚 **Best Resources to Learn:**
- Anthropic API docs — docs.anthropic.com
- Anthropic Cookbook — github.com/anthropics/anthropic-cookbook
- Anthropic SDK — github.com/anthropics/anthropic-sdk-python

- [ ] Messages API structure (similar to but different from OpenAI)
  - [ ] `system` as a top-level field
  - [ ] `messages` array
  - [ ] `model` selection (claude-opus-4-6, claude-sonnet-4-6, claude-haiku-4-5)
- [ ] Claude-specific parameters
  - [ ] `max_tokens` (required in Claude)
  - [ ] Temperature, top_p, top_k
  - [ ] `stop_sequences`
  - [ ] Extended thinking / reasoning (`thinking` parameter)
- [ ] Tool use with Claude
  - [ ] Tool definition with `input_schema`
  - [ ] Handling `tool_use` blocks in response
  - [ ] `tool_result` messages
  - [ ] Computer use tools (beta)
- [ ] Streaming with Claude
- [ ] Vision/multimodal with Claude
- [ ] Prompt caching (reduce cost for long system prompts)
  - [ ] `cache_control: {type: "ephemeral"}`
  - [ ] Cache lifetime and invalidation
- [ ] Batch API
- [ ] Model context protocol (MCP)

🏋️ **Exercises:**
1. Port your OpenAI chatbot to Claude — note the API differences
2. Use Claude's extended thinking on a hard math problem — observe the reasoning traces
3. Implement prompt caching for a system prompt and measure cost savings
4. Build a tool-use example: Claude can call a calculator function and a weather API

#### 1.2.3 Other Key APIs and Open-Source Models

📚 **Best Resources to Learn:**
- Google AI Studio — aistudio.google.com (free Gemini access)
- Together AI — together.ai (open-source model hosting)
- Groq — groq.com (ultra-fast inference)
- Ollama — ollama.ai (local model running)
- LiteLLM docs — docs.litellm.ai

- [ ] Google Gemini API (Google AI Studio / Vertex AI)
  - [ ] Gemini models: Flash vs Pro vs Ultra
  - [ ] Long context (1M+ tokens)
  - [ ] Native multimodal (video, audio, images, text)
- [ ] Running models locally with Ollama
  - [ ] Installing Ollama and pulling models (`ollama pull llama3.2`)
  - [ ] Ollama REST API
  - [ ] LLaMA, Mistral, Phi, Qwen models locally
  - [ ] Hardware requirements (VRAM, RAM)
- [ ] Hugging Face Inference API
  - [ ] Serverless inference endpoint
  - [ ] Dedicated endpoints
- [ ] LiteLLM — unified interface for all providers
  - [ ] Same API for OpenAI, Anthropic, Gemini, open-source
  - [ ] Fallback and load balancing
  - [ ] Cost tracking across providers
- [ ] Groq API (fast inference via LPU chips)
- [ ] Together AI, Fireworks AI, Replicate

🏋️ **Exercises:**
1. Install Ollama, run Llama 3.2, Mistral, and Phi locally — compare quality and speed
2. Use LiteLLM to build a provider-agnostic chatbot that can swap models via config
3. Run the same 10 prompts on GPT-4o, Claude 3.5 Sonnet, and Gemini Flash — compare outputs
4. Measure latency: tokens/second for local (Ollama) vs cloud (Groq) vs standard cloud (OpenAI)

🛠️ **PROJECT: AI Model Comparison Dashboard** — Web app where you enter a prompt and see responses from 5 providers side-by-side, with latency, cost, and token count for each.

---

### 1.3 Prompt Engineering

#### 1.3.1 Fundamentals of Prompting

📚 **Best Resources to Learn:**
- Anthropic Prompt Engineering Guide — docs.anthropic.com/en/docs/build-with-claude/prompt-engineering
- OpenAI Prompt Engineering Guide — platform.openai.com/docs/guides/prompt-engineering
- DeepLearning.AI "ChatGPT Prompt Engineering for Developers" (FREE) — learn.deeplearning.ai
- Prompt Engineering Guide — promptingguide.ai — **comprehensive reference**
- "Chain-of-Thought Prompting" paper — arxiv.org/abs/2201.11903

- [ ] Anatomy of a good prompt
  - [ ] Clear task description
  - [ ] Role / persona setting
  - [ ] Context provision
  - [ ] Output format specification
  - [ ] Examples (few-shot)
  - [ ] Constraints and guardrails
- [ ] Zero-shot prompting
- [ ] Few-shot prompting
  - [ ] Example selection strategies
  - [ ] Example ordering effects
  - [ ] Dynamic few-shot (retrieve relevant examples)
- [ ] Instruction following
  - [ ] Be specific and explicit
  - [ ] Use XML/markdown tags to structure prompts
  - [ ] Numbered lists vs prose instructions
  - [ ] Breaking complex tasks into steps
- [ ] Role prompting
  - [ ] System prompt vs user prompt for roles
  - [ ] Expert persona effects
- [ ] Chain-of-Thought (CoT) prompting
  - [ ] "Let's think step by step"
  - [ ] Zero-shot CoT
  - [ ] Few-shot CoT (with reasoning examples)
  - [ ] CoT for different task types (math, logic, planning)
- [ ] Self-Consistency
  - [ ] Sample multiple CoT paths, majority vote
  - [ ] When to use (high-stakes reasoning)
- [ ] Tree of Thoughts (ToT)
  - [ ] Exploring reasoning trees
  - [ ] Backtracking and evaluation
- [ ] ReAct (Reason + Act)
  - [ ] Interleaving reasoning and tool use
- [ ] Step-back prompting
- [ ] Least-to-most prompting (decompose, then solve sequentially)
- [ ] Generated Knowledge prompting
- [ ] Prompt chaining (multi-step pipelines)
- [ ] Output format control
  - [ ] JSON / structured output
  - [ ] Markdown formatting
  - [ ] Length control
  - [ ] "Answer in exactly N words" type constraints
- [ ] Handling long context
  - [ ] Lost in the middle problem (put important info at start/end)
  - [ ] Summarization strategies for long docs
- [ ] Common failure modes
  - [ ] Hallucination
  - [ ] Prompt injection attacks
  - [ ] Verbose / padded responses
  - [ ] Sycophancy ("you're right" even when wrong)

🏋️ **Exercises:**
1. Take 5 tasks (summarization, classification, extraction, Q&A, code generation), write bad prompts and then iteratively improve them — document what changed and why
2. Implement self-consistency: solve 10 math problems using CoT, sample 5 times, majority vote — compare accuracy to single-sample
3. Demonstrate the "lost in the middle" effect: hide the answer in different positions of a 20-document context, measure retrieval accuracy
4. Write prompts for JSON extraction from unstructured text — test on 20 varied inputs, measure parse success rate
5. Build a prompt versioning system: save prompts with metadata, version numbers, eval scores
6. Demonstrate sycophancy: present a wrong answer confidently, observe the model agree, then design a counter-prompt

#### 1.3.2 Advanced Prompting Techniques

📚 **Best Resources to Learn:**
- "Large Language Models are Zero-Shot Reasoners" (Kojima et al.) — arxiv.org/abs/2205.11916
- "Tree of Thoughts" paper — arxiv.org/abs/2305.10601
- "Reflexion" paper — arxiv.org/abs/2303.11366
- Langchain Expression Language (LCEL) docs — python.langchain.com

- [ ] Meta-prompting (model writes its own prompts)
- [ ] Prompt compression / distillation
  - [ ] LLMLingua
  - [ ] Selective context
- [ ] Automatic Prompt Optimization
  - [ ] DSPy framework (program synthesis over prompts)
  - [ ] APE (Automatic Prompt Engineer)
  - [ ] OPRO (Optimization by PROmpting)
- [ ] Prompt injection and defense
  - [ ] Direct injection
  - [ ] Indirect injection (via documents in context)
  - [ ] Defense strategies
- [ ] System prompt engineering
  - [ ] Persona persistence
  - [ ] Behavioral constraints
  - [ ] Format enforcement
  - [ ] Knowledge cutoff handling
- [ ] Context management strategies
  - [ ] Sliding window approaches
  - [ ] Summary buffers
  - [ ] Dynamic context selection
- [ ] Temperature and sampling strategy selection
  - [ ] Low temp for factual/extraction tasks
  - [ ] High temp for creative tasks
  - [ ] Top-p vs top-k tradeoffs

🏋️ **Exercises:**
1. Use DSPy to auto-optimize a prompt for a classification task — compare with hand-written prompt
2. Build a prompt injection demo: create a system prompt, then craft a user message that tries to override it — then patch it
3. Implement LLM-based prompt evaluation: use one LLM to score another LLM's outputs
4. Create a "prompt library" with 20 reusable prompt templates for common tasks (extraction, summarization, classification, etc.)

🛠️ **Mini-Project: Prompt Optimization Lab** — Take 5 tasks, implement baseline zero-shot → few-shot → CoT → self-consistency, measure accuracy on eval set, visualize improvement chain.

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## PHASE 2: EVALUATION & BENCHMARKING
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

> **Why second:** Before building complex systems, you must know how to measure whether they work. Evaluation skills apply to every phase that follows — prompt quality, RAG accuracy, agent reliability, fine-tuned model behavior, and production health all require evaluation. Learning this early means you can iterate with data, not guesswork.

---

### 2.1 Standard Benchmarks

📚 **Best Resources to Learn:**
- EleutherAI LM Evaluation Harness — github.com/EleutherAI/lm-evaluation-harness
- OpenAI Evals — github.com/openai/evals
- MT-Bench / Chatbot Arena (LMSYS) — lmsys.org/blog/2023-06-22-leaderboard

- [ ] Standard benchmarks
  - [ ] MMLU (knowledge across 57 subjects)
  - [ ] HellaSwag (commonsense reasoning)
  - [ ] TruthfulQA (hallucination)
  - [ ] GSM8K (math reasoning)
  - [ ] HumanEval / MBPP (code)
  - [ ] ARC-Challenge (reasoning)
  - [ ] MT-Bench (conversation quality)
- [ ] Evaluation harness
  - [ ] lm-evaluation-harness by EleutherAI
  - [ ] Running benchmarks locally
  - [ ] Comparing models on same benchmarks
- [ ] HELM (Holistic Evaluation of Language Models)
- [ ] BIG-Bench (beyond imitation games)
- [ ] AgentBench (agent evaluation)
- [ ] Evaluation pitfalls
  - [ ] Contamination (test data in training)
  - [ ] Gaming benchmarks (training on test distributions)
  - [ ] Metric vs real-world performance gap

🏋️ **Exercises:**
1. Run lm-evaluation-harness on LLaMA 3.2 3B and Mistral 7B — compare on MMLU, HellaSwag, GSM8K

---

### 2.2 Building Custom Evals

📚 **Best Resources to Learn:**
- OpenAI Evals framework — github.com/openai/evals
- DeepEval docs — docs.confident-ai.com
- TruLens docs — trulens.org

- [ ] Defining your own task and metrics
- [ ] LLM-as-judge (GPT-4 grading)
  - [ ] Criteria-based evaluation
  - [ ] Side-by-side comparison (A/B)
  - [ ] Bias in LLM judges (position, verbosity, self-enhancement)
- [ ] Human evaluation
- [ ] Eval dataset curation
  - [ ] Human-labeled QA pairs
  - [ ] Synthetic QA generation with LLM
  - [ ] LLM-curated difficult questions
- [ ] A/B testing LLM responses
  - [ ] Routing % of traffic to new prompt/model
  - [ ] Statistical significance testing

🏋️ **Exercises:**
1. Build a custom eval for a specific domain: 50 questions, LLM-as-judge scoring on faithfulness and relevance
2. Detect eval contamination: generate questions similar to a test set, check if model performs much better on those
3. Build a prompt versioning system: run the same eval set after every prompt change, track metric regressions

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## PHASE 3: RETRIEVAL AUGMENTED GENERATION (RAG)
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

> **Why third:** After you can prompt models and measure results, the next critical skill is giving LLMs access to external knowledge. RAG is the most widely used technique for grounding LLMs in proprietary or up-to-date data, and evaluation (Phase 2) now gives you tools to assess RAG quality rigorously.

---

### 3.1 Vector Databases and Embeddings

#### 3.1.1 Text Embeddings

📚 **Best Resources to Learn:**
- Sentence Transformers docs — sbert.net — **industry standard**
- OpenAI Embeddings Guide — platform.openai.com/docs/guides/embeddings
- MTEB Leaderboard — huggingface.co/spaces/mteb/leaderboard — **benchmark for embedding models**
- Jay Alammar "The Illustrated Word2Vec" — jalammar.github.io

- [ ] What is a text embedding (semantic vector representation)
- [ ] Embedding models
  - [ ] OpenAI `text-embedding-3-small` / `text-embedding-3-large`
  - [ ] Cohere `embed-v3`
  - [ ] Sentence Transformers (open-source)
    - [ ] `all-MiniLM-L6-v2` (fast, small)
    - [ ] `bge-large-en-v1.5` (BAAI, high quality)
    - [ ] `E5-large` (Microsoft)
    - [ ] `Nomic-embed-text`
    - [ ] `mxbai-embed-large`
  - [ ] ColBERT (late interaction models)
  - [ ] Binary embeddings (Matryoshka)
- [ ] Choosing embedding dimensions (tradeoff: storage vs quality)
- [ ] Matryoshka Representation Learning (MRL)
  - [ ] Truncating embeddings without quality loss
- [ ] Cosine similarity vs dot product vs Euclidean distance
- [ ] Batch embedding for efficiency
- [ ] Embedding normalization
- [ ] Cross-lingual embeddings
- [ ] Code embeddings
- [ ] Multi-modal embeddings (CLIP for images+text)

🏋️ **Exercises:**
1. Embed 500 sentences, visualize with UMAP — observe semantic clustering
2. Compute cosine similarity between 20 sentence pairs — validate that semantically similar = higher similarity
3. Compare 5 embedding models on a sentence similarity task using MTEB protocol
4. Implement semantic search: embed 1000 Wikipedia passages, query with natural language, return top-5
5. Demonstrate Matryoshka: truncate embeddings to 128/256/512/1536 dims, compare search quality

#### 3.1.2 Vector Databases

📚 **Best Resources to Learn:**
- Pinecone docs — docs.pinecone.io (hosted, easiest to start)
- Weaviate docs — weaviate.io/developers/weaviate
- Qdrant docs — qdrant.tech/documentation
- ChromaDB docs — docs.trychroma.com (local, open-source, great for prototyping)
- pgvector docs — github.com/pgvector/pgvector (vector search in PostgreSQL)
- FAISS — github.com/facebookresearch/faiss (Meta, for in-memory ANN)

- [ ] What a vector database does (ANN search at scale)
- [ ] Approximate Nearest Neighbor (ANN) algorithms
  - [ ] HNSW (Hierarchical Navigable Small World) — most common
  - [ ] IVF (Inverted File Index)
  - [ ] PQ (Product Quantization) for compression
  - [ ] ScaNN (Google)
  - [ ] FAISS indexes (Flat, IVF, HNSW)
- [ ] Core operations
  - [ ] Upsert / index documents
  - [ ] Query / similarity search
  - [ ] Metadata filtering (hybrid search)
  - [ ] Delete and update
- [ ] ChromaDB (local dev)
  - [ ] Collections
  - [ ] Persistent vs in-memory mode
  - [ ] Metadata filtering
- [ ] Pinecone (production hosted)
  - [ ] Indexes, namespaces
  - [ ] Serverless vs pod-based
  - [ ] Metadata filtering
- [ ] Weaviate
  - [ ] Schema definition
  - [ ] GraphQL query interface
  - [ ] Hybrid search (dense + sparse)
- [ ] Qdrant
  - [ ] Collections and payloads
  - [ ] Filtering strategies
  - [ ] Quantization options
- [ ] pgvector (PostgreSQL extension)
  - [ ] When to use (existing Postgres infra)
  - [ ] `<->` cosine distance operator
  - [ ] Index types (HNSW, IVF)
- [ ] Choosing a vector database (decision criteria)
- [ ] Vector DB performance benchmarks (ANN Benchmarks)

🏋️ **Exercises:**
1. Index 10,000 documents into ChromaDB, run 100 queries, measure avg latency
2. Implement the same search pipeline in Pinecone, FAISS, and ChromaDB — compare dev experience
3. Demonstrate metadata filtering: store products with {category, price, brand}, query "red shoes under $100"
4. Benchmark HNSW vs brute-force (Flat) at 1K, 10K, 100K documents — plot speed vs accuracy tradeoff
5. Add pgvector to a PostgreSQL database, implement semantic search alongside traditional SQL filters

---

### 3.2 Building RAG Systems

#### 3.2.1 Naive / Basic RAG

📚 **Best Resources to Learn:**
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020) — arxiv.org/abs/2005.11401 — **original RAG paper**
- LlamaIndex docs — docs.llamaindex.ai — **best for RAG pipelines**
- LangChain RAG tutorial — python.langchain.com/docs/tutorials/rag
- DeepLearning.AI "Building and Evaluating Advanced RAG" — learn.deeplearning.ai (FREE)
- "RAG Survey" — arxiv.org/abs/2312.10997

- [ ] RAG pipeline components
  - [ ] Ingestion (load → split → embed → store)
  - [ ] Retrieval (embed query → search → return chunks)
  - [ ] Generation (context + query → LLM → answer)
- [ ] Document loading
  - [ ] PDF: PyMuPDF, pdfplumber, pypdf
  - [ ] Word documents: python-docx
  - [ ] Web pages: BeautifulSoup, Trafilatura
  - [ ] Markdown, plain text
  - [ ] HTML with structure preservation
  - [ ] Spreadsheets (CSV, Excel)
  - [ ] Unstructured.io (handles many formats)
- [ ] Text chunking strategies
  - [ ] Fixed-size chunking (by character count)
  - [ ] Sentence splitting (NLTK, spaCy)
  - [ ] Recursive character text splitter (LangChain default)
  - [ ] Semantic chunking (embed sentences, split on semantic shifts)
  - [ ] Markdown-aware splitting (preserve headers)
  - [ ] Code-aware splitting
  - [ ] Chunk size selection (tradeoffs: 256 vs 512 vs 1024 tokens)
  - [ ] Overlap between chunks (prevent context loss at boundaries)
- [ ] Metadata enrichment
  - [ ] Source document, page number, section headers
  - [ ] Timestamps, authors
  - [ ] Custom metadata for filtering
- [ ] Basic retrieval
  - [ ] Top-K retrieval
  - [ ] Similarity threshold filtering
  - [ ] MMR (Maximal Marginal Relevance) for diversity
- [ ] Context assembly
  - [ ] Ordering retrieved chunks
  - [ ] Handling overlapping chunks
  - [ ] Context window management
- [ ] Generation with retrieved context
  - [ ] Prompt template for RAG
  - [ ] Citation and source attribution
  - [ ] "I don't know" handling when context is insufficient

🏋️ **Exercises:**
1. Build a basic RAG pipeline from scratch (no LangChain) — load a PDF, chunk it, embed it, query it
2. Compare chunking strategies: fixed 512 tokens vs sentence vs semantic — evaluate answer quality on 20 questions
3. Build a RAG system for your own documentation (README files, Notion export, etc.)
4. Implement source citation: after each answer, include "Sources: [doc1, p.3], [doc2, p.7]"
5. Test the "lost in the middle" problem in RAG: vary where the answer appears in retrieved chunks

#### 3.2.2 Advanced RAG Techniques

📚 **Best Resources to Learn:**
- "Advanced RAG Techniques" — towardsdatascience.com (search)
- LlamaIndex advanced RAG docs
- "RAPTOR" paper — arxiv.org/abs/2401.18059
- "HyDE" paper — arxiv.org/abs/2212.10496
- DeepLearning.AI "Advanced RAG" course — learn.deeplearning.ai

- [ ] **Query transformation**
  - [ ] HyDE (Hypothetical Document Embeddings) — generate hypothetical answer, embed that
  - [ ] Query decomposition (break complex query into sub-queries)
  - [ ] Step-back prompting (abstract from specific to general)
  - [ ] Multi-query retrieval (generate N query variants, union results)
  - [ ] Query expansion with LLM
- [ ] **Advanced indexing**
  - [ ] Hierarchical indexing (summary + detailed chunks)
  - [ ] RAPTOR (recursive tree-based indexing for multi-level summaries)
  - [ ] Small-to-big retrieval (child chunks → parent context)
  - [ ] Sentence window retrieval
  - [ ] ColBERT / late interaction for re-ranking
  - [ ] Knowledge graphs as index
- [ ] **Hybrid search**
  - [ ] Dense + sparse retrieval combination
  - [ ] BM25 (sparse/keyword search)
  - [ ] Reciprocal Rank Fusion (RRF) for merging results
  - [ ] Weaviate/Qdrant hybrid mode
- [ ] **Re-ranking**
  - [ ] Cross-encoder re-ranking (more accurate than bi-encoder)
  - [ ] Cohere Rerank
  - [ ] BGE-Reranker
  - [ ] LLM-based re-ranking
  - [ ] FlashRank (fast local re-ranker)
- [ ] **Contextual compression**
  - [ ] LLM-based context filtering (keep only relevant sentences)
  - [ ] Extractive compression vs abstractive
- [ ] **Multi-document handling**
  - [ ] Cross-document reasoning
  - [ ] Deduplication
  - [ ] Conflicting information handling
- [ ] **Iterative / agentic RAG**
  - [ ] FLARE (Forward-Looking Active REtrieval)
  - [ ] Self-RAG (model decides when to retrieve)
  - [ ] Corrective RAG (CRAG)
  - [ ] GraphRAG (Microsoft, knowledge graph + community summaries)
- [ ] **Multimodal RAG**
  - [ ] Image retrieval alongside text
  - [ ] Table extraction and querying
  - [ ] ColPali (visual document retrieval without OCR)
- [ ] **Streaming RAG**
  - [ ] Stream tokens while still retrieving
  - [ ] Progressive disclosure

🏋️ **Exercises:**
1. Implement HyDE: generate hypothetical answer → embed → retrieve — compare precision vs baseline on 30 questions
2. Implement multi-query retrieval: generate 3 query variants, union results, compare recall vs single query
3. Add BM25 + dense hybrid search using Qdrant or Weaviate, implement RRF fusion
4. Add a cross-encoder re-ranker (BGE or Cohere) after retrieval — measure improvement on a Q&A eval set
5. Build a small-to-big retrieval system: child sentence chunks → retrieve by similarity → return parent paragraph context
6. Implement RAPTOR: recursively cluster and summarize document chunks, build tree, query at multiple levels

#### 3.2.3 RAG Evaluation

📚 **Best Resources to Learn:**
- RAGAS library — github.com/explodinggradients/ragas — **standard RAG evaluation framework**
- "RAGAS: Automated Evaluation of RAG" paper — arxiv.org/abs/2309.15217
- TruLens docs — trulens.org (LLM evaluation)
- DeepEval docs — docs.confident-ai.com

- [ ] RAG-specific metrics
  - [ ] **Faithfulness** — is the answer grounded in retrieved context? (no hallucination)
  - [ ] **Answer Relevance** — does the answer address the question?
  - [ ] **Context Recall** — did retrieval find all necessary information?
  - [ ] **Context Precision** — are retrieved chunks relevant? (no noise)
  - [ ] **Context Entity Recall**
  - [ ] **Answer Correctness** — factual accuracy vs ground truth
- [ ] RAGAS framework
  - [ ] Generating synthetic evaluation datasets
  - [ ] Computing all RAGAS metrics
  - [ ] Analyzing bottlenecks (retrieval vs generation)
- [ ] Building eval datasets
  - [ ] Human-labeled QA pairs
  - [ ] Synthetic QA generation with LLM
  - [ ] LLM-curated difficult questions

🏋️ **Exercises:**
1. Build a RAG pipeline for a 100-page PDF, generate 50 synthetic QA pairs with GPT-4, run RAGAS evaluation
2. Ablation study: evaluate RAG with/without re-ranking, with/without hybrid search — show metric improvements
3. Implement an LLM judge that scores answers on faithfulness and relevance on a 1-5 scale
4. Create a RAG testing harness: run the same eval set after every pipeline change, track metric regressions

🛠️ **PROJECT: Enterprise Document Q&A System** — Full RAG system for a company's documentation (50+ PDFs). Features: hybrid search, re-ranking, streaming responses, source citations, RAGAS eval score ≥ 0.8 on held-out test set. Build as a FastAPI backend with a simple Streamlit or React UI.

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## PHASE 4: LLM AGENTS AND TOOL USE
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

> **Why fourth:** Agents combine everything from Phase 1–3: they use APIs, apply prompting strategies, and often perform retrieval. Building agents before fine-tuning (Phase 6) also helps you understand what behavioral gaps can't be closed with prompting alone — which clarifies when fine-tuning is actually warranted.

---

### 4.1 Agent Foundations

#### 4.1.1 What Are LLM Agents?

📚 **Best Resources to Learn:**
- "ReAct: Synergizing Reasoning and Acting" — arxiv.org/abs/2210.03629 — **foundational agent paper**
- "Reflexion" paper — arxiv.org/abs/2303.11366
- Lilian Weng "LLM Powered Autonomous Agents" blog — lilianweng.github.io/posts/2023-06-23-agent — **comprehensive overview**
- Andrew Ng "AI Agentic Design Patterns" — deeplearning.ai (short course)

- [ ] Agent definition: LLM + tools + memory + planning + action
- [ ] Agentic design patterns
  - [ ] Reflection (self-critique and improvement)
  - [ ] Tool use (calling functions/APIs)
  - [ ] Planning (task decomposition)
  - [ ] Multi-agent collaboration
- [ ] The ReAct loop: Thought → Action → Observation → Thought
- [ ] Agent vs RAG vs standard LLM (when to use each)
- [ ] Agent failure modes
  - [ ] Infinite loops
  - [ ] Wrong tool selection
  - [ ] Hallucinated tool calls
  - [ ] Context window exhaustion
  - [ ] Irreversible actions (safety concern)
- [ ] Agent reliability challenges
  - [ ] Success rates degrade with task complexity
  - [ ] Error propagation across steps
  - [ ] Grounding and verification

🏋️ **Exercises:**
1. Build a ReAct agent from scratch (no framework) that can do math, search Wikipedia, and answer questions
2. Implement a "reflection" loop: agent produces answer → critiques its own answer → revises → repeat N times

#### 4.1.2 Tools and Function Calling

📚 **Best Resources to Learn:**
- OpenAI Tool Use docs — platform.openai.com/docs/guides/function-calling
- Anthropic Tool Use docs — docs.anthropic.com/en/docs/build-with-claude/tool-use
- "Toolformer" paper — arxiv.org/abs/2302.04761

- [ ] Defining tools with JSON schema
  - [ ] Name, description, parameters
  - [ ] Required vs optional parameters
  - [ ] Type annotations (string, number, array, object, enum)
  - [ ] Description quality matters for tool selection
- [ ] Tool execution pattern
  - [ ] Detect tool call in LLM response
  - [ ] Execute the function
  - [ ] Return result back to LLM
  - [ ] Multi-turn tool use
- [ ] Parallel tool calls
- [ ] Nested / sequential tool calls
- [ ] Tool selection accuracy
- [ ] Tool error handling
  - [ ] Tool returns error → LLM retries or falls back
  - [ ] Timeout handling
- [ ] Common built-in tools
  - [ ] Web search (Tavily, Brave, SerpAPI)
  - [ ] Code execution (Python REPL, E2B)
  - [ ] Calculator / math
  - [ ] File system read/write
  - [ ] Database queries
  - [ ] Email / calendar (via APIs)
  - [ ] Image generation (DALL-E, Stable Diffusion)
  - [ ] Web browser / scraping
- [ ] Tool authorization and safety
  - [ ] Human-in-the-loop for destructive actions
  - [ ] Sandboxing code execution (E2B, Docker)
  - [ ] Rate limiting tools

🏋️ **Exercises:**
1. Build a tool-calling agent with 5 tools: web search, Wikipedia, calculator, weather API, currency converter
2. Implement proper error handling: tool fails → agent tries alternative approach
3. Create a "code interpreter" agent using E2B sandbox — agent writes Python, executes it, reads results
4. Build a database-querying agent: natural language → SQL → execute → present results

#### 4.1.3 Agent Memory Systems

📚 **Best Resources to Learn:**
- LlamaIndex Memory docs
- "MemGPT" paper — arxiv.org/abs/2310.08560
- Mem0 library — github.com/mem0ai/mem0

- [ ] Types of agent memory
  - [ ] **Sensory memory** — current context window (short-term)
  - [ ] **Short-term / working memory** — conversation history
  - [ ] **Long-term memory** — persistent facts across sessions
  - [ ] **Episodic memory** — past experiences / history
  - [ ] **Semantic memory** — general knowledge (model weights)
  - [ ] **Procedural memory** — how to do things (also in weights)
- [ ] Conversation history management
  - [ ] Message buffering (keep last N turns)
  - [ ] Sliding window with summarization
  - [ ] Token counting and trimming
- [ ] External memory storage
  - [ ] Vector DB for semantic recall
  - [ ] Key-value stores for entity memory
  - [ ] SQL/structured for facts
- [ ] Memory consolidation
  - [ ] Periodic summarization of old memories
  - [ ] Importance scoring (what to remember)
  - [ ] Forgetting curve (TTL for memories)
- [ ] Mem0 (persistent user memory)
- [ ] MemGPT architecture (OS-inspired paging)
- [ ] Zep (conversation memory platform)

🏋️ **Exercises:**
1. Build an agent with persistent memory across sessions using SQLite + vector search
2. Implement smart conversation history trimming: always keep system prompt + recent + summarize old
3. Build a personal assistant that remembers user preferences, past tasks, and key facts across multiple sessions

---

### 4.2 Agent Frameworks

#### 4.2.1 LangChain

📚 **Best Resources to Learn:**
- LangChain Python docs — python.langchain.com
- LangChain Expression Language (LCEL) guide
- "LangChain for LLM Application Development" — DeepLearning.AI (FREE)

- [ ] Core abstractions
  - [ ] Runnables and LCEL (pipe operator `|`)
  - [ ] Chains (sequential pipelines)
  - [ ] `PromptTemplate`, `ChatPromptTemplate`
  - [ ] Output parsers (Pydantic, JSON, string)
- [ ] LangChain agents
  - [ ] `create_react_agent`
  - [ ] `create_tool_calling_agent`
  - [ ] AgentExecutor
  - [ ] Streaming agent output
- [ ] Memory in LangChain
  - [ ] `ConversationBufferMemory`
  - [ ] `ConversationSummaryMemory`
  - [ ] `ConversationTokenBufferMemory`
- [ ] Document loaders and text splitters
- [ ] Vectorstores integration
- [ ] Callbacks and tracing (LangSmith)
- [ ] LangSmith for debugging and evaluation
  - [ ] Tracing LLM calls
  - [ ] Datasets and evals
  - [ ] Prompt management

🏋️ **Exercises:**
1. Build a multi-step chain with LCEL: classify → branch to different handlers → format output
2. Build a LangChain agent with 3 tools, trace execution in LangSmith
3. Implement a summarization chain for long documents (map-reduce pattern)

#### 4.2.2 LlamaIndex

📚 **Best Resources to Learn:**
- LlamaIndex docs — docs.llamaindex.ai
- LlamaIndex tutorials — docs.llamaindex.ai/en/stable/examples

- [ ] Core concepts: Documents, Nodes, Index, Query Engine
- [ ] Data connectors (SimpleDirectoryReader, web, databases)
- [ ] Indexes
  - [ ] VectorStoreIndex (main)
  - [ ] SummaryIndex (for summarization)
  - [ ] KnowledgeGraphIndex
  - [ ] DocumentSummaryIndex
- [ ] Retrievers and QueryEngines
  - [ ] Top-K retriever
  - [ ] BM25 retriever
  - [ ] RouterQueryEngine (multi-index routing)
- [ ] Sub-question query engine
- [ ] Response synthesizers
  - [ ] compact, refine, tree_summarize
- [ ] Agents with LlamaIndex
  - [ ] ReActAgent
  - [ ] FunctionCallingAgent
  - [ ] AgentWorkflow

🏋️ **Exercises:**
1. Build a RAG system with LlamaIndex: ingest 10 PDFs, query with RouterQueryEngine routing by topic
2. Use LlamaIndex sub-question query engine to answer complex multi-hop questions
3. Build an agent with LlamaIndex that has both retrieval tools and calculator tools

#### 4.2.3 LangGraph

📚 **Best Resources to Learn:**
- LangGraph docs — langchain-ai.github.io/langgraph
- LangGraph tutorials — langchain-ai.github.io/langgraph/tutorials

- [ ] Why LangGraph: stateful, cyclical agent graphs
- [ ] Core concepts: State, Nodes, Edges
- [ ] State management (TypedDict for state schema)
- [ ] Conditional edges (routing)
- [ ] Cycles and loops (for iterative agents)
- [ ] Checkpointing (persistence, resumability)
- [ ] Human-in-the-loop with LangGraph
  - [ ] `interrupt_before` / `interrupt_after`
  - [ ] Approving / rejecting tool calls
- [ ] Subgraphs (nested graphs)
- [ ] Streaming graph execution
- [ ] LangGraph Studio (visual debugging)
- [ ] Multi-agent patterns with LangGraph
  - [ ] Supervisor pattern
  - [ ] Hierarchical agents
  - [ ] Handoffs between agents

🏋️ **Exercises:**
1. Build a ReAct agent with LangGraph: nodes = LLM, tool execution; cycles = tool → llm → tool
2. Add human-in-the-loop: agent pauses before executing any write/delete tool, waits for approval
3. Build a research assistant graph: plan → search → summarize → evaluate → iterate if needed

#### 4.2.4 Multi-Agent Systems

📚 **Best Resources to Learn:**
- AutoGen docs — microsoft.github.io/autogen — **Microsoft multi-agent framework**
- CrewAI docs — docs.crewai.com
- "AgentBench" paper — arxiv.org/abs/2308.03688
- "More Agents Is All You Need" paper — arxiv.org/abs/2402.05120

- [ ] Multi-agent architectures
  - [ ] Supervisor / orchestrator pattern
  - [ ] Peer-to-peer collaboration
  - [ ] Hierarchical agents
  - [ ] Parallel agent execution
  - [ ] Debate / critique pattern (multiple agents argue)
- [ ] Agent specialization
  - [ ] Researcher, Writer, Reviewer, Executor
  - [ ] Domain-specific agents
- [ ] Inter-agent communication
  - [ ] Structured messages vs natural language
  - [ ] Shared state / blackboard
  - [ ] Message queues
- [ ] AutoGen
  - [ ] ConversableAgent
  - [ ] GroupChat
  - [ ] Code execution agent
- [ ] CrewAI
  - [ ] Crews, Agents, Tasks, Tools
  - [ ] Sequential vs hierarchical processes
- [ ] OpenAI Swarm (lightweight orchestration)
- [ ] MCP (Model Context Protocol)
  - [ ] What MCP is (standardized tool/context interface)
  - [ ] MCP servers (expose tools, resources, prompts)
  - [ ] MCP clients (connect to servers)
  - [ ] Building MCP servers
  - [ ] Existing MCP servers (filesystem, GitHub, Slack, etc.)

🏋️ **Exercises:**
1. Build a multi-agent research pipeline: Researcher agent → Critic agent → Writer agent (CrewAI or LangGraph)
2. Implement the debate pattern: two agents argue opposite positions, a judge agent decides the winner
3. Build an MCP server that exposes a company's internal database as tools for any MCP-compatible AI client
4. Implement parallel agent execution: fan-out multiple specialized agents, fan-in their results

🛠️ **PROJECT: Autonomous Research Assistant** — Multi-agent system that takes a research question, autonomously searches the web (Tavily), reads papers (ArXiv API), synthesizes findings, writes a 1,000-word report with citations, critiques its own report, and revises. Built with LangGraph with human-in-the-loop checkpoints.

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## PHASE 5: LLM SAFETY AND RELIABILITY
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

> **Why fifth:** Safety and structured outputs must be understood before you deploy anything to production (Phase 7) or fine-tune a model (Phase 6). Knowing failure modes, content risks, and output reliability requirements informs every design decision in the phases that follow.

---

### 5.1 Content Safety

📚 **Best Resources to Learn:**
- Llama Guard — meta-llama.github.io/llama-recipes/use_cases/responsible_ai/llama_guard
- Guardrails AI docs — docs.guardrailsai.com
- NeMo Guardrails docs — github.com/NVIDIA/NeMo-Guardrails
- "Constitutional AI" paper (Anthropic) — arxiv.org/abs/2212.08073

- [ ] Types of safety risks
  - [ ] Harmful content generation
  - [ ] Jailbreaks and prompt injection
  - [ ] PII leakage
  - [ ] Copyright violations
  - [ ] Misinformation / hallucination
  - [ ] Bias and discrimination
- [ ] Input guardrails
  - [ ] Keyword / regex filtering
  - [ ] ML-based toxicity classifiers (Perspective API)
  - [ ] LLM-based classification (Llama Guard, GPT-4 moderation)
  - [ ] PII detection (Presidio, AWS Comprehend)
  - [ ] Prompt injection detection
- [ ] Output guardrails
  - [ ] Factuality checking
  - [ ] Format validation (JSON schema, regex)
  - [ ] Length limits
  - [ ] Brand safety checks
  - [ ] Sensitive topic detection
- [ ] Guardrails libraries
  - [ ] Guardrails AI (validators framework)
  - [ ] NeMo Guardrails (NVIDIA, conversation flows)
  - [ ] Llama Guard (Meta's safety classifier)
- [ ] OpenAI Moderation API
- [ ] Anthropic's built-in Constitutional AI
- [ ] Rate limiting per user to prevent abuse

🏋️ **Exercises:**
1. Build a content moderation pipeline: input → Llama Guard → if safe, LLM → output → Guardrails output validator
2. Implement PII redaction using Presidio: detect names, emails, phone numbers, SSNs in user input
3. Red-team your chatbot: attempt 20 jailbreaks and document which succeed, then patch

---

### 5.2 Structured Outputs and Reliability

📚 **Best Resources to Learn:**
- Instructor library — github.com/jxnl/instructor — **best for structured outputs**
- Outlines library — github.com/outlines-dev/outlines (constrained generation)
- Pydantic docs — docs.pydantic.dev

- [ ] Structured output with Pydantic + Instructor
  - [ ] Define Pydantic models as output schema
  - [ ] `@instructor.patch` for OpenAI / Anthropic
  - [ ] Automatic retry on validation failure
  - [ ] Nested models, lists, enums
  - [ ] Streaming structured outputs
- [ ] JSON Schema enforcement
  - [ ] OpenAI structured outputs (`response_format`)
  - [ ] Outlines for constrained generation (local models)
  - [ ] Grammar-based constraints
- [ ] Retry and fallback patterns
  - [ ] Retry on parse failure (up to N times)
  - [ ] Fallback to a simpler output format
  - [ ] Partial output handling
- [ ] Output validation
  - [ ] Type checking
  - [ ] Range validation
  - [ ] Custom validators
  - [ ] Cross-field validation

🏋️ **Exercises:**
1. Use Instructor to extract structured data (events, people, dates) from 50 news articles — 100% parseable
2. Build an entity extraction API: input = free text, output = typed Pydantic model with validation
3. Implement retry-with-feedback: on validation failure, send error message back to LLM, ask it to fix

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## PHASE 6: FINE-TUNING AND MODEL ADAPTATION
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

> **Why sixth:** Fine-tuning is a power tool — it requires understanding what behavior you want (from Phases 1–5), how to evaluate it (Phase 2), and what safety constraints to preserve (Phase 5). Jumping to fine-tuning before building and evaluating prompt-based systems leads to wasted effort.

---

### 6.1 When and Why to Fine-Tune

📚 **Best Resources to Learn:**
- OpenAI Fine-Tuning Guide — platform.openai.com/docs/guides/fine-tuning
- "A Practical Guide to Fine-Tuning LLMs" — sebastianraschka.com (blog)
- DeepLearning.AI "Finetuning LLMs" — learn.deeplearning.ai (FREE)
- Hugging Face Fine-tuning course — huggingface.co/learn/nlp-course

- [ ] Fine-tuning vs prompting vs RAG decision framework
  - [ ] Use prompting when: task is general, data is scarce, speed of iteration matters
  - [ ] Use RAG when: knowledge needs to be up-to-date or proprietary
  - [ ] Use fine-tuning when: need consistent format/style, domain-specific vocabulary, latency/cost matters, behavior customization
- [ ] Types of fine-tuning
  - [ ] Supervised Fine-Tuning (SFT) — instruction following
  - [ ] Continued pretraining — domain adaptation
  - [ ] RLHF (Reinforcement Learning from Human Feedback)
  - [ ] DPO (Direct Preference Optimization) — simpler RLHF alternative
  - [ ] ORPO, SimPO, KTO — alternatives to DPO
- [ ] Full fine-tuning vs parameter-efficient fine-tuning (PEFT)
  - [ ] Full fine-tuning: update all parameters (expensive)
  - [ ] PEFT: update small fraction (much cheaper)
- [ ] Data requirements for fine-tuning
  - [ ] Quality > Quantity
  - [ ] Minimum dataset sizes (100–1000 examples for SFT)
  - [ ] Data formatting (chat format, instruction format)
  - [ ] Data deduplication and cleaning

🏋️ **Exercises:**
1. Fine-tune GPT-4o-mini via OpenAI API on a customer service dataset — compare before/after on 20 test queries
2. Decision exercise: given 5 different use cases, argue whether each needs prompting, RAG, or fine-tuning

---

### 6.2 Parameter-Efficient Fine-Tuning (PEFT)

#### 6.2.1 LoRA and Variants

📚 **Best Resources to Learn:**
- "LoRA" paper (Hu et al., 2021) — arxiv.org/abs/2106.09685 — **must read**
- "QLoRA" paper (Dettmers et al., 2023) — arxiv.org/abs/2305.14314
- Hugging Face PEFT library — huggingface.co/docs/peft
- Axolotl — github.com/axolotl-org/axolotl (fine-tuning framework)
- Unsloth — github.com/unslothai/unsloth (faster fine-tuning)

- [ ] LoRA (Low-Rank Adaptation)
  - [ ] Intuition: freeze base model, add low-rank weight matrices
  - [ ] Rank r and alpha hyperparameters
  - [ ] Which layers to apply LoRA to (QKV, all linear, etc.)
  - [ ] Merging LoRA weights back into base model
  - [ ] LoRA math: W = W₀ + BA (B: d×r, A: r×k, r << d)
  - [ ] Parameter count calculation
- [ ] QLoRA (Quantized LoRA)
  - [ ] 4-bit NormalFloat (NF4) quantization
  - [ ] Double quantization
  - [ ] Paged optimizer states
  - [ ] Enables fine-tuning LLaMA-65B on a single 48GB GPU
- [ ] LoRA variants
  - [ ] LoRA+
  - [ ] DoRA (Weight Decomposition LoRA)
  - [ ] LongLoRA (extending context)
  - [ ] GaLore (gradient low-rank projection)
  - [ ] Flora
- [ ] Other PEFT methods
  - [ ] Prefix tuning
  - [ ] Prompt tuning (soft prompts)
  - [ ] IA³ (Infused Adapter by Inhibiting and Amplifying)
  - [ ] Adapter layers

🏋️ **Exercises:**
1. Fine-tune LLaMA-3.2-3B with LoRA using PEFT + Transformers on a custom dataset (Google Colab free tier)
2. Compute parameter counts: how many trainable params with r=8 LoRA vs full fine-tuning for a 7B model?
3. Fine-tune with QLoRA on a 7B model — profile VRAM usage vs full fine-tuning

#### 6.2.2 Fine-Tuning Pipeline

📚 **Best Resources to Learn:**
- Hugging Face `trl` library — huggingface.co/docs/trl — SFT, DPO, RLHF
- Axolotl config examples — github.com/axolotl-org/axolotl/tree/main/examples
- Unsloth notebooks — github.com/unslothai/unsloth

- [ ] Dataset preparation
  - [ ] Chat template formatting (ChatML, LLaMA-3, Alpaca)
  - [ ] `apply_chat_template()`
  - [ ] Instruction-response pairs
  - [ ] Multi-turn conversation formatting
  - [ ] Dataset deduplication and quality filtering
- [ ] Training infrastructure
  - [ ] Hugging Face `Trainer` and `SFTTrainer`
  - [ ] `TrainingArguments`
  - [ ] Mixed precision training (BF16)
  - [ ] Gradient checkpointing (save memory)
  - [ ] Gradient accumulation (simulate large batch)
  - [ ] Packing / sequence packing for efficiency
- [ ] Hyperparameters for fine-tuning
  - [ ] Learning rate (2e-4 for LoRA, 1e-5 for full fine-tuning)
  - [ ] Epochs (1–3 for SFT to avoid overfitting)
  - [ ] Batch size
  - [ ] Warmup ratio
  - [ ] Weight decay
- [ ] Evaluation during training
  - [ ] Validation loss monitoring
  - [ ] Perplexity on held-out set
- [ ] DPO (Direct Preference Optimization)
  - [ ] Preference datasets (chosen vs rejected)
  - [ ] `DPOTrainer` from TRL
  - [ ] Beta hyperparameter
- [ ] Synthetic data generation for fine-tuning
  - [ ] Alpaca-style data generation (self-instruct)
  - [ ] GPT-4/Claude-generated training data
  - [ ] Evol-Instruct (more complex variants)
  - [ ] Magpie (self-synthesis from base models)
- [ ] Cloud platforms for fine-tuning
  - [ ] Modal (serverless GPU)
  - [ ] RunPod
  - [ ] Lambda Labs
  - [ ] Google Colab Pro
  - [ ] Vast.ai (cheap GPU rental)

🏋️ **Exercises:**
1. Generate 1,000 instruction-following examples using GPT-4o for a specific domain, fine-tune LLaMA 3.2 with QLoRA
2. Implement DPO: create preference pairs (model outputs rated good/bad), fine-tune with `DPOTrainer`
3. Ablation: compare fine-tuned model at epoch 1 vs 2 vs 3 — show epoch 3 often overfits
4. Deploy your fine-tuned model to Ollama locally and compare it to the base model on 20 test cases

🛠️ **PROJECT: Domain-Expert Model** — Fine-tune a 7B model (e.g., LLaMA 3.2) to be an expert in a specific domain (medical, legal, finance, cooking, etc.). Generate synthetic training data with GPT-4o, fine-tune with QLoRA + SFT + DPO, evaluate with domain-specific benchmark, deploy with Ollama or vLLM.

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## PHASE 7: PRODUCTION AI ENGINEERING
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

> **Why seventh:** Once you have working, safe, and optionally fine-tuned models, you're ready to expose them as real products. This phase covers API design, cost management, observability, and deployment — all the engineering concerns that make the difference between a demo and a production system.

---

### 7.1 Building AI-Powered APIs

#### 7.1.1 API Design for AI Applications

📚 **Best Resources to Learn:**
- FastAPI docs — fastapi.tiangolo.com
- "Building LLM-Powered Applications" (various blog posts)
- Full Stack FastAPI Template — github.com/fastapi/full-stack-fastapi-template

- [ ] FastAPI fundamentals
  - [ ] Route handlers, path params, query params
  - [ ] Request / response models with Pydantic
  - [ ] Dependency injection
  - [ ] Background tasks
  - [ ] Middleware (CORS, logging, auth)
  - [ ] OpenAPI docs generation
- [ ] Streaming responses (SSE / Server-Sent Events)
  - [ ] `StreamingResponse` in FastAPI
  - [ ] EventSource in frontend JavaScript
  - [ ] Token-by-token streaming
- [ ] Async programming for AI APIs
  - [ ] `async`/`await` with httpx / aiohttp
  - [ ] Concurrent API calls
  - [ ] Asyncio gather for parallel requests
- [ ] Request queuing and job management
  - [ ] Celery with Redis/RabbitMQ for async jobs
  - [ ] ARQ (async job queue)
  - [ ] Progress reporting via WebSockets
- [ ] Handling long-running AI tasks
  - [ ] Webhook callbacks
  - [ ] Polling endpoints
  - [ ] Task status tracking
- [ ] API versioning for AI endpoints
- [ ] Input validation and sanitization
  - [ ] Prompt injection defense in API layer
  - [ ] Content type validation

🏋️ **Exercises:**
1. Build a FastAPI service wrapping an LLM with streaming — test with curl and a simple HTML client
2. Add async concurrent requests: process a batch of 50 items using `asyncio.gather` with rate limiting
3. Build a job queue system: POST request starts async AI task, GET endpoint polls for results

#### 7.1.2 Caching and Cost Optimization

📚 **Best Resources to Learn:**
- Redis docs — redis.io/docs
- LangChain caching docs
- OpenAI prompt caching docs

- [ ] Semantic caching
  - [ ] Cache LLM responses by semantic similarity of query
  - [ ] GPTCache library
  - [ ] Redis with vector search
  - [ ] Cache hit rate measurement
- [ ] Exact-match caching
  - [ ] Hash-based cache key from prompt
  - [ ] Redis / Memcached for fast lookup
  - [ ] TTL policies
- [ ] Provider-level caching
  - [ ] OpenAI prompt caching (automatic, 50% discount)
  - [ ] Anthropic prompt caching
  - [ ] Cache-friendly prompt design
- [ ] Cost optimization strategies
  - [ ] Model routing: use GPT-4o-mini for easy tasks, GPT-4o for hard
  - [ ] Input compression / summarization
  - [ ] Output length control
  - [ ] Batch API (50% cost savings for async workloads)
  - [ ] Embedding caching
  - [ ] Cost alerting and budget limits
- [ ] Token counting before API calls
  - [ ] `tiktoken` for OpenAI
  - [ ] Estimating before sending

🏋️ **Exercises:**
1. Implement semantic cache: embed incoming query, if similarity > 0.95 with cached query, return cached response — measure cost savings on 1,000 queries
2. Build a model router: classify query complexity (easy/hard) with fast model, route to appropriate model
3. Track real API costs for a week of usage, identify top cost drivers, optimize

#### 7.1.3 Observability and Monitoring

📚 **Best Resources to Learn:**
- LangSmith docs — smith.langchain.com
- Langfuse docs — langfuse.com — **open-source alternative**
- Helicone docs — helicone.ai
- Arize Phoenix — phoenix.arize.com

- [ ] LLM-specific observability needs
  - [ ] Tracing multi-step chains and agents
  - [ ] Input/output logging
  - [ ] Token usage tracking
  - [ ] Latency measurement (TTFT, total latency)
  - [ ] Cost tracking per request
  - [ ] Error tracking
- [ ] LangSmith / Langfuse for LLM tracing
  - [ ] Automatic tracing with LangChain
  - [ ] Manual span annotation
  - [ ] Dataset creation for eval
  - [ ] Production monitoring dashboards
- [ ] Standard metrics to monitor
  - [ ] P50/P95/P99 latency
  - [ ] Error rate
  - [ ] Cost per request
  - [ ] Cache hit rate
  - [ ] User satisfaction signals (thumbs up/down)
- [ ] Structured logging for LLM apps
  - [ ] Log: model, prompt, response, tokens, latency, user_id
  - [ ] JSON logging for Elasticsearch / Loki ingestion
- [ ] Alerting
  - [ ] Cost spike alerts
  - [ ] Latency degradation alerts
  - [ ] Error rate alerts
- [ ] A/B testing LLM responses
  - [ ] Routing % of traffic to new prompt/model
  - [ ] Statistical significance testing

🏋️ **Exercises:**
1. Instrument a LangChain app with Langfuse — trace 100 queries, analyze latency distribution
2. Build a cost dashboard: query your logs, plot daily cost by model, cost per user, cost per feature
3. Implement user feedback collection (thumbs up/down), correlate with automatic evaluation scores

---

### 7.2 Deployment and Infrastructure

#### 7.2.1 Deploying LLM Applications

📚 **Best Resources to Learn:**
- Modal docs — modal.com/docs
- Fly.io docs — fly.io/docs
- Railway docs — railway.app/docs
- Docker docs — docs.docker.com

- [ ] Containerizing AI applications
  - [ ] Dockerfile for FastAPI + LLM dependencies
  - [ ] Multi-stage builds (smaller images)
  - [ ] GPU-enabled Docker images
  - [ ] Docker Compose for local dev (app + Redis + Postgres)
- [ ] Cloud platforms (no GPU needed for API wrappers)
  - [ ] Railway / Render / Fly.io (simple web apps)
  - [ ] Vercel / Cloudflare Workers (edge, lightweight)
  - [ ] AWS Lambda / Google Cloud Run (serverless)
  - [ ] Azure Container Apps / AWS App Runner
- [ ] GPU-based deployment (for self-hosted models)
  - [ ] Modal (serverless GPU, pay-per-second)
  - [ ] RunPod serverless
  - [ ] AWS SageMaker / Google Vertex AI endpoints
  - [ ] Hugging Face Inference Endpoints
- [ ] Self-hosting open-source models
  - [ ] Ollama (simplest, local + server mode)
  - [ ] vLLM (production-grade, OpenAI-compatible)
  - [ ] TGI — Text Generation Inference (Hugging Face)
  - [ ] Llamafile (single executable)
- [ ] Environment management
  - [ ] Secrets management (AWS Secrets Manager, HashiCorp Vault)
  - [ ] Configuration management
  - [ ] Environment-specific settings (dev/staging/prod)
- [ ] Auto-scaling
  - [ ] Horizontal scaling (multiple instances)
  - [ ] Queue-based auto-scaling
  - [ ] Cold start mitigation

🏋️ **Exercises:**
1. Dockerize your RAG application, deploy to Railway or Fly.io with a free tier
2. Deploy LLaMA 3.2 3B on Modal serverless GPU — test the OpenAI-compatible endpoint
3. Set up a CI/CD pipeline (GitHub Actions) that runs evals on every PR and blocks merge if scores drop

#### 7.2.2 Serving Open-Source Models

📚 **Best Resources to Learn:**
- vLLM docs — docs.vllm.ai — **production standard**
- Ollama docs — ollama.ai
- TGI docs — huggingface.co/docs/text-generation-inference

- [ ] vLLM
  - [ ] PagedAttention (key innovation — efficient KV cache)
  - [ ] Continuous batching (vs static batching)
  - [ ] OpenAI-compatible server (`vllm serve`)
  - [ ] Tensor parallelism (multi-GPU)
  - [ ] Quantized model serving (AWQ, GPTQ, INT8)
  - [ ] LoRA serving (multiple adapters)
  - [ ] Speculative decoding support
- [ ] TGI (Text Generation Inference)
  - [ ] Flash Attention integration
  - [ ] Continuous batching
  - [ ] Quantization support
  - [ ] Safetensors format
- [ ] Model quantization for deployment
  - [ ] GPTQ (post-training quantization)
  - [ ] AWQ (Activation-aware Weight Quantization)
  - [ ] GGUF (for llama.cpp / Ollama)
  - [ ] INT8 vs INT4 vs FP8 tradeoffs
  - [ ] VRAM requirements by quantization level
- [ ] llama.cpp
  - [ ] CPU inference
  - [ ] GGUF format
  - [ ] Metal (Mac GPU), CUDA, Vulkan backends

🏋️ **Exercises:**
1. Serve LLaMA 3.1 8B with vLLM — benchmark throughput (tokens/sec) at different batch sizes
2. Compare quantization: run same model in FP16 vs INT8 vs INT4 — compare quality and speed
3. Set up vLLM with multiple LoRA adapters — switch adapters per request

🛠️ **PHASE 7 CAPSTONE: Production AI Application** — Build and deploy a full production-grade AI application:
- FastAPI backend with streaming LLM responses
- RAG over a domain-specific knowledge base (your choice)
- Agent capabilities with 3+ tools
- Langfuse observability and cost tracking
- Guardrails for safety
- Deployed on Fly.io or Railway
- A/B testing two prompt versions
- RAGAS eval score ≥ 0.80 on held-out test set
- README with architecture diagram

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## PHASE 8: LLM INFERENCE MASTERY (CMU LLM INFERENCE)
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

> **Source:** CMU 11-663 "Inference Algorithms for Language Modeling" (Graham Neubig) + CMU 11-868 "Large Language Model Systems". This phase is optional but highly valuable for developers who need to optimize performance, reduce costs, or implement custom generation behaviors.
>
> **Why last:** This is the deepest technical content in the checklist. By this point you've built and deployed real systems (Phases 1–7), so you understand the practical problems that inference optimization solves: latency, cost, throughput, and output quality. Learning this theory before having real systems to optimize is premature.
>
> **Usefulness for AI Application Developers:** ★★★★★ — Understanding inference deeply unlocks: faster apps, lower costs, better output quality, custom generation strategies, and ability to self-host models effectively.

---

### 8.1 Decoding Algorithms

#### 8.1.1 Greedy and Beam Search

📚 **Best Resources to Learn:**
- CMU 11-663 lecture slides — phontron.com/class/lminference-fall2025
- "The Curious Case of Neural Text Degeneration" (Holtzman et al.) — arxiv.org/abs/1904.09751
- Jay Alammar "The Illustrated Beam Search"

- [ ] Greedy decoding
  - [ ] Always pick argmax token
  - [ ] Fast but often suboptimal
  - [ ] Deterministic (no randomness)
- [ ] Beam search
  - [ ] Maintain top-B hypotheses at each step
  - [ ] Beam width B: tradeoff (larger B = better quality but slower)
  - [ ] Normalized log-probability scoring
  - [ ] Diverse beam search
  - [ ] Constrained beam search (force certain tokens)
- [ ] Problems with beam search
  - [ ] Generic / bland outputs
  - [ ] Repetition loops
  - [ ] Doesn't always beat greedy for open-ended generation
  - [ ] Good for translation/summarization, bad for conversation
- [ ] Beam search variants
  - [ ] Top-K beam pruning
  - [ ] Minimum Bayes Risk (MBR) decoding
  - [ ] Best-of-N (sample N, pick best by reward model)

🏋️ **Exercises:**
1. Implement greedy decoding from scratch using a HuggingFace model's raw logits (no `generate()`)
2. Implement beam search from scratch (B=3,5,10), compare outputs with greedy on a translation task
3. Show the "generic text" problem with beam search: generate 20 story beginnings with beam vs sampling — blind rate quality
4. Implement Best-of-N: sample 10 completions, rank with a perplexity scorer, pick the best

#### 8.1.2 Sampling Strategies

📚 **Best Resources to Learn:**
- "The Curious Case of Neural Text Degeneration" — arxiv.org/abs/1904.09751 — **introduces nucleus sampling**
- "Locally Typical Sampling" (Meister et al.) — arxiv.org/abs/2202.00666
- "Contrastive Decoding" — arxiv.org/abs/2210.15097

- [ ] Temperature scaling
  - [ ] T=0: greedy (argmax)
  - [ ] T=1: standard sampling
  - [ ] T>1: more random, creative
  - [ ] T<1: more focused, less diverse
  - [ ] Softmax temperature: logits / T → softmax
- [ ] Top-K sampling
  - [ ] Keep only top K tokens, renormalize, sample
  - [ ] Fixed K regardless of probability distribution shape
  - [ ] Problem: sometimes K is too small/large
- [ ] Top-P (Nucleus) sampling
  - [ ] Keep smallest set of tokens summing to P probability
  - [ ] Dynamic K based on distribution entropy
  - [ ] P=0.9 or 0.95 common choices
  - [ ] Often better than top-K in practice
- [ ] Top-K + Top-P combination
- [ ] Min-P sampling
  - [ ] Keep tokens with probability ≥ P × max_probability
  - [ ] Better tail truncation
- [ ] Locally Typical Sampling
  - [ ] Remove tokens that are too surprising or too expected
  - [ ] Better text quality for stories
- [ ] Mirostat sampling
  - [ ] Adaptive target perplexity control
- [ ] Contrastive Decoding
  - [ ] Expert model minus amateur model logits
  - [ ] Better factuality, less repetition
- [ ] Logit processors and warpers
  - [ ] Repetition penalty (`repetition_penalty`)
  - [ ] Presence penalty, frequency penalty (OpenAI)
  - [ ] No-repeat-ngram-size
  - [ ] Forced tokens / bad words list

🏋️ **Exercises:**
1. Implement temperature sampling, top-K, top-P from scratch — generate 100 samples each at 5 settings, compare diversity
2. Plot entropy of the token distribution as generation proceeds — observe how it changes
3. Implement repetition penalty from scratch — test on a prompt that normally produces repetitive output
4. Build a "sampling strategy explorer" CLI: choose a prompt, interactively tweak all sampling params, observe outputs

#### 8.1.3 Advanced Decoding Methods

📚 **Best Resources to Learn:**
- "Chain-of-Thought Prompting" — arxiv.org/abs/2201.11903
- "Self-Consistency" — arxiv.org/abs/2203.11171
- "Tree of Thoughts" — arxiv.org/abs/2305.10601
- "Let's Verify Step by Step" (OpenAI process reward models) — arxiv.org/abs/2305.20050
- CMU 11-663 lectures on meta-generation algorithms

- [ ] Chain-of-Thought (CoT) as decoding strategy
  - [ ] Sequential reasoning as a decoding algorithm
  - [ ] CoT improves accuracy by ~10-30% on reasoning tasks
- [ ] Self-Consistency
  - [ ] Sample K diverse CoT paths
  - [ ] Majority vote on final answer
  - [ ] K=10–40 for best results
  - [ ] Cost-quality tradeoff
- [ ] Verification-augmented generation
  - [ ] Outcome Reward Models (ORM) — score final answer
  - [ ] Process Reward Models (PRM) — score each step
  - [ ] Let's Verify Step by Step (LVSL)
- [ ] Tree of Thoughts (ToT)
  - [ ] BFS / DFS through thought tree
  - [ ] Evaluation at each node
  - [ ] Backtracking when dead ends
- [ ] Monte Carlo Tree Search (MCTS) for LLMs
  - [ ] AlphaCode-style search
  - [ ] rStar (Microsoft)
- [ ] A* and heuristic search for generation
  - [ ] Lookahead decoding
  - [ ] Cost estimation for future tokens
- [ ] FLARE (Forward-Looking Active REtrieval)
  - [ ] Predict next sentence → if uncertain → retrieve → continue
- [ ] Speculative decoding (covered in efficiency section)
- [ ] Maieutic prompting
  - [ ] Build trees of entailment/contradiction relationships
  - [ ] Solve as satisfiability problem
- [ ] Non-monotonic generation
  - [ ] Infilling (fill in the middle — FIM)
  - [ ] Diffusion language models (overview)
- [ ] Minimum Bayes Risk (MBR) decoding
  - [ ] Generate N candidates, pick the one with highest average similarity to others
  - [ ] Better for summarization, translation

🏋️ **Exercises:**
1. Implement self-consistency for a math reasoning task: generate 20 CoT paths, majority vote — compare with single CoT on GSM8K benchmark
2. Implement Tree of Thoughts for a planning puzzle (24 game or Sudoku) — compare with standard CoT
3. Implement MBR decoding: generate 10 translation candidates, score with COMET/BLEURT, pick best
4. Build a PRM-guided beam search: score each reasoning step, prune low-scoring branches

🛠️ **PROJECT: LLM Reasoning Engine** — Build a library that implements: greedy, beam, top-p, self-consistency, Tree of Thoughts. Benchmark on GSM8K (math), HotpotQA (multi-hop), and a custom planning task. Show improvement of each method over baseline.

---

### 8.2 Efficient Inference

#### 8.2.1 The KV Cache

📚 **Best Resources to Learn:**
- Lilian Weng "Inference Optimization" blog post
- "Efficient Memory Management for Large LM Serving with PagedAttention" (vLLM paper) — arxiv.org/abs/2309.06180
- CMU 11-868 KV cache lectures

- [ ] What the KV cache is and why it exists
  - [ ] Attention needs K, V for all past tokens
  - [ ] Recomputing is wasteful — cache instead
  - [ ] KV cache memory: 2 × n_layers × n_heads × head_dim × seq_len × bytes
  - [ ] For LLaMA-2 70B: ~140GB for 4K context!
- [ ] KV cache during prefill vs decode phase
  - [ ] Prefill: process all prompt tokens at once (compute-bound)
  - [ ] Decode: generate one token at a time (memory-bound)
  - [ ] Prefill/decode disaggregation
- [ ] KV cache compression
  - [ ] GQA (Grouped Query Attention) — share KV across heads
  - [ ] MQA (Multi-Query Attention) — single KV for all heads
  - [ ] H2O (Heavy-Hitter Oracle) — evict unimportant tokens
  - [ ] SnapKV — compress by clustering similar tokens
  - [ ] StreamingLLM — sliding window with attention sinks
- [ ] PagedAttention (vLLM)
  - [ ] Virtual memory abstraction for KV cache
  - [ ] Non-contiguous memory (like OS paging)
  - [ ] Eliminates fragmentation
  - [ ] Enables effective batching
- [ ] Prefix caching
  - [ ] Reuse KV cache for shared prefixes (system prompts)
  - [ ] RadixAttention (SGLang) — tree-based caching
  - [ ] Automatic prefix caching in vLLM
- [ ] Quantizing the KV cache
  - [ ] INT8 / INT4 KV cache
  - [ ] FP8 KV cache (hardware support on H100)

🏋️ **Exercises:**
1. Compute KV cache size for LLaMA-3 8B at context lengths 2K, 8K, 32K, 128K in GB — see why it matters
2. Demonstrate prefix caching: measure time to process repeated system prompt with and without caching
3. Profile vLLM: run same batch with PagedAttention on/off (different backends), compare throughput

#### 8.2.2 Quantization

📚 **Best Resources to Learn:**
- "GPTQ" paper — arxiv.org/abs/2210.17323
- "AWQ" paper — arxiv.org/abs/2306.00978
- llama.cpp quantization formats
- bitsandbytes docs — huggingface.co/docs/bitsandbytes
- "The case for 4-bit quantization" (Tim Dettmers blog)

- [ ] Why quantization: less VRAM, faster compute
- [ ] Weight-only quantization
  - [ ] INT8 (W8A16): 2x VRAM reduction
  - [ ] INT4 (W4A16): 4x VRAM reduction
  - [ ] NF4 (QLoRA's 4-bit normal float)
- [ ] Activation quantization
  - [ ] W8A8: faster matrix multiply on hardware
  - [ ] W4A8
  - [ ] FP8 (hardware-native on H100, A100 80GB)
- [ ] Post-Training Quantization (PTQ) methods
  - [ ] Round-to-nearest (RTN) — naive, fast, quality loss
  - [ ] GPTQ — second-order PTQ, layer-by-layer
  - [ ] AWQ (Activation-aware Weight Quantization) — protects important weights
  - [ ] SqueezeLLM
  - [ ] SpQR
- [ ] Quantization-Aware Training (QAT)
  - [ ] Simulates quantization during training
  - [ ] Better quality than PTQ but expensive
- [ ] GGUF format (llama.cpp)
  - [ ] Q2_K, Q3_K, Q4_K_M, Q5_K_M, Q6_K, Q8_0
  - [ ] Quantization group sizes
- [ ] bitsandbytes (HuggingFace integration)
  - [ ] `load_in_8bit=True`, `load_in_4bit=True`
  - [ ] `BitsAndBytesConfig`
- [ ] Tradeoffs: size vs quality vs speed
  - [ ] Quality degradation at INT4 vs INT8
  - [ ] Some models are more quantization-sensitive

🏋️ **Exercises:**
1. Quantize LLaMA 3.2 3B to INT8 and INT4 using bitsandbytes — measure: VRAM, generation speed, quality on MMLU
2. Compare GPTQ vs AWQ vs RTN on same model — perplexity on WikiText-2, inference speed
3. Build a benchmark: run 50 questions on FP16, GPTQ-4bit, AWQ-4bit — compute accuracy and VRAM

#### 8.2.3 Speculative Decoding

📚 **Best Resources to Learn:**
- "Accelerating Large Language Model Decoding with Speculative Sampling" (DeepMind) — arxiv.org/abs/2302.01318
- "SpecInfer" paper — arxiv.org/abs/2305.09781
- CMU 11-663 speculative decoding lecture

- [ ] The fundamental problem: LLM decode is memory-bandwidth bound (1 token/forward pass)
- [ ] Speculative decoding idea
  - [ ] Small "draft" model generates K tokens fast
  - [ ] Large "target" model verifies all K in parallel
  - [ ] Accept/reject each draft token
  - [ ] Guarantees: same output distribution as target model
- [ ] Why this works: batched verification is nearly as fast as single-token
- [ ] Acceptance rate and speedup relationship
  - [ ] If draft model guesses 70% correctly → ~2-3x speedup
- [ ] Draft model choices
  - [ ] Smaller model same family (LLaMA 70B + LLaMA 8B draft)
  - [ ] Medusa heads (draft heads on top of target model)
  - [ ] EAGLE (feature-based draft model)
  - [ ] Self-speculative decoding (draft using lower layers)
  - [ ] n-gram retrieval-based drafting
- [ ] Speculative decoding in vLLM and TGI
- [ ] Limitations and failure modes
  - [ ] Only helps when draft acceptance rate is high
  - [ ] Bad for highly creative/random outputs

🏋️ **Exercises:**
1. Implement speculative decoding from scratch: use LLaMA-3.2-1B as draft, LLaMA-3.1-8B as target
2. Measure acceptance rate at different temperatures — plot acceptance rate vs temperature
3. Enable speculative decoding in vLLM, benchmark throughput vs baseline at different batch sizes

#### 8.2.4 Flash Attention and Hardware Optimization

📚 **Best Resources to Learn:**
- "FlashAttention" paper (Dao et al.) — arxiv.org/abs/2205.14135
- "FlashAttention-2" — arxiv.org/abs/2307.08691
- "FlashAttention-3" — arxiv.org/abs/2407.08608
- Tri Dao's blog on Flash Attention

- [ ] Standard attention memory complexity: O(n²) — the problem
- [ ] Flash Attention
  - [ ] Tiling to keep computation in SRAM (avoid HBM reads/writes)
  - [ ] Fused kernel (Q, K, V → output in one GPU kernel)
  - [ ] Memory: O(n) instead of O(n²)
  - [ ] Speed: 2-4x faster, same result (numerically stable recomputation)
- [ ] Flash Attention 2 improvements
  - [ ] Better parallelism across sequence dimension
  - [ ] Reduced non-matmul FLOPs
- [ ] Flash Attention 3 (H100 specific)
  - [ ] Asynchronous softmax + GEMM
  - [ ] FP8 support
- [ ] Enabling Flash Attention
  - [ ] HuggingFace: `attn_implementation="flash_attention_2"`
  - [ ] Requires CUDA device, flash-attn package
- [ ] Triton kernels
  - [ ] What Triton is (Python-based GPU kernel language)
  - [ ] How Flash Attention is implemented in Triton
  - [ ] Writing simple Triton kernels
- [ ] Continuous batching
  - [ ] Static batching: wait for entire batch, wasteful
  - [ ] Continuous batching: add requests dynamically as others complete
  - [ ] How vLLM and TGI implement it
- [ ] Tensor parallelism for inference
  - [ ] Splitting attention heads across GPUs
  - [ ] All-reduce communication overhead

🏋️ **Exercises:**
1. Benchmark standard attention vs Flash Attention on same sequence lengths (512, 2K, 8K, 32K) — memory and speed
2. Use `torch.compile` + Flash Attention on a small model — compare with baseline inference speed
3. Write a simple Triton kernel (e.g., fused softmax) following the Triton tutorials

---

### 8.3 LLM Serving Systems

#### 8.3.1 Production Serving Architecture

📚 **Best Resources to Learn:**
- vLLM docs — docs.vllm.ai
- SGLang docs — sgl-project.github.io/sglang
- "Sarathi-Serve" paper — arxiv.org/abs/2403.02310
- CMU 11-868 serving system lectures

- [ ] Serving system design goals
  - [ ] Throughput (tokens/second)
  - [ ] Latency (TTFT = time to first token, ITL = inter-token latency)
  - [ ] GPU utilization
  - [ ] Cost efficiency
- [ ] vLLM architecture
  - [ ] LLM engine, scheduler, worker
  - [ ] Continuous batching with PagedAttention
  - [ ] Async engine for high concurrency
  - [ ] Multi-GPU support (tensor + pipeline parallelism)
  - [ ] LoRA serving with multiple adapters
- [ ] SGLang (Stanford)
  - [ ] RadixAttention (automatic KV cache reuse)
  - [ ] `sgl.gen`, `sgl.select` primitives
  - [ ] Constrained decoding (JSON, regex)
  - [ ] Multi-call programs
- [ ] Disaggregated prefill/decode
  - [ ] Prefill (prompt processing) on one set of GPUs
  - [ ] Decode (generation) on another
  - [ ] Reduces prefill's impact on decode latency
- [ ] Request scheduling strategies
  - [ ] FCFS (First-Come-First-Served)
  - [ ] Priority scheduling
  - [ ] Length-aware scheduling
  - [ ] Preemption strategies
- [ ] Online vs offline serving
  - [ ] Online: interactive, latency-sensitive
  - [ ] Offline: batch jobs, throughput-optimized

🏋️ **Exercises:**
1. Deploy LLaMA 3.1 8B with vLLM, use the load testing tool (vllm benchmark_serving.py) — find the latency/throughput Pareto frontier
2. Compare vLLM vs TGI vs Ollama on same model: throughput at batch=1, 8, 32
3. Set up vLLM with multiple LoRA adapters — test routing different requests to different adapters

---

### 8.4 Structured and Constrained Generation

📚 **Best Resources to Learn:**
- Outlines library — github.com/outlines-dev/outlines
- Guidance library — github.com/guidance-ai/guidance
- Instructor library — github.com/jxnl/instructor
- LMQL docs — lmql.ai
- "KGFG" (Grammar-based constrained decoding) papers

- [ ] Why constrained generation
  - [ ] Guaranteed valid JSON, XML, code
  - [ ] No post-processing / retry needed
  - [ ] Works at decode time — no performance overhead with efficient implementation
- [ ] Outlines (constrained generation framework)
  - [ ] JSON schema generation
  - [ ] Regex-guided generation
  - [ ] Grammar-constrained generation (CFG)
  - [ ] Choice from predefined set
  - [ ] Fast token masks with index
- [ ] Guidance (Microsoft)
  - [ ] Interleaved control flow and generation
  - [ ] Selective generation with `{{gen}}`
  - [ ] Constrained choices with `{{select}}`
  - [ ] State machines for generation
- [ ] Logits processors
  - [ ] What they are: transform logits before sampling
  - [ ] Implementing custom constraints as logit processors
  - [ ] Temperature, top-p, repetition penalty are logit processors
- [ ] Structured output at the API level
  - [ ] OpenAI structured outputs (enforces JSON schema)
  - [ ] Anthropic's XML tag conventions
  - [ ] Instructor as wrapper
- [ ] Grammar-constrained generation
  - [ ] CFG (Context-Free Grammar) as constraints
  - [ ] SQL grammar, JSON grammar, Python grammar
  - [ ] Earley parsing for LLM decoding

🏋️ **Exercises:**
1. Use Outlines to generate valid JSON from a natural language description — 100% parse success on 100 samples
2. Use Guidance to build a controlled interview: ask questions, conditionally branch based on answers
3. Implement a custom logit processor that prevents the model from repeating any word used in the last 50 tokens
4. Build a SQL query generator using grammar-constrained generation — all outputs are valid SQL

🛠️ **PROJECT: High-Performance LLM Service** — Build a production inference service:
- vLLM backend with a 7B quantized model (AWQ or GPTQ)
- Speculative decoding with small draft model
- Structured output endpoints (JSON schema enforcement)
- KV cache prefix caching for shared system prompts
- Load test to achieve >100 tokens/second at P95 latency < 1 second
- Cost comparison: cloud API vs self-hosted (per 1M tokens)

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## BONUS: SPECIALIZED APPLICATION TRACKS
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

> These tracks extend your skills into specific domains. Complete them in any order, or in parallel with Phases 3–7, depending on your target use case.

---

### Track A: Multimodal AI Applications

- [ ] Vision-Language Models (VLMs)
  - [ ] OpenAI GPT-4o (vision, audio, text)
  - [ ] Anthropic Claude (vision)
  - [ ] Google Gemini (native multimodal)
  - [ ] LLaVA, InternVL, Qwen-VL (open-source)
- [ ] Image understanding tasks
  - [ ] Image captioning
  - [ ] Visual Q&A
  - [ ] Document parsing / OCR with LLMs
  - [ ] Chart and table understanding
- [ ] Document intelligence
  - [ ] PDF parsing beyond text (tables, figures, layouts)
  - [ ] ColPali / DocVQA (visual document retrieval)
  - [ ] Unstructured.io multimodal pipeline
- [ ] Audio with LLMs
  - [ ] Whisper for transcription
  - [ ] Voice-enabled chatbots (speech-to-text → LLM → text-to-speech)
  - [ ] Streaming voice pipeline
  - [ ] OpenAI Realtime API
- [ ] Video understanding (overview)
  - [ ] Gemini's native video processing
  - [ ] Frame extraction for non-native models

🏋️ **Exercises:**
1. Build a document understanding pipeline: PDF with tables/charts → GPT-4o → structured extraction
2. Build a voice chatbot: microphone → Whisper STT → Claude → TTS → speaker (under 2 second latency)
3. Build a receipt/invoice parser: photo → GPT-4o vision → structured JSON with all line items

---

### Track B: AI Coding Assistants

- [ ] Code generation models
  - [ ] GitHub Copilot (VS Code integration)
  - [ ] Claude for code (best for complex tasks)
  - [ ] DeepSeek-Coder (open-source, strong on code)
  - [ ] Codestral, CodeLlama, Starcoder2
- [ ] Code understanding and review
  - [ ] Automated code review pipeline
  - [ ] Security vulnerability detection
  - [ ] Documentation generation
  - [ ] Test case generation
- [ ] Fill-in-the-middle (FIM) for code completion
  - [ ] `<prefix><suffix><middle>` format
  - [ ] Cursor-aware completion
- [ ] Repository-level context
  - [ ] Retrieval over codebase (code embeddings)
  - [ ] Grep and AST-based retrieval
  - [ ] Microsoft GraphRAG for code
- [ ] Agentic coding (AI writes and runs code)
  - [ ] Claude Code, Devin, OpenHands, SWE-agent
  - [ ] SWE-bench evaluation
  - [ ] Code execution sandboxes (E2B, Docker)

🏋️ **Exercises:**
1. Build a code review bot: reads a PR diff, comments on potential bugs, style violations, security issues
2. Build a test generator: given a Python function, generate 10 pytest test cases using LLM
3. Implement a codebase Q&A RAG: embed all `.py` files, answer questions about the code

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## ULTIMATE CAPSTONE PROJECTS
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### 🏆 Capstone 1: AI-Powered SaaS Application
Build a production SaaS product with paying users:
- Domain of your choice (legal, medical, coding, education, etc.)
- RAG over domain-specific knowledge
- Fine-tuned model for domain expertise
- Multi-agent workflow (research, draft, review)
- User authentication and per-user memory
- Usage billing (track tokens, charge via Stripe)
- Production deployment with monitoring
- LLM observability dashboard (Langfuse)
- Safety and content moderation layer

### 🏆 Capstone 2: Open-Source LLM Serving Stack
Deploy and optimize a complete self-hosted inference stack:
- 70B model (quantized to AWQ/GPTQ INT4) on 2x A100s
- vLLM with PagedAttention and speculative decoding
- OpenAI-compatible API gateway
- Multiple LoRA adapters (one per use case)
- Load balancing across multiple vLLM instances
- Prometheus + Grafana monitoring
- Cost comparison vs OpenAI API at various scales
- Document: at what request volume does self-hosting break even?

### 🏆 Capstone 3: LLM Reasoning System
Build a state-of-the-art reasoning system:
- Implement 5+ decoding strategies (CoT, Self-Consistency, ToT, MCTS, MBR)
- Benchmark on GSM8K, MATH, HotpotQA, AIME
- Build a routing layer: select optimal decoding strategy per problem type
- Implement a learned verifier (PRM) to guide tree search
- Compare your system against published GPT-4 scores on benchmarks
- Write a technical report documenting findings

---

## Quick Reference: Phase Ordering Rationale

| Phase | Topic | Why Here |
|---|---|---|
| Phase 1 | LLM Foundations + APIs + Prompt Engineering | Prerequisites for everything |
| Phase 2 | Evaluation & Benchmarking | Learn to measure before building complex systems |
| Phase 3 | RAG | Give LLMs external knowledge; evaluate with Phase 2 tools |
| Phase 4 | Agents & Tool Use | Combine APIs + prompting + RAG into autonomous systems |
| Phase 5 | Safety & Reliability | Understand failure modes before production deployment |
| Phase 6 | Fine-Tuning | Customize behavior once you know what prompting can't fix |
| Phase 7 | Production Engineering | Deploy, scale, monitor, and optimize real systems |
| Phase 8 | Inference Mastery | Deep optimization of systems you've already built |

## Quick Reference: CMU LLM Inference Course Topics → Checklist Mapping

| CMU Course Topic | Checklist Location |
|---|---|
| Greedy / Beam Search | Phase 8, Section 8.1.1 |
| Temperature, Top-K, Top-P, Nucleus | Phase 8, Section 8.1.2 |
| CoT, Self-Consistency, Tree of Thoughts | Phase 8, Section 8.1.3 |
| Process Reward Models / Verification | Phase 8, Section 8.1.3 |
| KV Cache and PagedAttention | Phase 8, Section 8.2.1 |
| Quantization (GPTQ, AWQ, INT4) | Phase 8, Section 8.2.2 |
| Speculative Decoding | Phase 8, Section 8.2.3 |
| Flash Attention | Phase 8, Section 8.2.4 |
| vLLM, TGI, serving systems | Phase 8, Section 8.3.1 |
| Constrained generation (Outlines, Guidance) | Phase 8, Section 8.4 |
| RAG | Phase 3 |
| Function calling / Tool Use | Phase 4, Section 4.1.2 |
| Multi-agent systems | Phase 4, Section 4.2.4 |
| MCP (Model Context Protocol) | Phase 4, Section 4.2.4 |
| Evaluation methods | Phase 2 |
| Fine-tuning / PEFT / LoRA / DPO | Phase 6 |
