"""
Working Example: Retrieval-Augmented Generation (RAG)
Covers indexing, retrieval, reranking, advanced RAG patterns,
and evaluation.
"""
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_rag")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def cosine_sim(a, b):
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    return (b @ a)


# ── 1. RAG pipeline overview ──────────────────────────────────────────────────
def rag_overview():
    print("=== Retrieval-Augmented Generation (RAG) ===")
    print()
    print("  RAG pipeline:")
    print("    1. Ingest:    chunk documents → embed → store in vector DB")
    print("    2. Retrieve:  embed query → ANN search → top-k chunks")
    print("    3. Generate:  LLM synthesises answer from retrieved context")
    print()
    print("  Why RAG?")
    why = [
        ("Freshness",       "Knowledge beyond training cutoff"),
        ("Private data",    "Company documents, code, customer data"),
        ("Accuracy",        "Ground answer in retrieved facts; reduce hallucination"),
        ("Attribution",     "Cite source documents"),
        ("Cost",            "Cheaper than fine-tuning; update without retraining"),
    ]
    for w, d in why:
        print(f"  {w:<16} {d}")


# ── 2. Document ingestion ─────────────────────────────────────────────────────
def document_ingestion():
    print("\n=== Document Ingestion ===")
    print()
    print("  Chunking strategies:")
    strategies = [
        ("Fixed-size",        "Split every N tokens; simple; no semantic awareness"),
        ("Sentence-level",    "Split on sentence boundaries; better coherence"),
        ("Recursive char",    "Split by paragraph → sentence → word; LangChain default"),
        ("Semantic",          "Embed → split where similarity drops; best quality"),
        ("Document-aware",    "PDF/HTML/MD parser; preserve structure; headers"),
        ("Hierarchical",      "Parent chunk (512 tok) + child chunk (128 tok) for retrieve"),
    ]
    for s, d in strategies:
        print(f"  {s:<18} {d}")
    print()
    print("  Embedding models:")
    models = [
        ("text-embedding-3-small","OpenAI; 1536d; fast; cheap"),
        ("text-embedding-3-large","OpenAI; 3072d; best quality"),
        ("E5-large-v2",           "Microsoft; 1024d; strong BEIR benchmark"),
        ("BGE-m3",                "BAAI; multilingual; 1024d; ColBERT support"),
        ("Nomic-embed-text",      "Open; 768d; Apache 2.0; long context"),
        ("Cohere embed v3",       "Matryoshka embeddings; Cohere API"),
    ]
    for m, d in models:
        print(f"  {m:<28} {d}")


# ── 3. Retrieval demo ─────────────────────────────────────────────────────────
def retrieval_demo():
    print("\n=== Retrieval Demo ===")
    print()

    rng = np.random.default_rng(0)
    d = 16   # embedding dim

    # Simulated corpus chunks
    chunks = [
        "Python is a high-level programming language.",
        "Machine learning is a subset of artificial intelligence.",
        "Neural networks are inspired by the human brain.",
        "The transformer architecture revolutionised NLP in 2017.",
        "Gradient descent optimises neural network weights.",
        "Retrieval-augmented generation improves LLM factual accuracy.",
        "Docker containers package software and its dependencies.",
        "Kubernetes orchestrates containerised applications at scale.",
    ]

    # Random embeddings (in practice: sentence transformer)
    chunk_embeds = rng.normal(0, 1, (len(chunks), d))
    chunk_embeds /= np.linalg.norm(chunk_embeds, axis=1, keepdims=True)

    # Make query embedding similar to a few relevant chunks
    query = "How does retrieval-augmented generation work?"
    query_embed = chunk_embeds[5] + rng.normal(0, 0.3, d)  # similar to chunk 5
    query_embed /= np.linalg.norm(query_embed)

    sims = cosine_sim(query_embed, chunk_embeds)
    top_k = 3
    top_idx = np.argsort(-sims)[:top_k]

    print(f"  Query: '{query}'")
    print(f"  Top-{top_k} retrieved chunks:")
    for rank, idx in enumerate(top_idx):
        print(f"    {rank+1}. [sim={sims[idx]:.3f}] {chunks[idx]}")
    print()
    print("  Vector databases:")
    vecdbs = [
        ("Pinecone",   "Managed; serverless; metadata filtering; popular"),
        ("Weaviate",   "Open-source; hybrid (dense + BM25); GraphQL"),
        ("Qdrant",     "Open-source; Rust; fast; product quantisation"),
        ("Chroma",     "Lightweight; Python; great for prototyping"),
        ("pgvector",   "PostgreSQL extension; SQL + vectors; production-ready"),
        ("FAISS",      "Meta; library (not DB); exact or approximate; research"),
    ]
    for v, d in vecdbs:
        print(f"  {v:<12} {d}")


# ── 4. Advanced RAG ───────────────────────────────────────────────────────────
def advanced_rag():
    print("\n=== Advanced RAG Patterns ===")
    print()
    patterns = [
        ("HyDE",           "Hypothetical Document Embeddings; generate doc then embed"),
        ("RAG Fusion",     "Multiple query variants; reciprocal rank fusion"),
        ("Reranking",      "Cross-encoder re-scores top-N; Cohere/BAAI reranker"),
        ("Multi-vector",   "ColBERT: late interaction; token-level matching"),
        ("Parent-child",   "Retrieve small chunks; expand to parent for context"),
        ("FLARE",          "Active retrieval; retrieve only when uncertain"),
        ("Corrective RAG", "Verify retrieved docs; web search if insufficient"),
        ("Self-RAG",       "Model decides when to retrieve; generates citations"),
        ("GraphRAG",       "Community summaries from knowledge graph; Microsoft"),
    ]
    for p, d in patterns:
        print(f"  {p:<18} {d}")
    print()
    print("  RAG evaluation metrics:")
    evals = [
        ("Faithfulness",    "Is the answer grounded in retrieved context? (no hallucination)"),
        ("Answer relevance","Does the answer address the question?"),
        ("Context precision","Are retrieved docs relevant to the query?"),
        ("Context recall",   "Are all needed docs retrieved?"),
    ]
    for e, d in evals:
        print(f"  {e:<20} {d}")
    print()
    print("  RAGAS: automated RAG evaluation framework (ragas.io)")


if __name__ == "__main__":
    rag_overview()
    document_ingestion()
    retrieval_demo()
    advanced_rag()
