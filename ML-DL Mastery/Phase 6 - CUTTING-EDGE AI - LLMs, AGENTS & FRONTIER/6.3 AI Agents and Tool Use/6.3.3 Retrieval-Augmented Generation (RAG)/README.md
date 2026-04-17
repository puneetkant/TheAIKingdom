# 6.3.3 Retrieval-Augmented Generation (RAG)

RAG augments LLM generation with dynamically retrieved documents, reducing hallucination and enabling up-to-date knowledge without retraining. A retriever (dense or sparse) fetches relevant chunks; the generator conditions on them. This folder builds a miniature RAG pipeline with TF-IDF retrieval, cosine similarity, and end-to-end generation simulation.

## Files

| File | Description |
|------|-------------|
| `working_example2.py` | TF-IDF document index, cosine retrieval, retrieved context grounding, BLEU vs no-RAG |
| `working_example.ipynb` | Interactive notebook version |
| `output/` | `rag_retrieval.png` |

## Quick Start

```bash
python working_example2.py
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| Dense retrieval | FAISS + bi-encoder embeddings (DPR, ColBERT) |
| Sparse retrieval | BM25 / TF-IDF keyword matching |
| Chunking | Split docs into overlapping windows for indexing |
| Re-ranking | Cross-encoder re-orders top-k candidates |
| Hybrid RAG | Combine dense + sparse scores (RRF) |

## Learning Resources

- Lewis et al. *RAG* (2020)
- Karpukhin et al. *DPR* (2020)
- LangChain / LlamaIndex documentation
