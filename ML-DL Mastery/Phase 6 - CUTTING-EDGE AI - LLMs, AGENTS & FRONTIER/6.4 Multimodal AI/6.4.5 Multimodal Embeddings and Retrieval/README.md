# 6.4.5 Multimodal Embeddings and Retrieval

Multimodal embeddings project images, text, audio, and video into a shared vector space enabling cross-modal retrieval (text query → image results, image query → text results). Key metrics include Recall@k, MRR, and nDCG. This folder builds a cross-modal retrieval benchmark with PCA visualisation and cosine similarity matrices.

## Files

| File | Description |
|------|-------------|
| `working_example2.py` | Cross-modal retrieval sim, Recall@k, MRR, PCA 2D scatter, cosine similarity heatmap |
| `working_example.ipynb` | Interactive notebook version |
| `output/` | `multimodal_retrieval.png` |

## Quick Start

```bash
python working_example2.py
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| Cross-modal retrieval | Query one modality, retrieve another |
| Recall@k | Fraction of queries where correct item is in top-k |
| MRR | Mean Reciprocal Rank of first correct result |
| Alignment | Training to pull matching pairs together in embed space |
| FAISS | Approximate nearest-neighbour library for large indexes |

## Learning Resources

- Radford et al. *CLIP* (2021)
- ImageBind (Meta, 2023)
- Johnson et al. *FAISS* (2019)
