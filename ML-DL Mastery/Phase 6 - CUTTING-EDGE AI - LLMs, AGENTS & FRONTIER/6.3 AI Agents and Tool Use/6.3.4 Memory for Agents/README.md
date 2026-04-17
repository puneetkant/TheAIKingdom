# 6.3.4 Memory for Agents

Long-running agents need structured memory: in-context (conversation history), external (vector store retrieval), episodic (event logs), and semantic (knowledge summaries). This folder implements a simple memory manager that stores, retrieves, and forgets memories based on recency and relevance scores.

## Files

| File | Description |
|------|-------------|
| `working_example2.py` | Memory store with recency decay, relevance scoring, retrieval simulation, memory utilisation chart |
| `working_example.ipynb` | Interactive notebook version |
| `output/` | `memory_agents.png` |

## Quick Start

```bash
python working_example2.py
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| In-context memory | Conversation window; limited by context length |
| Vector store | External long-term memory via embedding search |
| Episodic memory | Time-stamped event log; retrieved by recency |
| Semantic memory | Distilled facts and summaries |
| Forgetting curve | Ebbinghaus decay: retention ∝ e^(−t/s) |

## Learning Resources

- Park et al. *Generative Agents* (2023)
- Zhong et al. *MemGPT* (2023)
- Langchain Memory documentation
