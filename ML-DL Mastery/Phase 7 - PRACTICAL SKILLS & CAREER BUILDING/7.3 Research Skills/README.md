# 7.3 Research Skills

Effective ML research requires systematic paper reading (abstract → conclusion → figures → methods), critical analysis, ablation study design, and reproducibility. Tools like Connected Papers, Semantic Scholar, and Zotero help manage literature. This folder implements a paper tracker, citation network analysis, and ablation study template.

## Files

| File | Description |
|------|-------------|
| `working_example2.py` | Paper tracker, citation DAG, ablation study scorer, reading progress tracker |
| `working_example.ipynb` | Interactive notebook version |
| `output/` | `research_skills.png` |

## Quick Start

```bash
python working_example2.py
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| Ablation study | Remove one component at a time; measure impact |
| Citation network | Graph of paper citations; find influential ancestors |
| Reproducibility | Code + data + hyperparams to replicate results |
| Peer review | Evaluate novelty, soundness, significance, clarity |
| Paper tracker | Prioritise reading by relevance and citation count |

## Learning Resources

- Karpathy *How to Read a Paper* (3-pass method)
- Connected Papers (connectedpapers.com)
- Semantic Scholar API
