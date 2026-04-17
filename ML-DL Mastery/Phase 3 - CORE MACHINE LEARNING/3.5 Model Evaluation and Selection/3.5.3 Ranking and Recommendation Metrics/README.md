# 3.5.3 Ranking & Recommendation Metrics

NDCG, MAP, Hit Rate, Recall@K — for search, recommendation, and ranking systems.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Basic precision@K, recall@K |
| `working_example2.py` | Manual NDCG/MAP, regression-as-ranking hit rate |
| `working_example.ipynb` | Interactive: NDCG@K for two ranking systems → GBM recall@K |

## Quick Reference

```python
import numpy as np

def dcg(r, k=None):
    r = np.asarray(r[:k], float)
    return np.sum(r / np.log2(np.arange(2, len(r)+2)))

def ndcg(r, ideal, k=None):
    i = dcg(sorted(ideal, reverse=True), k)
    return dcg(r, k) / i if i else 0.0

# Recall@K for recommendation
pred_rank = np.argsort(-y_scores)
recall_at_k = len(relevant & set(pred_rank[:k])) / len(relevant)
```

## Metric Formulas

$$\text{DCG@k} = \sum_{i=1}^{k} \frac{\text{rel}_i}{\log_2(i+1)} \qquad \text{NDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}}$$

## Learning Resources
- [Evaluation Measures in Information Retrieval](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval))
- [RecSys metrics overview](https://arxiv.org/abs/2012.10185)

Explore this topic with a small practical project or coding exercise.

## What to build

- Try a small hands-on exercise focused on this topic.
- Keep the code in `project.py` in this folder.
- Add notes, examples, or results inside this directory.

## Suggestions

1. Read the checklist topic and identify one practice task.
2. Write code in `project.py` that illustrates the main concept.
3. Run your code and iterate until it works.

## Notes

- Use Python and standard libraries when possible.
- For data topics, install `numpy`, `pandas`, `matplotlib` as needed.
