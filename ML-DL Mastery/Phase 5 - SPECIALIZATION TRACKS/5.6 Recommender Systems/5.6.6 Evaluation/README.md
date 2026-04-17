# 5.6.6 Evaluation

Precision@k, Recall@k, NDCG, MRR, coverage, diversity, hit rate, A/B testing.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Offline evaluation protocol |
| `working_example2.py` | Full metrics suite: P@k, R@k, NDCG, MRR — random vs smart |
| `working_example.ipynb` | Interactive: metric functions + Precision@k curve |

## Quick Reference

```python
# Precision@k
def precision_at_k(rec, rel, k):
    return len(set(rec[:k]) & set(rel)) / k

# Recall@k
def recall_at_k(rec, rel, k):
    return len(set(rec[:k]) & set(rel)) / len(rel)

# NDCG@k (normalised discounted cumulative gain)
def ndcg_at_k(rec, rel, k):
    dcg  = sum(1/np.log2(i+2) for i,r in enumerate(rec[:k]) if r in rel)
    idcg = sum(1/np.log2(i+2) for i in range(min(len(rel), k)))
    return dcg / idcg

# MRR (Mean Reciprocal Rank)
mrr = mean(1/(rank of first relevant item) for each user)

# Coverage
coverage = len(unique recommended items) / total items
```

## Metric Cheat-Sheet

| Metric | Range | Higher = | Captures |
|--------|-------|----------|---------|
| P@k | [0,1] | Better | Relevance in top-k |
| R@k | [0,1] | Better | Recall coverage |
| NDCG@k | [0,1] | Better | Rank quality |
| MRR | [0,1] | Better | First hit position |
| Coverage | [0,1] | Better | Catalogue coverage |

## Learning Resources
- [RecSys evaluation guide](https://arxiv.org/abs/1911.07698)

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
