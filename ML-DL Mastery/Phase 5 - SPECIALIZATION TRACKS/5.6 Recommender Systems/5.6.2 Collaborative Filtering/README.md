# 5.6.2 Collaborative Filtering

User-user CF, item-item CF, matrix factorisation, SVD, ALS, implicit feedback.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | User-based CF with cosine similarity |
| `working_example2.py` | SGD matrix factorisation on synthetic ratings |
| `working_example.ipynb` | Interactive: user similarity matrix |

## Quick Reference

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# User-user CF
R_filled = np.nan_to_num(R)                   # fill missing with 0
user_sim = cosine_similarity(R_filled)         # (n_users, n_users)

# Predict rating for user u, item i
def predict(u, i, R, user_sim, k=5):
    sims = user_sim[u].copy(); sims[u] = 0
    top_k = np.argsort(sims)[::-1][:k]
    top_k = [v for v in top_k if not np.isnan(R[v, i])]
    numer = sum(user_sim[u, v] * R[v, i] for v in top_k)
    denom = sum(abs(user_sim[u, v]) for v in top_k) + 1e-8
    return numer / denom

# Matrix factorisation (SGD)
P[u] += lr * (err * Q[i] - reg * P[u])
Q[i] += lr * (err * P[u] - reg * Q[i])
```

## CF Approaches

| Approach | Similarity | Scalability | Cold-start |
|----------|-----------|-------------|-----------|
| User-user | Cosine/Pearson | Poor (large n_users) | Poor |
| Item-item | Cosine | Better (stable items) | Poor |
| SVD/MF | Learned | Excellent | Poor |
| ALS | Learned | Excellent | Poor |

## Learning Resources
- [Surprise library](https://surprise.readthedocs.io/)
- [Netflix prize retrospective](https://arxiv.org/abs/1202.1112)

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
