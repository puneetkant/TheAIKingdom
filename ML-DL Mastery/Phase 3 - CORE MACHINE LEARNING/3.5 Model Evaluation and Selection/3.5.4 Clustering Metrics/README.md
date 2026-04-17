# 3.5.4 Clustering Metrics

Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz, ARI, NMI.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Elbow method, inertia |
| `working_example2.py` | Full internal+external metric sweep across k=2..10 |
| `working_example.ipynb` | Interactive: metric table → external ARI/NMI |

## Quick Reference

```python
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                              calinski_harabasz_score, adjusted_rand_score)
from sklearn.cluster import KMeans

km = KMeans(k, random_state=42, n_init=10)
labels = km.fit_predict(X_scaled)

sil = silhouette_score(X_scaled, labels)        # higher = better
dbi = davies_bouldin_score(X_scaled, labels)    # lower = better
chi = calinski_harabasz_score(X_scaled, labels) # higher = better

# External (when ground truth known)
ari = adjusted_rand_score(true_labels, pred_labels)
```

## Metric Summary

| Metric | Range | Better |
|--------|-------|--------|
| Silhouette | [-1, 1] | Higher |
| Davies-Bouldin | [0, ∞) | Lower |
| Calinski-Harabasz | [0, ∞) | Higher |
| ARI | [-1, 1] | Higher |

## Learning Resources
- [sklearn clustering metrics](https://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation)

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
