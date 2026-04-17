# 3.3.1 Clustering

K-Means (elbow method, silhouette), DBSCAN, Agglomerative/Hierarchical clustering.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | K-Means from scratch, centroid update loop |
| `working_example2.py` | K-Means elbow+sil, DBSCAN eps sweep, Agglomerative linkage |
| `working_example.ipynb` | Interactive: k sweep → PCA scatter → DBSCAN → Agglomerative |

## Quick Reference

```python
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

# K-Means (always scale features!)
km = KMeans(n_clusters=4, n_init=10, random_state=42)
labels = km.fit_predict(X_scaled)
sil = silhouette_score(X_scaled, labels)

# DBSCAN (density-based, no k needed)
db = DBSCAN(eps=0.5, min_samples=10)
labels = db.fit_predict(X_scaled)   # -1 = noise
```

## Choosing k
- **Elbow**: plot inertia vs k, find "elbow"
- **Silhouette**: -1 to 1, higher = better separation

## Learning Resources
- [StatQuest: K-Means](https://youtu.be/4b5d3muPQmA)
- [sklearn Clustering guide](https://scikit-learn.org/stable/modules/clustering.html)

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
