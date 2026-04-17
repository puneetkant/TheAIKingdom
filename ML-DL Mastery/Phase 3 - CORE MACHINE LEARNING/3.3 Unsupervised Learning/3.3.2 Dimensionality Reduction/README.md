# 3.3.2 Dimensionality Reduction

PCA (scree plot, explained variance, reconstruction), t-SNE, UMAP, LDA.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | PCA from scratch (SVD), projection visualisation |
| `working_example2.py` | PCA scree + cumvar, reconstruction error, PCA + Ridge pipeline |
| `working_example.ipynb` | Interactive: scree → cumvar → reconstruction → PCA+Ridge sweep |

## Quick Reference

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Always scale first!
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=0.95)    # keep 95% variance
X_pca = pca.fit_transform(X_scaled)
print(pca.n_components_)        # actual components retained

# Explained variance
evr = pca.explained_variance_ratio_
print(evr.cumsum())

# t-SNE (visualization only, not preprocessing)
from sklearn.manifold import TSNE
X_2d = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_pca)
```

## ML Connections
- **Preprocessing**: reduce features before KNN/SVM (fight CoD)
- **Visualization**: t-SNE / UMAP for cluster structure
- **Compression**: PCA autoencoder baseline
- **Feature extraction**: SVD → LSA for text

## Learning Resources
- [StatQuest: PCA Step-by-Step](https://youtu.be/FgakZw6K1QQ)
- [sklearn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

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
