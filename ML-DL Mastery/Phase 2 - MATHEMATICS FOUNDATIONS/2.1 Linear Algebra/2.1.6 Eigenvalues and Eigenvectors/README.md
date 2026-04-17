# 2.1.6 Eigenvalues and Eigenvectors

Eigendecomposition: PCA via covariance eigenvectors, power iteration, Markov chain stationary distribution.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | 2×2 eigen decomposition, diagonalisation |
| `working_example2.py` | Cal Housing PCA (eigh), power iteration, Markov stationary, explained variance |
| `working_example.ipynb` | Interactive: eigen basics → PCA scatter → Markov |

## Quick Reference

```python
import numpy as np

# General matrix
vals, vecs = np.linalg.eig(A)

# Symmetric (covariance) — more stable
vals, vecs = np.linalg.eigh(A)    # eigenvalues in ascending order

# PCA
Xc = X - X.mean(axis=0)
cov = Xc.T @ Xc / (len(X) - 1)
vals, vecs = np.linalg.eigh(cov)
# Sort descending
idx = np.argsort(vals)[::-1]
W = vecs[:, idx[:2]]             # top-2 eigenvectors
X_pca = Xc @ W
```

## ML Connections
- **PCA**: eigenvectors of covariance = principal components
- **Spectral clustering**: eigenvectors of graph Laplacian
- **PageRank**: dominant eigenvector of web link matrix
- **Markov chains**: stationary = eigenvector with λ=1

## Learning Resources
- [3Blue1Brown: Eigenvectors and values](https://youtu.be/PFDu9oVAE-g)
- **Book:** *Mathematics for Machine Learning* Ch. 4.2–4.4
- [StatQuest: PCA step-by-step](https://youtu.be/FgakZw6K1QQ)

Implement vector operations and visualizations.

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
