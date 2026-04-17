# 2.1.9 Inner Product Spaces

Euclidean and weighted inner products, Cauchy-Schwarz, kernel trick (linear/poly/RBF), Gram matrix, Fourier projection.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Inner product axioms, angle, Gram-Schmidt |
| `working_example2.py` | Cauchy-Schwarz, kernel functions, Gram matrix, Fourier projection |
| `working_example.ipynb` | Interactive: inner products → kernel trick → RBF Gram matrix → Fourier projection |

## Quick Reference

```python
import numpy as np

# Euclidean inner product
np.dot(x, y)

# Weighted inner product  ⟨x,y⟩_W = x^T W y
x @ W @ y

# Cosine angle
cos = np.dot(x,y) / (np.linalg.norm(x) * np.linalg.norm(y))

# RBF kernel
k_rbf = lambda x,y,g=0.5: np.exp(-g * np.dot(x-y, x-y))

# Gram matrix (pairwise kernel matrix)
K = np.array([[k_rbf(X[i], X[j]) for j in range(n)] for i in range(n)])
```

## ML Connections
- **SVM kernel trick** — implicit high-dimensional inner products via kernels
- **Gaussian Process** — RBF Gram matrix as covariance
- **Attention mechanism** — `softmax(QK^T / √d)` is a scaled inner product
- **Cosine similarity** in embeddings / NLP

## Learning Resources
- [StatQuest: Kernel trick](https://youtu.be/Q7vT0--5VII)
- **Book:** *Pattern Recognition and Machine Learning* (Bishop) Ch. 6

Explore dot products and orthogonality.

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
