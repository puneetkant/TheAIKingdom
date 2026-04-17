# 2.1.7 Singular Value Decomposition (SVD)

SVD for PCA, low-rank approximation, pseudo-inverse, condition number — on Cal Housing data.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Manual SVD, truncated reconstruction |
| `working_example2.py` | Cal Housing: SVD decomp, low-rank approx, PCA scatter, pseudo-inverse, condition number |
| `working_example.ipynb` | Interactive: SVD → low-rank error → PCA scatter → pseudo-inverse |

## Quick Reference

```python
import numpy as np

U, s, Vt = np.linalg.svd(X, full_matrices=False)

# Reconstruct exactly
X_hat = U @ np.diag(s) @ Vt

# Low-rank approximation (keep top k)
k = 2
X_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

# PCA scores
scores = U * s   # (n, k)

# Pseudo-inverse  A⁺ = V Σ⁻¹ Uᵀ
A_plus = np.linalg.pinv(A)

# Condition number
cond = s[0] / s[-1]
```

## ML Connections
- **PCA** = SVD of centred data matrix
- **Collaborative filtering** = low-rank matrix factorisation
- **LSA** (Latent Semantic Analysis) = SVD of term-doc matrix
- **Pseudo-inverse** = generalised least-squares solution

## Dataset
- **California Housing** — [scikit-learn/california-housing on HuggingFace](https://huggingface.co/datasets/scikit-learn/california-housing)

## Learning Resources
- [StatQuest: SVD visually explained](https://youtu.be/nbBvuuNVfco)
- **Book:** *Mathematics for Machine Learning* Ch. 4.5
- [NumPy SVD docs](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html)

Decompose matrices with SVD and inspect components.

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
