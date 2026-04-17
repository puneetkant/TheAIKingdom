# 2.1.10 Numerical Linear Algebra

Condition numbers, QR vs normal equations, sparse matrices, Jacobi iterative solver.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Float precision demo, catastrophic cancellation |
| `working_example2.py` | Stability comparison (QR vs NE), sparse adjacency, Jacobi convergence plot |
| `working_example.ipynb` | Interactive: condition → stability → Jacobi iteration |

## Quick Reference

```python
import numpy as np

# Condition number
_, s, _ = np.linalg.svd(A, full_matrices=False)
cond = s[0] / s[-1]

# Prefer QR over normal equations for ill-conditioned A
Q, R = np.linalg.qr(A)
x = np.linalg.solve(R, Q.T @ b)   # stable

# Jacobi iteration: Ax=b, A = D + R
for _ in range(max_iter):
    x = (b - R @ x) / d   # d = np.diag(A)
```

## Key Rules
- **Never form A^T A** for least-squares — use QR or `np.linalg.lstsq`
- Condition number > 1e8 → results may lose 8 digits of precision
- Jacobi converges iff A is strictly diagonally dominant

## Learning Resources
- **Book:** *Numerical Linear Algebra* — Trefethen & Bau (MIT Press)
- [Fast.ai NLA course](https://www.fast.ai/)

Use numerical methods in linear algebra.

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
