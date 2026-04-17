# 2.1.8 Matrix Decompositions (Additional)

QR, Cholesky, LU decompositions with ML applications: QR least-squares, Cholesky multivariate sampling.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Decomp by hand: row operations, partial pivoting |
| `working_example2.py` | QR least-squares, Cholesky sampling from N(0,Σ), LU via scipy |
| `working_example.ipynb` | Interactive: QR → Cholesky → LU |

## Quick Reference

```python
import numpy as np
from scipy.linalg import lu

# QR (A = QR): Q orthonormal, R upper triangular
Q, R = np.linalg.qr(A)
x = np.linalg.solve(R, Q.T @ b)   # least squares

# Cholesky (A = LL^T, A must be SPD)
L = np.linalg.cholesky(A)
samples = np.random.randn(n, d) @ L.T  # ~ N(0, A)

# LU (PA = LU)
P, L, U = scipy.linalg.lu(A)
```

## Learning Resources
- **Book:** *Numerical Linear Algebra* (Trefethen & Bau)
- [Gilbert Strang 18.065: Matrix methods](https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/)

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
