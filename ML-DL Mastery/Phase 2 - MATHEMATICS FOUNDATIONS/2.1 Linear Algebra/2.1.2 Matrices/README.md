# 2.1.2 Matrices

Matrix operations for ML: OLS via normal equations, covariance/correlation matrix, broadcasting, softmax weight matrix.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Pure-Python matrix ops: multiply, transpose, inverse |
| `working_example2.py` | Cal Housing: OLS, covariance heatmap, broadcasting standardisation, softmax weights |
| `working_example.ipynb` | Interactive: load data → OLS → correlation heatmap → broadcasting |

## Run

```bash
python working_example.py
python working_example2.py   # saves output/corr_matrix.png
jupyter lab working_example.ipynb
```

## Matrix Cheat Sheet

```python
import numpy as np

A = np.array([[1,2],[3,4]])   # 2×2 matrix

# Basic ops
A.T                            # transpose
A @ B                          # matrix multiply (preferred over np.dot)
A * B                          # element-wise (Hadamard)
np.linalg.inv(A)              # inverse (use solve instead)
np.linalg.solve(A, b)         # solve Ax=b (more stable than inv)
np.linalg.det(A)              # determinant
np.trace(A)                   # sum of diagonal

# Stacking
np.hstack([A, B])             # column-wise
np.vstack([A, B])             # row-wise

# Broadcasting (auto-expands shape)
X - X.mean(axis=0)            # subtract column means (n×d) - (d,)
```

## ML Connections
- **Design matrix** $X \in \mathbb{R}^{n \times d}$: rows=samples, cols=features
- **Normal equations**: $\mathbf{w} = (X^TX)^{-1}X^T\mathbf{y}$
- **Covariance matrix**: $\Sigma = \frac{1}{n-1}X_c^T X_c$
- **Weight matrix** in neural nets: $Z = XW + b$

## Learning Resources
- [3Blue1Brown: Linear transformations](https://www.3blue1brown.com/topics/linear-algebra)
- **Book:** *Mathematics for Machine Learning* Ch. 2 (Deisenroth)
- [NumPy matrix operations](https://numpy.org/doc/stable/reference/routines.linalg.html)

Work with matrix math and transformations.

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
