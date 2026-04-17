# 1.2.1 NumPy

Vectorized array operations, broadcasting, linear algebra, and indexing — the foundation of every ML framework.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Array creation, shapes, reshaping, basic ops, random |
| `working_example2.py` | Cal Housing: load → feature engineering → OLS via normal equations → PCA (eigh) → indexing → speed test |
| `working_example.ipynb` | Interactive: load data, stats, broadcasting, OLS, scatter + histogram |

## Run

```bash
python working_example.py
python working_example2.py    # downloads cal_housing.csv
jupyter lab working_example.ipynb
```

## Quick Reference

```python
import numpy as np

# Array creation
a = np.array([1,2,3], dtype=np.float32)
z = np.zeros((3,4));  o = np.ones((3,4))
r = np.random.randn(100, 8)     # standard normal

# Shape and indexing
a.shape, a.ndim, a.size
a[1:3], a[:, 0]                 # slice, column
a[a > 0]                        # boolean index
a[np.array([0,2,4])]            # fancy index

# Broadcasting
(X - X.mean(axis=0)) / X.std(axis=0)   # z-score all features

# Linear algebra
X.T @ X                          # gram matrix
np.linalg.solve(A, b)            # solve linear system
np.linalg.eigh(cov)              # eigendecomposition
np.linalg.svd(X, full_matrices=False)

# Reduction
X.sum(axis=0)  # column sums
X.mean(axis=1) # row means
np.percentile(y, 75)
```

## Dataset
- **California Housing** — [scikit-learn/california-housing on HuggingFace](https://huggingface.co/datasets/scikit-learn/california-housing)

## Learning Resources
- [NumPy docs](https://numpy.org/doc/stable/)
- [NumPy quickstart](https://numpy.org/doc/stable/user/quickstart.html)
- [Real Python: NumPy tutorial](https://realpython.com/numpy-tutorial/)
- [Stanford CS231n NumPy guide](https://cs231n.github.io/python-numpy-tutorial/)
- **Book:** *Python for Data Analysis* (Wes McKinney) Ch. 4
- **Book:** *Numerical Python* (Robert Johansson)

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
