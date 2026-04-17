# 2.1.3 Systems of Linear Equations

Solving Ax=b: `np.linalg.solve`, `lstsq` for over/under-determined systems, Gaussian elimination, condition numbers.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Cramer's rule, row reduction, basic examples |
| `working_example2.py` | Cal Housing OLS via lstsq, manual Gaussian elimination, condition number analysis |
| `working_example.ipynb` | Interactive: solve → OLS → condition number → underdetermined |

## Run

```bash
python working_example.py
python working_example2.py
jupyter lab working_example.ipynb
```

## Quick Reference

```python
import numpy as np

# Square system (unique solution)
x = np.linalg.solve(A, b)

# Overdetermined (n>d) — least squares
w, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)

# Check condition number
cond = np.linalg.cond(A)    # large → ill-conditioned → unstable

# Rank
np.linalg.matrix_rank(A)
```

## Case Summary
| System | n vs d | Solution |
|--------|--------|----------|
| Exact | n=d, full rank | `solve` |
| Overdetermined | n>d | `lstsq` (min ‖Ax−b‖) |
| Underdetermined | n<d | `lstsq` (min ‖x‖) |

## Learning Resources
- [Khan Academy: Systems of equations](https://www.khanacademy.org/math/algebra/x2f8bb11595b61c86:systems-of-equations)
- **Book:** *Mathematics for Machine Learning* Ch. 2.3
- [Gilbert Strang MIT 18.06](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/)

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
