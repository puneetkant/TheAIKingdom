# 2.1.5 Vector Spaces

Basis, rank, linear independence, Gram-Schmidt orthonormalisation, null space via SVD, feature space analysis.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Span, basis, subspace examples |
| `working_example2.py` | Independence check, Gram-Schmidt, null space, Cal Housing feature rank |
| `working_example.ipynb` | Interactive: rank → Gram-Schmidt → null space |

## Quick Reference

```python
import numpy as np

np.linalg.matrix_rank(A)           # rank (number of independent cols/rows)
U, s, Vt = np.linalg.svd(A)        # null space = Vt rows with s≈0
# Gram-Schmidt → Q with orthonormal columns
```

## Key Concepts
- **Span**: all linear combos of a set of vectors
- **Basis**: minimal spanning set (linearly independent)
- **Null space**: {x : Ax = 0} — reveals redundancy in features
- **Column space**: range of A — what outputs are reachable

## Learning Resources
- [3Blue1Brown: Linear algebra playlist](https://www.3blue1brown.com/topics/linear-algebra)
- **Book:** *Mathematics for Machine Learning* Ch. 2.4–2.6

Explore linear combinations and spans.

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
