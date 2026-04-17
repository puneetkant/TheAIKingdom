# 2.1.4 Determinants

Determinant properties, invertibility check, geometric area interpretation, 3×3 examples.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | 2×2 / 3×3 det by hand, cofactor expansion |
| `working_example2.py` | Properties, invertibility, parallelogram area visualisation |
| `working_example.ipynb` | Interactive: det properties → invertibility → area plot → 3×3 |

## Quick Reference

```python
import numpy as np

np.linalg.det(A)           # determinant
np.linalg.matrix_rank(A)   # rank check

# Properties
# det(AB) = det(A) * det(B)
# det(A.T) = det(A)
# det(cA) = c^n * det(A)
# det=0 ↔ singular ↔ columns linearly dependent
```

## Geometric meaning: area / volume scaling factor

## Learning Resources
- [3Blue1Brown: The determinant (video)](https://youtu.be/Ip3X9LOh2dk)
- **Book:** *Mathematics for Machine Learning* Ch. 4.1

Compute determinants and matrix properties.

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
