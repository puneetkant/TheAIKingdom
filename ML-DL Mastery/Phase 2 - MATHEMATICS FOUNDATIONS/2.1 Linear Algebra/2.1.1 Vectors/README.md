# 2.1.1 Vectors

ML-focused vector operations: norms, normalisation, cosine similarity, projection, SGD gradient vectors.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Pure-Python vector ops: dot, cross, norms, angle |
| `working_example2.py` | Cal Housing: L1/L2/L∞ norms, normalisation, cosine sim, projection, SGD gradient, speed benchmark |
| `working_example.ipynb` | Interactive: norms → normalisation → projection plot → SGD loss curve |

## Run

```bash
python working_example.py
python working_example2.py   # saves output/vectors.png
jupyter lab working_example.ipynb
```

## Vector Cheat Sheet

```python
import numpy as np

a = np.array([1., 2., 3.])
b = np.array([4., 5., 6.])

# Arithmetic
a + b;  a - b;  2 * a               # element-wise
np.dot(a, b)                         # dot product  a·b = |a||b|cos θ
np.cross(a[:2], b[:2])              # cross product (2D)

# Norms
np.linalg.norm(a)          # L2 (Euclidean)
np.linalg.norm(a, 1)       # L1 (Manhattan)
np.linalg.norm(a, np.inf)  # L∞ (Chebyshev)

# Normalise to unit vector
a_hat = a / np.linalg.norm(a)

# Cosine similarity
cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Projection of b onto a
proj = (np.dot(a, b) / np.dot(a, a)) * a
```

## ML Connections
- **Feature vectors**: each data point is a vector $\mathbf{x} \in \mathbb{R}^d$
- **Weight vector**: $\mathbf{w}$ in $y = \mathbf{w} \cdot \mathbf{x} + b$
- **Gradient vector**: $\nabla_w L$ points in direction of steepest ascent
- **Cosine similarity**: text embeddings, recommendation systems

## Dataset
- **California Housing** — [scikit-learn/california-housing on HuggingFace](https://huggingface.co/datasets/scikit-learn/california-housing)

## Learning Resources
- [3Blue1Brown: Essence of Linear Algebra (YouTube)](https://www.3blue1brown.com/topics/linear-algebra)
- [Khan Academy: Vectors](https://www.khanacademy.org/math/linear-algebra/vectors-and-spaces)
- **Book:** *Mathematics for Machine Learning* (Deisenroth) Ch. 3.1–3.3 (free PDF)
- [NumPy linalg docs](https://numpy.org/doc/stable/reference/routines.linalg.html)

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
