# 2.2.2 Multivariable Calculus

Partial derivatives, numerical gradient/Jacobian/Hessian, gradient descent on Rosenbrock surface.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Partial derivatives by hand, directional derivative |
| `working_example2.py` | Numerical gradient/Jacobian/Hessian, GD on Rosenbrock with contour plot |
| `working_example.ipynb` | Interactive: gradient → Hessian → GD contour |

## Quick Reference

```python
import numpy as np

# Numerical gradient (central difference)
def gradient(f, x, h=1e-6):
    g = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        xp, xm = x.copy(), x.copy()
        xp[i] += h; xm[i] -= h
        g[i] = (f(xp) - f(xm)) / (2 * h)
    return g

# Jacobian: J_ij = df_i/dx_j
# Hessian: H_ij = d²f/dx_i dx_j

# GD step
x -= lr * gradient(f, x)
```

## Key Concepts
| Term | Formula | ML role |
|------|---------|---------|
| Gradient | ∇f = [∂f/∂x₁, …] | GD direction |
| Jacobian | J ∈ ℝ^{m×n} | Backprop chain rule |
| Hessian | H ∈ ℝ^{n×n} | Newton's method, curvature |

## Learning Resources
- [3Blue1Brown: Partial derivatives](https://youtu.be/AXqhWeUEtQU)
- **Book:** *Mathematics for ML* Ch. 5

Implement derivatives, integrals, and optimization examples.

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
