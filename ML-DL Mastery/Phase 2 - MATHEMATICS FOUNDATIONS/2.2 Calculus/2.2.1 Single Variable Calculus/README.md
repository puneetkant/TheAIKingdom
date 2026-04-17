# 2.2.1 Single Variable Calculus

Numerical derivatives, chain rule, Taylor series, 1D gradient descent — essential foundations for backpropagation.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Limit definition, power rule, product rule by hand |
| `working_example2.py` | Central-difference accuracy, chain rule verification, Taylor series plot, GD convergence |
| `working_example.ipynb` | Interactive: central diff → chain rule → Taylor → GD |

## Quick Reference

```python
import numpy as np

# Central difference (2nd order accurate, O(h^2) error)
df = lambda f, x, h=1e-7: (f(x+h) - f(x-h)) / (2*h)

# Second derivative
d2f = lambda f, x, h=1e-5: (f(x+h) - 2*f(x) + f(x-h)) / h**2

# Taylor: sin(x) ≈ x - x³/6 + x⁵/120 - x⁷/5040

# 1D gradient descent
x = x_init
for _ in range(n): x -= lr * df(x)
```

## ML Connections
- **Backprop** = repeated chain rule application
- **Learning rate** = step size in gradient descent
- **Activation functions** — sigmoid, ReLU, tanh derivatives

## Learning Resources
- [3Blue1Brown: Essence of Calculus](https://youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)
- [Khan Academy: Calculus](https://www.khanacademy.org/math/calculus-1)

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
