# 2.2.4 Optimization Theory

Convexity, gradient descent, SGD mini-batch, Momentum, Adam, Newton's method — implemented from scratch.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Lagrange multipliers, constrained optimisation |
| `working_example2.py` | Convexity (Hessian PSD), GD/SGD/Momentum/Adam comparison, Newton's method |
| `working_example.ipynb` | Interactive: convexity → optimizer comparison → Newton |

## Quick Reference

```python
# Gradient Descent
for _ in range(T):
    w -= lr * grad(w)

# Adam
m = b1*m + (1-b1)*g        # 1st moment
v = b2*v + (1-b2)*g**2     # 2nd moment
m_hat = m / (1-b1**t)
v_hat = v / (1-b2**t)
w -= lr * m_hat / (sqrt(v_hat) + eps)

# Newton: x_{t+1} = x_t - f'(x)/f''(x)
x -= df(x) / d2f(x)
```

## Optimizer Summary

| Method | Pros | Cons |
|--------|------|------|
| GD | Simple, stable | Slow, needs full data |
| SGD | Fast per step | Noisy convergence |
| Momentum | Faster than GD | Overshoots |
| Adam | Adaptive lr | Memory overhead |

## Learning Resources
- [Sebastian Ruder: Overview of GD algorithms](https://www.ruder.io/optimizing-gradient-descent/)
- [CS229 Optimization Notes (Stanford)](https://cs229.stanford.edu/notes2020spring/cs229-notes1.pdf)

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
