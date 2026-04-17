# 2.2.3 Matrix Calculus

Gradients of matrix expressions: MSE, Ridge regression, Softmax Jacobian — numerically verified.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Trace gradient, d/dX tr(AX), by-hand Jacobian |
| `working_example2.py` | Numerical verification: d/dw‖Xw-y‖², softmax Jacobian, Ridge GD convergence |
| `working_example.ipynb` | Interactive: linear grad → softmax Jacobian → Ridge GD |

## Key Identities

$$\frac{\partial}{\partial \mathbf{w}} \|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2 = 2\mathbf{X}^T(\mathbf{X}\mathbf{w} - \mathbf{y})$$

$$\frac{\partial \text{softmax}(\mathbf{z})}{\partial \mathbf{z}} = \text{diag}(\mathbf{s}) - \mathbf{s}\mathbf{s}^T$$

## Quick Reference

```python
# Verify with central difference
def num_grad(f, w, h=1e-7):
    return [(f(w+h*e)-f(w-h*e))/(2*h) for e in np.eye(len(w))]

# MSE gradient
g = 2 * X.T @ (X @ w - y)

# Ridge gradient
g = 2 * X.T @ (X @ w - y) + 2 * lam * w

# Softmax Jacobian
J = np.diag(s) - np.outer(s, s)
```

## ML Connections
- **Backpropagation** = matrix chain rule applied layer by layer
- **Normal equations** = setting gradient = 0 → w* = (X^TX)^{-1}X^Ty
- **Softmax** gradient needed for cross-entropy backprop

## Learning Resources
- [The Matrix Calculus You Need for Deep Learning](https://explained.ai/matrix-calculus/)
- **Book:** *Deep Learning* (Goodfellow) Appendix

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
