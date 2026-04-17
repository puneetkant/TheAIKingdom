# 4.1.1 The Neuron & Perceptron

McCulloch-Pitts neuron model, Perceptron learning rule, convergence theorem, XOR limitation.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | AND/OR/XOR truth tables with single neuron |
| `working_example2.py` | Numpy Perceptron class, linearly separable vs circles |
| `working_example.ipynb` | Interactive: perceptron training → error plot → XOR failure |

## Quick Reference

```python
class Perceptron:
    def fit(self, X, y):
        self.w = np.zeros(X.shape[1]); self.b = 0.
        for epoch in range(n_iter):
            for xi, yi in zip(X, y):
                pred = 1 if np.dot(self.w, xi) + self.b >= 0 else -1
                delta = lr * (yi - pred)
                self.w += delta * xi
                self.b += delta
```

## Key Facts

- **Convergence theorem**: Perceptron converges iff data is linearly separable
- **XOR problem**: Not linearly separable → need hidden layers
- **Step function**: $\hat{y} = \text{sign}(\mathbf{w}^T\mathbf{x} + b)$

## Learning Resources
- [sklearn Perceptron](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html)
- Minsky & Papert, *Perceptrons* (1969)

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
