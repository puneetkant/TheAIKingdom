# 4.1.5 Backpropagation

Chain rule, gradient computation for W and b, full numpy training loop.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Scalar chain rule demo, autograd comparison |
| `working_example2.py` | TwoLayerNet class with forward/backward, train on moons |
| `working_example.ipynb` | Interactive: Net class → training loop → loss plot → test acc |

## Quick Reference

```python
# Output layer (BCE + sigmoid): dL/dz2 = (a2 - y) / n
dz2 = (a2 - y) / n
dW2 = a1.T @ dz2
db2 = dz2.sum(axis=0)

# Hidden layer (ReLU): dL/dz1 = dL/da1 ⊙ ReLU'(z1)
da1 = dz2 @ W2.T
dz1 = da1 * (z1 > 0)    # ReLU derivative
dW1 = X.T @ dz1
db1 = dz1.sum(axis=0)

# Update
W -= lr * dW; b -= lr * db
```

## Backprop Equations

$$\frac{\partial L}{\partial W^{[l]}} = \frac{1}{n} A^{[l-1]T} \cdot \delta^{[l]}$$
$$\delta^{[l]} = (\delta^{[l+1]} W^{[l+1]T}) \odot g'^{[l]}(z^{[l]})$$

## Learning Resources
- [CS231n backprop notes](https://cs231n.github.io/optimization-2/)
- [3Blue1Brown: Backpropagation calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8)

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
