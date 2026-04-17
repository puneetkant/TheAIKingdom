# 4.2.2 Optimizers (Detailed)

SGD → Momentum → RMSProp → Adam — adaptive learning rates from scratch.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | SGD vs Adam convergence on quadratic |
| `working_example2.py` | All 4 optimizers: quadratic bowl + moons NN |
| `working_example.ipynb` | Interactive: implement SGD/Adam → compare convergence |

## Quick Reference

```python
import torch.optim as optim

# Common optimizers
optim.SGD(params, lr=0.01, momentum=0.9)
optim.Adam(params, lr=1e-3, betas=(0.9, 0.999))
optim.RMSprop(params, lr=0.01, alpha=0.99)
optim.AdamW(params, lr=1e-3, weight_decay=1e-2)  # Adam + decoupled L2
```

## Optimizer Comparison

| Optimizer | Updates | Adaptive | Best for |
|-----------|---------|----------|----------|
| SGD | `lr*g` | No | Large batch, CV |
| Momentum | `lr*v` (EMA of g) | No | Stable landscapes |
| RMSProp | `lr*g/√s` | Per-param | RNNs |
| Adam | bias-corrected m/√v | Per-param | Most deep learning |

## Learning Resources
- [Adam paper (Kingma & Ba 2014)](https://arxiv.org/abs/1412.6980)
- [SGD vs Adam discussion](https://arxiv.org/abs/1712.07628)

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
