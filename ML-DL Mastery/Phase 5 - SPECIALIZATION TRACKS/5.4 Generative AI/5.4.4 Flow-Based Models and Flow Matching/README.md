# 5.4.4 Flow-Based Models and Flow Matching

Normalizing flows, change-of-variables, affine coupling, RealNVP, Glow, flow matching (CFM).

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | RealNVP coupling layer stack |
| `working_example2.py` | Affine coupling forward/inverse on 2-D data |
| `working_example.ipynb` | Interactive: log-prob computation + flow matching path |

## Quick Reference

```python
# Change of variables
log_p(x) = log_p(z) + log |det(dz/dx)|

# Affine coupling layer (RealNVP)
# Split x into x1, x2; only transform x2
s, t = net(x1)            # scale and shift from x1
z2   = x2 * exp(s) + t   # forward
log_det = s.sum(dim=-1)   # log-Jacobian (trivial)
# Inverse: x2 = (z2 - t) * exp(-s)

# Flow matching training objective
t = Uniform(0, 1).sample()
xt = (1-t) * x0 + t * x1        # linear interpolation
vt_pred = model(xt, t)
loss = MSE(vt_pred, x1 - x0)    # match vector field
```

## Model Comparison

| Model | Exact likelihood | Invertible | Generation |
|-------|-----------------|-----------|-----------|
| RealNVP | ✓ | ✓ | Good |
| Glow | ✓ | ✓ | Good |
| FFJORD | ✓ | ✓ | Expensive |
| Flow Matching | ✗ | ✓ | Excellent |

## Learning Resources
- [RealNVP paper](https://arxiv.org/abs/1605.08803)
- [Flow Matching paper](https://arxiv.org/abs/2210.02747)

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
