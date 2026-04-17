# 2.3.6 Limit Theorems

Law of Large Numbers, Central Limit Theorem, bootstrap confidence intervals, Monte Carlo integration.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Proof sketch LLN, Chebyshev bound |
| `working_example2.py` | LLN running average, CLT histogram grid, bootstrap CI, MC integration |
| `working_example.ipynb` | Interactive: LLN → CLT → bootstrap → MC |

## Quick Reference

```python
import numpy as np

# LLN: running average
running = np.cumsum(samples) / np.arange(1, len(samples)+1)

# CLT: standardised sample mean ~ N(0,1) as n -> ∞
z = (means - mu) / (sigma / np.sqrt(n))

# Bootstrap 95% CI
boot = [np.random.choice(data, len(data), replace=True).mean() for _ in range(10_000)]
ci = np.percentile(boot, [2.5, 97.5])

# Monte Carlo integration: ∫ f(x) dx ≈ (1/N) Σ f(xᵢ),  xᵢ ~ Uniform
est = np.mean(f(np.random.uniform(a, b, N))) * (b - a)
se  = np.std(f(np.random.uniform(a, b, N))) / np.sqrt(N)
```

## ML Connections
- **Mini-batch SGD** relies on CLT — batch mean ≈ population gradient
- **Bootstrap** → model uncertainty / confidence intervals
- **Monte Carlo** → variational inference, MCMC

## Learning Resources
- [StatQuest: LLN & CLT](https://youtu.be/YAlJCEDH2uY)
- [MIT 18.650 Lecture notes](https://ocw.mit.edu/courses/18-650-statistics-for-applications-fall-2016/)

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
