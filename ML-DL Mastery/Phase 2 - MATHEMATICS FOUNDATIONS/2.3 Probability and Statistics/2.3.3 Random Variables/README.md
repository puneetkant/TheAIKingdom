# 2.3.3 Random Variables

Discrete and continuous RVs — PMF, PDF, CDF; expectation, variance, skewness, kurtosis.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | PMF/CDF by hand for Bernoulli and Binomial |
| `working_example2.py` | Binomial/Poisson/Gaussian simulation, Gaussian PDF+CDF plot, moments of Exponential |
| `working_example.ipynb` | Interactive: discrete → continuous PDF → CDF → moments |

## Quick Reference

```python
import numpy as np

# Discrete
np.random.binomial(n, p, size=N)
np.random.poisson(lam, size=N)

# Continuous
np.random.normal(mu, sigma, size=N)
np.random.exponential(scale, size=N)

# Moments
E = np.mean(s)
Var = np.var(s)
skew = np.mean(((s - E) / s.std())**3)
kurt = np.mean(((s - E) / s.std())**4) - 3
```

## Key Distributions

| Distribution | E[X] | Var[X] |
|-------------|------|--------|
| Bernoulli(p) | p | p(1-p) |
| Binomial(n,p) | np | np(1-p) |
| Poisson(λ) | λ | λ |
| Gaussian(μ,σ²) | μ | σ² |
| Exponential(λ) | 1/λ | 1/λ² |

## Learning Resources
- [StatQuest: Distributions](https://youtu.be/oI3hZJqXJuc)
- [Seeing Theory](https://seeing-theory.brown.edu/)

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
